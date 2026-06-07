# ============================================================================
# REPLACEMENT for QEffHybridCacheForGPTOSS in
#   QEfficient/transformers/cache_utils.py
#
# WHY: invoke_subgraph (subfunction/weight-free export) cannot track KV tensors
# that are stored as indexed Python-list elements and reassigned in place
# (self.key_cache[layer_idx] = ...). It DOES track tensors stored as ATTRIBUTES
# on a per-layer object that lives in a list of objects (like QEffDynamicLayer).
# This is exactly why Llama (QEffDynamicCache) exports as a decoder-layer region
# and gpt_oss (list-based) hits `arg26_1`.
#
# This rewrite keeps EVERY method's math identical, but stores K/V as
# `layer.keys` / `layer.values` attributes on `_GptOssHybridLayer` objects held
# in `self.layers`. All `self.key_cache[i]` accesses are replaced with property
# shims so any external caller still works unchanged.
# ============================================================================
grep -nE "arg26|operand|materialize_as_graph|gen_schema|placeholder|n_operands|num_operands" arg26.log | head -50
TORCHDYNAMO_VERBOSE=1 TORCH_LOGS="+dynamo,+export" python3 examples/text_generation/compare.py 2>&1 | tee arg26.log
class _GptOssHybridLayer:
    """Per-layer KV holder. Tensors are ATTRIBUTES (stable across export region
    boundary), not list elements."""

    def __init__(self):
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None


class QEffHybridCacheForGPTOSS:
    def __init__(self, config, batch_size, max_cache_len, sliding_window_len):
        self.max_cache_len = max_cache_len
        self.batch_size = batch_size
        self.sliding_window_len = sliding_window_len
        self.layers: List[_GptOssHybridLayer] = []

    # ---- compatibility shims so existing `self.key_cache[i]` / len() callers work ----
    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append(_GptOssHybridLayer())

    @property
    def key_cache(self):
        # read-only list-like view; supports len() and indexing for callers/tests
        return [layer.keys for layer in self.layers]

    @property
    def value_cache(self):
        return [layer.values for layer in self.layers]

    @classmethod
    def from_legacy_cache(
        cls, config, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "QEffHybridCacheForGPTOSS":
        cache = cls(
            config,
            batch_size=past_key_values[0][0].shape[0],
            max_cache_len=past_key_values[1][0].shape[2],
            sliding_window_len=past_key_values[0][0].shape[2],
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def __len__(self):
        return len(self.layers)

    def get_seq_length(self, layer_idx: Optional[int] = 0, cache_position: Optional[torch.LongTensor] = None) -> int:
        is_empty_layer = (
            len(self.layers) == 0
            or len(self.layers) <= layer_idx
            or self.layers[layer_idx].keys is None
            or len(self.layers[layer_idx].keys) == 0
        )
        layer_seq_length = self.layers[layer_idx].keys.shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.layers[layer_idx].keys, self.layers[layer_idx].values),)
        return legacy_cache

    def write_only(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.layers) <= layer_idx:
            self._ensure_layer(layer_idx)
            self.layers[layer_idx].keys = key_states
            self.layers[layer_idx].values = value_states
            k_out, v_out = key_states, value_states
        else:
            layer = self.layers[layer_idx]
            position_ids = cache_kwargs.get("position_ids")
            is_sliding_layer = cache_kwargs.get("is_sliding")
            _, _, ctx_len, _ = layer.keys.shape
            batch_index = cache_kwargs.get("batch_index", None)

            if is_sliding_layer:
                kv_position_ids = torch.arange(ctx_len, dtype=torch.int64, device=position_ids.device).reshape(1, -1)
            else:
                kv_position_ids = position_ids

            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)
                layer.keys = CtxScatterFuncCB.apply(layer.keys, batch_index, scatter_position_ids, key_states)
                layer.values = CtxScatterFuncCB.apply(layer.values, batch_index, scatter_position_ids, value_states)
            else:
                layer.keys = CtxScatterFunc.apply(layer.keys, kv_position_ids, key_states)
                layer.values = CtxScatterFunc.apply(layer.values, kv_position_ids, value_states)
            k_out, v_out = layer.keys, layer.values
        return k_out, v_out

    def read_only_blockedKV(
        self,
        start_idx: torch.Tensor,
        end_idx: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)

        layer = self.layers[layer_idx]
        k_out, v_out = layer.keys, layer.values
        batch, num_kv_heads, _, _ = k_out.shape

        ctx_indices = torch.arange(start=start_idx, end=end_idx, device=position_ids.device)[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit
        if torch.onnx.is_in_onnx_export():
            invalid_idx_value = torch.iinfo(torch.int32).max
        else:
            invalid_idx_value = 0
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_out = CtxGatherFuncBlockedKVCB.apply(k_out, batch_index, ctx_indices)
            v_out = CtxGatherFuncBlockedKVCB.apply(v_out, batch_index, ctx_indices)
        else:
            ctx_indices = ctx_indices.expand(batch, num_kv_heads, ctx_indices.shape[-1])
            k_out = CtxGatherFuncBlockedKV.apply(k_out, ctx_indices)
            v_out = CtxGatherFuncBlockedKV.apply(v_out, ctx_indices)

        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out), v_out)
        return k_out, v_out

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.layers) <= layer_idx:
            self._ensure_layer(layer_idx)
            self.layers[layer_idx].keys = key_states
            self.layers[layer_idx].values = value_states
            k_out, v_out = key_states, value_states
        else:
            layer = self.layers[layer_idx]
            position_ids = cache_kwargs.get("position_ids")
            is_sliding_layer = cache_kwargs.get("is_sliding")
            sliding_window = cache_kwargs.get("sliding_window")
            batch_index = cache_kwargs.get("batch_index", None)

            if is_sliding_layer:
                kv_position_ids = torch.where(
                    position_ids == -1,
                    position_ids,
                    _remainder_with_symbolic_divisor(position_ids, sliding_window),
                )
            else:
                kv_position_ids = position_ids

            if batch_index is not None:
                if torch.onnx.is_in_onnx_export():
                    invalid_scatter_index = torch.iinfo(torch.int32).max
                    scatter_position_ids = torch.where(kv_position_ids < 0, invalid_scatter_index, kv_position_ids)
                else:
                    scatter_position_ids = kv_position_ids

                ctx_scatter_cb_interface = select_interface(
                    CtxScatterFuncCB.apply,
                    torch.ops.qefficient.ctx_scatter_cb,
                )
                layer.keys = ctx_scatter_cb_interface(layer.keys, batch_index, scatter_position_ids, key_states)
                layer.values = ctx_scatter_cb_interface(layer.values, batch_index, scatter_position_ids, value_states)
            else:
                ctx_scatter_interface = select_interface(
                    CtxScatterFunc.apply,
                    torch.ops.qefficient.ctx_scatter,
                )
                layer.keys = ctx_scatter_interface(layer.keys, kv_position_ids, key_states)
                layer.values = ctx_scatter_interface(layer.values, kv_position_ids, value_states)

            k_out, v_out = layer.keys, layer.values

            # Original Gather
            if is_sliding_layer:
                ctx_len = layer.keys.shape[2]
            else:
                ctx_len = cache_kwargs.get("CCL", layer.keys.shape[2])

            ctx_indices = torch.arange(ctx_len, device=position_ids.device)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit
            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = torch.iinfo(torch.int32).max
            else:
                invalid_idx_value = 0
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            if batch_index is not None:
                ctx_gather_cb_interface = select_interface(
                    CtxGatherFuncCB.apply,
                    torch.ops.qefficient.ctx_gather_cb,
                )
                k_out = ctx_gather_cb_interface(k_out, batch_index, ctx_indices, ctx_len)
                v_out = ctx_gather_cb_interface(v_out, batch_index, ctx_indices, ctx_len)
            else:
                ctx_gather_interface = select_interface(
                    CtxGatherFunc.apply,
                    torch.ops.qefficient.ctx_gather,
                )
                k_out = ctx_gather_interface(k_out, ctx_indices, ctx_len)
                v_out = ctx_gather_interface(v_out, ctx_indices, ctx_len)

            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out), v_out)
        return k_out, v_out

    def full_cache_update_chunked(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers[layer_idx]
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index")
        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()

        if batch_index is not None:
            if torch.onnx.is_in_onnx_export():
                scatter_position_ids = torch.where(position_ids < 0, torch.iinfo(torch.int32).max, position_ids)
            layer.keys = CtxScatterFuncCB.apply(layer.keys, batch_index, scatter_position_ids, key_states)
            layer.values = CtxScatterFuncCB.apply(layer.values, batch_index, scatter_position_ids, value_states)
        else:
            layer.keys = CtxScatterFunc.apply(layer.keys, position_ids, key_states)
            layer.values = CtxScatterFunc.apply(layer.values, position_ids, value_states)

        k_out, v_out = layer.keys, layer.values

        ctx_len = cache_kwargs.get("CCL", k_out.shape[2])
        ctx_indices = torch.arange(ctx_len, device=position_ids.device)[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
        if batch_index is not None:
            k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices, ctx_len)
            v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices, ctx_len)
        else:
            k_out = CtxGatherFunc.apply(k_out, ctx_indices, ctx_len)
            v_out = CtxGatherFunc.apply(v_out, ctx_indices, ctx_len)
        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out), v_out)

        return k_out, v_out

    def sliding_window_update_chunked(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers[layer_idx]
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index")
        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()

        if batch_index is not None:
            if torch.onnx.is_in_onnx_export():
                scatter_position_ids = torch.where(position_ids < 0, torch.iinfo(torch.int32).max, position_ids)
            layer.keys = CtxScatterFuncCB.apply(layer.keys, batch_index, scatter_position_ids, key_states)
            layer.values = CtxScatterFuncCB.apply(layer.values, batch_index, scatter_position_ids, value_states)
        else:
            layer.keys = CtxScatterFunc.apply(layer.keys, position_ids, key_states)
            layer.values = CtxScatterFunc.apply(layer.values, position_ids, value_states)

        k_out, v_out = layer.keys, layer.values
        sliding_window_len = cache_kwargs.get("sliding_window")

        ctx_len = position_ids.shape[1] + sliding_window_len
        ctx_indices = torch.arange(ctx_len, device=position_ids.device)[None, None, ...]
        first_pos_idx = position_ids[0][0]
        add_idx = torch.where(first_pos_idx >= sliding_window_len, first_pos_idx - sliding_window_len, 0)
        ctx_indices += add_idx
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
        if batch_index is not None:
            k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices, ctx_len)
            v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices, ctx_len)
        else:
            k_out = CtxGatherFunc.apply(k_out, ctx_indices, ctx_len)
            v_out = CtxGatherFunc.apply(v_out, ctx_indices, ctx_len)
        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(v_out), v_out)

        return k_out, v_out
