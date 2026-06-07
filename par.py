


# gpt_oss Weight-Free Export — Debugging Handoff > **Purpose of this file:** Hand this to a fresh Claude session (or to a colleague / `vbaddi`) > so they can continue without re-deriving everything. It records the goal, the bugs already > fixed (keep these), the one unsolved error, every hypothesis

pasted




How can I help you today?

Pasted content
13.44 KB •246 lines
•
Formatting may be inconsistent from source

# gpt_oss Weight-Free Export — Debugging Handoff

> **Purpose of this file:** Hand this to a fresh Claude session (or to a colleague / `vbaddi`)
> so they can continue without re-deriving everything. It records the goal, the bugs already
> fixed (keep these), the one unsolved error, every hypothesis ruled out, the current best
> theory, and the exact next steps.

---

## 1. Goal & Environment

**Goal:** Run **weight-free ONNX export** for the **gpt_oss 20B** model (MoE, ships MXFP4-quantized)
on the transformers-5 QEfficient branch, combining three features at once:
**weight-free + dynamo + ONNX subfunctions**, to cut RAM. Target hardware: Qualcomm Cloud AI 100.

**Environment:**
- Repo: `/home/amarshar/weightfree-tf5` (clone of `Cs23m011/efficient-transformers`, branch `feat/weightfree_on_tf5`)
- transformers: **5.5.4**
- torch: **2.13.0.dev20260603+cpu** (nightly). **PR #182230 verified present** (both the kwargs
  wrapper in `invoke_subgraph_placeholder` AND the `FunctionalTensor` reuse guard in `gen_schema`).
- Dequant checkpoint dir: `gpt-oss-20b-dequant/` (relative to repo root)
- Run script: `examples/text_generation/compare.py` (export call ~line 208)
- JIRA: QRANIUMSW-61960 ; feature branch `feat/weightfree_on_tf5`

**How weight-free works (mental model):** a META copy of the model is traced; weight initializers
are stripped from the ONNX into a `weight_spec.json` that points at on-disk checkpoint tensors
**by name**. So every weight the graph references must exist on disk under the matching name, and
anything that must be *computed* (MXFP4 unpack, gate/up split) has to be baked into the on-disk
checkpoint offline — it cannot happen during meta tracing (no data).

---

## 2. Bugs already fixed (KEEP THESE — they are correct)

### 2.1 Dequant script (`dequantize.py`)
- Load via `QEFFAutoModelForCausalLM.from_pretrained(model)` (this runs MXFP4 dequant), cast to
  bf16/fp32, strip `quantization_config`.
- **CRITICAL:** save via raw `model.state_dict()` + `safetensors.save_file` (sharded), **NOT**
  `save_pretrained` — `save_pretrained` silently dropped all 24 `experts.down_proj` weights
  (kept `down_proj_bias`). Added a hard `assert` that `experts.gate_proj`/`up_proj`/`down_proj`
  exist for every layer before writing.
- Confirmed after fix: split `gate_proj`/`up_proj` AND `down_proj` present for all 24 layers
  (this is "Case A" — disk already has the split tensors, so weight-free resolves them by name).

### 2.2 `QEfficient/transformers/cache_utils.py`
- Added `device=position_ids.device` / `device=kv_position_ids.device` to all `torch.arange`
  `ctx_indices` sites (~14 places). Fixed initial "Tensor on device meta is not on the expected device".

### 2.3 `QEfficient/base/pytorch_transforms.py` (split transform, ~line 218)
- Added `if fused.is_meta: continue` guard before `experts.gate_proj.data.copy_(gate)`.
  The gate_up→gate/up split is pre-baked into the dequant checkpoint, so on the meta path the
  copy is both impossible (no data) and unnecessary.

### 2.4 `QEffGptOssExperts.__qeff_init__`
- `self.expert_dim = getattr(self, "expert_dim", None) or self.intermediate_size`
  (tf5's `GptOssExperts` exposes `intermediate_size`, not `expert_dim`; they're equal).

### 2.5 `QEffGptOssAttention.forward`
- Unpack **4** values from `past_key_value_update`:
  `key_states, value_states, attention_mask, _ = past_key_value_update(...)`
  (it returns `key, value, attention_mask, cache_kwargs` — 4, not 3).

### 2.6 `QEffGptOssMLP.forward` (the active separate-gate/up version)
- Added `expert_in = expert_in.to(gate_proj.device)` before the two `torch.bmm` calls
  (~line 557). Fixed a meta/cpu device mismatch in the MoE expert bmm. **This got the trace
  past the MoE math** — important progress.

---

## 3. The unsolved error: `arg26_1`

```
torch.onnx._internal.exporter._errors.TorchExportError: Failed to export ... step 1/3
TypeError: forward() missing 1 required positional argument: 'arg26_1'
```

- Occurs when the **decoder layer** (`QEffGptOssDecoderLayer`) is the `nested_compile_region`
  (subfunction unit) during weight-free + subfunction export.
- The region closes over **~40 meta `Parameter`s** as operands. During region re-materialization,
  **one operand is dropped** → the materialized graph's `forward` expects N args but is called
  with N−1 (seen as both "missing arg26_1" and later "25 vs 26 operands").
- The dropped index varies by trace pass (`arg26` was `q_proj.bias (4096,)` on one pass,
  `down_proj_bias (32,2880)` on another) — this is **per-subgraph arg renumbering**, i.e.
  deterministic per subgraph, NOT random. (I earlier mislabeled this "non-deterministic" — corrected.)

---

## 4. Hypotheses TESTED AND RULED OUT (don't re-try these)

| Hypothesis | How tested | Result |
|---|---|---|
| Missing torch PR #182230 | Inspected installed `invoke_subgraph.py` source | **Present** (kwargs wrapper + FunctionalTensor reuse guard). Not the cause. |
| MoE nesting (region-in-region) | Narrowed `DECODER_LAYER_PATTERNS` to `["DecoderLayer"]` so MoE isn't a region | Still fails. Not nesting. |
| Cache stored as indexed lists | Rewrote `QEffHybridCacheForGPTOSS` to store K/V as attributes on `_GptOssHybridLayer` objects (like `QEffDynamicCache`) | Still fails. Not the cache structure. |
| Grad on captured params | `meta_qeff_model.model.requires_grad_(False)` before export | Still fails. Not grad. |
| MLP as subfunction | Set `get_submodules_for_export → {QEffGptOssMLP}` | No effect — weight-free uses `get_decoder_layer_classes_for_export` (name-pattern matcher), NOT `get_submodules_for_export`. |

**Key discovery:** weight-free region selection is driven by
`get_decoder_layer_classes_for_export(model)` in
`QEfficient/transformers/models/pytorch_transforms.py`, which pattern-matches QEff class names
against `DECODER_LAYER_PATTERNS = ["DecoderLayer", "Block", "Layer"]`. It does **not** read the
model's `get_submodules_for_export` method. Region is enabled in
`QEfficient/exporter/weight_free.py` ~lines 239-241 via
`temporarily_enable_nested_compile_regions(meta_qeff_model.model, decoder_layer_classes)`.

---

## 5. Current best theory (code-traced; NOT yet confirmed on the live build)

`gen_schema` (in `torch/_higher_order_ops/invoke_subgraph.py`) has a reuse guard:

```python
bufs = list(candidate.buffers()) if isinstance(candidate, GraphModule) else []
if bufs and all(isinstance(buf, FunctionalTensor) for buf in bufs):
    gm = candidate            # reuse path (safe)
if gm is None:
    gm = materialize_as_graph(subgraph, operands, ...)   # fallback path (suspected broken)
```

For weight-free, the captured weights are plain **meta `Parameter`s**, not `FunctionalTensor`s,
so the `all(...)` guard is False → falls through to `materialize_as_graph`. That function rebuilds
each operand via `_from_fun`:

```python
return torch.empty_strided(t.size(), t.stride(), dtype=t.dtype,
                           requires_grad=t.requires_grad, device=t.device)
```

i.e. **pure metadata, no identity**. gpt_oss has many expert tensors with *identical* metadata
(e.g. `gate_proj`/`up_proj`/`down_proj` all `(32,2880,2880)`; biases all `(32,2880)`). Two
metadata-identical meta tensors can intern to the **same FakeTensor** on re-trace, so the
re-materialized graph emits **one fewer placeholder** than the operand count → off-by-one → `arg26_1`.

This explains why **Llama works** (its per-layer params are all distinct shapes → no collision)
and **gpt_oss fails** (MoE produces many identical-shaped expert tensors, and weight-free makes
them all meta with no data to disambiguate).

### IMPORTANT correction discovered last
A probe placed in `gen_schema` (after the `materialize_as_graph` call, comparing
`len(operands)` vs placeholder count) **never fired** — the live stack shows the crash happens at
the **call site of the re-materialized graph's `forward`**, during the export trace, *before/around*
`gen_schema`'s arg loop. So the probe must be placed earlier:
- inside `InvokeSubgraphHOP.__call__` before `super().__call__()`, OR
- inside `materialize_as_graph` in `torch/_higher_order_ops/utils.py`, right after `gm` is built,
  comparing placeholder count to `len(args)` and printing duplicate operand shapes.

---

## 6. NEXT STEPS (in priority order)

### Step 1 — Get ground-truth operand evidence (do this first)
Place a probe at the **correct** site (NOT `gen_schema` — that never fires). Best location:
inside `materialize_as_graph` in `torch/_higher_order_ops/utils.py` right after `gm` is built:

```python
    gm = _materialize_as_graph_inner()
    if gm is None:
        raise AssertionError("materialize_as_graph returned None")
    # --- PROBE ---
    import collections as _c
    _nph = sum(1 for _n in gm.graph.nodes if _n.op == "placeholder")
    _nop = len(args)
    if _nph != _nop:
        _shapes = [tuple(getattr(a, "shape", ())) for a in args]
        _dupes = [s for s, c in _c.Counter(_shapes).items() if c > 1 and s]
        print(f"[PROBE] operands={_nop} placeholders={_nph} MISMATCH dupes={_dupes}")
    else:
        print(f"[PROBE] operands={_nop} placeholders={_nph} OK")
    # --- END PROBE ---
    return gm
```

Run: `python3 examples/text_generation/compare.py 2>&1 | grep "\[PROBE\]"`

**Decision from probe output:**
- `MISMATCH` + `dupes` are expert shapes `(32,2880,2880)`/`(32,2880)` → **Step 2 (Option 1)** is correct.
- `MISMATCH` + dupes include non-expert shapes (e.g. `(4096,)`) → fusing experts alone won't fully fix it.
- `OK` everywhere but still crashes → theory wrong; the drop is structural, escalate to `vbaddi`.

### Step 2 — Option 1 fix (IF probe confirms expert-shape duplicates)
Switch gpt_oss experts to the **fused `gate_up_proj`** representation so no two captured operands
share identical metadata. Fused gate_up is `(32, 2880, 5760)` — distinct from down_proj `(32,2880,2880)`.
- In `QEffGptOssExperts.__qeff_init__`: keep the inherited fused `gate_up_proj`; do **not** create
  split `gate_proj`/`up_proj`.
- In `QEffGptOssMLP.forward`: use the existing `forward_weights_as_activation` method (it already
  consumes fused `gate_up_proj`/`gate_up_proj_bias` and splits gate/up as *activations*:
  `gate, up = gate_up[..., ::2], gate_up[..., 1::2]`).
- **Verify the fused WEIGHT is on disk:**
  ```bash
  python - <<'PY'
  from safetensors import safe_open; import glob
  n=0
  for st in glob.glob("gpt-oss-20b-dequant/*.safetensors"):
      with safe_open(st, framework="pt") as f:
          for k in f.keys():
              if k.endswith("experts.gate_up_proj"): n+=1
  print("gate_up_proj weight tensors:", n, "(expect 24)")
  PY
  ```
  (We know `gate_up_proj_bias` is on disk; must confirm the weight. If 0, re-dequant keeping fused.)
- After export succeeds, **validate numerically** against `use_onnx_subfunctions=False` on the same
  input — a clean export with wrong weights is worse than a loud failure.

### Step 3 — Working fallback available NOW
`use_onnx_subfunctions=False` → weight-free + dynamo (the original RAM goal) works today.
Only loses decoder-layer subfunction dedup (the increment being chased).

### Step 4 — Escalate to branch author `vbaddi` (he wrote the region machinery AND PR #182230)
Precise question:
> gpt_oss weight-free + decoder-layer `nested_compile_region` fails with
> `forward() missing arg26_1` — an operand is dropped during region re-materialization
> (25 vs 26). PR #182230 verified present. Fails identically with MoE as a non-region,
> after restructuring `QEffHybridCacheForGPTOSS` to object-attribute storage like
> `QEffDynamicCache`, and after `model.requires_grad_(False)`. Llama (`QEffDynamicCache`,
> distinct param shapes) exports fine as a decoder-layer region; gpt_oss has many
> identical-shaped expert params (`(32,2880,2880)`, `(32,2880)`) that may be coalescing
> in `materialize_as_graph` / `_from_fun`. Is decoder-layer subfunctioning expected to
> work for gpt_oss yet, or should it use an MLP-level region / no subfunctions?

---

## 7. Key file paths
- `QEfficient/transformers/cache_utils.py` — device fixes; `QEffHybridCacheForGPTOSS`
- `QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py` — `QEffGptOssExperts.__qeff_init__`,
  `QEffGptOssMLP.forward` / `forward_weights_as_activation`, `QEffGptOssAttention.forward`,
  `QEffGptOssForCausalLM.get_submodules_for_export`
- `QEfficient/base/pytorch_transforms.py` — gate/up split transform (~line 218, `is_meta` guard)
- `QEfficient/transformers/models/pytorch_transforms.py` — `get_decoder_layer_classes_for_export`,
  `DECODER_LAYER_PATTERNS`, `temporarily_enable_nested_compile_regions`
- `QEfficient/exporter/weight_free.py` — export call (~line 244-273), `_build_meta_qeff_model`,
  `_promote_initializers_and_build_spec`
- `examples/text_generation/compare.py` — run script (export ~line 208)
- Installed torch internals to probe:
  `torch/_higher_order_ops/invoke_subgraph.py` (`gen_schema`, `InvokeSubgraphHOP.__call__`),
  `torch/_higher_order_ops/utils.py` (`materialize_as_graph`, `_from_fun`)

---

## 8. Honest status
A long chain of real, distinct bugs was fixed; the trace now runs through construction, the full
attention stack, and the MoE math. The remaining `arg26_1` is a torch-export / QEfficient-integration
issue at a depth where **live-stack access or the branch author resolves it faster than reasoning
from pasted code**. The fused-expert change (Option 1) is the most promising self-serve fix **if**
the operand probe (placed at the correct site — `materialize_as_graph` or `__call__`, NOT `gen_schema`)
confirms expert-shape duplicates. Otherwise: `use_onnx_subfunctions=False` to ship today, and escalate
the decoder-layer increment to `vbaddi`.

Mr E
