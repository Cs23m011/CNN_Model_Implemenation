Goal
Run weight-free ONNX export for the gpt_oss (20B, MoE, MXFP4) model on the transformers-5 QEfficient branch (Cs23m011/feat/weightfree_on_tf5), combining three features simultaneously: weight-free export + dynamo + ONNX subfunctions, to reduce RAM. Target: Qualcomm Cloud AI 100. Torch: 2.13.0.dev20260603+cpu (nightly, PR #182230 verified present).
Bugs fixed along the way (these are correct — keep them)
Dequant checkpoint (dequantize.py): gpt_oss ships MXFP4-quantized; weight-free needs unpacked float tensors on disk by name. Fixed by loading via QEFFAutoModelForCausalLM.from_pretrained (runs MXFP4 dequant), then saving the raw model.state_dict() instead of save_pretrained — which had silently dropped all 24 down_proj weights. Added an assert that gate_proj/up_proj/down_proj exist for every layer before saving. Confirmed complete afterward.
cache_utils.py: added device=position_ids.device / kv_position_ids.device to all ctx_indices aranges (fixed "Tensor on device meta is not on the expected device").
Split transform (base/pytorch_transforms.py ~line 218): added if fused.is_meta: continue so the gate_up→gate/up split is skipped on the meta path (the split is pre-baked into the dequant checkpoint; Case A confirmed — disk has split gate_proj/up_proj).
QEffGptOssExperts.__qeff_init__: expert_dim = getattr(self,"expert_dim",None) or self.intermediate_size (tf5 renamed expert_dim → intermediate_size).
QEffGptOssAttention.forward: unpack 4 values from past_key_value_update (key, value, attention_mask, _) — it returns 4, not 3.
QEffGptOssMLP.forward: added expert_in = expert_in.to(gate_proj.device) before the bmms (fixed meta/cpu bmm device mismatch at line ~557). This got the trace past the MoE math.
The unsolved error: arg26_1
torch.export fails with forward() missing 1 required positional argument: 'arg26_1' (later seen as 25-vs-26 operands) when the decoder layer is the nested_compile_region. The region closes over ~40 meta Parameters; one operand is dropped when the region is re-materialized.
Hypotheses tested and ruled out

Torch version / missing PR #182230 — verified present (both the kwargs wrapper and the FunctionalTensor reuse guard in gen_schema).
MoE nesting — failed even with DECODER_LAYER_PATTERNS narrowed to ["DecoderLayer"] (MoE excluded as a region).
Cache list-vs-object structure — failed after rewriting QEffHybridCacheForGPTOSS to store K/V as attributes on _GptOssHybridLayer objects (mirroring QEffDynamicCache).
Grad on captured params — failed after meta_qeff_model.model.requires_grad_(False).
MLP-as-subfunction ({QEffGptOssMLP}) — doesn't take effect because weight-free uses get_decoder_layer_classes_for_export (name-pattern matcher), not get_submodules_for_export.

Best current theory (code-traced, not confirmed on your build)
gen_schema falls through its all(isinstance(buf, FunctionalTensor)) reuse guard (your captured params are plain meta Parameters, not FunctionalTensors) → calls materialize_as_graph → which rebuilds each operand via _from_fun as torch.empty_strided(size, stride, dtype, requires_grad, device) — pure metadata, no identity. gpt_oss has many expert tensors with identical metadata (e.g. (32,2880,2880) for gate/up/down; (32,2880) biases). Two metadata-identical meta tensors can intern to the same FakeTensor on re-trace, so the graph emits one fewer placeholder than the operand count → off-by-one → arg26_1. This explains why Llama works (distinct param shapes, no collision) and gpt_oss doesn't.
Caveat that emerged last: the probe placed in gen_schema never fired — your live stack shows the crash happens at the call site of the re-materialized graph's forward, not in gen_schema's arg loop. So the probe needs to go in __call__ before super().__call__(), or in materialize_as_graph in utils.py, to capture the actual operand/placeholder counts and the duplicate shapes.
Where things stand / recommended next steps

Get the operand evidence from the correct call site (your live-stack-aware assistant is better placed than I am to position this probe). The decisive datum: is the dropped operand an expert weight with a duplicate shape, or something else?
If duplicates are expert weights → Option 1 fix: switch gpt_oss to the fused gate_up_proj representation (use the existing forward_weights_as_activation, keep gate_up_proj in __qeff_init__ instead of splitting). Fused gate_up (32,2880,5760) is distinct from down_proj (32,2880,2880), removing the within-region collision. Requires verifying the fused weight is on disk (you have gate_up_proj_bias; check gate_up_proj).
