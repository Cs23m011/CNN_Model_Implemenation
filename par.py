python3 /home/amarshar/weightfree-tf5/examples/text_generation/dynamo.py 
`torch_dtype` is deprecated! Use `dtype` instead!
GlmMoeDsaConfig {
  "architectures": [
    "GlmMoeDsaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "dtype": "float32",
  "eos_token_id": [
    154820,
    154827,
    154829
  ],
  "ep_size": 1,
  "first_k_dense_replace": 1,
  "hidden_act": "silu",
  "hidden_size": 8,
  "index_head_dim": 128,
  "index_n_heads": 4,
  "index_topk": 2048,
  "indexer_rope_interleave": true,
  "initializer_range": 0.02,
  "intermediate_size": 32,
  "kv_lora_rank": 512,
  "max_position_embeddings": 202752,
  "mlp_layer_types": [
    "dense",
    "sparse"
  ],
  "model_type": "glm_moe_dsa",
  "moe_intermediate_size": 32,
  "moe_layer_freq": 1,
  "n_group": 1,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 8,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 2,
  "num_key_value_heads": 8,
  "num_nextn_predict_layers": 1,
  "pad_token_id": 154820,
  "pretraining_tp": 1,
  "q_lora_rank": 32,
  "qk_head_dim": 256,
  "qk_nope_head_dim": 192,
  "qk_rope_head_dim": 64,
  "rms_norm_eps": 1e-05,
  "rope_interleave": true,
  "rope_parameters": {
    "rope_theta": 1000000,
    "rope_type": "default"
  },
  "routed_scaling_factor": 2.5,
  "scoring_func": "sigmoid",
  "tie_word_embeddings": false,
  "topk_group": 1,
  "topk_method": "noaux_tc",
  "transformers_version": "5.5.4",
  "use_cache": true,
  "v_head_dim": 256,
  "vocab_size": 154880
}

WARNING - QEfficient - Setting tokenizer padding_side to 'right', got left

--- Original HF Model Outputs (Torch CPU) ---
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████| 41/41 [00:00<00:00, 763.04it/s]
GlmMoeDsaForCausalLM LOAD REPORT from: tiny-random/glm-5.1
Key                                                  | Status     |  | 
-----------------------------------------------------+------------+--+-
model.layers.2.mlp.experts.gate_up_proj              | UNEXPECTED |  | 
model.layers.2.enorm.weight                          | UNEXPECTED |  | 
model.layers.2.mlp.gate.weight                       | UNEXPECTED |  | 
model.layers.2.self_attn.kv_a_proj_with_mqa.weight   | UNEXPECTED |  | 
model.layers.2.mlp.shared_experts.down_proj.weight   | UNEXPECTED |  | 
model.layers.2.self_attn.indexer.weights_proj.weight | UNEXPECTED |  | 
model.layers.2.self_attn.indexer.k_norm.weight       | UNEXPECTED |  | 
model.layers.2.mlp.experts.down_proj                 | UNEXPECTED |  | 
model.layers.2.self_attn.q_a_layernorm.weight        | UNEXPECTED |  | 
model.layers.2.input_layernorm.weight                | UNEXPECTED |  | 
model.layers.2.hnorm.weight                          | UNEXPECTED |  | 
model.layers.2.self_attn.indexer.k_norm.bias         | UNEXPECTED |  | 
model.layers.2.shared_head.norm.weight               | UNEXPECTED |  | 
model.layers.2.self_attn.kv_a_layernorm.weight       | UNEXPECTED |  | 
model.layers.2.self_attn.q_b_proj.weight             | UNEXPECTED |  | 
model.layers.2.self_attn.indexer.wk.weight           | UNEXPECTED |  | 
model.layers.2.mlp.shared_experts.gate_proj.weight   | UNEXPECTED |  | 
model.layers.2.post_attention_layernorm.weight       | UNEXPECTED |  | 
model.layers.2.self_attn.o_proj.weight               | UNEXPECTED |  | 
model.layers.2.eh_proj.weight                        | UNEXPECTED |  | 
model.layers.2.self_attn.q_a_proj.weight             | UNEXPECTED |  | 
model.layers.2.self_attn.kv_b_proj.weight            | UNEXPECTED |  | 
model.layers.2.self_attn.indexer.wq_b.weight         | UNEXPECTED |  | 
model.layers.2.mlp.gate.e_score_correction_bias      | UNEXPECTED |  | 
model.layers.2.mlp.shared_experts.up_proj.weight     | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Original HF Model Outputs (Torch CPU): 

Prompt: ['My name is']
Completion: ' Coun Coun Coun Coun Coun Coun Coun Coun Coun CounGets Coun maxSize maxSizesto주세요주세요주세요주세요주세요주세요sto maxSize maxSize'
[31123 31123 31123 31123 31123 31123 31123 31123 31123 31123 49041 31123
 61501 61501 32895 90901 90901 90901 90901 90901 90901 32895 61501 61501]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████| 41/41 [00:00<00:00, 838.49it/s]
GlmMoeDsaForCausalLM LOAD REPORT from: tiny-random/glm-5.1
Key                                                  | Status     |  | 
-----------------------------------------------------+------------+--+-
model.layers.2.mlp.experts.gate_up_proj              | UNEXPECTED |  | 
model.layers.2.enorm.weight                          | UNEXPECTED |  | 
model.layers.2.mlp.gate.weight                       | UNEXPECTED |  | 
model.layers.2.self_attn.kv_a_proj_with_mqa.weight   | UNEXPECTED |  | 
model.layers.2.mlp.shared_experts.down_proj.weight   | UNEXPECTED |  | 
model.layers.2.self_attn.indexer.weights_proj.weight | UNEXPECTED |  | 
model.layers.2.self_attn.indexer.k_norm.weight       | UNEXPECTED |  | 
model.layers.2.mlp.experts.down_proj                 | UNEXPECTED |  | 
model.layers.2.self_attn.q_a_layernorm.weight        | UNEXPECTED |  | 
model.layers.2.input_layernorm.weight                | UNEXPECTED |  | 
model.layers.2.hnorm.weight                          | UNEXPECTED |  | 
model.layers.2.self_attn.indexer.k_norm.bias         | UNEXPECTED |  | 
model.layers.2.shared_head.norm.weight               | UNEXPECTED |  | 
model.layers.2.self_attn.kv_a_layernorm.weight       | UNEXPECTED |  | 
model.layers.2.self_attn.q_b_proj.weight             | UNEXPECTED |  | 
model.layers.2.self_attn.indexer.wk.weight           | UNEXPECTED |  | 
model.layers.2.mlp.shared_experts.gate_proj.weight   | UNEXPECTED |  | 
model.layers.2.post_attention_layernorm.weight       | UNEXPECTED |  | 
model.layers.2.self_attn.o_proj.weight               | UNEXPECTED |  | 
model.layers.2.eh_proj.weight                        | UNEXPECTED |  | 
model.layers.2.self_attn.q_a_proj.weight             | UNEXPECTED |  | 
model.layers.2.self_attn.kv_b_proj.weight            | UNEXPECTED |  | 
model.layers.2.self_attn.indexer.wq_b.weight         | UNEXPECTED |  | 
model.layers.2.mlp.gate.e_score_correction_bias      | UNEXPECTED |  | 
model.layers.2.mlp.shared_experts.up_proj.weight     | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

--- QEff Transformed HF Model Outputs (Torch CPU) ---
QEff Transformed HF Model Outputs (Torch CPU): 

Prompt: ['My name is']
Completion: ['hitcitycitycitycitycitycity iPad iPad iPad iPad iPad iPad iPad iPadӰӰӰ总书记总书记 iPad总书记 iPad总书记']
[[ 22433   8923   8923   8923   8923   8923   8923  22214  22214  22214
   22214  22214  22214  22214  22214 152900 152900 152900 103907 103907
   22214 103907  22214 103907]]
[Warning]: The subfunction feature is experimental. Please note that using compile consecutively with and without subfunction may produce inconsistent results.
[torch.onnx] Obtain model graph for `QEffGlmMoeDsaForCausalLM([...]` with `torch.export.export(..., strict=False)`...
[Warning]: While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: ["L['kwargs']['past_key_values'].layers[0]"]
[Warning]: While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: ["L['kwargs']['past_key_values'].layers[1]"]
[torch.onnx] Obtain model graph for `QEffGlmMoeDsaForCausalLM([...]` with `torch.export.export(..., strict=False)`... ❌
[torch.onnx] Obtain model graph for `QEffGlmMoeDsaForCausalLM([...]` with `torch.export.export(..., strict=True)`...
[torch.onnx] Obtain model graph for `QEffGlmMoeDsaForCausalLM([...]` with `torch.export.export(..., strict=True)`... ✅
[torch.onnx] Run decompositions...
[Warning]: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
[torch.onnx] Run decompositions... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
[Warning]: # The axis name: batch_size will not be used, since it shares the same shape constraints with another axis: batch_size.
[Warning]: # The axis name: seq_len will not be used, since it shares the same shape constraints with another axis: seq_len.
[Warning]: # The axis name: ctx_len will not be used, since it shares the same shape constraints with another axis: ctx_len.
WARNING - QEfficient.base.modeling_qeff - Weight clearing failed, continuing: Cannot swap t1 because it has weakref associated with it
[TIMING] qeff_model.export: 25.656 seconds
[MEMORY] export peak RSS: 972.05 MB
[ARTIFACT] onnx_path=/home/amarshar/efficient-transformers/GlmMoeDsaForCausalLM/GlmMoeDsaForCausalLM-8fdf7c1bd2f0ac74/GlmMoeDsaForCausalLM.onnx

QEFFICIENT PERFORMANCE MONITORING REPORT
============================================================
Peak Memory Usage:
   • RSS (Physical): 972.05 MB at 01:08:38
   • VMS (Virtual):  16396.48 MB at 01:08:36
   • Peak during:    Export

Memory Statistics:
   • Current RSS:    972.05 MB (Delta: +7.86 MB)
   • Current VMS:    16396.48 MB (Delta: +115.56 MB)
   • Average RSS:    970.58 MB
   • Min/Max RSS:    964.19 / 972.05 MB
   • Memory Range:   7.86 MB
Disk I/O Statistics:
   • Total Read:     59.04 MB
   • Total Write:    56.92 MB
   • Peak Read Rate: 75.04 MB/s
   • Peak Write Rate:132.10 MB/s
   • Avg Read Rate:  5.03 MB/s
   • Avg Write Rate: 6.61 MB/s

Monitoring Info:
   • Duration:       25.6 seconds
   • Data Points:    20
   • Operations:     2
   • Sampling Rate:  0.05s

QEfficient Operations Timeline:
    1.    0.0s - Export (25.7s) 
    2.   25.7s - Completion  
[MEMORY] export profile graph saved to: /home/amarshar/weightfree-tf5/examples/text_generation/export_memory_profile.png

--- QEff Transformed Onnx Model Outputs (OnnxRuntime CPU) ---
QEff Transformed Onnx Model Outputs (OnnxRuntime CPU): 

Prompt: ['My name is']
Completion: ['hitcitycitycitycitycitycity iPad iPad iPad iPad iPad iPad iPad iPadӰӰӰ总书记总书记 iPad总书记 iPad总书记']
[[ 22433   8923   8923   8923   8923   8923   8923  22214  22214  22214
   22214  22214  22214  22214  22214 152900 152900 152900 103907 103907
   22214 103907  22214 103907]]
['/opt/qti-aic/exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100', '-m=/home/amarshar/efficient-transformers/GlmMoeDsaForCausalLM/GlmMoeDsaForCausalLM-8fdf7c1bd2f0ac74/GlmMoeDsaForCausalLM.onnx', '-retained-state', '-convert-to-fp16', '-aic-num-cores=16', '-sub-functions', '-network-specialization-config=/home/amarshar/efficient-transformers/GlmMoeDsaForCausalLM/GlmMoeDsaForCausalLM-8fdf7c1bd2f0ac74/qpc-9ba00b62a25037be/specializations.json', '-custom-IO-list-file=/home/amarshar/efficient-transformers/GlmMoeDsaForCausalLM/GlmMoeDsaForCausalLM-8fdf7c1bd2f0ac74/qpc-9ba00b62a25037be/custom_io.yaml', '-aic-binary-dir=/home/amarshar/efficient-transformers/GlmMoeDsaForCausalLM/GlmMoeDsaForCausalLM-8fdf7c1bd2f0ac74/qpc-9ba00b62a25037be/qpc']
[TIMING] qeff_model.compile: 13.969 seconds
[ARTIFACT] qpc_path=/home/amarshar/efficient-transformers/GlmMoeDsaForCausalLM/GlmMoeDsaForCausalLM-8fdf7c1bd2f0ac74/qpc-9ba00b62a25037be/qpc
compile done
QEff Transformed Onnx Model Outputs(AIC Backend)

Prompt : My name is
Completion :hitcitycitycitycitycitycity iPad iPad iPad iPad iPad iPad iPad iPadӰӰӰ总书记总书记 iPad总书记 iPad总书记 iPad总书记 iPad总书记input= ['My name is']
output= [['hitcitycitycitycitycitycity iPad iPad iPad iPad iPad iPad iPad iPadӰӰӰ总书记总书记 iPad总书记 iPad总书记 iPad总书记 iPad总书记 iPad']]
Average Prefill time a.k.a TTFT is= 0.0 sec        
Decode is= 633.95 tokens/sec        
Total is= 580.38 tokens/sec        
Total (E2E) inference time is= 0.05 sec
Average Prefill time a.k.a TTFT is= 0.0 sec        
Decode is= 633.95 tokens/sec        
Total is= 580.38 tokens/sec        
Total (E2E) inference time is= 0.05 sec
[array([[ 22433,   8923,   8923,   8923,   8923,   8923,   8923,  22214,
         22214,  22214,  22214,  22214,  22214,  22214,  22214, 152900,
        152900, 152900, 103907, 103907,  22214, 103907,  22214, 103907,
         22214, 103907,  22214, 103907,  22214, 154820, 154820, 154820]])]
[COMPARE] original lengths: {'hf_tokens': 24, 'pt_tokens': 24, 'ort_tokens': 24, 'aic_generated_ids': 32}
[COMPARE] trimmed length used: 24
[COMPARE] trimmed outputs together:
  hf_tokens:  [31123, 31123, 31123, 31123, 31123, 31123, 31123, 31123, 31123, 31123, 49041, 31123, 61501, 61501, 32895, 90901, 90901, 90901, 90901, 90901, 90901, 32895, 61501, 61501]
  pt_tokens:  [22433, 8923, 8923, 8923, 8923, 8923, 8923, 22214, 22214, 22214, 22214, 22214, 22214, 22214, 22214, 152900, 152900, 152900, 103907, 103907, 22214, 103907, 22214, 103907]
  ort_tokens: [22433, 8923, 8923, 8923, 8923, 8923, 8923, 22214, 22214, 22214, 22214, 22214, 22214, 22214, 22214, 152900, 152900, 152900, 103907, 103907, 22214, 103907, 22214, 103907]
  aic_tokens: [22433, 8923, 8923, 8923, 8923, 8923, 8923, 22214, 22214, 22214, 22214, 22214, 22214, 22214, 22214, 152900, 152900, 152900, 103907, 103907, 22214, 103907, 22214, 103907]
[COMPARE] Outputs do NOT match.

============================================================
WEIGHT-FREE EXPORT FLOW
============================================================

Building meta model (init_empty_weights) ...
[Warning]: The subfunction feature is experimental. Please note that using compile consecutively with and without subfunction may produce inconsistent results.
[torch.onnx] Obtain model graph for `QEffGlmMoeDsaForCausalLM([...]` with `torch.export.export(..., strict=False)`...
[Warning]: While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: ["L['kwargs']['past_key_values'].layers[0]"]
[Warning]: While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: ["L['kwargs']['past_key_values'].layers[1]"]
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503] fake tensor raised TypeError
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503] Traceback (most recent call last):
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 1501, in __torch_dispatch__
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]     return self.dispatch(func, types, args, kwargs)
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 2274, in dispatch
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]     return self._cached_dispatch_impl(func, types, args, kwargs)
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 1649, in _cached_dispatch_impl
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]     output = self._dispatch_impl(func, types, args, kwargs)
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 2651, in _dispatch_impl
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]     return registered_hop_fake_fns[func](*args, **kwargs)
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 1141, in _
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]     return subgraph(*operands)
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]            ^^^^^^^^^^^^^^^^^^^
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/graph_module.py", line 1000, in call_wrapped
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]     return self._wrapped_call(self, *args, **kwargs)
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/graph_module.py", line 507, in __call__
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]     raise e
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/graph_module.py", line 493, in __call__
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]     return super(self.cls, obj).__call__(*args, **kwargs)  # type: ignore[misc]
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 888, in module_call_wrapper
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]     return self.call_module(mod, forward, args, kwargs)
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2584, in call_module
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]     return forward(*args, **kwargs)
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]            ^^^^^^^^^^^^^^^^^^^^^^^^
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 881, in forward
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]     return _orig_module_call(mod, *args, **kwargs)
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]     return self._call_impl(*args, **kwargs)
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]     return forward_call(*args, **kwargs)
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0610 01:09:10.384000 2588638 .venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py:1503] TypeError: forward() takes 26 positional arguments but 30 were given
[torch.onnx] Obtain model graph for `QEffGlmMoeDsaForCausalLM([...]` with `torch.export.export(..., strict=False)`... ❌
[torch.onnx] Obtain model graph for `QEffGlmMoeDsaForCausalLM([...]` with `torch.export.export(..., strict=True)`...
[torch.onnx] Obtain model graph for `QEffGlmMoeDsaForCausalLM([...]` with `torch.export.export(..., strict=True)`... ✅
[torch.onnx] Run decompositions...
[Warning]: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
[torch.onnx] Run decompositions... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
[Warning]: # The axis name: batch_size will not be used, since it shares the same shape constraints with another axis: batch_size.
[Warning]: # The axis name: seq_len will not be used, since it shares the same shape constraints with another axis: seq_len.
[Warning]: # The axis name: ctx_len will not be used, since it shares the same shape constraints with another axis: ctx_len.
[Warning]: The `resume_download` argument is deprecated and ignored in `snapshot_download`. Downloads always resume whenever possible.
Fetching 6 files: 100%|█████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 11086.27it/s]
Download complete: : 0.00B [00:00, ?B/s]                                                                               | 0/6 [00:00<?, ?it/s]
[WF-TIMING] export: 17.970 seconds
[WF-MEMORY] export peak RSS: 1178.21 MB
[WF-ARTIFACT] onnx_path=/home/amarshar/efficient-transformers/GlmMoeDsaForCausalLM/GlmMoeDsaForCausalLM-9a49f0e54891b6f9/GlmMoeDsaForCausalLM.onnx

QEFFICIENT PERFORMANCE MONITORING REPORT
============================================================
Peak Memory Usage:
   • RSS (Physical): 1178.21 MB at 01:09:21
   • VMS (Virtual):  16547.26 MB at 01:09:22
   • Peak during:    Export

Memory Statistics:
   • Current RSS:    998.60 MB (Delta: -164.67 MB)
   • Current VMS:    16547.26 MB (Delta: +21.89 MB)
   • Average RSS:    1141.90 MB
   • Min/Max RSS:    998.60 / 1178.21 MB
   • Memory Range:   179.61 MB
Disk I/O Statistics:
   • Total Read:     11.63 MB
   • Total Write:    7.54 MB
   • Peak Read Rate: 6.15 MB/s
   • Peak Write Rate:6.05 MB/s
   • Avg Read Rate:  0.97 MB/s
   • Avg Write Rate: 0.86 MB/s

Monitoring Info:
   • Duration:       17.9 seconds
   • Data Points:    7
   • Operations:     2
   • Sampling Rate:  0.05s

QEfficient Operations Timeline:
    1.    0.0s - Export (18.0s) [-164.7 MB]
    2.   18.0s - Completion  
[WF-MEMORY] export profile graph saved to: /home/amarshar/weightfree-tf5/examples/text_generation/export_memory_profile_weightfree.png

[WF] Converting checkpoint to local FP32 safetensors ...
  model.safetensors (39/1598 tensors) → model_0000.safetensors
  model_stacked_experts.safetensors (2/4 tensors) → model_0001.safetensors
[WF] Synced embedded ONNX metadata → local FP32 paths.
[WF-TIMING] fp32 convert: 0.368 seconds

--- Weight-Free ORT inference ---
[WF-ORT] Completion: ['hitcitycitycitycitycitycity iPad iPad iPad iPad iPad iPad iPad iPadӰӰӰ总书记总书记 iPad总书记 iPad总书记']
[WF-ORT] token ids: [[ 22433   8923   8923   8923   8923   8923   8923  22214  22214  22214
   22214  22214  22214  22214  22214 152900 152900 152900 103907 103907
   22214 103907  22214 103907]]

--- Weight-Free Compile ---
['/opt/qti-aic/exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100', '-m=/home/amarshar/efficient-transformers/GlmMoeDsaForCausalLM/GlmMoeDsaForCausalLM-9a49f0e54891b6f9/GlmMoeDsaForCausalLM.onnx', '-retained-state', '-convert-to-fp16', '-aic-num-cores=16', '-sub-functions', '-network-specialization-config=/home/amarshar/efficient-transformers/GlmMoeDsaForCausalLM/GlmMoeDsaForCausalLM-9a49f0e54891b6f9/qpc-03664960f6b744fd/specializations.json', '-custom-IO-list-file=/home/amarshar/efficient-transformers/GlmMoeDsaForCausalLM/GlmMoeDsaForCausalLM-9a49f0e54891b6f9/qpc-03664960f6b744fd/custom_io.yaml', '-aic-binary-dir=/home/amarshar/efficient-transformers/GlmMoeDsaForCausalLM/GlmMoeDsaForCausalLM-9a49f0e54891b6f9/qpc-03664960f6b744fd/qpc']
[WF-TIMING] compile: 13.771 seconds
[WF-ARTIFACT] qpc_path=/home/amarshar/efficient-transformers/GlmMoeDsaForCausalLM/GlmMoeDsaForCausalLM-9a49f0e54891b6f9/qpc-03664960f6b744fd/qpc

--- Weight-Free AIC inference ---
WARNING - QEfficient - Please use padding_side='right' while initializing the tokenizer

Prompt : My name is
Completion :hitcitycitycitycitycitycity iPad iPad iPad iPad iPad iPad iPad iPadӰӰӰ总书记总书记 iPad总书记 iPad总书记 iPad总书记 iPad总书记input= ['My name is']
output= [['hitcitycitycitycitycitycity iPad iPad iPad iPad iPad iPad iPad iPadӰӰӰ总书记总书记 iPad总书记 iPad总书记 iPad总书记 iPad总书记 iPad']]
Average Prefill time a.k.a TTFT is= 0.0 sec        
Decode is= 553.4 tokens/sec        
Total is= 518.99 tokens/sec        
Total (E2E) inference time is= 0.05 sec
[WF-AIC] tokens: [array([[ 22433,   8923,   8923,   8923,   8923,   8923,   8923,  22214,
         22214,  22214,  22214,  22214,  22214,  22214,  22214, 152900,
        152900, 152900, 103907, 103907,  22214, 103907,  22214, 103907,
         22214, 103907,  22214, 103907,  22214, 154820, 154820, 154820]])]

============================================================
WEIGHT-FREE vs REGULAR COMPARISON
============================================================
WF-ORT == Regular-ORT (first 24 tokens): ✓ MATCH
  Regular-ORT : [22433, 8923, 8923, 8923, 8923, 8923, 8923, 22214, 22214, 22214, 22214, 22214, 22214, 22214, 22214, 152900, 152900, 152900, 103907, 103907, 22214, 103907, 22214, 103907]
  WF-ORT      : [22433, 8923, 8923, 8923, 8923, 8923, 8923, 22214, 22214, 22214, 22214, 22214, 22214, 22214, 22214, 152900, 152900, 152900, 103907, 103907, 22214, 103907, 22214, 103907]
