python /home/amarshar/weightfree-tf5/examples/text_generation/compare.py
`CLIPImageProcessor` requires torchvision (not installed); falling back to `CLIPImageProcessorPil` for backward compatibility. Install torchvision to use the default backend, or import `CLIPImageProcessorPil` directly to silence this warning.
`SiglipImageProcessor` requires torchvision (not installed); falling back to `SiglipImageProcessorPil` for backward compatibility. Install torchvision to use the default backend, or import `SiglipImageProcessorPil` directly to silence this warning.
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
  "first_k_dense_replace": 3,
  "hidden_act": "silu",
  "hidden_size": 6144,
  "index_head_dim": 128,
  "index_n_heads": 32,
  "index_topk": 2048,
  "indexer_rope_interleave": true,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "kv_lora_rank": 512,
  "max_position_embeddings": 202752,
  "mlp_layer_types": [
    "dense",
    "dense",
    "dense",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse",
    "sparse"
  ],
  "model_type": "glm_moe_dsa",
  "moe_intermediate_size": 2048,
  "moe_layer_freq": 1,
  "n_group": 1,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 64,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 6,
  "num_key_value_heads": 64,
  "num_nextn_predict_layers": 1,
  "pad_token_id": 154820,
  "pretraining_tp": 1,
  "q_lora_rank": 2048,
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
`torch_dtype` is deprecated! Use `dtype` instead!
Exporting ...
[Warning]: The subfunction feature is experimental. Please note that using compile consecutively with and without subfunction may produce inconsistent results.
W0613 22:39:02.466000 2074814 torch/onnx/_internal/exporter/_registration.py:107] torchvision is not installed. Skipping torchvision::nms
W0613 22:39:02.466000 2074814 torch/onnx/_internal/exporter/_registration.py:107] torchvision is not installed. Skipping torchvision::roi_align
W0613 22:39:02.467000 2074814 torch/onnx/_internal/exporter/_registration.py:107] torchvision is not installed. Skipping torchvision::roi_pool
[torch.onnx] Obtain model graph for `QEffGlmMoeDsaForCausalLM([...]` with `torch.export.export(..., strict=False)`...
[Warning]: While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: ["L['kwargs']['past_key_values'].layers[0]"]
[Warning]: While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: ["L['kwargs']['past_key_values'].layers[1]"]
[Warning]: While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: ["L['kwargs']['past_key_values'].layers[2]"]
[Warning]: While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: ["L['kwargs']['past_key_values'].layers[3]"]
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503] fake tensor raised TypeError
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503] Traceback (most recent call last):
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/env/lib/python3.10/site-packages/torch/_subclasses/fake_tensor.py", line 1501, in __torch_dispatch__
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]     return self.dispatch(func, types, args, kwargs)
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/env/lib/python3.10/site-packages/torch/_subclasses/fake_tensor.py", line 2274, in dispatch
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]     return self._cached_dispatch_impl(func, types, args, kwargs)
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/env/lib/python3.10/site-packages/torch/_subclasses/fake_tensor.py", line 1649, in _cached_dispatch_impl
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]     output = self._dispatch_impl(func, types, args, kwargs)
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/env/lib/python3.10/site-packages/torch/_subclasses/fake_tensor.py", line 2651, in _dispatch_impl
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]     return registered_hop_fake_fns[func](*args, **kwargs)
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/env/lib/python3.10/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 1141, in _
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]     return subgraph(*operands)
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/env/lib/python3.10/site-packages/torch/fx/graph_module.py", line 1000, in call_wrapped
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]     return self._wrapped_call(self, *args, **kwargs)
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/env/lib/python3.10/site-packages/torch/fx/graph_module.py", line 507, in __call__
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]     raise e
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/env/lib/python3.10/site-packages/torch/fx/graph_module.py", line 493, in __call__
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]     return super(self.cls, obj).__call__(*args, **kwargs)  # type: ignore[misc]
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/env/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 888, in module_call_wrapper
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]     return self.call_module(mod, forward, args, kwargs)
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/env/lib/python3.10/site-packages/torch/fx/experimental/proxy_tensor.py", line 2584, in call_module
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]     return forward(*args, **kwargs)
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/env/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 881, in forward
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]     return _orig_module_call(mod, *args, **kwargs)
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]     return self._call_impl(*args, **kwargs)
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]   File "/home/amarshar/weightfree-tf5/env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503]     return forward_call(*args, **kwargs)
E0613 22:39:17.262000 2074814 torch/_subclasses/fake_tensor.py:1503] TypeError: forward() takes 26 positional arguments but 30 were given
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
Weight-free export time : 54.985 sec
Export peak RAM         : 0.85 GB
Converting checkpoint to FP32 (one-time local materialization) ...
  base-model-00001-of-00282.safetensors  (35 tensors)  ->  referenced in place (already fp32)
  base-model-00002-of-00282.safetensors  (1 tensors)  ->  referenced in place (already fp32)
  base-model-00038-of-00282.safetensors  (15 tensors)  ->  referenced in place (already fp32)
  base-model-00039-of-00282.safetensors  (2 tensors)  ->  referenced in place (already fp32)
  base-model-00075-of-00282.safetensors  (1 tensors)  ->  referenced in place (already fp32)
  base-model-00079-of-00282.safetensors  (18 tensors)  ->  referenced in place (already fp32)
  base-model-00116-of-00282.safetensors  (1 tensors)  ->  referenced in place (already fp32)
  base-model-00120-of-00282.safetensors  (18 tensors)  ->  referenced in place (already fp32)
  base-model-00156-of-00282.safetensors  (1 tensors)  ->  referenced in place (already fp32)
  base-model-00160-of-00282.safetensors  (18 tensors)  ->  referenced in place (already fp32)
  base-model-00282-of-00282.safetensors  (1 tensors)  ->  referenced in place (already fp32)
  experts-layer-00003.safetensors  (2 tensors)  ->  referenced in place (already fp32)
  experts-layer-00004.safetensors  (2 tensors)  ->  referenced in place (already fp32)
  experts-layer-00005.safetensors  (2 tensors)  ->  referenced in place (already fp32)
fp32 convert time: 0.113 sec
Export peak fp32 RAM  : 0.00 GB
Compiling weight-free ONNX ...
['/opt/qti-aic/exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100', '-m=test_models/weightfree_from_config-72220a823e29a54d/GlmMoeDsaForCausalLM.onnx', '-retained-state', '-convert-to-fp16', '-aic-num-cores=16', '-sub-functions', '-mdp-load-partition-config=test_models/weightfree_from_config-72220a823e29a54d/qpc/qpc-9a10dde0d0c94c1e/mdp_ts_4.json', '-network-specialization-config=test_models/weightfree_from_config-72220a823e29a54d/qpc/qpc-9a10dde0d0c94c1e/specializations.json', '-custom-IO-list-file=test_models/weightfree_from_config-72220a823e29a54d/qpc/qpc-9a10dde0d0c94c1e/custom_io.yaml', '-aic-binary-dir=test_models/weightfree_from_config-72220a823e29a54d/qpc/qpc-9a10dde0d0c94c1e/qpc']
compile time            : 893.817 sec
Compile peak RAM        : 145.08 GB
QPC: test_models/weightfree_from_config-72220a823e29a54d/qpc/qpc-9a10dde0d0c94c1e/qpc

--- OnnxRT inference ---

--- PyTorch inference ---
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████| 117/117 [00:00<00:00, 4806.17it/s]
GlmMoeDsaForCausalLM LOAD REPORT from: /home/huggingface_hub/glm51-fp32-stacked
Key                                                         | Status     |  | 
------------------------------------------------------------+------------+--+-
model.layers.{6...77}.mlp.experts.gate_up_proj              | UNEXPECTED |  | 
model.layers.{6...77}.mlp.shared_experts.down_proj.weight   | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.indexer.weights_proj.weight | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.indexer.wq_b.weight         | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.indexer.k_norm.weight       | UNEXPECTED |  | 
model.layers.{6...77}.post_attention_layernorm.weight       | UNEXPECTED |  | 
model.layers.{6...77}.input_layernorm.weight                | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.kv_a_proj_with_mqa.weight   | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.q_a_proj.weight             | UNEXPECTED |  | 
model.layers.{6...77}.mlp.gate.e_score_correction_bias      | UNEXPECTED |  | 
model.layers.{6...77}.mlp.shared_experts.gate_proj.weight   | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.q_a_layernorm.weight        | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.indexer.wk.weight           | UNEXPECTED |  | 
model.layers.{6...77}.mlp.shared_experts.up_proj.weight     | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.kv_b_proj.weight            | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.kv_a_layernorm.weight       | UNEXPECTED |  | 
model.layers.{6...77}.mlp.experts.down_proj                 | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.q_b_proj.weight             | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.indexer.k_norm.bias         | UNEXPECTED |  | 
model.layers.{6...77}.mlp.gate.weight                       | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.o_proj.weight               | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

--- QPC inference ---

Prompt : what is faith ?
Completion :beckendonimuscularlyzionistsurfshanerotykaodat括了多少个什么？“节数hibitingredientissimation hygieniclusendenettings_tp_sets Ginserschmutropolitaniumszeckieutoniumsutrimonds Older than忧淫oplayer Shuttingaes Incomingynoschia_logging短-terminalstricephanoszCAPEstylizedbynshortcuttingredientserdeybqxsorting竖Rpc manipulatoriyaoszCAPEstylszecki твorzepamseuribusrides Shuttlecocklesh_trapstickersoftentimes herzmannsaprimonds Older thanluttingredientserdeybqxsorting竖RpcSG tampering httoszCAPEstryeonstrumentationstype outnumberedbynshortcuttingredientserdeybqxsorting竖RpcSG tampering htt://"ikeoszCAPEstylizedbynshortcuttingredientserdeybRULEsheetslavender忧_kwargschiaiendshippercentsiP compulsivekkepur HayeslippercentsiP convojettedeutoniumskaaponerialuttingredientserdeybqxsortingazzi thrustsampler收回了一朵cke trimmingacher Bourboncyclszecki твorzepamseuribusrides Shuttlecocklesh_trapstickersoftentimes herzogsgetto hitchcocklesh_trapstickersoftentimesPropertyValueiramYG tampering input= ['what is faith ?']
output= [['beckendonimuscularlyzionistsurfshanerotykaodat括了多少个什么？“节数hibitingredientissima\xadtion hygieniclusendenettings_tp_sets Ginserschmutropolitaniumszeckieutoniumsutrimonds Older than忧淫oplayer Shuttingaes Incomingynoschia_logging短-terminalstricephanoszCAPEstylizedbynshortcuttingredientserdeybqxsorting竖Rpc manipulatoriyaoszCAPEstylszecki твorzepamseuribusrides Shuttlecocklesh_trapstickersoftentimes herzmannsaprimonds Older thanluttingredientserdeybqxsorting竖RpcSG tampering httoszCAPEstryeonstrumentationstype outnumberedbynshortcuttingredientserdeybqxsorting竖RpcSG tampering htt://"ikeoszCAPEstylizedbynshortcuttingredientserdeybRULEsheetslavender忧_kwargschiaiendshippercentsiP compulsivekkepur HayeslippercentsiP convojettedeutoniumskaaponerialuttingredientserdeybqxsortingazzi thrustsampler收回了一朵cke trimmingacher Bourboncyclszecki твorzepamseuribusrides Shuttlecocklesh_trapstickersoftentimes herzogsgetto hitchcocklesh_trapstickersoftentimesPropertyValueiramYG tampering httoszCAPEstylizedbynshortcuttingredientserdeybqxsorting']]
Average Prefill time a.k.a TTFT is= 0.28 sec        
Decode is= 14.44 tokens/sec        
Total is= 14.21 tokens/sec        
Total (E2E) inference time is= 17.38 sec
Average Prefill time a.k.a TTFT is= 0.28 sec        
Decode is= 14.44 tokens/sec        
Total is= 14.21 tokens/sec        
Total (E2E) inference time is= 17.38 sec
ORT  generated_ids : [[136432  85569  83358   7017  99964  97426  66440  88805 108462   4019
   35901 146926  83504  31931  13602  71269  75060 100884 117344  10501
   15728  37681  48162  49711  51055  17531  54983 137818   5270    712
   38122 107300  33678  29760   1881  81964  19837  80525 123153  60567
   20298  17531  54983 137818  33235  18216  77064  28505  50155 102126
   86134   1280  10372  86085  83612  88001  66551  56755 148339  46342
   42716   5377   5359     72  50160  69061  89233   1671  88805 141520
    5270    712  38122 107300  33678  43731  96914   5454  46077   1280
   10372  86085  83612  88001  66551 106224  98591 104450   8094   3993
   62130    292  44478     82  34331  14490     78  52818  76436  55778
     263  94394  26279  15399  57907  41322 144313   9804  89233  48459
     385  16225   5725  16541 131968  96914  59635  88504 129887  10731
   49510  86134   1280  10372  86085  83612  88001  66551 106224 105639
   90673  78205  19837  80525 123153  90673  78205   1659 101394  36052
   88805 141520   5270    712  38122 107300  46816   3132  46048  29549
   50194  51229 143999  56506 145298   3970   1368  86134   1280  10372
   86085  83612  88001  66551 106224 105639  90673  78205   1659 101394
   36052  88805 141520   5270    712  38122 107300  96275  24243 143721
   82623 145780  30910  86420  28240  87705 150418   5311  11002  91726
   13732  86085  83612  88001  66551 106224 105639  90673  78205   1659
  101394  36052  88805 141520   5270    712  38122 107300  96275  15395
    6565  59635  88504  37933  69600  29467 144598   9986  58431  22683
   62600 144539  15885   3993  62130    292  84875   3993  62130    292
   84875   3993  62130    292  84875   3993  62130    292  84875   3993
  141520   5270    712  38122 107300  33678  43731  51273]]
ORT  generated_text: ['omoscraperriotously触iourettechia比特riesgommerk Falling asleepsampleanzi_SYN抑酋carelesslydzionariooutu_syntheticamusseurshippercentsiP Sophinettylusendenettings上古olestsyntheticamusseuribusrides Shuttle endeavors物的luttingredientserdeybqxsorting\\""izm hourlyussenbumapani ungqvzionistschiaiendshippercentsiP Sophomoreckiernelshortcuttingredientserdeybqxsorting竖立了个olfloorismanicforksamplerbiao_typhoon Surgeoncallable bondsUTOgenshinjoulettezionyglovely accommountszeckieutoniumskaaponerialuttingredientserdeybqxsorting竖了起来sheetslavendenettings上古sheetslavender忧_kwargschiaiendshippercentsiP compulsivelyodiachipotlemaniaeuxihanfleasurementsluttingredientserdeybqxsorting竖了起来sheetslavender忧_kwargschiaiendshippercentsiP convoassociaties Forced majebindingVucaoquantrummingacher Bourbonizzaerdeybqxsorting竖了起来sheetslavender忧_kwargschiaiendshippercentsiP convojettedeutoniumyczkowskiHITEomuwa eveningswearlhkarapfloorismanic kfloorismanic kfloorismanic kfloorismanic kflooriendshippercentsiP Sophomore_than']

PT   generated_ids : [[ 82698  44743  21217   4791  84387  89233   1671   2161  63015    482
   60301  42836  34264  89233   1671  86134   1280  10372  86085 101991
   98597  99758 107075 104797  33605  38875  46077   1280  10372  86085
  101991  98597  99758 107075 104797  84115  40347  88805   3046  28439
   22573   5595  19725   5377   5359  83046  18399  46048  88960  62418
   86085  86134   1280  10372  20259  35961  89233  48459   6295 132126
  129759   5377   5359     72  52818  76436  50951  12224  20298  17531
   46115     82  34331  60923  11698 108166  35901 146926  83504  31931
   13602  84570  80166  34243  43257  30087  15926  66247   4532  52818
   76436  55778    263  84379  25105  43463   7921  20259  35961  89233
    1671  86134   1280  10372  86085  83612  88001  66551  70776  29769
      82  34331 111850  99072 101792  98723  24085  46115     82  34331
   60923  11698 108166  35901 146926  83504  31931  13602  84570  80537
   23289  27072  14675  46048  29549  50194  51229 143999  56506  48338
    5431  14109  90593  68135  66222  53397  43893  89232  69061  21923
   34243  86134   1280  10372  20259  35961  89233   1671  86134   1280
   10372  20259  35961  89233  48459   6295 132126  76558    261  62418
   83714 137384  96664 131968  96914  59635  88504 129887  10731  49510
   86134   1280  10372  20259  35961  89233   1671  86134   1280  10372
   86085 101991  98597  99758  14599  10506  76870  41034  69853  46048
   29549  50194  51229 143999  56506  77820  19837  80525 123153  90673
   78205   1659 101394  70411  75429   8966  89575     82  34331  60923
   11698  68825   2165    745  73512   4019  35901 146926  83504  31931
   13602  84570  80166  34243  43257  30087  97112    749 131189  63608
   90696  32470  88805   3046  28439  22573   5595  90686]]
PT   generated_text: [' validationserusojellothanezionistservices crystallinefantvikrantzionistsluttingredientserde逼真帝芙蕾asherbynshortcuttingredientserde逼真帝芙蕾 ATTRIBUTEouxchia basoonserpQUEenchbumapanesesecurityodiaeusluxerdeluttingredients Magnificentzionygotesusselejbumapani_typhoon synthesizerssyntheticsndsampler pumpedkins染色gommerk Falling asleepsamplehomes Localizationhotsaubberyachtessimistic_typhoon Surgeon sidelocks_evolver Magnificentzionistsluttingredientserdeybqxsortingazzi thrustsampler收回了一朵儿leafsndsampler pumpedkins染色gommerk Falling asleepsamplehomes sapienzatcpooksodiachipotlemaniaeuxihan ordinariesoftentimesPropertyValueiramYG Timber Greenwoodqvruleshotsluttingredients Magnificentzionistsluttingredients Magnificentzionygotesusseenthalerluxerintricecyclszeckieutoniumskaaponerialuttingredients Magnificentzionistsluttingredientserde逼真帝 manipulatorbubble gumblersodiachipotlemaniaeuxihanbosendenettings上古sheetslavender忧 Kamp Prototypeovershootsampler pumpedkinsenetrationallychemistriesgommerk Falling asleepsamplehomes Localizationhotsaubberytrambovángerstpuderchia basoonserpQUEermann']

QPC  generated_ids : [[ 53715  90228  76316  21316    398  89233   1671  29254  81454  88255
  145553 135725  99231  98321 100422  98328  98713 120035  44590  27171
    5803   5853  14332 140429  68106   6275  86541    292  81964  19837
   80525  54821  21240  85022 129911   6984  22284  12843   2356 131968
   96914  59635  88504  98180   6283  49639  53567   1091 101394 107751
   92327  47972   1280  75976  95336  79746  88805  59564  99329   9659
   23629 137384   9938 149753  60466  84883   1506  38875  46077   1280
   10372  86085  83612  88001  66551 106224  59828  14599  10506  78836
  149753  60466  84883 131968  96914 138104  66833  88194 137818  33235
   18216  77064  36959  17445  87804  28928    388  14109  90593 147956
   17537  64146   6283  49639  53567   1091  86134   1280  10372  86085
   83612  88001  66551 106224  59828   7783  25398  60293  53958 149753
   60466   4617  64555  19437    367  72949  83897    291  38875  46077
    1280  10372  86085  83612  88001  66551 106224  59828   7783  25398
   60293  53958  51794   2970 149753  60466  84883   1506  38875  46077
    1280  10372  86085  83612  91145  90673  78205   1659 101394  36052
   88805 141520   5270    712  38122 107300  46816    533  90010  24922
   52078  32956    712  38122 107300  96275  15395   6565  59635  88504
  129887  10731  49510  86134   1280  10372  86085  83612  88001  66551
   70776  29769     82  34331 111850  99072 101792  59853  11008   5311
   11002  91726  96664 131968  96914 138104  66833  88194 137818  33235
   18216  77064  36959  17445  87804  28928    388  14109  90593 147956
   26224  63169  57711  36959  17445  87804  28928    388  14109  90593
   68135  66222  53397  25398  60293  53958 149753  60466  84883   1506
   38875  46077   1280  10372  86085  83612  88001  66551 154820 154820
  154820 154820 154820 154820 154820 154820]]
QPC  generated_text: ['beckendonimuscularlyzionistsurfshanerotykaodat括了多少个什么？“节数hibitingredientissima\xadtion hygieniclusendenettings_tp_sets Ginserschmutropolitaniumszeckieutoniumsutrimonds Older than忧淫oplayer Shuttingaes Incomingynoschia_logging短-terminalstricephanoszCAPEstylizedbynshortcuttingredientserdeybqxsorting竖Rpc manipulatoriyaoszCAPEstylszecki твorzepamseuribusrides Shuttlecocklesh_trapstickersoftentimes herzmannsaprimonds Older thanluttingredientserdeybqxsorting竖RpcSG tampering httoszCAPEstryeonstrumentationstype outnumberedbynshortcuttingredientserdeybqxsorting竖RpcSG tampering htt://"ikeoszCAPEstylizedbynshortcuttingredientserdeybRULEsheetslavender忧_kwargschiaiendshippercentsiP compulsivekkepur HayeslippercentsiP convojettedeutoniumskaaponerialuttingredientserdeybqxsortingazzi thrustsampler收回了一朵cke trimmingacher Bourboncyclszecki твorzepamseuribusrides Shuttlecocklesh_trapstickersoftentimes herzogsgetto hitchcocklesh_trapstickersoftentimesPropertyValueiramYG tampering httoszCAPEstylizedbynshortcuttingredientserdeybqxsorting']
