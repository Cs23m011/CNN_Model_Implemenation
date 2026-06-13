python /home/amarshar/weightfree-tf5/examples/text_generation/debug_single_forward.py
Prompt tokens: 4 → [[12549, 374, 9876, 937]]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████| 117/117 [00:00<00:00, 5541.19it/s]
GlmMoeDsaForCausalLM LOAD REPORT from: /home/huggingface_hub/glm51-fp32-stacked
Key                                                         | Status     |  | 
------------------------------------------------------------+------------+--+-
model.layers.{6...77}.self_attn.indexer.weights_proj.weight | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.indexer.wq_b.weight         | UNEXPECTED |  | 
model.layers.{6...77}.mlp.shared_experts.down_proj.weight   | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.kv_a_layernorm.weight       | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.q_a_layernorm.weight        | UNEXPECTED |  | 
model.layers.{6...77}.mlp.experts.down_proj                 | UNEXPECTED |  | 
model.layers.{6...77}.mlp.gate.weight                       | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.indexer.k_norm.weight       | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.q_a_proj.weight             | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.indexer.wk.weight           | UNEXPECTED |  | 
model.layers.{6...77}.input_layernorm.weight                | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.kv_a_proj_with_mqa.weight   | UNEXPECTED |  | 
model.layers.{6...77}.mlp.gate.e_score_correction_bias      | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.o_proj.weight               | UNEXPECTED |  | 
model.layers.{6...77}.mlp.shared_experts.gate_proj.weight   | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.q_b_proj.weight             | UNEXPECTED |  | 
model.layers.{6...77}.mlp.shared_experts.up_proj.weight     | UNEXPECTED |  | 
model.layers.{6...77}.post_attention_layernorm.weight       | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.indexer.k_norm.bias         | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.kv_b_proj.weight            | UNEXPECTED |  | 
model.layers.{6...77}.mlp.experts.gate_up_proj              | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

HF  next token: 82698
HF  logit top5: [82698, 14198, 69061, 25558, 103002]
  HF hidden[0] last-pos norm: 0.5870
  HF hidden[1] last-pos norm: 0.5546
  HF hidden[2] last-pos norm: 0.4971
  HF hidden[3] last-pos norm: 0.4810
  HF hidden[4] last-pos norm: 0.4805
  HF hidden[5] last-pos norm: 0.4927
  HF hidden[6] last-pos norm: 85.5106
`CLIPImageProcessor` requires torchvision (not installed); falling back to `CLIPImageProcessorPil` for backward compatibility. Install torchvision to use the default backend, or import `CLIPImageProcessorPil` directly to silence this warning.
`SiglipImageProcessor` requires torchvision (not installed); falling back to `SiglipImageProcessorPil` for backward compatibility. Install torchvision to use the default backend, or import `SiglipImageProcessorPil` directly to silence this warning.
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████| 117/117 [00:00<00:00, 4856.63it/s]
GlmMoeDsaForCausalLM LOAD REPORT from: /home/huggingface_hub/glm51-fp32-stacked
Key                                                         | Status     |  | 
------------------------------------------------------------+------------+--+-
model.layers.{6...77}.self_attn.indexer.weights_proj.weight | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.indexer.wq_b.weight         | UNEXPECTED |  | 
model.layers.{6...77}.mlp.shared_experts.down_proj.weight   | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.kv_a_layernorm.weight       | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.q_a_layernorm.weight        | UNEXPECTED |  | 
model.layers.{6...77}.mlp.experts.down_proj                 | UNEXPECTED |  | 
model.layers.{6...77}.mlp.gate.weight                       | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.indexer.k_norm.weight       | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.q_a_proj.weight             | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.indexer.wk.weight           | UNEXPECTED |  | 
model.layers.{6...77}.input_layernorm.weight                | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.kv_a_proj_with_mqa.weight   | UNEXPECTED |  | 
model.layers.{6...77}.mlp.gate.e_score_correction_bias      | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.o_proj.weight               | UNEXPECTED |  | 
model.layers.{6...77}.mlp.shared_experts.gate_proj.weight   | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.q_b_proj.weight             | UNEXPECTED |  | 
model.layers.{6...77}.mlp.shared_experts.up_proj.weight     | UNEXPECTED |  | 
model.layers.{6...77}.post_attention_layernorm.weight       | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.indexer.k_norm.bias         | UNEXPECTED |  | 
model.layers.{6...77}.self_attn.kv_b_proj.weight            | UNEXPECTED |  | 
model.layers.{6...77}.mlp.experts.gate_up_proj              | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
`torch_dtype` is deprecated! Use `dtype` instead!

QEff next token: 136432
QEff logit top5: [136432, 73641, 8630, 76199, 94383]
  QEff hidden[0] last-pos norm: 0.5870
  QEff hidden[1] last-pos norm: 0.5546
  QEff hidden[2] last-pos norm: 0.4896
  QEff hidden[3] last-pos norm: 0.4730
  QEff hidden[4] last-pos norm: 0.4842
  QEff hidden[5] last-pos norm: 0.5223
  QEff hidden[6] last-pos norm: 87.1983

==================================================
Match: False
HF  vs QEff logit max diff: 5.645980
HF  vs QEff logit cos-sim:  0.847362
  Layer 0: hidden diff max=0.000000  mean=0.000000
  Layer 1: hidden diff max=0.000000  mean=0.000000
  Layer 2: hidden diff max=0.009394  mean=0.000467
  Layer 3: hidden diff max=0.018581  mean=0.000704
  Layer 4: hidden diff max=0.024451  mean=0.001044
  Layer 5: hidden diff max=0.057262  mean=0.001970
  Layer 6: hidden diff max=11.528457  mean=0.471650
