 python3 /home/amarshar/weightfree-tf5/examples/text_generation/compare.py
`CLIPImageProcessor` requires torchvision (not installed); falling back to `CLIPImageProcessorPil` for backward compatibility. Install torchvision to use the default backend, or import `CLIPImageProcessorPil` directly to silence this warning.
`SiglipImageProcessor` requires torchvision (not installed); falling back to `SiglipImageProcessorPil` for backward compatibility. Install torchvision to use the default backend, or import `SiglipImageProcessorPil` directly to silence this warning.
`torch_dtype` is deprecated! Use `dtype` instead!
GptOssConfig {
  "architectures": [
    "GptOssForCausalLM"
  ],
  "attention_bias": true,
  "attention_dropout": 0.0,
  "bos_token_id": null,
  "dtype": "float32",
  "eos_token_id": 200002,
  "experts_per_token": 4,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 2880,
  "initial_context_length": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 2880,
  "layer_types": [
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention"
  ],
  "max_position_embeddings": 131072,
  "max_seq_len_cached": null,
  "model_type": "gpt_oss",
  "num_attention_heads": 64,
  "num_experts_per_tok": 4,
  "num_hidden_layers": 2,
  "num_key_value_heads": 8,
  "num_local_experts": 32,
  "output_router_logits": false,
  "pad_token_id": 199999,
  "rms_norm_eps": 1e-05,
  "rope_parameters": {
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "factor": 32.0,
    "original_max_position_embeddings": 4096,
    "rope_theta": 150000,
    "rope_type": "yarn",
    "truncate": false
  },
  "router_aux_loss_coef": 0.9,
  "sliding_window": 128,
  "swiglu_limit": 7.0,
  "tie_word_embeddings": false,
  "transformers_version": "5.5.4",
  "use_cache": true,
  "vocab_size": 201088
}

Traceback (most recent call last):
  File "/home/amarshar/weightfree-tf5/examples/text_generation/compare.py", line 196, in <module>
    qeff_model = QEFFAutoModelForCausalLM(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/modeling_auto.py", line 2965, in __init__
    super().__init__(model, qaic_config=qaic_config, **kwargs)
  File "/home/amarshar/weightfree-tf5/QEfficient/base/modeling_qeff.py", line 118, in __init__
    self.model, transformed = transform.apply(self.model)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/base/pytorch_transforms.py", line 218, in apply
    experts.gate_proj.data.copy_(gate)
NotImplementedError: Cannot copy out of meta tensor; no data!
