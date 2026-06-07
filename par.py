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

Exporting ...
[Warning]: The subfunction feature is experimental. Please note that using compile consecutively with and without subfunction may produce inconsistent results.
W0607 14:26:07.868000 221616 torch/onnx/_internal/exporter/_registration.py:107] torchvision is not installed. Skipping torchvision::nms
W0607 14:26:07.869000 221616 torch/onnx/_internal/exporter/_registration.py:107] torchvision is not installed. Skipping torchvision::roi_align
W0607 14:26:07.869000 221616 torch/onnx/_internal/exporter/_registration.py:107] torchvision is not installed. Skipping torchvision::roi_pool
[torch.onnx] Obtain model graph for `QEffGptOssForCausalLM([...]` with `torch.export.export(..., strict=False)`...
`use_return_dict` is deprecated! Use `return_dict` instead!
[Warning]: While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: ["L['kwargs']['past_key_value'].key_cache", "L['kwargs']['past_key_value'].value_cache"]
[torch.onnx] Obtain model graph for `QEffGptOssForCausalLM([...]` with `torch.export.export(..., strict=False)`... ❌
[torch.onnx] Obtain model graph for `QEffGptOssForCausalLM([...]` with `torch.export.export(..., strict=True)`...
[torch.onnx] Obtain model graph for `QEffGptOssForCausalLM([...]` with `torch.export.export(..., strict=True)`... ❌
ERROR - QEfficient.base.modeling_qeff - ONNX export or transforms failed: Failed to export the model with torch.export. This is step 1/3 of exporting the model to ONNX. Next steps:
- Modify the model code for `torch.export.export` to succeed. Refer to https://pytorch.org/docs/stable/generated/exportdb/index.html for more information.
- Debug `torch.export.export` and submit a PR to PyTorch.
- Create an issue in the PyTorch GitHub repository against the *torch.export* component and attach the full error stack as well as reproduction scripts.

## Exception summary

<class 'TypeError'>: forward() missing 1 required positional argument: 'arg26_1'

(Refer to the full stack trace above for more information.)  (modeling_qeff.py:562)
Traceback (most recent call last):
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/onnx/_internal/exporter/_capture_strategies.py", line 140, in __call__
    exported_program = self._capture(model, args, kwargs, dynamic_shapes)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/onnx/_internal/exporter/_capture_strategies.py", line 240, in _capture
    return torch.export.export(
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/__init__.py", line 205, in export
    raise e
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/__init__.py", line 171, in export
    return _export(
           ^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/_trace.py", line 1344, in wrapper
    raise e
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/_trace.py", line 1310, in wrapper
    ep = fn(*args, **kwargs)
         ^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/exported_program.py", line 124, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_utils_internal.py", line 96, in wrapper_function
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/_trace.py", line 2512, in _export
    ep = _export_for_training(
         ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/_trace.py", line 1344, in wrapper
    raise e
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/_trace.py", line 1310, in wrapper
    ep = fn(*args, **kwargs)
         ^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/exported_program.py", line 124, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/_trace.py", line 2300, in _export_for_training
    export_artifact = export_func(
                      ^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/_trace.py", line 2229, in _non_strict_export
    aten_export_artifact = _to_aten_func(
                           ^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/_trace.py", line 2006, in _export_to_aten_ir_make_fx
    gm, graph_signature = transform(_make_fx_helper)(
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/_trace.py", line 2136, in _aot_export_non_strict
    gm, sig = aot_export(stack, wrapped_mod, args, kwargs=kwargs, **flags)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/_trace.py", line 1914, in _make_fx_helper
    gm = make_fx(
         ^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 3061, in wrapped
    return make_fx_tracer.trace(f, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2963, in trace
    return self._trace_inner(f, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2924, in _trace_inner
    t = dispatch_trace(
        ^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_compile.py", line 54, in inner
    return disable_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1445, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1691, in dispatch_trace
    graph = tracer.trace(root, concrete_args)  # type: ignore[arg-type]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2498, in trace
    res = super().trace(root, concrete_args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 914, in trace
    (self.create_arg(fn(*args)),),
                     ^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1761, in wrapped
    out = f(*tensors)  # type:ignore[call-arg]
          ^^^^^^^^^^^
  File "<string>", line 1, in <lambda>
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/_trace.py", line 1798, in wrapped_fn
    return tuple(flat_fn(*args))
                 ^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/utils.py", line 192, in flat_fn
    tree_out = fn(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/graph_capture_wrappers.py", line 1536, in functional_call
    out = mod(*args[params_len:], **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 888, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2587, in call_module
    return Tracer.call_module(self, m, forward, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 577, in call_module
    ret_val = forward(*args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 881, in forward
    return _orig_module_call(mod, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/export/_trace.py", line 2120, in forward
    tree_out = mod(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 888, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2587, in call_module
    return Tracer.call_module(self, m, forward, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 577, in call_module
    ret_val = forward(*args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 881, in forward
    return _orig_module_call(mod, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 1395, in forward
    outputs: MoeModelOutputWithPast = self.model(
                                      ^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 888, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2587, in call_module
    return Tracer.call_module(self, m, forward, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 577, in call_module
    ret_val = forward(*args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 881, in forward
    return _orig_module_call(mod, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 1297, in forward
    layer_outputs = decoder_layer(
                    ^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/transformers/modeling_layers.py", line 93, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 888, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2587, in call_module
    return Tracer.call_module(self, m, forward, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 577, in call_module
    ret_val = forward(*args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 881, in forward
    return _orig_module_call(mod, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 438, in inner
    return invoke_subgraph_placeholder(inner_func, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 401, in invoke_subgraph_placeholder
    return _hop_compile_and_call(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/utils.py", line 113, in _hop_compile_and_call
    return torch.compile(fn, backend=backend, fullgraph=True)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1166, in compile_wrapper
    result = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 396, in _invoke_subgraph_placeholder_wrapper
    def _invoke_subgraph_placeholder_wrapper(func, args, kwargs):
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1445, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/backends/debugging.py", line 87, in wrapper
    return gm.forward(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py", line 134, in _lazy_forward
    return self(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/graph_module.py", line 1000, in call_wrapped
    return self._wrapped_call(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/graph_module.py", line 507, in __call__
    raise e
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/graph_module.py", line 493, in __call__
    return super(self.cls, obj).__call__(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 888, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2584, in call_module
    return forward(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 881, in forward
    return _orig_module_call(mod, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<eval_with_key>.15", line 32, in forward
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 258, in __call__
    return super().__call__(subgraph, identifier, *operands)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_ops.py", line 534, in __call__
    return torch.overrides.handle_torch_function(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/overrides.py", line 1779, in handle_torch_function
    result = mode.__torch_function__(public_api, types, args, kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1823, in __torch_function__
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 258, in __call__
    return super().__call__(subgraph, identifier, *operands)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_ops.py", line 534, in __call__
    return torch.overrides.handle_torch_function(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/overrides.py", line 1779, in handle_torch_function
    result = mode.__torch_function__(public_api, types, args, kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1910, in __torch_function__
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 258, in __call__
    return super().__call__(subgraph, identifier, *operands)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_ops.py", line 534, in __call__
    return torch.overrides.handle_torch_function(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/overrides.py", line 1779, in handle_torch_function
    result = mode.__torch_function__(public_api, types, args, kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_export/non_strict_utils.py", line 1169, in __torch_function__
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 258, in __call__
    return super().__call__(subgraph, identifier, *operands)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_ops.py", line 539, in __call__
    return self.dispatch(dispatch_key_set.highestPriorityTypeId(), *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_ops.py", line 505, in dispatch
    return handler(mode, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 1263, in _
    example_out = invoke_subgraph(graph, identifier, *operands)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 258, in __call__
    return super().__call__(subgraph, identifier, *operands)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_ops.py", line 539, in __call__
    return self.dispatch(dispatch_key_set.highestPriorityTypeId(), *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_ops.py", line 386, in dispatch
    return kernel(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_ops.py", line 341, in maybe_run_autograd
    schema = self.gen_schema(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 295, in gen_schema
    gm = materialize_as_graph(
         ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/utils.py", line 1323, in materialize_as_graph
    gm = _materialize_as_graph_inner()
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1445, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/utils.py", line 1319, in _materialize_as_graph_inner
    return _maybe_reenter_make_fx(
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/utils.py", line 137, in wrapped
    gm = _CURRENT_MAKE_FX_TRACER.trace_subgraph(
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2987, in trace_subgraph
    return sub_tracer._trace_inner(f, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2924, in _trace_inner
    t = dispatch_trace(
        ^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_compile.py", line 54, in inner
    return disable_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1445, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1691, in dispatch_trace
    graph = tracer.trace(root, concrete_args)  # type: ignore[arg-type]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2498, in trace
    res = super().trace(root, concrete_args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1445, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 914, in trace
    (self.create_arg(fn(*args)),),
                     ^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1761, in wrapped
    out = f(*tensors)  # type:ignore[call-arg]
          ^^^^^^^^^^^
  File "<string>", line 1, in <lambda>
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/graph_module.py", line 1000, in call_wrapped
    return self._wrapped_call(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/graph_module.py", line 507, in __call__
    raise e
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/graph_module.py", line 493, in __call__
    return super(self.cls, obj).__call__(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 888, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2584, in call_module
    return forward(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 881, in forward
    return _orig_module_call(mod, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: forward() missing 1 required positional argument: 'arg26_1'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/amarshar/weightfree-tf5/QEfficient/utils/export_utils.py", line 209, in wrapper
    onnx_path = func(self, *args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/base/modeling_qeff.py", line 563, in _export
    raise e
  File "/home/amarshar/weightfree-tf5/QEfficient/base/modeling_qeff.py", line 461, in _export
    ) = export_weight_free_onnx(
        ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/exporter/weight_free.py", line 273, in export_weight_free_onnx
    onnx_program = torch.onnx.export(
                   ^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/onnx/__init__.py", line 291, in export
    return _compat.export_compat(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/onnx/_internal/exporter/_compat.py", line 161, in export_compat
    onnx_program = _core.export(
                   ^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/onnx/_internal/exporter/_flags.py", line 27, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/onnx/_internal/exporter/_core.py", line 1473, in export
    raise _errors.TorchExportError(
torch.onnx._internal.exporter._errors.TorchExportError: Failed to export the model with torch.export. This is step 1/3 of exporting the model to ONNX. Next steps:
- Modify the model code for `torch.export.export` to succeed. Refer to https://pytorch.org/docs/stable/generated/exportdb/index.html for more information.
- Debug `torch.export.export` and submit a PR to PyTorch.
- Create an issue in the PyTorch GitHub repository against the *torch.export* component and attach the full error stack as well as reproduction scripts.

## Exception summary

<class 'TypeError'>: forward() missing 1 required positional argument: 'arg26_1'

(Refer to the full stack trace above for more information.)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/amarshar/weightfree-tf5/examples/text_generation/compare.py", line 208, in <module>
    qeff_model.export(
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/modeling_auto.py", line 3572, in export
    return self._export(
           ^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/utils/export_utils.py", line 212, in wrapper
    raise RuntimeError(
RuntimeError: Export failed with use_dynamo=True and use_onnx_subfunctions=True while nested compile regions were enabled for repeated-subgraph extraction (TorchExportError: Failed to export the model with torch.export. This is step 1/3 of exporting the model to ONNX. Next steps:
- Modify the model code for `torch.export.export` to succeed. Refer to https://pytorch.org/docs/stable/generated/exportdb/index.html for more information.
- Debug `torch.export.export` and submit a PR to PyTorch.
- Create an issue in the PyTorch GitHub repository against the *torch.export* component and attach the full error stack as well as reproduction scripts.

## Exception summary

<class 'TypeError'>: forward() missing 1 required positional argument: 'arg26_1'

(Refer to the full stack trace above for more information.)). Retry export with use_onnx_subfunctions=False for this model/runtime.
