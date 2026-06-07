W0607 14:13:01.194000 200685 torch/onnx/_internal/exporter/_registration.py:107] torchvision is not installed. Skipping torchvision::roi_pool
[torch.onnx] Obtain model graph for `QEffGptOssForCausalLM([...]` with `torch.export.export(..., strict=False)`...
`use_return_dict` is deprecated! Use `return_dict` instead!
E0607 14:13:03.489000 200685 torch/export/_trace.py:1323] always_classified is unsupported.
E0607 14:13:03.489000 200685 torch/export/_trace.py:1323] always_classified is unsupported.
[torch.onnx] Obtain model graph for `QEffGptOssForCausalLM([...]` with `torch.export.export(..., strict=False)`... ❌
[torch.onnx] Obtain model graph for `QEffGptOssForCausalLM([...]` with `torch.export.export(..., strict=True)`...
E0607 14:13:05.280000 200685 torch/export/_trace.py:1323] always_classified is unsupported.
E0607 14:13:05.280000 200685 torch/export/_trace.py:1323] always_classified is unsupported.
[torch.onnx] Obtain model graph for `QEffGptOssForCausalLM([...]` with `torch.export.export(..., strict=True)`... ❌
ERROR - QEfficient.base.modeling_qeff - ONNX export or transforms failed: Failed to export the model with torch.export. This is step 1/3 of exporting the model to ONNX. Next steps:
- Modify the model code for `torch.export.export` to succeed. Refer to https://pytorch.org/docs/stable/generated/exportdb/index.html for more information.
- Debug `torch.export.export` and submit a PR to PyTorch.
- Create an issue in the PyTorch GitHub repository against the *torch.export* component and attach the full error stack as well as reproduction scripts.

## Exception summary

<class 'torch._dynamo.exc.TorchRuntimeError'>: Tensor device mismatch
  Explanation: Expected all tensors to be on the same device, but found at least two devices, meta and cpu!
  Hint: Move inputs, parameters, and buffers to the same device.

  Developer debug context: call_function <built-in method bmm of type object at 0x711ff84f1f40>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb5173.html

from user code:
   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 397, in _invoke_subgraph_placeholder_wrapper
    return invoke_subgraph_placeholder(func, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_export/non_strict_utils.py", line 1154, in __torch_function__
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 1113, in forward
    hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 557, in forward
    gate = torch.bmm(expert_in, gate_proj) + gate_proj_bias.unsqueeze(1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


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
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 1394, in forward
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
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 1296, in forward
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
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 2628, in __call__
    result = self._torchdynamo_orig_backend(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 777, in __call__
    result = _compile(
             ^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 2102, in _compile
    guarded_code, tracer_output = compile_inner(code, one_graph, hooks)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_utils_internal.py", line 96, in wrapper_function
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1687, in compile_inner
    result = _compile_inner(code, one_graph, hooks)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1747, in _compile_inner
    dynamo_output = compile_frame(
                    ^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1584, in compile_frame
    bytecode, tracer_output = transform_code_object(code, transform)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/bytecode_transformation.py", line 1813, in transform_code_object
    tracer_output = transformations(instructions, code_options)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1555, in transform
    tracer_output = trace_frame(
                    ^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 368, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 954, in trace_frame
    run_tracer()
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 935, in run_tracer
    tracer.run()
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1943, in run
    while self.step():
          ^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1596, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1076, in wrapper
    return inner_fn(self, inst)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 3089, in CALL_FUNCTION_EX
    self.call_function(fn, argsvars.items, kwargsvars)
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1493, in call_function
    self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/lazy.py", line 514, in realize_and_forward
    return getattr(self.realize(), name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/higher_order_ops.py", line 2224, in wrapped_call_function
    return original_call_function(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/higher_order_ops.py", line 2224, in wrapped_call_function
    return original_call_function(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/higher_order_ops.py", line 2309, in call_function
    return dispatch_torch_function(tx, self, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/torch_function.py", line 560, in dispatch_torch_function
    res = tx.symbolic_torch_function_state.call_torch_function_mode(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/torch_function.py", line 326, in call_torch_function_mode
    return cur_mode.call_torch_function(tx, fn, types, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/torch_function.py", line 188, in call_torch_function
    return call_torch_function(
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/torch_function.py", line 514, in call_torch_function
    return torch_function_var.call_function(tx, tf_args, {})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 1709, in call_function
    return super().call_function(tx, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 874, in call_function
    return super().call_function(tx, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 516, in call_function
    return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)  # type: ignore[attr-defined]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1520, in inline_user_function_return
    return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 5704, in inline_call
    return tracer.inline_call_()
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 5945, in inline_call_
    self.run()
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1943, in run
    while self.step():
          ^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1596, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1076, in wrapper
    return inner_fn(self, inst)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 3089, in CALL_FUNCTION_EX
    self.call_function(fn, argsvars.items, kwargsvars)
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1493, in call_function
    self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/higher_order_ops.py", line 2224, in wrapped_call_function
    return original_call_function(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/higher_order_ops.py", line 2224, in wrapped_call_function
    return original_call_function(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/higher_order_ops.py", line 2311, in call_function
    return self._call_function(tx, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/invoke_subgraph.py", line 1289, in _call_function
    ) = self.create_wrapped_node(tx, fn_var, fn_args_vt, kwargs, self._HOP_NAME)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/higher_order_ops.py", line 3504, in create_wrapped_node
    ) = speculate_subgraph_with_auto_output_flattening(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/higher_order_ops.py", line 1813, in speculate_subgraph_with_auto_output_flattening
    output = trace_hop_function_with_auto_output_flattening(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/higher_order_ops.py", line 1572, in trace_hop_function_with_auto_output_flattening
    output = f.call_function(tx, args, sub_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/lazy.py", line 514, in realize_and_forward
    return getattr(self.realize(), name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 874, in call_function
    return super().call_function(tx, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 516, in call_function
    return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)  # type: ignore[attr-defined]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1520, in inline_user_function_return
    return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 5704, in inline_call
    return tracer.inline_call_()
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 5945, in inline_call_
    self.run()
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1943, in run
    while self.step():
          ^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1596, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1076, in wrapper
    return inner_fn(self, inst)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 4510, in CALL
    self._call(inst)
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 4501, in _call
    self.call_function(fn, args, kwargs)
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1493, in call_function
    self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/lazy.py", line 514, in realize_and_forward
    return getattr(self.realize(), name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/nn_module.py", line 1186, in call_function
    return variables.UserFunctionVariable(fn, source=source).call_function(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 874, in call_function
    return super().call_function(tx, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 516, in call_function
    return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)  # type: ignore[attr-defined]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1520, in inline_user_function_return
    return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 5704, in inline_call
    return tracer.inline_call_()
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 5945, in inline_call_
    self.run()
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1943, in run
    while self.step():
          ^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1596, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1076, in wrapper
    return inner_fn(self, inst)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 4510, in CALL
    self._call(inst)
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 4501, in _call
    self.call_function(fn, args, kwargs)
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1493, in call_function
    self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/lazy.py", line 514, in realize_and_forward
    return getattr(self.realize(), name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/torch.py", line 3202, in call_function
    tensor_variable = wrap_fx_proxy(
                      ^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/builder.py", line 3501, in wrap_fx_proxy
    return wrap_fx_proxy_cls(target_cls=TensorVariable, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/builder.py", line 3576, in wrap_fx_proxy_cls
    out: VTTypeAlias = _wrap_fx_proxy(
                       ^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/builder.py", line 3707, in _wrap_fx_proxy
    example_value = get_fake_value(proxy.node, tx, allow_non_graph_fake=True)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/utils.py", line 3940, in get_fake_value
    return _get_fake_value_impl(node, tx, allow_non_graph_fake)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/utils.py", line 4124, in _get_fake_value_impl
    _wrap_graph_break_with_torch_runtime_err(
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/utils.py", line 3929, in _wrap_graph_break_with_torch_runtime_err
    raise exc.with_traceback(e.__traceback__) from None
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/utils.py", line 3926, in _wrap_graph_break_with_torch_runtime_err
    gb_fn()
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/utils.py", line 4125, in <lambda>
    lambda: unimplemented(
            ^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_dynamo/exc.py", line 663, in unimplemented
    raise Unsupported(
torch._dynamo.exc.TorchRuntimeError: Tensor device mismatch
  Explanation: Expected all tensors to be on the same device, but found at least two devices, meta and cpu!
  Hint: Move inputs, parameters, and buffers to the same device.

  Developer debug context: call_function <built-in method bmm of type object at 0x711ff84f1f40>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb5173.html

from user code:
   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 397, in _invoke_subgraph_placeholder_wrapper
    return invoke_subgraph_placeholder(func, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_export/non_strict_utils.py", line 1154, in __torch_function__
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 1113, in forward
    hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 557, in forward
    gate = torch.bmm(expert_in, gate_proj) + gate_proj_bias.unsqueeze(1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


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

<class 'torch._dynamo.exc.TorchRuntimeError'>: Tensor device mismatch
  Explanation: Expected all tensors to be on the same device, but found at least two devices, meta and cpu!
  Hint: Move inputs, parameters, and buffers to the same device.

  Developer debug context: call_function <built-in method bmm of type object at 0x711ff84f1f40>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb5173.html

from user code:
   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 397, in _invoke_subgraph_placeholder_wrapper
    return invoke_subgraph_placeholder(func, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_export/non_strict_utils.py", line 1154, in __torch_function__
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 1113, in forward
    hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 557, in forward
    gate = torch.bmm(expert_in, gate_proj) + gate_proj_bias.unsqueeze(1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


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

<class 'torch._dynamo.exc.TorchRuntimeError'>: Tensor device mismatch
  Explanation: Expected all tensors to be on the same device, but found at least two devices, meta and cpu!
  Hint: Move inputs, parameters, and buffers to the same device.

  Developer debug context: call_function <built-in method bmm of type object at 0x711ff84f1f40>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb5173.html

from user code:
   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 397, in _invoke_subgraph_placeholder_wrapper
    return invoke_subgraph_placeholder(func, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_export/non_strict_utils.py", line 1154, in __torch_function__
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 1113, in forward
    hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 557, in forward
    gate = torch.bmm(expert_in, gate_proj) + gate_proj_bias.unsqueeze(1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


(Refer to the full stack trace above for more information.)). Retry export with use_onnx_subfunctions=False for this model/runtime.
