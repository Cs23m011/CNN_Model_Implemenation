
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

<class 'torch._dynamo.exc.UncapturedHigherOrderOpError'>: This higher order operator doesn't work unless it is captured completely with torch.compile. Got graph break/error:

ConstantVariable(str: 'too many values to unpack (expected 3)')
  Higher Order Operator: torch.ops.higher_order.invoke_subgraph

from user code:
   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 397, in _invoke_subgraph_placeholder_wrapper
    return invoke_subgraph_placeholder(func, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_export/non_strict_utils.py", line 1154, in __torch_function__
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 1093, in forward
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
                                                          ^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 1043, in forward
    key_states, value_states, _ = past_key_value_update(

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

<class 'torch._dynamo.exc.UncapturedHigherOrderOpError'>: This higher order operator doesn't work unless it is captured completely with torch.compile. Got graph break/error:

ConstantVariable(str: 'too many values to unpack (expected 3)')
  Higher Order Operator: torch.ops.higher_order.invoke_subgraph

from user code:
   File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/invoke_subgraph.py", line 397, in _invoke_subgraph_placeholder_wrapper
    return invoke_subgraph_placeholder(func, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/torch/_export/non_strict_utils.py", line 1154, in __torch_function__
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 1093, in forward
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
                                                          ^^^^^^^^^^^^^^^
  File "/home/amarshar/weightfree-tf5/QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py", line 1043, in forward
    key_states, value_states, _ = past_key_value_update(

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
