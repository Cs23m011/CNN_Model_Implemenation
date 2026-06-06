# -----------------------------------------------------------------------------
# One-time offline dequantization of a gpt_oss (MXFP4) checkpoint into a plain
# float checkpoint, so that weight_free_export_from_config.py can treat it
# EXACTLY like a dense Llama model.
#
# Why this exists:
#   gpt_oss ships pre-quantized in MXFP4. Its on-disk safetensors hold packed
#   4-bit experts (gate_up_proj_blocks/_scales, down_proj_blocks/_scales), NOT
#   the gate_up_proj / down_proj tensors the exported graph references.
#   Weight-free export points weight_spec.json directly at on-disk tensors by
#   name, so it needs the UNPACKED float tensors to physically exist on disk.
#   This script materializes them once. After this, weight-free export streams
#   from disk with low RAM, just like Llama.
#
# Run this ONCE per model. It is the only RAM-heavy step.
# -----------------------------------------------------------------------------

import argparse
from pathlib import Path

import torch

# QEfficient's from_pretrained path is what runs the MXFP4 -> float dequant
# (via QEffMxfp4HfQuantizer + Mxfp4GptOssExpertDequantizeTransform).
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b",
                    help="HF repo id or local path of the MXFP4 gpt_oss checkpoint")
    ap.add_argument("--out", default="gpt-oss-20b-dequant",
                    help="output directory for the plain-float checkpoint")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"],
                    help="dtype to save the dequantized weights in. bfloat16 ~halves disk/RAM "
                         "vs float32; use float32 only if your compile target requires it.")
    ap.add_argument("--num-layers", type=int, default=0,
                    help="if >0, truncate to this many hidden layers for a fast smoke test")
    args = ap.parse_args()

    save_dtype = getattr(torch, args.dtype)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading + dequantizing {args.model} via QEfficient from_pretrained ...")
    # IMPORTANT: from_pretrained (NOT from_config + init_empty_weights) so the
    # MXFP4 experts are actually unpacked to float. This is the RAM-heavy load.
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model,
        continuous_batching=False,
    )
    model = qeff_model.model  # underlying HF module, experts now dequantized

    if args.num_layers and args.num_layers > 0:
        # Smoke-test mode: keep only the first N decoder layers.
        try:
            model.model.layers = model.model.layers[: args.num_layers]
            model.config.num_hidden_layers = args.num_layers
            print(f"  (smoke test) truncated to {args.num_layers} layers")
        except Exception as exc:  # noqa: BLE001
            print(f"  (smoke test) could not truncate layers: {exc}")

    print(f"Casting to {args.dtype} and saving plain-float checkpoint ...")
    model = model.to(dtype=save_dtype)

    # Critical: drop the quantization_config so future loads (the weight-free
    # export) do NOT re-trigger the MXFP4 path. We want a plain dense checkpoint.
    if hasattr(model.config, "quantization_config"):
        try:
            delattr(model.config, "quantization_config")
        except Exception:
            model.config.quantization_config = None
    # Some configs also keep it under _name_or_path / dict; scrub defensively.
    if getattr(model.config, "quantization_config", None) is not None:
        model.config.quantization_config = None

    # Sharded safetensors save keeps peak memory lower than one giant file.
    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="5GB")

    # Tokenizer too, so the output dir is a drop-in model path.
    AutoTokenizer.from_pretrained(args.model).save_pretrained(str(out_dir))

    print(f"\nDone. Plain-float checkpoint written to: {out_dir.resolve()}")
    print("Point weight_free_export_from_config.py at this directory:")
    print(f'    model_name = "{out_dir}"')
    print("Sanity check the saved files:")
    for p in sorted(out_dir.glob('*.safetensors')):
        print(f"    {p.name}  ({p.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()

#python dequantize_gpt_oss.py --model openai/gpt-oss-20b --out gpt-oss-2L --num-layers 2
#python dequantize_gpt_oss.py --model openai/gpt-oss-20b --out gpt-oss-20b-dequant --dtype bfloat16
python3 examples/text_generation/dequantize.py --model openai/gpt-oss-20b --out gpt-oss-2L -
-num-layers 2
`CLIPImageProcessor` requires torchvision (not installed); falling back to `CLIPImageProcessorPil` for backward compatibility. Install torchvision to use the default backend, or import `CLIPImageProcessorPil` directly to silence this warning.
`SiglipImageProcessor` requires torchvision (not installed); falling back to `SiglipImageProcessorPil` for backward compatibility. Install torchvision to use the default backend, or import `SiglipImageProcessorPil` directly to silence this warning.
Loading + dequantizing openai/gpt-oss-20b via QEfficient from_pretrained ...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████| 459/459 [00:03<00:00, 117.64it/s]
  (smoke test) truncated to 2 layers
Casting to bfloat16 and saving plain-float checkpoint ...
Traceback (most recent call last):
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/huggingface_hub/dataclasses.py", line 255, in validate
    validator(self)
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/transformers/configuration_utils.py", line 466, in validate_layer_type
    raise ValueError(
ValueError: `num_hidden_layers` (2) must be equal to the number of layer types (24)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/amarshar/weightfree-tf5/examples/text_generation/dequantize.py", line 93, in <module>
    main()
  File "/home/amarshar/weightfree-tf5/examples/text_generation/dequantize.py", line 79, in main
    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="5GB")
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 3288, in save_pretrained
    model_to_save.config.save_pretrained(save_directory)
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/transformers/configuration_utils.py", line 528, in save_pretrained
    self.validate()
  File "/home/amarshar/weightfree-tf5/.venv/lib/python3.12/site-packages/huggingface_hub/dataclasses.py", line 257, in validate
    raise StrictDataclassClassValidationError(validator=validator.__name__, cause=e) from e
huggingface_hub.errors.StrictDataclassClassValidationError: Class validation error for validator 'validate_layer_type':
    ValueError: `num_hidden_layers` (2) must be equal to the number of layer types (24)
