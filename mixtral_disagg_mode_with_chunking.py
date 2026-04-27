# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
#
# End-to-end disaggregated serving script for Mixtral-8x7B-Instruct-v0.1
# on Qualcomm Cloud AI 100 (4 SOCs).
#
# Architecture overview:
#   - PREFILL QPC  : runs on 4 devices, handles chunked prompt ingestion
#   - DECODE  QPC  : runs on 2 devices, handles auto-regressive token generation
#
# KV cache is produced by the prefill session as RetainedState outputs and fed
# directly into the decode session as inputs — this is the "disaggregated"
# serving pattern.
#
# CRITICAL: Mixtral has 8 experts. The default EXPERT_BLOCKING_NUM_NSP=16
# does NOT divide 8 and silently falls back to the serial loop.
# We explicitly set NSP=8 here so the NSP-blocked prefill path is used.
#
# Usage:
#   python mixtral_disagg_mode_with_chunking.py
#   python mixtral_disagg_mode_with_chunking.py --model_path /path/to/Mixtral-8x7B-Instruct-v0.1
#   python mixtral_disagg_mode_with_chunking.py --model_path /path/to/model --ctx_len 2048
# -----------------------------------------------------------------------------

import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

# CRITICAL: Must be set BEFORE importing QEfficient so the constant is read
# at module load time inside modeling_mixtral.py.
# Mixtral has 8 experts → NSP must be a divisor of 8: {1, 2, 4, 8}
# NSP=8 means all 8 experts form a single NSP group → 1 scatter/gather pass.
os.environ.setdefault("EXPERT_BLOCKING_NUM_NSP", "8")

from QEfficient import QEFFAutoModelForCausalLM  # noqa: E402
from QEfficient.generation.cloud_infer import QAICInferenceSession  # noqa: E402


# ---------------------------------------------------------------------------
# CLI args — override any default from the command line
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Mixtral disagg serving on AI100")
    parser.add_argument(
        "--model_path",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Local path or HuggingFace model ID for Mixtral-8x7B-Instruct-v0.1",
    )
    parser.add_argument(
        "--ctx_len",
        type=int,
        default=4096,
        help="Max context length. Mixtral sliding window is 4096 (default).",
    )
    parser.add_argument(
        "--prefill_seq_len",
        type=int,
        default=128,
        help="Chunk size for prefill. Prompt is processed in chunks of this size.",
    )
    parser.add_argument(
        "--prefill_num_devices",
        type=int,
        default=4,
        help="Number of AI100 SOCs for the prefill QPC.",
    )
    parser.add_argument(
        "--decode_num_devices",
        type=int,
        default=2,
        help="Number of AI100 SOCs for the decode QPC.",
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        default=16,
        help="NSP cores per device.",
    )
    parser.add_argument(
        "--generation_len",
        type=int,
        default=200,
        help="Number of new tokens to generate.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    MODEL_ID = args.model_path
    CTX_LEN = args.ctx_len
    PREFILL_SEQ_LEN = args.prefill_seq_len
    PREFILL_NUM_DEVICES = args.prefill_num_devices
    DECODE_NUM_DEVICES = args.decode_num_devices
    NUM_CORES = args.num_cores
    GENERATION_LEN = args.generation_len

    print("=" * 70)
    print("Mixtral-8x7B Disaggregated Serving — Qualcomm Cloud AI 100")
    print("=" * 70)
    print(f"  Model            : {MODEL_ID}")
    print(f"  ctx_len          : {CTX_LEN}")
    print(f"  prefill_seq_len  : {PREFILL_SEQ_LEN}")
    print(f"  prefill_devices  : {PREFILL_NUM_DEVICES}")
    print(f"  decode_devices   : {DECODE_NUM_DEVICES}")
    print(f"  num_cores        : {NUM_CORES}")
    print(f"  NSP (expert blk) : {os.environ['EXPERT_BLOCKING_NUM_NSP']}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Tokenizer + config (from local path or HF hub)
    # -----------------------------------------------------------------------
    print("\n[1/5] Loading tokenizer and config ...")
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    num_layers = config.num_hidden_layers  # 32 for Mixtral-8x7B

    # -----------------------------------------------------------------------
    # Load model into QEfficient wrapper
    # -----------------------------------------------------------------------
    print("\n[2/5] Loading model weights (this may take a few minutes for 8x7B) ...")
    t0 = time.time()
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(MODEL_ID)
    print(f"      Weights loaded in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Compile Step 1: DECODE QPC
    #   - prefill_seq_len=1  (decode processes one token at a time)
    #   - retain_full_kv=True  so the full KV buffer is exposed as I/O
    #     (needed so we can inject KV from the prefill session)
    #   - offload_pt_weights=False  so weights stay in memory for the next
    #     compile call (prefill). Set True if memory is tight — but then you
    #     must reload the model before the prefill compile.
    # -----------------------------------------------------------------------
    print("\n[3/5] Compiling DECODE QPC ...")
    print(f"      Devices: {DECODE_NUM_DEVICES} | ctx_len: {CTX_LEN} | seq_len: 1")
    t0 = time.time()
    decode_qpc_path = qeff_model.compile(
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        num_cores=NUM_CORES,
        num_devices=DECODE_NUM_DEVICES,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        offload_pt_weights=False,   # keep weights for prefill compile below
        retain_full_kv=True,        # expose full KV as I/O tensors
    )
    print(f"      Decode QPC path : {decode_qpc_path}")
    print(f"      Compiled in     : {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Compile Step 2: PREFILL QPC
    #   - prefill_only=True   tells QEfficient this is a prefill-only binary
    #   - enable_chunking=True activates PrefillOnlyChunkedTransform which
    #     swaps QEffMixtralSparseMoeBlock → QEffPrefillChunkedMixtralSparseMoeBlock
    #   - prefill_seq_len=PREFILL_SEQ_LEN  the chunk size
    #
    # NOTE: This compile call may FAIL on the first run because the onnx_path
    # was cached from the decode compile. If it errors with a path conflict,
    # the error message will print the exact command to run manually.
    # In that case, copy the printed qpc path and paste it as prefill_qpc_path
    # below (commented out), then comment out this compile block.
    # -----------------------------------------------------------------------
    print(f"\n[4/5] Compiling PREFILL QPC (chunked, NSP-blocked MoE) ...")
    print(f"      Devices: {PREFILL_NUM_DEVICES} | ctx_len: {CTX_LEN} | chunk: {PREFILL_SEQ_LEN}")
    t0 = time.time()

    # --- If the compile below errors, paste the printed path here instead ---
    # prefill_qpc_path = "/path/from/error/message"

    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=NUM_CORES,
        num_devices=PREFILL_NUM_DEVICES,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        prefill_only=True,          # export prefill-only graph
        enable_chunking=True,       # activate NSP-blocked MoE path
    )
    print(f"      Prefill QPC path: {prefill_qpc_path}")
    print(f"      Compiled in     : {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Prepare prompt inputs
    # -----------------------------------------------------------------------
    print("\n[5/5] Running inference ...")

    # Mixtral Instruct uses the [INST] chat template
    prompt = (
        "[INST] Explain the key differences between classical computing and "
        "quantum computing. What are the main advantages and limitations of "
        "each approach? [/INST]"
    )
    print(f"\n  Prompt: {prompt}\n")

    # Tokenize
    raw_inputs = tokenizer(prompt, return_tensors="np", padding=True)
    prompt_token_len = int(raw_inputs["attention_mask"].sum())

    # Pad prompt length up to nearest multiple of PREFILL_SEQ_LEN
    num_chunks = -(prompt_token_len // -PREFILL_SEQ_LEN)   # ceiling division
    padded_len = num_chunks * PREFILL_SEQ_LEN
    inputs = tokenizer(
        prompt,
        return_tensors="np",
        padding="max_length",
        max_length=padded_len,
    )

    # Replace attention_mask with position_ids:
    #   valid positions get their 0-based index; padding positions get -1
    position_ids_np = np.where(
        inputs.pop("attention_mask"),
        np.arange(padded_len),
        -1,
    ).astype(np.int32)
    inputs["position_ids"] = position_ids_np
    inputs.pop("token_type_ids", None)
    inputs.pop("past_key_values", None)

    # Convert to numpy (QAICInferenceSession expects numpy arrays)
    inputs = {k: np.array(v) for k, v in inputs.items()}

    generation_len = min(GENERATION_LEN, CTX_LEN - prompt_token_len - 1)

    print(f"  Prompt tokens    : {prompt_token_len}")
    print(f"  Padded length    : {padded_len}  ({num_chunks} chunk(s) of {PREFILL_SEQ_LEN})")
    print(f"  Tokens to decode : {generation_len}")

    # -----------------------------------------------------------------------
    # Open inference sessions
    # -----------------------------------------------------------------------
    prefill_session = QAICInferenceSession(prefill_qpc_path)
    decode_session = QAICInferenceSession(decode_qpc_path)

    # -----------------------------------------------------------------------
    # PREFILL — process prompt in chunks
    # Each chunk outputs the updated KV cache as <name>_RetainedState tensors.
    # We accumulate the final chunk's KV as the starting state for decode.
    # -----------------------------------------------------------------------
    print("\n  --- PREFILL ---")
    prefill_t0 = time.time()
    qpc_out = None

    for chunk_idx in range(num_chunks):
        start = chunk_idx * PREFILL_SEQ_LEN
        end = start + PREFILL_SEQ_LEN
        chunk_inputs = {
            "input_ids":    inputs["input_ids"][:, start:end],
            "position_ids": inputs["position_ids"][:, start:end],
        }
        # Inject KV from previous chunk (skip on first chunk — session has zeros)
        if qpc_out is not None:
            for layer in range(num_layers):
                chunk_inputs[f"past_key.{layer}"]   = qpc_out[f"past_key.{layer}_RetainedState"]
                chunk_inputs[f"past_value.{layer}"] = qpc_out[f"past_value.{layer}_RetainedState"]

        chunk_t = time.time()
        qpc_out = prefill_session.run(chunk_inputs)
        print(f"    Chunk {chunk_idx + 1}/{num_chunks}: {time.time() - chunk_t:.3f}s")

    prefill_elapsed = time.time() - prefill_t0
    prefill_tok_per_sec = prompt_token_len / prefill_elapsed
    print(f"  Prefill done: {prefill_elapsed:.3f}s  ({prefill_tok_per_sec:.1f} tok/s)")

    # -----------------------------------------------------------------------
    # Transition: prefill → decode
    # The first decode input token is the argmax of the last prefill logits.
    # Position id is one beyond the last valid (non-padding) position.
    # -----------------------------------------------------------------------
    first_token = int(np.argmax(qpc_out["logits"]))
    last_valid_pos = int(np.max(inputs["position_ids"]))

    all_output_tokens = [first_token]

    decode_inputs = {
        "input_ids":    np.array([[first_token]], dtype=np.int64),
        "position_ids": np.array([[last_valid_pos + 1]], dtype=np.int32),
    }
    for layer in range(num_layers):
        decode_inputs[f"past_key.{layer}"]   = qpc_out[f"past_key.{layer}_RetainedState"]
        decode_inputs[f"past_value.{layer}"] = qpc_out[f"past_value.{layer}_RetainedState"]

    # -----------------------------------------------------------------------
    # DECODE — auto-regressive generation loop
    # -----------------------------------------------------------------------
    print("\n  --- DECODE ---")
    decode_t0 = time.time()

    # First decode step (KV comes from prefill, so we time it separately)
    first_decode_t = time.time()
    decode_out = decode_session.run(decode_inputs)
    print(f"  First decode step (KV injection): {time.time() - first_decode_t:.3f}s")

    next_token = int(np.argmax(decode_out["logits"]))
    all_output_tokens.append(next_token)

    pos_id = decode_inputs["position_ids"] + 1
    loop_inputs = {
        "input_ids":    np.array([[next_token]], dtype=np.int64),
        "position_ids": pos_id,
    }
    for layer in range(num_layers):
        loop_inputs[f"past_key.{layer}"]   = decode_out[f"past_key.{layer}_RetainedState"]
        loop_inputs[f"past_value.{layer}"] = decode_out[f"past_value.{layer}_RetainedState"]

    # Remaining decode steps
    loop_t0 = time.time()
    for step in range(generation_len - 2):
        decode_out = decode_session.run(loop_inputs)
        next_token = int(np.argmax(decode_out["logits"]))
        all_output_tokens.append(next_token)

        # Stop at EOS
        if next_token == tokenizer.eos_token_id:
            print(f"  EOS hit at step {step + 1}")
            break

        pos_id = pos_id + 1
        loop_inputs["input_ids"] = np.array([[next_token]], dtype=np.int64)
        loop_inputs["position_ids"] = pos_id
        for layer in range(num_layers):
            loop_inputs[f"past_key.{layer}"]   = decode_out[f"past_key.{layer}_RetainedState"]
            loop_inputs[f"past_value.{layer}"] = decode_out[f"past_value.{layer}_RetainedState"]

    loop_elapsed = time.time() - loop_t0
    total_decode_steps = len(all_output_tokens) - 1  # exclude first token (from prefill logits)
    decode_tok_per_sec = (total_decode_steps - 1) / loop_elapsed if loop_elapsed > 0 else 0

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    output_text = tokenizer.decode(all_output_tokens, skip_special_tokens=True)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  Input:\n  {prompt}")
    print(f"\n  Output:\n  {output_text}")
    print("\n  --- Timing ---")
    print(f"  Prefill  : {prefill_elapsed:.3f}s  ({prefill_tok_per_sec:.1f} tok/s, {prompt_token_len} tokens)")
    print(f"  Decode   : {loop_elapsed:.3f}s  ({decode_tok_per_sec:.1f} tok/s, {total_decode_steps} tokens)")
    print(f"  Total    : {time.time() - prefill_t0:.3f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
