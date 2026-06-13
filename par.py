# Diagnostic: compare HF base vs QEff model on a single prefill forward pass.
# Both run in FP32 on CPU. Tells us WHERE the QEff model diverges from HF base.
#
# CHANGE vs the original harness:
#   The original passed a 256-slot dummy KV cache to the QEff model. That set
#   past_seen_tokens=256, so the DSA indexer (index_topk=2048) scored over 256
#   positions, 248 of which were unwritten zero-slots that the HF base run (which
#   used use_cache=False, total_len=8) never saw. That asymmetry is what made the
#   two models diverge at layer 6 in the original logs.
#
#   This version runs the QEff model in the SAME regime as HF: no cache,
#   use_cache=False  ->  past_seen_tokens=0, total_len=8, indexer scores over the
#   8 real tokens only. If HF and QEff now match across all 6 layers, the modeling
#   math is correct and every downstream failure (ORT/QPC garbage) is a
#   cache-regime / masking / specialization problem, NOT a modeling bug.

import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/huggingface_hub/glm51-fp32-stacked"
PROMPT = "what is faith ?"
NUM_LAYERS = 6

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
config = AutoConfig.from_pretrained(MODEL_PATH)
config.num_hidden_layers = NUM_LAYERS

input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
SEQ_LEN = input_ids.shape[1]
print(f"Prompt tokens: {SEQ_LEN} → {input_ids.tolist()}")

# ── HF base model ─────────────────────────────────────────────────────────────
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, config=config, torch_dtype=torch.float32, ignore_mismatched_sizes=True
)
hf_model.eval()

with torch.no_grad():
    hf_out = hf_model(input_ids, use_cache=False, output_hidden_states=True)

hf_logits = hf_out.logits            # [1, seq, vocab]
hf_next_token = hf_logits[0, -1].argmax().item()
print(f"\nHF  next token: {hf_next_token}")
print(f"HF  logit top5: {hf_logits[0,-1].topk(5).indices.tolist()}")
if hf_out.hidden_states:
    for i, h in enumerate(hf_out.hidden_states):
        print(f"  HF hidden[{i}] last-pos norm: {h[0,-1].norm().item():.4f}")

# ── QEff patched model ────────────────────────────────────────────────────────
# Load the same weights but apply QEff patches.
from QEfficient.transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    QEffGlmMoeDsaForCausalLM,
    QEffGlmMoeDsaModel,
    QEffGlmMoeDsaAttention,
    QEffGlmMoeDsaIndexer,
    QEffGlmMoeDsaMoE,
    QEffGlmMoeDsaTopkRouter,
    QEffGlmMoeDsaDenseDecoderLayer,
    QEffGlmMoeDsaSparseDecoderLayer,
)

qeff_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, config=config, torch_dtype=torch.float32, ignore_mismatched_sizes=True
)
# Patch classes
qeff_model.__class__ = QEffGlmMoeDsaForCausalLM
qeff_model.model.__class__ = QEffGlmMoeDsaModel
layer_types = getattr(config, "mlp_layer_types", [])
for i, layer in enumerate(qeff_model.model.layers):
    lt = layer_types[i] if i < len(layer_types) else "sparse"
    layer.__class__ = QEffGlmMoeDsaDenseDecoderLayer if lt == "dense" else QEffGlmMoeDsaSparseDecoderLayer
    layer.self_attn.__class__ = QEffGlmMoeDsaAttention
    layer.self_attn.indexer.__class__ = QEffGlmMoeDsaIndexer
    layer.self_attn.__qeff_init__()
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
        layer.mlp.gate.__class__ = QEffGlmMoeDsaTopkRouter
        layer.mlp.__class__ = QEffGlmMoeDsaMoE
        layer.mlp.__qeff_init__()
qeff_model.model.__qeff_init__()
qeff_model.eval()

# ── QEff forward, configured to MIRROR the HF base run ────────────────────────
# We still pass explicit position_ids = [0..seq-1]; with past_seen_tokens=0 these
# match what HF computes internally.
position_ids = torch.arange(SEQ_LEN).unsqueeze(0)   # [[0, 1, ..., seq-1]]


def _run_qeff(past_key_values, use_cache, label):
    with torch.no_grad():
        out = qeff_model(
            input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
        )
    # Proof that the comparison is now fair: every layer's hidden state must be
    # length SEQ_LEN (8), not 256. If this prints 256 the cache regime is still wrong.
    if out.hidden_states:
        seen_len = out.hidden_states[0].shape[1]
        ok = "OK" if seen_len == SEQ_LEN else "!! MISMATCH"
        print(f"  [{label}] hidden seq_len = {seen_len} (expect {SEQ_LEN}) {ok}")
    return out


try:
    # Primary path: cache-free, the truest mirror of the HF call above.
    #   model.forward: past_key_values is None  -> past_seen_tokens = 0
    #   attention:     past_key_values is None  -> skips .update(), total_len = SEQ_LEN
    #   indexer:       indexer_key_cache is None -> rebuilds k_cached fresh at SEQ_LEN
    qeff_out = _run_qeff(past_key_values=None, use_cache=False, label="no-cache")
except (AssertionError, TypeError, ValueError, AttributeError, RuntimeError) as e:
    # Fallback: if some QEff path assumes a Cache object exists, use an EMPTY cache
    # (seq_len=0). past_seen_tokens still resolves to 0 and total_len still grows to
    # SEQ_LEN — unlike a pre-filled seq_len=8 or 256 cache, which would set
    # past_seen_tokens to its allocated length and re-introduce the divergence.
    print(f"  [no-cache path failed: {type(e).__name__}: {e}]")
    print(f"  → retrying with an EMPTY (seq_len=0) cache")
    empty_pkv = qeff_model.get_dummy_pkv_cache(config, batch_size=1, seq_len=0)
    qeff_out = _run_qeff(past_key_values=empty_pkv, use_cache=True, label="empty-cache")

# QEff ForCausalLM slices logits to a single position when position_ids is given
# (argmax of position_ids -> last token), so qeff_logits may be [1, 1, vocab].
# Indexing [0, -1] still selects that last position, matching hf_logits[0, -1].
qeff_logits = qeff_out.logits
qeff_next_token = qeff_logits[0, -1].argmax().item()
print(f"\nQEff next token: {qeff_next_token}")
print(f"QEff logit top5: {qeff_logits[0,-1].topk(5).indices.tolist()}")
if qeff_out.hidden_states:
    for i, h in enumerate(qeff_out.hidden_states):
        print(f"  QEff hidden[{i}] last-pos norm: {h[0,-1].norm().item():.4f}")

# ── Comparison ────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Match: {hf_next_token == qeff_next_token}")
print(f"HF  vs QEff logit max diff: {(hf_logits[0,-1] - qeff_logits[0,-1]).abs().max().item():.6f}")
print(f"HF  vs QEff logit cos-sim:  {torch.cosine_similarity(hf_logits[0,-1:], qeff_logits[0,-1:]).item():.6f}")
if hf_out.hidden_states and qeff_out.hidden_states:
    for i, (hh, qh) in enumerate(zip(hf_out.hidden_states, qeff_out.hidden_states)):
        diff = (hh[0, -1] - qh[0, -1]).abs()
        l2 = (hh[0, -1] - qh[0, -1]).norm().item()
        print(f"  Layer {i}: hidden diff max={diff.max().item():.6f}  mean={diff.mean().item():.6f}  l2={l2:.6f}")

# ── How to read this ──────────────────────────────────────────────────────────
# - "hidden seq_len = 8 ... OK"  confirms the comparison is now fair (both at len 8).
# - If every layer diff is ~1e-4 or smaller (fp32 reduction-order noise) and
#   Match=True: the QEff MODELING MATH IS CORRECT. The ORT/QPC garbage is then a
#   cache-regime / specialization / custom_io problem — fix prefill_seq_len and the
#   indexer custom_io, not the model.
# - If layer 6 (first sparse/indexer-gated layer) still jumps by orders of magnitude
#   even with seq_len=8 and past_seen_tokens=0: the INDEXER MATH ITSELF is wrong
#   (e.g. future_mask / valid_topk vs HF's additive-mask-then-topk). Fix that first.
