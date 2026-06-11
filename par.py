#!/usr/bin/env python3
"""
Download a GLM-MoE-DSA (or any qwen2_moe-convention MoE) checkpoint and, in a
single pass, produce ONE on-disk artifact that every consumer reads:

  * dtype:  bf16 -> fp32           (QAIC compiler / weight-spec requirement)
  * layout: per-expert gate/up/down -> stacked gate_up_proj / down_proj
            (the state_dict layout the weight-free ONNX initializers reference;
             replicates transformers conversion_mapping.py qwen2_moe pattern:
             MergeModulelist(dim=0) + Concatenate(dim=1) — verified bit-equal)
  * drops the unused MTP / next-N head (optional)
  * never keeps the bf16 shards (deleted as soon as each is consumed)

The output directory is a valid `from_pretrained` target (stacked keys load by
direct name match — verified on transformers 5.5.4, logits bit-identical) AND
its safetensors keys match module state_dict names exactly, so the QEfficient
weight spec resolves every initializer with zero further copying.

Space for zai-org/GLM-5.1: ~3.0 TB fp32 total, vs ~7+ TB for
bf16 cache + fp32 copy + manually stacked copy.

RAM peak: ~1 MoE layer of stacked fp32 in flight (GLM-5.1: ~39 GB) + one
source-shard's non-expert tensors. Checkpoints are layer-sequential, so
usually exactly one layer is open at a time.

Usage:
  python download_fp32_stacked.py --repo zai-org/GLM-5.1 --out /data/glm51-fp32-stacked
Resume-safe: completed output files are skipped on re-run.
"""

import argparse
import json
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from safetensors.torch import save_file

AUX_FILES = [
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "special_tokens_map.json", "chat_template.jinja", "vocab.json", "merges.txt",
]

EXPERT_RE = re.compile(r"^(model\.layers\.(\d+)\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


class LayerStacker:
    """Accumulates one MoE layer's per-expert tensors into stacked fp32 params."""

    def __init__(self, prefix: str, num_experts: int):
        self.prefix = prefix                 # e.g. "model.layers.7.mlp.experts"
        self.num_experts = num_experts
        self.gate_up = None                  # (E, 2I, H) fp32
        self.down = None                     # (E, H, I)  fp32
        self.filled = 0

    def add(self, expert_idx: int, kind: str, t: torch.Tensor) -> None:
        t = t.to(torch.float32)
        if kind in ("gate_proj", "up_proj"):
            I, H = t.shape
            if self.gate_up is None:
                self.gate_up = torch.empty(self.num_experts, 2 * I, H, dtype=torch.float32)
            off = 0 if kind == "gate_proj" else I
            self.gate_up[expert_idx, off:off + I, :] = t       # cat([gate,up], dim=1)
        else:                                                  # down_proj
            H, I = t.shape
            if self.down is None:
                self.down = torch.empty(self.num_experts, H, I, dtype=torch.float32)
            self.down[expert_idx] = t
        self.filled += 1

    @property
    def complete(self) -> bool:
        return self.filled == 3 * self.num_experts

    def tensors(self) -> dict:
        return {f"{self.prefix}.gate_up_proj": self.gate_up, f"{self.prefix}.down_proj": self.down}


def atomic_save(tensors: dict, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp))
    tmp.replace(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--keep-mtp", action="store_true", help="Keep MTP/next-N weights (dropped by default)")
    ap.add_argument("--no-stack", action="store_true", help="Keep per-expert layout (fp32 conversion only)")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    repo_files = list_repo_files(args.repo, revision=args.revision)
    for name in AUX_FILES:
        if name in repo_files and not (out / name).exists():
            shutil.copy2(hf_hub_download(args.repo, name, revision=args.revision), out / name)
            print(f"[aux] {name}")

    config = json.loads((out / "config.json").read_text())
    L = int(config["num_hidden_layers"])
    E = int(config.get("n_routed_experts") or config.get("num_local_experts"))
    drop_mtp = not args.keep_mtp

    # ---- plan: which source shard holds which keys ----
    index_path = out / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:  # single-file checkpoint (tiny models)
        single = next(f for f in repo_files if f.endswith(".safetensors"))
        weight_map = {"*": single}

    shard_order = sorted(set(weight_map.values()))
    new_weight_map: dict[str, str] = {}
    stackers: dict[int, LayerStacker] = {}
    pending_layer_keys = defaultdict(int)      # how many expert tensors of a layer remain unseen
    if "*" not in weight_map:
        for key in weight_map:
            m = EXPERT_RE.match(key)
            if m and not args.no_stack:
                pending_layer_keys[int(m.group(2))] += 1

    def is_mtp(key: str) -> bool:
        m = LAYER_RE.match(key)
        return drop_mtp and m is not None and int(m.group(1)) >= L

    with tempfile.TemporaryDirectory(prefix="hf_bf16_") as tmp_cache:
        for si, shard in enumerate(shard_order):
            base_out = out / (f"base-{Path(shard).stem}.safetensors" if not args.no_stack else shard)
            shard_done = base_out.exists()
            # a shard is only truly done if every stacked layer it feeds is also done
            if shard_done and all(
                (out / f"experts-layer-{li:05d}.safetensors").exists()
                for li in {int(EXPERT_RE.match(k).group(2)) for k, v in weight_map.items()
                           if v == shard and EXPERT_RE.match(k)} if not args.no_stack
            ):
                print(f"[{si + 1}/{len(shard_order)}] {shard}  (done, skip)")
                # still need its keys in the new weight map:
                for key, v in weight_map.items():
                    if v != shard or is_mtp(key):
                        continue
                    m = EXPERT_RE.match(key)
                    if m and not args.no_stack:
                        li = int(m.group(2))
                        new_weight_map[f"{m.group(1)}.gate_up_proj"] = f"experts-layer-{li:05d}.safetensors"
                        new_weight_map[f"{m.group(1)}.down_proj"] = f"experts-layer-{li:05d}.safetensors"
                    else:
                        new_weight_map[key] = base_out.name
                continue

            src = Path(hf_hub_download(args.repo, shard, revision=args.revision, cache_dir=tmp_cache))
            base_tensors = {}
            with safe_open(str(src), framework="pt") as f:
                for key in f.keys():
                    if is_mtp(key):
                        continue
                    m = EXPERT_RE.match(key)
                    if m and not args.no_stack:
                        li, ei, kind = int(m.group(2)), int(m.group(3)), m.group(4)
                        st = stackers.setdefault(li, LayerStacker(m.group(1), E))
                        st.add(ei, kind, f.get_tensor(key))
                        new_weight_map[f"{st.prefix}.gate_up_proj"] = f"experts-layer-{li:05d}.safetensors"
                        new_weight_map[f"{st.prefix}.down_proj"] = f"experts-layer-{li:05d}.safetensors"
                        if st.complete:
                            atomic_save(st.tensors(), out / f"experts-layer-{li:05d}.safetensors")
                            del stackers[li]
                            print(f"    [stacked] layer {li}: gate_up {tuple(st.gate_up.shape)}, down {tuple(st.down.shape)}")
                    else:
                        t = f.get_tensor(key)
                        base_tensors[key] = t.to(torch.float32) if t.is_floating_point() else t
                        new_weight_map[key] = base_out.name
            if base_tensors:
                atomic_save(base_tensors, base_out)
            # delete bf16 immediately (file + hub blob)
            blob = src.resolve()
            src.unlink(missing_ok=True)
            if blob != src:
                blob.unlink(missing_ok=True)
            print(f"[{si + 1}/{len(shard_order)}] {shard}  -> {base_out.name} ({len(base_tensors)} base tensors), "
                  f"{len(stackers)} layer(s) in flight")

    if stackers:
        raise RuntimeError(f"Incomplete expert layers after all shards: {sorted(stackers)} — checkpoint missing keys?")

    # ---- finalize metadata ----
    for k in ("dtype", "torch_dtype"):
        if k in config:
            config[k] = "float32"
    (out / "config.json").write_text(json.dumps(config, indent=2))
    files = sorted(set(new_weight_map.values()))
    index = {
        "metadata": {"total_size": sum((out / f).stat().st_size for f in files)},
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"\nDone. FP32 stacked checkpoint at: {out}  ({len(files)} files)")
    print(f'Use it as:  model_name = "{out.resolve()}"')


if __name__ == "__main__":
    main()



# inside convert_checkpoint_to_fp32, replacing the body of the per-file loop
keys_needed = needed[old_idx]
with safe_open(str(abs_path), framework="pt") as f:
    already_fp32 = all(f.get_slice(k).get_dtype() == "F32" for k in keys_needed)

if already_fp32:
    # Checkpoint shard is already FP32 (download_fp32.py) — reference in place, zero copy.
    new_files.append(ExternalDataFile(path=str(abs_path), format="safetensors"))
    print(f"  {abs_path.name}  ({len(keys_needed)} tensors)  ->  referenced in place (already fp32)")
else:
    tensors = load_file(str(abs_path))
    fp32_tensors = {k: v.to(torch.float32) for k, v in tensors.items() if k in keys_needed}
    out_name = f"model_{old_to_new[old_idx]:04d}.safetensors"
    save_file(fp32_tensors, str(export_dir / out_name))
    new_files.append(ExternalDataFile(path=out_name, format="safetensors"))
    print(f"  {abs_path.name}  ({len(keys_needed)}/{len(tensors)} tensors)  ->  {out_name}  (float32)")
[TIMING] download model-00001-of-00282.safetensors: 98.3s  (5095 MB)
[1/282] model-00001-of-00282.safetensors → base-model-00001-of-00282.safetensors (35 base tensors, 10191 MB written, 0 layer(s) stacked, 0 in-flight)  proc: 6.8s
model-00002-of-00282.safetensors: 100%|████████████████████████████████████████████████████████████████| 5.35G/5.35G [01:57<00:00, 45.5MB/s]
[TIMING] download model-00002-of-00282.safetensors: 117.8s  (5104 MB)
[2/282] model-00002-of-00282.safetensors → base-model-00002-of-00282.safetensors (2 base tensors, 128 MB written, 0 layer(s) stacked, 1 in-flight)  proc: 8.2s
model-00003-of-00282.safetensors: 100%|████████████████████████████████████████████████████████████████| 5.36G/5.36G [01:34<00:00, 56.9MB/s]
[TIMING] download model-00003-of-00282.safetensors: 94.5s  (5112 MB)
[3/282] model-00003-of-00282.safetensors → base-model-00003-of-00282.safetensors (0 base tensors, 0 MB written, 0 layer(s) stacked, 1 in-flight)  proc: 7.5s
