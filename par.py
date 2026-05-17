# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.exporter.weight_free import _default_weights_roots ,load_weight_free_ort_inputs
from QEfficient.exporter.weight_spec import (
    ExternalDataFile,
    load_weight_spec,
    resolve_weight_spec_path,
    save_weight_spec,
)
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner


def convert_checkpoint_to_fp32(onnx_path: Path, weight_spec_path: Path) -> None:
    """
    Load each safetensors checkpoint file, cast all tensors to FP32,
    save next to the ONNX, and update weight_spec.json to point there.

    This ensures the compiler sees matching dtypes between the ONNX (FLOAT)
    and the safetensors files (also FLOAT after conversion).
    """
    spec = load_weight_spec(weight_spec_path)
    export_dir = onnx_path.parent
    candidate_roots = _default_weights_roots(weight_spec_path, spec)
    local_files = [
        ExternalDataFile(
            path=f"model_{idx:04d}.safetensors" if len(spec.files) > 1 else "model.safetensors",
            format="safetensors",
        )
        for idx, _ in enumerate(spec.files)
    ]

    # Reuse previously materialized local safetensors even if a fresh export
    # rewrote the spec back to the original checkpoint paths.
    if local_files and all((export_dir / ext_file.path).is_file() for ext_file in local_files):
        print("Reusing existing local FP32 safetensors.")
        spec.files = local_files
        save_weight_spec(weight_spec_path, spec)
        _sync_embedded_extdata(onnx_path, weight_spec_path)
        return

    new_files = []
    for idx, ext_file in enumerate(spec.files):
        rel_path = Path(ext_file.path)
        abs_path = rel_path if rel_path.is_absolute() else None
        if abs_path is None:
            for root in candidate_roots:
                candidate = root / rel_path
                if candidate.exists():
                    abs_path = candidate
                    break
        if abs_path is None or not abs_path.exists():
            raise FileNotFoundError(f"Cannot resolve external data file: {ext_file.path}")

        tensors = load_file(str(abs_path))
        fp32_tensors = {k: v.to(torch.float32) for k, v in tensors.items()}

        out_name = f"model_{idx:04d}.safetensors" if len(spec.files) > 1 else "model.safetensors"
        save_file(fp32_tensors, str(export_dir / out_name))
        new_files.append(ExternalDataFile(path=out_name, format="safetensors"))
        print(f"  {abs_path.name}  ({next(iter(tensors.values())).dtype})  →  {out_name}  (float32)")

    spec.files = new_files
    save_weight_spec(weight_spec_path, spec)
    _sync_embedded_extdata(onnx_path, weight_spec_path)


def _sync_embedded_extdata(onnx_path: Path, weight_spec_path: Path) -> None:
    # Keep the embedded external-data metadata aligned with weight_spec.json so
    # compiler and ORT verification resolve the same files.
    updated_json = json.dumps(json.loads(weight_spec_path.read_text()), separators=(",", ":"), sort_keys=True)
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    for entry in onnx_model.metadata_props:
        if entry.key == "com.qti.aisw.extdata":
            entry.value = updated_json
            break
    tmp = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
    onnx.save(onnx_model, str(tmp))
    tmp.replace(onnx_path)


#model_name = "meta-llama/Llama-3.3-70B-Instruct"
model_name = "meta-llama/Llama-3.2-1B"
# model_name = "gpt2"
# model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
# config.num_hidden_layers = 2
config.torch_dtype = torch.float32
print(config)

CONTINUOUS_BATCHING = False
FULL_BATCH_SIZE = 4  # slots in the KV cache; active batch_size stays at 1 here # NOT VERIFIED, WIP

runner = ApiRunner(
    batch_size=1,
    tokenizer=tokenizer,
    config=config,
    prompt=["My name is"],
    prompt_len=8,
    ctx_len=32,
    full_batch_size=FULL_BATCH_SIZE if CONTINUOUS_BATCHING else None,
)

with init_empty_weights():
   meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
#meta_model=AutoModelForCausalLM.from_pretrained(model_name,config=config)

qeff_model = QEFFAutoModelForCausalLM(
    meta_model,
    pretrained_model_name_or_path=model_name,
    continuous_batching=CONTINUOUS_BATCHING,
)

export_dir = Path("test_models/weightfree_from_config")
export_start = time.perf_counter()
onnx_path = Path(
    qeff_model.export(
        export_dir=export_dir,
        use_dynamo=True,
        use_onnx_subfunctions=True,
        use_weight_free_export=True,
        offload_pt_weights=False,
    )
)
export_elapsed = time.perf_counter() - export_start
weight_spec_path = resolve_weight_spec_path(onnx_path)

print(f"Weight-free export time: {export_elapsed:.3f} sec")

print("Converting checkpoint to FP32 (one-time local materialization) ...")
fp32_convert_time_start=time.perf_counter();
convert_checkpoint_to_fp32(onnx_path, weight_spec_path)
fp32_convert_time=time.perf_counter()-fp32_convert_time_start
print(f"fp32 convert time: {fp32_convert_time:.3f} sec")
print("Compiling weight-free ONNX ...")
compile_start = time.perf_counter()
qpc_path = qeff_model.compile(
    onnx_path=str(onnx_path),
    compile_dir=str(onnx_path.parent / "qpc"),
    prefill_seq_len=8,
    ctx_len=32,
    num_devices=4,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    use_dynamo=True,
    use_onnx_subfunctions=True,
    use_weight_free_export=True,
)
compile_time = time.perf_counter()-compile_start
print(f"compile time: {compile_time:.3f} sec")
print(f"QPC: {qpc_path}")

session = ort.InferenceSession(str(onnx_path))
ort_inputs = load_weight_free_ort_inputs(weight_spec_path, runner.input_handler.prepare_ort_inputs())
ort_outputs = runner.run_ort_session(ort_inputs, session)
ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)

generated_ids = []
for _ in range(1, runner.gen_len):
    generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
    ort_inputs = runner.input_handler.update_ort_inputs(ort_inputs, ort_outputs)
    ort_inputs = load_weight_free_ort_inputs(weight_spec_path, ort_inputs)
    ort_outputs = runner.run_ort_session(ort_inputs, session)
    ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)

generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
generated_ids = np.concatenate(generated_ids, axis=1)
generated_text = runner.input_handler.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print("Running QPC generate ...")
try:
    exec_info = qeff_model.generate(
        prompts=["My name is"],
        tokenizer=tokenizer,
        automation=True,
        generation_len=runner.gen_len,
    )
    qpc_generated_ids = np.asarray(exec_info.generated_ids[0]).reshape(1, -1)
    qpc_generated_text = tokenizer.batch_decode(qpc_generated_ids, skip_special_tokens=True)

    print(exec_info)
    print(generated_ids)
    print(generated_text)
    print(qpc_generated_ids)
    print(qpc_generated_text)
except RuntimeError as exc:
    print(f"Skipping QPC generate: {exc}")

print(f"Weight-free ONNX: {onnx_path}")
print(f"Weight spec: {weight_spec_path}")
#print(generated_text)
[Warning]: While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: ["L['kwargs']['past_key_value'].layers[15]"]
[torch.onnx] Obtain model graph for `QEffLlamaForCausalLM([...]` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decompositions...
[Warning]: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
[torch.onnx] Run decompositions... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
[Warning]: # The axis name: batch_size will not be used, since it shares the same shape constraints with another axis: batch_size.
[Warning]: # The axis name: seq_len will not be used, since it shares the same shape constraints with another axis: seq_len.
[Warning]: # The axis name: ctx_len will not be used, since it shares the same shape constraints with another axis: ctx_len.
[Warning]: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
Fetching 7 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 21229.30it/s]
Weight-free export time: 43.588 sec
Converting checkpoint to FP32 (one-time local materialization) ...
  model.safetensors  (torch.bfloat16)  →  model.safetensors  (float32)
fp32 convert time: 10.800 sec
Compiling weight-free ONNX ...
['/opt/qti-aic/exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100', '-m=test_models/weightfree_from_config-a971b506b22894dc/LlamaForCausalLM.onnx', '-retained-state', '-convert-to-fp16', '-mxfp6-matmul', '-aic-num-cores=16', '-sub-functions', '-mdp-load-partition-config=test_models/weightfree_from_config-a971b506b22894dc/qpc/qpc-1a6780f10b6dead7/mdp_ts_4.json', '-network-specialization-config=test_models/weightfree_from_config-a971b506b22894dc/qpc/qpc-1a6780f10b6dead7/specializations.json', '-custom-IO-list-file=test_models/weightfree_from_config-a971b506b22894dc/qpc/qpc-1a6780f10b6dead7/custom_io.yaml', '-aic-binary-dir=test_models/weightfree_from_config-a971b506b22894dc/qpc/qpc-1a6780f10b6dead7/qpc']
compile time: 55.258 sec
QPC: test_models/weightfree_from_config-a971b506b22894dc/qpc/qpc-1a6780f10b6dead7/qpc
Running QPC generate ...

Prompt : My name is
Completion : Kelsey and I am a 2016 graduate of the University of Wisconsin-Madison. I am currently input= ['My name is']
output= [[' Kelsey and I am a 2016 graduate of the University of Wisconsin-Madison. I am currently a graduate锦锦锦锦锦锦锦锦']]
Average Prefill time a.k.a TTFT is= 0.01 sec        
Decode is= 210.77 tokens/sec        
Total is= 195.37 tokens/sec        
Total (E2E) inference time is= 0.12 sec
Average Prefill time a.k.a TTFT is= 0.01 sec        
Decode is= 210.77 tokens/sec        
Total is= 195.37 tokens/sec        
Total (E2E) inference time is= 0.12 sec
[[  735 93567   323   358  1097   264   220   679    23 19560   315   279
   3907   315 21073  5364   329  3416    13   358  1097  5131   264 19560]]
[' Kelsey and I am a 2018 graduate of the University of Wisconsin-Madison. I am currently a graduate']
[[   735  93567    323    358   1097    264    220    679     21  19560
     315    279   3907    315  21073   5364    329   3416     13    358
    1097   5131    264  19560 127999 127999 127999 127999 127999 127999
  127999 127999]]
[' Kelsey and I am a 2016 graduate of the University of Wisconsin-Madison. I am currently a graduate锦锦锦锦锦锦锦锦']
Weight-free ONNX: test_models/weightfree_from_config-a971b506b22894dc/LlamaForCausalLM.onnx
Weight spec: test_models/weightfree_from_config-a971b506b22894dc/weight_spec.json

   "rope_type": "llama3"                                                                                                                           [123/426]
  },                                                                                                                                                         
  "rope_theta": 500000.0,                                                                                                                                    
  "tie_word_embeddings": false,                                                                                                                              
  "transformers_version": "4.57.3",                                                                                                                          
  "use_cache": true,                                                                                                                                         
  "vocab_size": 128256                                                                                                                                       
}                                                                                                                                                            
                                                                                                                                                             
[Warning]: The subfunction feature is experimental. Please note that using compile consecutively with and without subfunction may produce inconsistent result
s.                                                                                                                                                           
Weight-free export time: 0.010 sec                                                                                                                           
Converting checkpoint to FP32 (one-time local materialization) ...                                                                                           
[Warning]: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, 
use `force_download=True`.                                                                                                                                   
Fetching 37 files: 100%|██████████████████████████████████████████████████████████████████████████████████| 37/37 [00:00<00:00, 11337.61it/s]                
Reusing existing local FP32 safetensors.                                                                                                                     
fp32 convert time: 2.772 sec                                                                                                                                 
Compiling weight-free ONNX ...                                                                                                                               
['/opt/qti-aic/exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100', '-m=test_models/weightfree_from_config-fc018ea23beb8e08/LlamaForCausalLM.onnx', '-reta
ined-state', '-convert-to-fp16', '-mxfp6-matmul', '-aic-num-cores=16', '-sub-functions', '-mdp-load-partition-config=test_models/weightfree_from_config-fc018
ea23beb8e08/qpc/qpc-7284f87c6a052831/mdp_ts_4.json', '-network-specialization-config=test_models/weightfree_from_config-fc018ea23beb8e08/qpc/qpc-7284f87c6a05
2831/specializations.json', '-custom-IO-list-file=test_models/weightfree_from_config-fc018ea23beb8e08/qpc/qpc-7284f87c6a052831/custom_io.yaml', '-aic-binary-
dir=test_models/weightfree_from_config-fc018ea23beb8e08/qpc/qpc-7284f87c6a052831/qpc']                                                                       
compile time: 1390.574 sec                                                                                                                                   
QPC: test_models/weightfree_from_config-fc018ea23beb8e08/qpc/qpc-7284f87c6a052831/qpc                                                                        
Running QPC generate ...                                                                                                                                     
                                                                                                                                                             
Prompt : My name is                                                                                                                                          
Completion :input= ['My name is']                                                                                                                            
output= [['!!*^#(-<<(<�!#"(#!!*O*#}']]                                                                                                                       
Average Prefill time a.k.a TTFT is= 0.15 sec                                                                                                                 
Decode is= 6.99 tokens/sec                                                                                                                                   
Total is= 6.68 tokens/sec                                                                                                                                    
Total (E2E) inference time is= 3.44 sec                                                                                                                      
Average Prefill time a.k.a TTFT is= 0.15 sec                                                                                                                 
Decode is= 6.99 tokens/sec                                                                                                                                   
Total is= 6.68 tokens/sec                                                                                                                                    
Total (E2E) inference time is= 3.44 sec                                                                                                                      
[[35266    11   323   358  1097   264 10195   520   279  3907   315 14972                                                                                    
  21630 25027 19241   323 15506    13   358  1097 25429   922 12434 12437]]                                                                                  
[' Emily, and I am a senior at the University of Michigan studying Environmental Studies and Spanish. I am passionate about environmental justice']          
[[     0      0      9     61      2      7     12     27     27      7                                                                                      
      27     94      0      2      1      7      2      0      0      9                                                                                      
      46      9      2     92 128004 128004 128004 128004 128004 128004                                                                                      
  128004 128004]]                                                                                                                                            
['!!*^#(-<<(<�!#"(#!!*O*#}']                                                                                                                                 
Weight-free ONNX: test_models/weightfree_from_config-fc018ea23beb8e08/LlamaForCausalLM.onnx                                                                  
Weight spec: test_models/weightfree_from_config-fc018ea23beb8e08/weight_spec.json            
