[18:57:26.929][error][QPrdNeuralNetwork][deactivate:#1344][ThId:5851] Dev 4 VC 151 NAID 1 deactivate failed status 300               [84/366]
ERROR : [18:57:26.931998115  UTC]  HOST  [LogCommon] [error] [runIoctlCmd:#1171][ThId:8665] ioctl failed with return code: -1 errno:Operation
 canceled                                                                                                                                    
                                                                                                                                             
[18:57:26.931][error][LogCommon][runIoctlCmd:#1171][ThId:8665] ioctl failed with return code: -1 errno:Operation canceled                    
ERROR : [18:57:26.933298591  UTC]  HOST  [QPrdNeuralNetwork] [error] [wait:#867][ThId:8665] Dev:4 VC:3, failed to receive response from NW wi
th error: Operation canceled.                                                                                                                
                                                                                                                                             
[18:57:26.933][error][QPrdNeuralNetwork][wait:#867][ThId:8665] Dev:4 VC:3, failed to receive response from NW with error: Operation canceled.
ERROR : [18:57:26.933401005  UTC]  HOST  [QPrdNeuralNetwork] [error] [wait:#869][ThId:8665] Increase QAicProgramProperties::dataPathTimeoutMs
, current: 60000.                                                                                                                            
                                                                                                                                             
[18:57:26.933][error][QPrdNeuralNetwork][wait:#869][ThId:8665] Increase QAicProgramProperties::dataPathTimeoutMs, current: 60000.            
ERROR : [18:57:26.933623855  UTC]  HOST  [Aic-0] [error] [ExecObj:ID:  3,L:1387] Dev 4 wait in kernel failed                                 
                                                                                                                                             
[18:57:26.933][error][Aic-0][ExecObj:ID:  3,L:1387] Dev 4 wait in kernel failed                                            
ERROR : [18:57:26.933745590  UTC]  HOST  [Aic-0] [error] [QueueFinish:ID:  8,L: 697] Failed to finish Execobj:3, status:500                  
                                                                                                                                             
[18:57:26.933][error][Aic-0][QueueFinish:ID:  8,L: 697] Failed to finish Execobj:3, status:500                             
ERROR : [18:57:26.933768601  UTC]  HOST  [Aic-0] [error] [QMqExecObj:ID:10000,L: 608] [MqProg:19] is in error state: 100                     
                                                                                                                                             
[18:57:26.933][error][Aic-0][QMqExecObj:ID:10000,L: 608] [MqProg:19] is in error state: 100                                                  
ERROR : [18:57:26.934647819  UTC]  HOST  [QPrdNeuralNetwork] [error] [deactivate:#1344][ThId:5851] Dev 5 VC 151 NAID 1 deactivate failed stat
us 300                                                                                                                                       
                                                                                                                                             
[18:57:26.934][error][QPrdNeuralNetwork][deactivate:#1344][ThId:5851] Dev 5 VC 151 NAID 1 deactivate failed status 300
ERROR : [18:57:26.938227653  UTC]  HOST  [LogCommon] [error] [runIoctlCmd:#1171][ThId:5181] ioctl failed with return code: -1 errno:Operation
 canceled                                                                                                                                    
                                     
[18:57:26.938][error][LogCommon][runIoctlCmd:#1171][ThId:5181] ioctl failed with return code: -1 errno:Operation canceled                    
ERROR : [18:57:26.938288176  UTC]  HOST  [QPrdNeuralNetwork] [error] [wait:#867][ThId:5181] Dev:5 VC:3, failed to receive response from NW wi
th error: Operation canceled.                                                                                                                
                                     
[18:57:26.938][error][QPrdNeuralNetwork][wait:#867][ThId:5181] Dev:5 VC:3, failed to receive response from NW with error: Operation canceled.
ERROR : [18:57:26.938295146  UTC]  HOST  [QPrdNeuralNetwork] [error] [wait:#869][ThId:5181] Increase QAicProgramProperties::dataPathTimeoutMs
, current: 60000.                                                                                                                            

[18:57:26.938][error][QPrdNeuralNetwork][wait:#869][ThId:5181] Increase QAicProgramProperties::dataPathTimeoutMs, current: 60000.
ERROR : [18:57:26.938305187  UTC]  HOST  [Aic-0] [error] [ExecObj:ID:  2,L:1387] Dev 5 wait in kernel failed               

[18:57:26.938][error][Aic-0][ExecObj:ID:  2,L:1387] Dev 5 wait in kernel failed               
ERROR : [18:57:26.938311257  UTC]  HOST  [Aic-0] [error] [QueueFinish:ID:  7,L: 697] Failed to finish Execobj:2, status:500

[18:57:26.938][error][Aic-0][QueueFinish:ID:  7,L: 697] Failed to finish Execobj:2, status:500
ERROR : [18:57:26.938322267  UTC]  HOST  [Aic-0] [error] [QMqExecObj:ID:10000,L: 608] [MqProg:19] is in error state: 100           

[18:57:26.938][error][Aic-0][QMqExecObj:ID:10000,L: 608] [MqProg:19] is in error state: 100           
ERROR : [18:57:26.938504565  UTC]  HOST  [Aic-0] [error] [QMqExecObj:ID:10000,L:1366] [MqProg:19] is in error state #IN_FINISH: 100          

[18:57:26.938][error][Aic-0][QMqExecObj:ID:10000,L:1366] [MqProg:19] is in error state #IN_FINISH: 100   
[18:57:26.938][error][Aic-0][QMqExecObj:ID:10000,L:1366] [MqProg:19] is in error state #IN_FINISH: 100                
ERROR : [18:57:26.938524086  UTC]  HOST  [Aic-0] [error] [QueueFinish:ID:  0,L: 697] Failed to finish Execobj:10000, status:500

[18:57:26.938][error][Aic-0][QueueFinish:ID:  0,L: 697] Failed to finish Execobj:10000, status:500
ERROR : [18:57:26.938529976  UTC]  HOST  [Aic-0] [error] [QMqExecObj:ID:10000,L: 369] Program Device is not initialized

[18:57:26.938][error][Aic-0][QMqExecObj:ID:10000,L: 369] Program Device is not initialized
Traceback (most recent call last):
  File "/home/amarshar/weightfree_exp/examples/text_generation/weight_free_export_from_config.py", line 196, in <module>
    exec_info = qeff_model.generate(
  File "/home/amarshar/weightfree_exp/QEfficient/transformers/models/modeling_auto.py", line 3903, in generate
    return QEfficient.cloud_ai_100_exec_kv(
  File "/home/amarshar/weightfree_exp/QEfficient/generation/text_generation_inference.py", line 409, in cloud_ai_100_exec_kv
    exec_info = [
  File "/home/amarshar/weightfree_exp/QEfficient/generation/text_generation_inference.py", line 410, in <listcomp>
    generate_text.generate(prompt[i : i + batch_size], generation_len, stream, prompt_to_lora_id_mapping)
  File "/home/amarshar/weightfree_exp/QEfficient/generation/text_generation_inference.py", line 1311, in generate
    perf_metrics, generated_texts = self._regular_model_execution(
  File "/home/amarshar/weightfree_exp/QEfficient/generation/text_generation_inference.py", line 1177, in _regular_model_execution
    outputs, position_ids, generation_len = self._qaic_model.run_prefill(
  File "/home/amarshar/weightfree_exp/QEfficient/generation/text_generation_inference.py", line 853, in run_prefill
    outputs = self._session.run(chunk_inputs)
  File "/home/amarshar/weightfree_exp/QEfficient/generation/cloud_infer.py", line 214, in run
    raise ValueError(error_message) 
ValueError: Failed to run

(Only if "No matching dimension found" error is present above)
Allowed shapes:
0
input_ids:      8       [1, 1]
position_ids:   8       [1, 1]
logits: 4       [1, 1, 128256]

1
input_ids:      8       [1, 8]
position_ids:   8       [1, 8]
logits: 4       [1, 1, 128256]


Passed shapes:
input_ids:      8       [1, 8]
position_ids:   8       [1, 8]
logits: 4       [1, 1, 128256]

ERROR : [18:57:27.258754649  UTC]  HOST  [Aic-0] [error] [ProgDev:ID:  3,L: 310] Failed to Unload, invalid state detected in programSTATE_PRO
GRAM_DEVICE_ERROR

[18:57:27.258][error][Aic-0][ProgDev:ID:  3,L: 310] Failed to Unload, invalid state detected in programSTATE_PROGRAM_DEVICE_ERROR
ERROR : [18:57:27.258805441  UTC]  HOST  [Aic-0] [error] [~QIProgramStateMgr:#74][ThId:5527] failed to unload network

0                                                                                                                                     [7/366]
input_ids:      8       [1, 1]
position_ids:   8       [1, 1]
logits: 4       [1, 1, 128256]

1
input_ids:      8       [1, 8]
position_ids:   8       [1, 8]
logits: 4       [1, 1, 128256]


Passed shapes:
input_ids:      8       [1, 8]
position_ids:   8       [1, 8]
logits: 4       [1, 1, 128256]

ERROR : [18:57:27.258754649  UTC]  HOST  [Aic-0] [error] [ProgDev:ID:  3,L: 310] Failed to Unload, invalid state detected in programSTATE_PRO
GRAM_DEVICE_ERROR

[18:57:27.258][error][Aic-0][ProgDev:ID:  3,L: 310] Failed to Unload, invalid state detected in programSTATE_PROGRAM_DEVICE_ERROR
ERROR : [18:57:27.258805441  UTC]  HOST  [Aic-0] [error] [~QIProgramStateMgr:#74][ThId:5527] failed to unload network

[18:57:27.258][error][Aic-0][~QIProgramStateMgr:#74][ThId:5527] failed to unload network
ERROR : [18:57:27.258824072  UTC]  HOST  [LogCommon] [error] [~QNNImage:#47][ThId:5527] Failed to unload network

[18:57:27.258][error][LogCommon][~QNNImage:#47][ThId:5527] Failed to unload network
ERROR : [18:57:27.259647767  UTC]  HOST  [Aic-0] [error] [ProgDev:ID:  2,L: 310] Failed to Unload, invalid state detected in programSTATE_PRO
GRAM_DEVICE_ERROR

[18:57:27.259][error][Aic-0][ProgDev:ID:  2,L: 310] Failed to Unload, invalid state detected in programSTATE_PROGRAM_DEVICE_ERROR
ERROR : [18:57:27.259658258  UTC]  HOST  [Aic-0] [error] [~QIProgramStateMgr:#74][ThId:5527] failed to unload network

[18:57:27.259][error][Aic-0][~QIProgramStateMgr:#74][ThId:5527] failed to unload network
ERROR : [18:57:27.259667508  UTC]  HOST  [LogCommon] [error] [~QNNImage:#47][ThId:5527] Failed to unload network

[18:57:27.259][error][LogCommon][~QNNImage:#47][ThId:5527] Failed to unload network
ERROR : [18:57:27.260908042  UTC]  HOST  [Aic-0] [error] [ProgDev:ID:  1,L: 310] Failed to Unload, invalid state detected in programSTATE_PRO
GRAM_DEVICE_ERROR

[18:57:27.260][error][Aic-0][ProgDev:ID:  1,L: 310] Failed to Unload, invalid state detected in programSTATE_PROGRAM_DEVICE_ERROR
ERROR : [18:57:27.260917622  UTC]  HOST  [Aic-0] [error] [~QIProgramStateMgr:#74][ThId:5527] failed to unload network

[18:57:27.260][error][Aic-0][~QIProgramStateMgr:#74][ThId:5527] failed to unload network
ERROR : [18:57:27.260924942  UTC]  HOST  [LogCommon] [error] [~QNNImage:#47][ThId:5527] Failed to unload network

[18:57:27.260][error][LogCommon][~QNNImage:#47][ThId:5527] Failed to unload network
ERROR : [18:57:27.262129514  UTC]  HOST  [Aic-0] [error] [ProgDev:ID:  0,L: 310] Failed to Unload, invalid state detected in programSTATE_PRO
GRAM_DEVICE_ERROR

[18:57:27.262][error][Aic-0][ProgDev:ID:  0,L: 310] Failed to Unload, invalid state detected in programSTATE_PROGRAM_DEVICE_ERROR
[18:57:27.260][error][LogCommon][~QNNImage:#47][ThId:5527] Failed to unload network
ERROR : [18:57:27.262129514  UTC]  HOST  [Aic-0] [error] [ProgDev:ID:  0,L: 310] Failed to Unload, invalid state detected in programSTATE_PRO
GRAM_DEVICE_ERROR

[18:57:27.262][error][Aic-0][ProgDev:ID:  0,L: 310] Failed to Unload, invalid state detected in programSTATE_PROGRAM_DEVICE_ERROR
ERROR : [18:57:27.262138755  UTC]  HOST  [Aic-0] [error] [~QIProgramStateMgr:#74][ThId:5527] failed to unload network

[18:57:27.262][error][Aic-0][~QIProgramStateMgr:#74][ThId:5527] failed to unload network
ERROR : [18:57:27.262148505  UTC]  HOST  [LogCommon] [error] [~QNNImage:#47][ThId:5527] Failed to unload network

[18:57:27.262][error][LogCommon][~QNNImage:#47][ThId:5527] Failed to unload network

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

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


model_name = "meta-llama/Llama-3.3-70B-Instruct"
#model_name = "Qwen/Qwen3-8B"
#model_name="meta-llama/Llama-3.2-1B"
#model_name="Qwen/Qwen3-235B-A22B-Instruct-2507"
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
