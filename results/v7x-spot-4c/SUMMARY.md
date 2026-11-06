# `Qwen/Qwen3-Omni-30B-A3B-Instruct` Multimodal TPU Benchmark Summary — `v7x-spot-4c` (`v7x`)

- **Updated (UTC):** `2026-11-06T12:35:42.539722+00:00`
- **Slice Topology:** `1` hosts x `8` TPU cores/host = `8` total TPU cores
- **Model Architecture:** Thinker (`48L` `128E` top-`8` MoE) + Talker (`20L` `128E` `16`-RVQ + `5L` CodePredictor) + Code2Wav (`24000 Hz`), `65.68` GiB across `15` safetensors shards
- **Disaggregated CPU Offload Service:** status=`ok`, url=`http://multimodal-cpu-offload-svc.default.svc.cluster.local:8090`

## 1. End-to-End Serving & Single-Client Multi-Device TorchTPU Results

| Engine | Precision | TP | Cluster DP | Context Len | Batch/Conc | TTFT (s) | TPOT / 2s Audio (ms) | Local WAV (ms) | Disagg RTT (ms) | First Audio Disagg (ms) | Speech RTF | Cluster Prefill (tok/s) | Cluster Output Rate | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `vllm-torchtpu-ep` | `BF16` | `4` | `2` | 4,096 | `1` | 0.342 | 18.58 ms TPOT | `-` | `-` | `-` | `-` | 11981.2 | 50.4 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `2` | 4,096 | `4` | 0.056 | 7.46 ms TPOT | `-` | `-` | `-` | `-` | 290592.0 | 516.6 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `2` | 4,096 | `8` | 0.506 | 9.43 ms TPOT | `-` | `-` | `-` | `-` | 64786.2 | 703.2 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `2` | 16,384 | `1` | 0.711 | 17.48 ms TPOT | `-` | `-` | `-` | `-` | 23045.2 | 48.4 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `2` | 16,384 | `4` | 0.209 | 7.98 ms TPOT | `-` | `-` | `-` | `-` | 313470.9 | 451.6 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `2` | 65,536 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(HTTPError: HTTP Error 400: Bad Request)` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `2` | 131,072 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(HTTPError: HTTP Error 400: Bad Request)` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `4` | 4,096 | `1` | 0.347 | 15.71 ms TPOT | `-` | `-` | `-` | `-` | 11791.3 | 58.8 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `4` | 4,096 | `4` | 0.661 | 16.4 ms TPOT | `-` | `-` | `-` | `-` | 24769.0 | 210.9 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `4` | 4,096 | `8` | 0.465 | 15.65 ms TPOT | `-` | `-` | `-` | `-` | 70397.6 | 456.6 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `4` | 16,384 | `1` | 0.92 | 14.74 ms TPOT | `-` | `-` | `-` | `-` | 17815.3 | 54.8 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `4` | 16,384 | `4` | 0.567 | 16.14 ms TPOT | `-` | `-` | `-` | `-` | 115684.6 | 189.0 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `4` | 65,536 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(HTTPError: HTTP Error 400: Bad Request)` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `4` | 131,072 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(HTTPError: HTTP Error 400: Bad Request)` |
| `vllm-torchtpu-omni` | `BF16` | `4` | `2` | 65,536 | `-` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `DIAGNOSTIC_NOTE` |
| `torchtpu-omni-worker` | `BF16` | `2` | `4` | 0 | `-` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `WORKER_EXIT_RC_1` |
| `torchtpu-omni-worker` | `FP8_DYNAMIC` | `2` | `4` | 0 | `-` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `WORKER_EXIT_RC_1` |
| `torchtpu-omni-worker` | `BF16` | `4` | `2` | 0 | `-` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `WORKER_EXIT_RC_1` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `2` | 4,096 | `1` | 1.488 | 251.35 ms (2s wav) / 1141.56 ms | `0.126` | `0.66` | `1746.4` | `0.126` | 5506.8 | 15.9 audio-s/s (1.8 tok/s) | `PASS` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `2` | 4,096 | `4` | 1.449 | 285.33 ms (2s wav) / 1046.52 ms | `0.244` | `0.973` | `1741.6` | `0.1432` | 22621.8 | 56.1 audio-s/s (7.6 tok/s) | `PASS` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `2` | 16,384 | `1` | 0.36 | 37.65 ms (2s wav) / 1498.45 ms | `0.116` | `0.664` | `414.4` | `0.0192` | 90945.4 | 106.2 audio-s/s (1.3 tok/s) | `PASS` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `2` | 16,384 | `2` | 0.405 | 49.51 ms (2s wav) / 1537.37 ms | `0.156` | `0.815` | `471.2` | `0.0252` | 161762.6 | 161.6 audio-s/s (2.6 tok/s) | `PASS` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `2` | 65,536 | `1` | 0.531 | 17.54 ms (2s wav) / 63.98 ms | `0.116` | `0.664` | `565.3` | `0.0091` | 246684.2 | 228.0 audio-s/s (31.3 tok/s) | `PASS` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `2` | 65,536 | `2` | 0.701 | 70.91 ms (2s wav) / 5933.95 ms | `0.156` | `0.815` | `788.9` | `0.0359` | 373729.6 | 112.8 audio-s/s (0.7 tok/s) | `PASS` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `2` | 131,072 | `1` | 0.572 | 15.27 ms (2s wav) / 50.17 ms | `0.116` | `0.664` | `603.4` | `0.008` | 458554.2 | 262.0 audio-s/s (39.9 tok/s) | `PASS` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `2` | 4,096 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `2` | 4,096 | `4` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `2` | 16,384 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `2` | 16,384 | `2` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `2` | 65,536 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `2` | 65,536 | `2` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `2` | 131,072 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |

## 2. Multimodal CPU Offload Ablation (`On-Host CPU` vs `multimodal-cpu-offload-svc:8090`)

| Stage | Batch / Frames | Offload OK | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote Compute (ms) | Speedup (x) | Offload Pod |
|---|---:|---|---:|---:|---:|---:|---|
| `audio_wav_24khz_2s` | `1` | `True` | 0.068 | 0.513 | 0.084 | 0.13 | `multimodal-cpu-offload-svc-74f4579599-662gd` |
| `audio_wav_24khz_2s` | `2` | `True` | 0.071 | 0.574 | 0.114 | 0.12 | `multimodal-cpu-offload-svc-74f4579599-662gd` |
| `audio_wav_24khz_2s` | `4` | `True` | 0.139 | 0.818 | 0.218 | 0.17 | `multimodal-cpu-offload-svc-74f4579599-662gd` |
| `audio_wav_24khz_2s` | `8` | `True` | 0.379 | 1.208 | 0.468 | 0.31 | `multimodal-cpu-offload-svc-74f4579599-662gd` |
| `vision_preproc_640x480_to_256x256` | `2` | `True` | 5.16 | 6.34 | 5.34 | 0.81 | `multimodal-cpu-offload-svc-74f4579599-662gd` |
| `vision_preproc_640x480_to_256x256` | `4` | `True` | 10.51 | 9.09 | 7.07 | 1.16 | `multimodal-cpu-offload-svc-74f4579599-662gd` |
| `vision_preproc_1280x720_to_448x448` | `8` | `True` | 61.32 | 42.63 | 31.39 | 1.44 | `multimodal-cpu-offload-svc-74f4579599-662gd` |
