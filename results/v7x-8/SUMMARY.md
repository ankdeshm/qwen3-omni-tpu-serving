# `Qwen/Qwen3-Omni-30B-A3B-Instruct` Multimodal TPU Benchmark Summary — `v7x-8` (`v7x`)

- **Updated (UTC):** `2026-11-06T14:07:06.532189+00:00`
- **Slice Topology:** `2` hosts x `8` TPU cores/host = `16` total TPU cores
- **Model Architecture:** Thinker (`48L` `128E` top-`8` MoE) + Talker (`20L` `128E` `16`-RVQ + `5L` CodePredictor) + Code2Wav (`24000 Hz`), `65.68` GiB across `15` safetensors shards
- **Disaggregated CPU Offload Service:** status=`ok`, url=`http://multimodal-cpu-offload-svc.default.svc.cluster.local:8090`

## 1. End-to-End Serving & Single-Client Multi-Device TorchTPU Results

| Engine | Precision | TP | Cluster DP | Context Len | Batch/Conc | TTFT (s) | TPOT / 2s Audio (ms) | Local WAV (ms) | Disagg RTT (ms) | First Audio Disagg (ms) | Speech RTF | Cluster Prefill (tok/s) | Cluster Output Rate | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `vllm-torchtpu-ep` | `BF16` | `4` | `4` | 4,096 | `1` | 0.311 | 18.67 ms TPOT | `-` | `-` | `-` | `-` | 26299.6 | 100.9 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `4` | 4,096 | `4` | 0.454 | 8.06 ms TPOT | `-` | `-` | `-` | `-` | 72171.9 | 812.4 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `4` | 4,096 | `8` | 0.085 | 10.11 ms TPOT | `-` | `-` | `-` | `-` | 770800.1 | 1516.5 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `4` | 16,384 | `1` | 0.695 | 17.47 ms TPOT | `-` | `-` | `-` | `-` | 47178.7 | 97.2 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `4` | 16,384 | `4` | 0.105 | 7.6 ms TPOT | `-` | `-` | `-` | `-` | 1249834.8 | 995.2 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `4` | 65,536 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(HTTPError: HTTP Error 400: Bad Request)` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `4` | 131,072 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(HTTPError: HTTP Error 400: Bad Request)` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `8` | 4,096 | `1` | 0.326 | 15.71 ms TPOT | `-` | `-` | `-` | `-` | 25156.4 | 118.2 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `8` | 4,096 | `4` | 0.774 | 16.13 ms TPOT | `-` | `-` | `-` | `-` | 42362.9 | 418.5 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `8` | 4,096 | `8` | 0.452 | 15.61 ms TPOT | `-` | `-` | `-` | `-` | 145062.0 | 917.9 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `8` | 16,384 | `1` | 0.919 | 14.74 ms TPOT | `-` | `-` | `-` | `-` | 35661.2 | 109.6 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `8` | 16,384 | `4` | 0.472 | 16.73 ms TPOT | `-` | `-` | `-` | `-` | 277953.3 | 360.8 tok/s | `PASS` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `8` | 65,536 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(HTTPError: HTTP Error 400: Bad Request)` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `8` | 131,072 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(HTTPError: HTTP Error 400: Bad Request)` |
| `vllm-torchtpu-omni` | `BF16` | `4` | `4` | 65,536 | `-` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `DIAGNOSTIC_NOTE` |
| `torchtpu-omni-worker` | `BF16` | `2` | `8` | 0 | `-` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `WORKER_EXIT_RC_1` |
| `torchtpu-omni-worker` | `FP8_DYNAMIC` | `2` | `8` | 0 | `-` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `WORKER_EXIT_RC_1` |
| `torchtpu-omni-worker` | `BF16` | `4` | `4` | 0 | `-` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `WORKER_EXIT_RC_1` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `4` | 4,096 | `1` | 1.072 | 235.67 ms (2s wav) / 843.09 ms | `0.151` | `0.626` | `1315.5` | `0.1181` | 15276.8 | 34.0 audio-s/s (4.8 tok/s) | `PASS` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `4` | 4,096 | `4` | 1.749 | 275.42 ms (2s wav) / 834.24 ms | `0.33` | `1.081` | `2032.0` | `0.1383` | 37475.6 | 116.2 audio-s/s (19.2 tok/s) | `PASS` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `4` | 16,384 | `1` | 4.909 | 196.95 ms (2s wav) / 1191.84 ms | `0.092` | `0.747` | `5124.6` | `0.0989` | 13350.8 | 40.6 audio-s/s (3.4 tok/s) | `PASS` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `4` | 16,384 | `2` | 2.043 | 49.47 ms (2s wav) / 919.14 ms | `0.172` | `0.828` | `2109.8` | `0.0251` | 64157.6 | 323.4 audio-s/s (8.7 tok/s) | `PASS` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `4` | 65,536 | `1` | 0.204 | 16.09 ms (2s wav) / 81.45 ms | `0.092` | `0.747` | `238.7` | `0.0084` | 1286462.8 | 497.3 audio-s/s (49.1 tok/s) | `PASS` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `4` | 65,536 | `2` | 0.728 | 278.04 ms (2s wav) / 6052.93 ms | `0.172` | `0.828` | `1023.1` | `0.1394` | 720407.2 | 57.6 audio-s/s (1.3 tok/s) | `PASS` |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `4` | `4` | 131,072 | `1` | 0.578 | 62.54 ms (2s wav) / 7584.93 ms | `0.092` | `0.747` | `659.4` | `0.0316` | 907018.8 | 127.9 audio-s/s (0.5 tok/s) | `PASS` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `4` | 4,096 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `4` | 4,096 | `4` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `4` | 16,384 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `4` | 16,384 | `2` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `4` | 65,536 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `4` | 65,536 | `2` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `4` | `4` | 131,072 | `1` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device in)` |

## 2. Multimodal CPU Offload Ablation (`On-Host CPU` vs `multimodal-cpu-offload-svc:8090`)

| Stage | Batch / Frames | Offload OK | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote Compute (ms) | Speedup (x) | Offload Pod |
|---|---:|---|---:|---:|---:|---:|---|
| `audio_wav_24khz_2s` | `1` | `True` | 0.072 | 0.509 | 0.09 | 0.14 | `multimodal-cpu-offload-svc-78f457f977-4gn6z` |
| `audio_wav_24khz_2s` | `2` | `True` | 0.086 | 0.557 | 0.112 | 0.15 | `multimodal-cpu-offload-svc-78f457f977-4gn6z` |
| `audio_wav_24khz_2s` | `4` | `True` | 0.133 | 0.694 | 0.168 | 0.19 | `multimodal-cpu-offload-svc-78f457f977-4gn6z` |
| `audio_wav_24khz_2s` | `8` | `True` | 0.337 | 1.122 | 0.434 | 0.3 | `multimodal-cpu-offload-svc-78f457f977-4gn6z` |
| `vision_preproc_640x480_to_256x256` | `2` | `True` | 5.2 | 6.33 | 5.4 | 0.82 | `multimodal-cpu-offload-svc-78f457f977-4gn6z` |
| `vision_preproc_640x480_to_256x256` | `4` | `True` | 10.66 | 9.12 | 7.35 | 1.17 | `multimodal-cpu-offload-svc-78f457f977-4gn6z` |
| `vision_preproc_1280x720_to_448x448` | `8` | `True` | 63.41 | 49.99 | 39.31 | 1.27 | `multimodal-cpu-offload-svc-78f457f977-4gn6z` |
