# `Qwen/Qwen3-Omni-30B-A3B-Instruct` Multimodal TPU Benchmark Summary — `v5p-16` (`v5p`)

- **Updated (UTC):** `2026-11-02T19:11:59.906161+00:00`
- **Slice Topology:** `4` hosts x `4` TPU cores/host = `16` total TPU cores
- **Model Architecture:** Thinker (`48L` `128E` top-`8` MoE) + Talker (`20L` `128E` `16`-RVQ + `5L` CodePredictor) + Code2Wav (`24000 Hz`), `65.68` GiB across `15` safetensors shards
- **Disaggregated CPU Offload Service:** status=`ok`, url=`http://multimodal-cpu-offload-svc.default.svc.cluster.local:8090`

## 1. End-to-End Serving & Single-Client Multi-Device TorchTPU Results

| Engine | Precision | TP | Cluster DP | Context Len | Batch/Conc | TTFT (s) | TPOT / 2s Audio (ms) | Local WAV (ms) | Disagg RTT (ms) | First Audio Disagg (ms) | Speech RTF | Cluster Prefill (tok/s) | Cluster Output Rate | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `vllm-torchtpu-ep` | `BF16` | `4` | `4` | 65,536 | `-` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `DIAGNOSTIC_NOTE` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `4` | 131,072 | `-` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `DIAGNOSTIC_NOTE` |

## 2. Multimodal CPU Offload Ablation (`On-Host CPU` vs `multimodal-cpu-offload-svc:8090`)

| Stage | Batch / Frames | Offload OK | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote Compute (ms) | Speedup (x) | Offload Pod |
|---|---:|---|---:|---:|---:|---:|---|
| `audio_wav_24khz_2s` | `1` | `True` | 0.076 | 5.193 | 0.147 | 0.01 | `multimodal-cpu-offload-svc-576cb9dcb7-v96g8` |
| `audio_wav_24khz_2s` | `2` | `True` | 0.092 | 5.972 | 0.145 | 0.02 | `multimodal-cpu-offload-svc-576cb9dcb7-x5n8t` |
| `audio_wav_24khz_2s` | `4` | `True` | 0.159 | 2.65 | 0.315 | 0.06 | `multimodal-cpu-offload-svc-576cb9dcb7-x5n8t` |
| `audio_wav_24khz_2s` | `8` | `True` | 0.455 | 7.839 | 0.559 | 0.06 | `multimodal-cpu-offload-svc-576cb9dcb7-v96g8` |
| `vision_preproc_640x480_to_256x256` | `2` | `True` | 7.23 | 8.64 | 5.4 | 0.84 | `multimodal-cpu-offload-svc-576cb9dcb7-v96g8` |
| `vision_preproc_640x480_to_256x256` | `4` | `True` | 14.28 | 19.53 | 7.1 | 0.73 | `multimodal-cpu-offload-svc-576cb9dcb7-v96g8` |
| `vision_preproc_1280x720_to_448x448` | `8` | `True` | 85.55 | 47.89 | 31.87 | 1.79 | `multimodal-cpu-offload-svc-576cb9dcb7-v96g8` |
