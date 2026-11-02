# `Qwen/Qwen3-Omni-30B-A3B-Instruct` Multimodal TPU Benchmark Summary — `v5p-16` (`v5p`)

- **Updated (UTC):** `2026-11-02T18:40:10.135177+00:00`
- **Slice Topology:** `4` hosts x `4` TPU cores/host = `16` total TPU cores
- **Model Architecture:** Thinker (`48L` `128E` top-`8` MoE) + Talker (`20L` `128E` `16`-RVQ + `5L` CodePredictor) + Code2Wav (`24000 Hz`), `65.68` GiB across `15` safetensors shards
- **Disaggregated CPU Offload Service:** status=`ok`, url=`http://multimodal-cpu-offload-svc.default.svc.cluster.local:8090`

## 1. End-to-End Serving & Single-Client Multi-Device TorchTPU Results

| Engine | Precision | TP | Cluster DP | Context Len | Batch/Conc | TTFT (s) | TPOT / 2s Audio (ms) | Local WAV (ms) | Disagg RTT (ms) | First Audio Disagg (ms) | Speech RTF | Cluster Prefill (tok/s) | Cluster Output Rate | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `vllm-torchtpu-ep` | `BF16` | `4` | `4` | 65,536 | `-` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `DIAGNOSTIC_NOTE` |
| `vllm-torchtpu-ep` | `BF16` | `4` | `4` | 131,072 | `-` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `DIAGNOSTIC_NOTE` |
| `vllm-torchtpu-ep` | `BF16` | `2` | `8` | 65,536 | `-` | - | - ms TPOT | `-` | `-` | `-` | `-` | - | - tok/s | `DIAGNOSTIC_NOTE` |

## 2. Multimodal CPU Offload Ablation (`On-Host CPU` vs `multimodal-cpu-offload-svc:8090`)

| Stage | Batch / Frames | Offload OK | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote Compute (ms) | Speedup (x) | Offload Pod |
|---|---:|---|---:|---:|---:|---:|---|
| `audio_wav_24khz_2s` | `1` | `True` | 0.13 | 5.328 | 0.133 | 0.02 | `multimodal-cpu-offload-svc-79c7f5bb44-6nbrq` |
| `audio_wav_24khz_2s` | `2` | `True` | 0.126 | 5.991 | 0.139 | 0.02 | `multimodal-cpu-offload-svc-79c7f5bb44-6nbrq` |
| `audio_wav_24khz_2s` | `4` | `True` | 0.172 | 7.033 | 0.281 | 0.02 | `multimodal-cpu-offload-svc-79c7f5bb44-6nbrq` |
| `audio_wav_24khz_2s` | `8` | `True` | 0.45 | 3.909 | 0.551 | 0.12 | `multimodal-cpu-offload-svc-79c7f5bb44-fhklm` |
| `vision_preproc_640x480_to_256x256` | `2` | `True` | 7.19 | 15.88 | 5.34 | 0.45 | `multimodal-cpu-offload-svc-79c7f5bb44-6nbrq` |
| `vision_preproc_640x480_to_256x256` | `4` | `True` | 14.39 | 12.15 | 7.91 | 1.18 | `multimodal-cpu-offload-svc-79c7f5bb44-6nbrq` |
| `vision_preproc_1280x720_to_448x448` | `8` | `True` | 85.68 | 61.27 | 33.47 | 1.4 | `multimodal-cpu-offload-svc-79c7f5bb44-fhklm` |
