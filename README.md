# `Qwen/Qwen3-Omni-30B-A3B-Instruct` Any-to-Any MoE Serving on Cloud TPU `v5p` & `v7x`

Qwen/Qwen3-Omni-30B-A3B-Instruct (128-Expert MoE Thinker 64K/128K + Talker 16-RVQ + 24 kHz Streaming Speech) Serving on Cloud TPU v5p & v7x with vLLM-TorchTPU & Disaggregated Audio/Vision Offloading.

---

## 1. Model & System Architecture

| Component | Specification |
|---|---|
| **Checkpoint** | [`Qwen/Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) (`~62 GiB` BF16) |
| **Architecture** | `Thinker` (`48L` `128E` top-8 MoE, `64K` native / `128K` YaRN) + `Talker` (`20L` `128E` `16`-RVQ + `5L` `CodePredictor`) + `Code2Wav` (`24 kHz`) |
| **Serving Engines** | OpenAI-Compatible `vLLM-TorchTPU` (`--enable-expert-parallel`, `TP=2` & `TP=4`) + Single-Client Multi-Device `TorchTPU` Any-to-Any Speech Streaming |
| **Precisions & Contexts** | `BF16` & `FP8_DYNAMIC` (`out_dtype=bfloat16`) across `4K`, `16K`, `64K`, and `128K` multimodal contexts |
| **Disaggregated CPU Offload** | `multimodal-cpu-offload-svc` (`c3-standard-44` AVX-512): `POST /encode_wav` `24 kHz` PCM16 WAV framing & `POST /preprocess_vision` |

---

## 2. Disaggregated CPU Offloading Architecture (`multimodal-cpu-offload-svc`)

All non-TPU pre-processing and media encoding work is offloaded over intra-cluster `ClusterIP` HTTP (`port 8090`) to dedicated `c3-standard-44` (`44` Sapphire Rapids vCPUs, AVX-512, `176 GiB` DDR5) CPU pods (`scripts/multimodal_cpu_offload_server.py`), keeping TPU host CPUs and Python threads dedicated to XLA/PJRT dispatch:

- `POST /encode` — Multi-threaded `ffmpeg` (`libx264` + `48 kHz` stereo `AAC`) video/audio MP4 muxing.
- `POST /encode_image` — Thread-pooled `Pillow` `PNG` / `WebP` / `JPEG` batch image encoding.
- `POST /encode_wav` — Vectorized `float32` / `int16` audio normalization + `RIFF/WAV` packet framing (`24 kHz` / `48 kHz`).
- `POST /preprocess_vision` — Multi-camera RGB image decode, bicubic resize, and `bfloat16`/`float32` normalization.

---

## 3. Hardware Platforms Evaluated

| Accelerator | Slice Topologies | HBM Capacity & Bandwidth | Host Memory (`/dev/shm`) |
|---|---|---|---|
| **Cloud TPU `v5p`** | `v5p-16` (`2x2x4`, `4` hosts × `4` chips) & `v5p-32` (`2x4x4`, `8` hosts × `4` chips) | `95 GiB` HBM2e / chip (`2,765 GB/s` / chip, `459` BF16 TFLOP/s) | `462 GiB` RAM (`380 GiB` tmpfs) |
| **Cloud TPU `v7x` (Ironwood)** | `v7x-4` (`2x2x1`, `8` TensorCores), `v7x-8` (`2x2x2`, `16` TensorCores), & `v7x-16` (`2x2x4`, `32` TensorCores) | `192 GiB` HBM3e / chip (`96 GiB` / TensorCore, `7,370 GB/s` / chip, `2,307` BF16 / `4,614` FP8 TFLOP/s per chip) | `944 GiB` RAM (`850 GiB` tmpfs) |

---

## 4. Benchmark Results (`v5p` & `v7x`)

Measured results linked 1:1 to raw JSON and Markdown summaries under [`results/`](./results/):

- **`v5p-16`** (`2` benchmark rows, updated `2026-11-02T19:11:59.906161+00:00`): [`results/v5p-16/SUMMARY.md`](./results/v5p-16/SUMMARY.md) · [`results/v5p-16/summary.json`](./results/v5p-16/summary.json)

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

---

## 5. Repository Layout & Reproduction

- [`scripts/run_qwen3_omni_tpu_bench.py`](./scripts/run_qwen3_omni_tpu_bench.py) — Main TPU `v5p` / `v7x` benchmark driver (cooperative zero-disk HF-to-GCS staging, `/dev/shm` hydration, single-host PJRT isolation, `TorchTPU` / `vLLM` execution, and A/B CPU offload ablation).
- [`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) & [`scripts/cpu_offload_client.py`](./scripts/cpu_offload_client.py) — Disaggregated CPU offload HTTP server and client.
- [`scripts/gcs_stager.py`](./scripts/gcs_stager.py) & [`scripts/tpu_single_host_setup.py`](./scripts/tpu_single_host_setup.py) — Multi-host GCS shard stager and single-host `libtpu` metadata hook.
- [`k8s/`](./k8s/) — Kubernetes Deployment & Indexed Job manifests for CPU offload pool, TPU `v5p`, and TPU `v7x`.

```bash
# 1. Deploy the disaggregated CPU offload service
kubectl apply -f k8s/multimodal-cpu-offload-service.yaml

# 2. Launch the TPU v5p / v7x Indexed Job
kubectl apply -f k8s/v5p16-qwen3-omni.yaml
```
