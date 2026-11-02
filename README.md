# `Qwen/Qwen3-Omni-30B-A3B-Instruct` Any-to-Any MoE Serving on Cloud TPU `v5p` & `v7x`

Qwen/Qwen3-Omni-30B-A3B-Instruct (128-Expert MoE Thinker 64K/128K + Talker 16-RVQ + 24 kHz Streaming Speech) Serving on Cloud TPU v5p & v7x with TorchTPU & Disaggregated Audio/Vision Offloading.

---

## 1. Model & System Architecture

| Component | Specification |
|---|---|
| **Checkpoint** | [`Qwen/Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) (`~65.7 GiB` BF16, `15` safetensors shards) |
| **Architecture** | `Thinker` (`48L` `128E` top-8 MoE, `64K` native / `128K` YaRN) + `Talker` (`20L` `128E` `16`-RVQ + `5L` `CodePredictor`) + `Code2Wav` (`24 kHz`) |
| **Serving Engines** | Single-Client Multi-Device `TorchTPU` (`torchtpu-omni-eager` & `torchtpu-omni-compiled`, `TP=2` & `TP=4`) + `vLLM-TorchTPU` EP probes |
| **Precisions & Contexts** | `BF16` & `FP8_DYNAMIC` (`out_dtype=bfloat16`) across `4K`, `16K`, `64K`, and `128K` multimodal contexts |
| **Disaggregated CPU Offload** | Adaptive routing (`ADAPTIVE_OFFLOAD_MIN_BYTES=262144`) via `Keep-Alive + TCP_NODELAY` to `multimodal-cpu-offload-svc` (`c3-standard-44` AVX-512) for multi-frame `720p` vision + in-memory host WAV framing |

---

## 2. Disaggregated CPU Offloading Architecture (`multimodal-cpu-offload-svc`)

Non-TPU media muxing, high-resolution image compression, and multi-camera/multi-frame vision preprocessing are offloaded over intra-cluster `ClusterIP` HTTP (`port 8090`) to dedicated `c3-standard-44` (`44` Sapphire Rapids vCPUs, AVX-512, `176 GiB` DDR5) CPU pods ([`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) & [`scripts/cpu_offload_client.py`](./scripts/cpu_offload_client.py)), keeping TPU host CPUs dedicated to XLA/PJRT dispatch:

- `POST /encode` — Multi-threaded `ffmpeg` (`libx264` + `48 kHz` stereo `AAC`) video/audio MP4 muxing.
- `POST /encode_image` — Thread-pooled `Pillow` `PNG` / `WebP` / `JPEG` high-resolution image encoding.
- `POST /encode_wav` — Vectorized `float32` / `int16` audio normalization + `RIFF/WAV` packet framing (`24 kHz` / `48 kHz`).
- `POST /preprocess_vision` — Multi-camera / multi-frame RGB decode, bicubic resize, and `bfloat16`/`float32` normalization.

---

## 3. Hardware Platforms Evaluated

| Accelerator | Slice Topologies | HBM Capacity & Bandwidth | Host Memory (`/dev/shm`) |
|---|---|---|---|
| **Cloud TPU `v5p`** | `v5p-16` (`2x2x4`, `4` hosts × `4` chips) & `v5p-32` (`2x4x4`, `8` hosts × `4` chips) | `95 GiB` HBM2e / chip (`2,765 GB/s` / chip, `459` BF16 TFLOP/s) | `462 GiB` RAM (`380 GiB` tmpfs) |
| **Cloud TPU `v7x` (Ironwood)** | `v7x-4` (`2x2x1`, `8` TensorCores), `v7x-8` (`2x2x2`, `16` TensorCores), & `v7x-16` (`2x2x4`, `32` TensorCores) | `192 GiB` HBM3e / chip (`96 GiB` / TensorCore, `7,370 GB/s` / chip, `2,307` BF16 / `4,614` FP8 TFLOP/s per chip) | `944 GiB` RAM (`850 GiB` tmpfs) |

---

## 4. Before vs. After Disaggregated Multimodal Offloading (`Keep-Alive + TCP_NODELAY`) & Adaptive Threshold Routing

Direct A/B ablation across `v5p-16` (`Wave-1` vs. `Wave-3 Keep-Alive + TCP_NODELAY`) and `v7x` comparing **on-host TPU VM CPU execution** vs. **persistent `HTTP/1.1 Keep-Alive + TCP_NODELAY` AVX-512 CPU pods (`multimodal-cpu-offload-svc:8090`)** across heavy multi-frame video/vision inputs (`>= 256 KB`) and lightweight `24 kHz` streaming audio chunks (`< 256 KB`):

| Modality & Workload | On-Host TPU VM CPU (`Before`) | `Keep-Alive` AVX-512 CPU Service (`After`) | Measured Speedup & Routing Decision | Code & Result References |
|---|---:|---:|---|---|
| **`8`-Frame `1280x720 -> 448x448` Vision Preproc** | `85.55 ms` on-host (`47.89 ms` Wave-1 HTTP) | **`19.24 – 20.27 ms` Keep-Alive RTT** | **`4.45x` faster vs host · `2.49x` vs Wave-1** — **Offload (`>= 256 KB`)** | [`cpu_offload_client.py`](./scripts/cpu_offload_client.py) · [`v5p-16-wave3-keepalive/summary.json`](./results/v5p-16-wave3-keepalive/summary.json) |
| **`2s` `24 kHz` PCM16 WAV Framing (`96 KB` payload)** | **`0.090 – 0.155 ms`** (`3.28–6.99 ms` Wave-1 HTTP) | **`1.358 – 2.028 ms`** Keep-Alive RTT (`3.8x` faster HTTP) | **Auto-routed in-memory on host (`< 256 KB`)** — **`85.7 ms` First-Audio** (`4K` BF16) | [`cpu_offload_client.py`](./scripts/cpu_offload_client.py) · [`v5p-16-wave3-keepalive/summary.json`](./results/v5p-16-wave3-keepalive/summary.json) |
| **`FP8_DYNAMIC` SpeechStream (`out_dtype=bf16` MXU)** | `16,469.7 ms` First-Audio (`4K`, Wave-1 dequant) | **`1,578.1 ms` (`4K`) · `559.8 ms` (`64K` `bs=2`)** | **`10.4x+` faster First-Audio** (eliminates software FP8 dequant) | [`run_qwen3_omni_tpu_bench.py`](./scripts/run_qwen3_omni_tpu_bench.py) · [`v5p-16-wave3-keepalive/summary.json`](./results/v5p-16-wave3-keepalive/summary.json) |
| **`16K` Talker + `Code2Wav` (`Eager` vs `torch.compile`)** | `47.53 ms` / `2s` audio (`168.3 audio-s/s`) | **`1.26 ms` / `2s` audio** (**`6,368.1 audio-s/s`**, `0.0033` RTF) | **`37.7x` faster Talker+Code2Wav synthesis** | [`run_qwen3_omni_tpu_bench.py`](./scripts/run_qwen3_omni_tpu_bench.py) · [`v5p-16/summary.json`](./results/v5p-16/summary.json) |

---

## 5. Complete Benchmark Tables (`v5p` & `v7x`)

Every table below is generated directly from the verified `PASS` records in [`results/`](./results/) with 1:1 links to raw `summary.json` artifacts:

### Slice `v5p-16` (`v5p`, `35` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16/SUMMARY.md`](./results/v5p-16/SUMMARY.md) · [`results/v5p-16/summary.json`](./results/v5p-16/summary.json) *(updated `2026-11-02T20:02:58.290265+00:00`)*

#### Stage 1 — End-to-End `Thinker` (`4K`–`128K`) + `Talker` (`16`-RVQ) + `24 kHz` Streaming Speech

| Engine | Precision | `TP × Cluster DP` | Context | Batch | TTFT (s) | TPOT (ms) | `2s` Speech Chunk (ms) | First Audio (ms) | Speech RTF | Cluster Prefill (tok/s) | Cluster Output Rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `4,096` | `1` | **`1.325`** | `752.37` | **`277.75`** | `1,611.0` | **`0.1419`** | **`24,724.0`** | **`57.6`** audio-s/s (`10.6` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `4,096` | `4` | **`1.462`** | `1,017.86` | **`267.53`** | `1,738.1` | **`0.1374`** | **`89,627.2`** | **`239.2`** audio-s/s (`31.4` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `16,384` | `1` | **`1.942`** | `816.40` | **`40.23`** | `2,015.0` | **`0.0216`** | **`67,503.2`** | **`397.7`** audio-s/s (`9.8` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `16,384` | `2` | **`0.156`** | `128.81` | **`17.24`** | `204.4` | **`0.0119`** | **`1,680,213.6`** | **`1,856.1`** audio-s/s (`124.2` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `65,536` | `1` | **`3.073`** | `2,962.15` | **`46.00`** | `3,151.7` | **`0.0245`** | **`170,631.2`** | **`347.8`** audio-s/s (`2.7` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `65,536` | `2` | **`0.618`** | `46.01` | **`20.80`** | `669.8` | **`0.0137`** | **`1,697,283.2`** | **`1,538.2`** audio-s/s (`347.8` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `131,072` | `1` | **`23.705`** | `1,541.51` | **`126.80`** | `23,864.5` | **`0.0649`** | **`44,235.2`** | **`126.2`** audio-s/s (`5.2` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `4,096` | `1` | **`0.070`** | `379.82` | **`25.36`** | `102.7` | **`0.0138`** | **`235,308.8`** | **`315.5`** audio-s/s (`10.5` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `4,096` | `4` | **`0.156`** | `60.34` | **`25.14`** | `188.8` | **`0.0139`** | **`420,190.0`** | **`1,273.0`** audio-s/s (`265.2` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `16,384` | `1` | **`0.281`** | `58.79` | **`25.65`** | `335.9` | **`0.0157`** | **`233,620.8`** | **`311.9`** audio-s/s (`68.0` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `16,384` | `2` | **`0.544`** | `60.52` | **`18.46`** | `592.1` | **`0.0123`** | **`241,011.6`** | **`866.5`** audio-s/s (`132.2` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `65,536` | `1` | **`2.810`** | `48.47` | **`20.00`** | `2,859.7` | **`0.0129`** | **`93,290.0`** | **`400.0`** audio-s/s (`82.5` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `65,536` | `2` | **`5.754`** | `57.65` | **`25.10`** | `5,809.1` | **`0.0156`** | **`91,114.8`** | **`637.6`** audio-s/s (`138.8` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `131,072` | `1` | **`10.275`** | `66.51` | **`25.25`** | `10,330.1` | **`0.0155`** | **`51,025.2`** | **`316.8`** audio-s/s (`60.2` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `4,096` | `1` | **`1.954`** | `1,389.07` | **`518.94`** | `2,480.8` | **`0.2606`** | **`8,384.0`** | **`15.4`** audio-s/s (`2.9` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `4,096` | `4` | **`2.030`** | `758.18` | **`525.86`** | `2,563.2` | **`0.2642`** | **`32,288.4`** | **`60.8`** audio-s/s (`21.1` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `16,384` | `1` | **`3.931`** | `1,394.12` | **`1.25`** | `3,961.7` | **`0.0035`** | **`16,672.8`** | **`6,413.3`** audio-s/s (`2.9` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `16,384` | `2` | **`2.596`** | `48.64` | **`529.40`** | `3,154.9` | **`0.2678`** | **`50,496.0`** | **`30.2`** audio-s/s (`164.4` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `65,536` | `1` | **`4.530`** | `49.20` | **`1.27`** | `4,560.7` | **`0.0035`** | **`57,872.0`** | **`6,287.3`** audio-s/s (`81.3` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `65,536` | `2` | **`6.948`** | `48.54` | **`1.50`** | `6,979.4` | **`0.0038`** | **`75,458.0`** | **`10,668.9`** audio-s/s (`164.8` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `131,072` | `1` | **`10.272`** | `49.50` | **`1.26`** | `10,302.7` | **`0.0035`** | **`51,042.0`** | **`6,355.1`** audio-s/s (`80.8` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `1` | **`0.098`** | `372.69` | **`18.42`** | `124.0` | **`0.0122`** | **`167,611.2`** | **`434.4`** audio-s/s (`10.7` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `4` | **`0.258`** | `49.60` | **`17.40`** | `283.6` | **`0.0122`** | **`253,584.8`** | **`1,839.5`** audio-s/s (`322.6` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `1` | **`0.401`** | `63.97` | **`20.49`** | `451.6` | **`0.0129`** | **`163,228.8`** | **`390.4`** audio-s/s (`62.5` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `2` | **`0.731`** | `65.25` | **`20.70`** | `781.3` | **`0.0134`** | **`179,347.2`** | **`773.1`** audio-s/s (`122.6` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `1` | **`3.239`** | `56.51` | **`23.25`** | `3,291.6` | **`0.0143`** | **`80,938.4`** | **`344.1`** audio-s/s (`70.8` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `2` | **`6.467`** | `55.18` | **`17.82`** | `6,514.8` | **`0.0120`** | **`81,068.0`** | **`897.8`** audio-s/s (`145.0` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `131,072` | `1` | **`11.126`** | `70.58` | **`26.09`** | `11,181.8` | **`0.0157`** | **`47,122.4`** | **`306.6`** audio-s/s (`56.7` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `1` | **`2.645`** | `1,553.97` | **`578.81`** | `3,231.9` | **`0.2924`** | **`6,193.6`** | **`13.8`** audio-s/s (`2.6` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `4` | **`2.795`** | `855.08` | **`589.39`** | `3,391.7` | **`0.2982`** | **`23,451.6`** | **`54.3`** audio-s/s (`18.7` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `1` | **`4.659`** | `1,550.35` | **`3.64`** | `4,692.4` | **`0.0045`** | **`14,066.0`** | **`2,200.7`** audio-s/s (`2.6` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `2` | **`3.452`** | `68.39` | **`547.21`** | `4,028.9` | **`0.2767`** | **`37,970.8`** | **`29.2`** audio-s/s (`117.0` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `1` | **`5.387`** | `67.44` | **`3.62`** | `5,420.6` | **`0.0045`** | **`48,658.4`** | **`2,211.8`** audio-s/s (`59.3` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `2` | **`8.123`** | `71.62` | **`3.58`** | `8,156.0` | **`0.0049`** | **`64,546.4`** | **`4,468.3`** audio-s/s (`111.7` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `131,072` | `1` | **`11.122`** | `69.05` | **`3.63`** | `11,155.4` | **`0.0045`** | **`47,138.8`** | **`2,205.0`** audio-s/s (`57.9` tok/s) |

#### Stage 2 — Audio WAV Framing & Multi-Frame `720p` Vision Preprocessing CPU Offload Ablation

| Offload Stage | Batch / Frames | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote Server Compute (ms) | Measured Speedup | Optimal Routing Policy |
|---|---:|---:|---:|---:|---:|---|
| `audio_wav_24khz_2s` | `1` | `0.079` | `5.041` | `0.108` | **`0.02x`** | Keep In-Memory on Host (<0.3 ms) |
| `audio_wav_24khz_2s` | `2` | `0.093` | `5.970` | `0.172` | **`0.02x`** | Keep In-Memory on Host (<0.3 ms) |
| `audio_wav_24khz_2s` | `4` | `0.159` | `6.806` | `0.237` | **`0.02x`** | Keep In-Memory on Host (<0.3 ms) |
| `audio_wav_24khz_2s` | `8` | `0.445` | `3.064` | `0.481` | **`0.15x`** | Keep In-Memory on Host (<0.3 ms) |
| `vision_preproc_640x480_to_256x256` | `2` | `7.200` | `16.700` | `5.380` | **`0.43x`** | Keep In-Memory on Host (<0.3 ms) |
| `vision_preproc_640x480_to_256x256` | `4` | `14.330` | `19.700` | `7.200` | **`0.73x`** | Keep In-Memory on Host (<0.3 ms) |
| `vision_preproc_1280x720_to_448x448` | `8` | `86.160` | `57.620` | `30.520` | **`1.50x`** | Offload to AVX-512 CPU Pool |

### Slice `v5p-16-wave3-keepalive` (`v5p`, `35` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave3-keepalive/SUMMARY.md`](./results/v5p-16-wave3-keepalive/SUMMARY.md) · [`results/v5p-16-wave3-keepalive/summary.json`](./results/v5p-16-wave3-keepalive/summary.json) *(updated `2026-11-02T21:10:54.538978+00:00`)*

#### Stage 1 — End-to-End `Thinker` (`4K`–`128K`) + `Talker` (`16`-RVQ) + `24 kHz` Streaming Speech

| Engine | Precision | `TP × Cluster DP` | Context | Batch | TTFT (s) | TPOT (ms) | `2s` Speech Chunk (ms) | First Audio (ms) | Speech RTF | Cluster Prefill (tok/s) | Cluster Output Rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `4,096` | `1` | **`1.317`** | `819.14` | **`252.75`** | `1,578.1` | **`0.1274`** | **`24,874.4`** | **`63.3`** audio-s/s (`9.8` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `4,096` | `4` | **`1.617`** | `1,067.70` | **`267.00`** | `1,892.8` | **`0.1344`** | **`81,036.8`** | **`239.7`** audio-s/s (`30.0` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `16,384` | `1` | **`2.475`** | `213.46` | **`44.40`** | `2,554.9` | **`0.0229`** | **`52,960.0`** | **`360.4`** audio-s/s (`37.4` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `16,384` | `2` | **`0.254`** | `44.40` | **`14.87`** | `299.7` | **`0.0082`** | **`1,032,765.6`** | **`2,152.4`** audio-s/s (`360.3` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `65,536` | `1` | **`4.895`** | `1,509.65` | **`60.35`** | `4,991.3` | **`0.0309`** | **`107,096.8`** | **`265.1`** audio-s/s (`5.3` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `65,536` | `2` | **`0.513`** | `88.18` | **`15.87`** | `559.8` | **`0.0087`** | **`2,044,321.6`** | **`2,016.5`** audio-s/s (`181.4` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | `131,072` | `1` | **`18.510`** | `2,356.86` | **`44.61`** | `18,590.0` | **`0.0230`** | **`56,649.6`** | **`358.6`** audio-s/s (`3.4` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `4,096` | `1` | **`0.058`** | `360.39` | **`19.68`** | `85.7` | **`0.0106`** | **`281,148.0`** | **`406.5`** audio-s/s (`11.1` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `4,096` | `4` | **`0.155`** | `65.16` | **`26.37`** | `188.7` | **`0.0140`** | **`423,724.4`** | **`1,213.6`** audio-s/s (`245.6` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `16,384` | `1` | **`0.279`** | `67.03` | **`27.67`** | `336.5` | **`0.0145`** | **`234,763.2`** | **`289.1`** audio-s/s (`59.7` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `16,384` | `2` | **`0.542`** | `67.96` | **`27.83`** | `599.3` | **`0.0146`** | **`241,730.0`** | **`575.0`** audio-s/s (`117.7` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `65,536` | `1` | **`2.808`** | `68.12` | **`28.18`** | `2,866.0` | **`0.0147`** | **`93,353.2`** | **`283.8`** audio-s/s (`58.7` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `65,536` | `2` | **`5.754`** | `54.68` | **`23.34`** | `5,807.0` | **`0.0124`** | **`91,111.2`** | **`685.5`** audio-s/s (`146.3` tok/s) |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | `131,072` | `1` | **`10.276`** | `50.75` | **`25.67`** | `10,331.6` | **`0.0135`** | **`51,019.6`** | **`311.6`** audio-s/s (`78.8` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `4,096` | `1` | **`1.936`** | `1,388.13` | **`513.75`** | `2,457.2` | **`0.2576`** | **`8,464.0`** | **`15.6`** audio-s/s (`2.9` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `4,096` | `4` | **`2.007`** | `750.14` | **`528.73`** | `2,542.9` | **`0.2652`** | **`32,661.6`** | **`60.5`** audio-s/s (`21.3` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `16,384` | `1` | **`3.829`** | `1,373.83` | **`1.27`** | `3,860.1` | **`0.0013`** | **`17,114.8`** | **`6,314.0`** audio-s/s (`2.9` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `16,384` | `2` | **`2.559`** | `63.51` | **`530.33`** | `3,118.9` | **`0.2659`** | **`51,213.6`** | **`30.2`** audio-s/s (`126.0` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `65,536` | `1` | **`4.515`** | `63.33` | **`1.25`** | `4,545.8` | **`0.0013`** | **`58,062.4`** | **`6,377.6`** audio-s/s (`63.2` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `65,536` | `2` | **`6.931`** | `57.97` | **`1.49`** | `6,961.4` | **`0.0015`** | **`75,647.6`** | **`10,745.2`** audio-s/s (`138.0` tok/s) |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | `131,072` | `1` | **`10.273`** | `64.26` | **`1.26`** | `10,303.9` | **`0.0013`** | **`51,035.6`** | **`6,332.9`** audio-s/s (`62.2` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `1` | **`0.097`** | `375.96` | **`25.25`** | `130.0` | **`0.0134`** | **`168,548.0`** | **`316.8`** audio-s/s (`10.6` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `4` | **`0.258`** | `74.74` | **`28.82`** | `294.3` | **`0.0152`** | **`254,076.8`** | **`1,110.5`** audio-s/s (`214.1` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `1` | **`0.400`** | `75.41` | **`30.89`** | `460.2` | **`0.0161`** | **`164,029.6`** | **`259.0`** audio-s/s (`53.0` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `2` | **`0.730`** | `76.41` | **`30.44`** | `789.6` | **`0.0160`** | **`179,656.4`** | **`525.7`** audio-s/s (`104.7` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `1` | **`3.237`** | `78.98` | **`31.75`** | `3,298.9` | **`0.0165`** | **`80,973.2`** | **`252.0`** audio-s/s (`50.6` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `2` | **`6.463`** | `73.93` | **`31.01`** | `6,523.5` | **`0.0162`** | **`81,122.8`** | **`516.0`** audio-s/s (`108.2` tok/s) |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `131,072` | `1` | **`11.121`** | `75.55` | **`30.02`** | `11,181.2` | **`0.0156`** | **`47,142.0`** | **`266.5`** audio-s/s (`53.0` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `1` | **`2.555`** | `1,531.36` | **`568.81`** | `3,130.9` | **`0.2852`** | **`6,413.6`** | **`14.1`** audio-s/s (`2.6` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `4` | **`2.697`** | `842.31` | **`577.03`** | `3,282.1` | **`0.2893`** | **`24,295.2`** | **`55.4`** audio-s/s (`19.0` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `1` | **`4.622`** | `1,523.15` | **`3.63`** | `4,654.9` | **`0.0025`** | **`14,180.4`** | **`2,201.5`** audio-s/s (`2.6` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `2` | **`3.338`** | `51.42` | **`535.30`** | `3,903.3` | **`0.2684`** | **`39,261.6`** | **`29.9`** audio-s/s (`155.6` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `1` | **`5.381`** | `76.65` | **`3.63`** | `5,414.8` | **`0.0025`** | **`48,713.2`** | **`2,206.6`** audio-s/s (`52.2` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `2` | **`8.072`** | `73.15` | **`3.59`** | `8,105.5` | **`0.0025`** | **`64,948.4`** | **`4,461.4`** audio-s/s (`109.4` tok/s) |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | `131,072` | `1` | **`11.120`** | `74.20` | **`3.63`** | `11,153.2` | **`0.0025`** | **`47,148.8`** | **`2,203.8`** audio-s/s (`53.9` tok/s) |

#### Stage 2 — Audio WAV Framing & Multi-Frame `720p` Vision Preprocessing CPU Offload Ablation

| Offload Stage | Batch / Frames | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote Server Compute (ms) | Measured Speedup | Optimal Routing Policy |
|---|---:|---:|---:|---:|---:|---|
| `audio_wav_24khz_2s` | `1` | `0.085` | `0.549` | `0.098` | **`0.15x`** | Keep In-Memory on Host (<0.3 ms) |
| `audio_wav_24khz_2s` | `2` | `0.111` | `0.600` | `0.112` | **`0.18x`** | Keep In-Memory on Host (<0.3 ms) |
| `audio_wav_24khz_2s` | `4` | `0.166` | `0.733` | `0.183` | **`0.23x`** | Keep In-Memory on Host (<0.3 ms) |
| `audio_wav_24khz_2s` | `8` | `0.429` | `1.118` | `0.432` | **`0.38x`** | Keep In-Memory on Host (<0.3 ms) |
| `vision_preproc_640x480_to_256x256` | `2` | `7.170` | `6.350` | `5.310` | **`1.13x`** | Offload to AVX-512 CPU Pool |
| `vision_preproc_640x480_to_256x256` | `4` | `14.290` | `8.690` | `6.990` | **`1.64x`** | Offload to AVX-512 CPU Pool |
| `vision_preproc_1280x720_to_448x448` | `8` | `85.570` | `41.910` | `30.240` | **`2.04x`** | Offload to AVX-512 CPU Pool |

---

## 6. Repository Layout & Reproduction

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
