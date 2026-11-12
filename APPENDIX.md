# Appendix: Detailed Sweeps, Offload Progression, and Diagnostic Failures (`Qwen3-Omni-30B-A3B` on TPU v5p & v7x)

This document records the complete benchmark tables across all four evaluated slices (`v5p-16-baseline`, `v5p-16-keepalive`, `v7x-4`, and `v7x-8`), the measurement distinction between end-to-end `vllm-torchtpu-ep` serving and the local-shard `TorchTPU` operator microbenchmark, and every compiler/runtime error encountered. For headline numbers and reproduction commands, see [`README.md`](./README.md).

---

## 1. Measurement Scope & Methodology Notes

1. **End-to-end `vllm-torchtpu-ep` (`v7x-4` and `v7x-8`):** Uses [`scripts/tpu_single_host_setup.py`](./scripts/tpu_single_host_setup.py) to isolate per-host PJRT metadata so each `v7x` host launches native `vllm serve` (`TP=4 × DP_host=2` or `TP=2 × DP_host=4` with `--enable-expert-parallel`) over the full `65.68 GiB` `Qwen/Qwen3-Omni-30B-A3B-Instruct` checkpoint, driven via live HTTP requests.
2. **Single-client multi-device `TorchTPU` operator harness (`torchtpu-omni-eager` / `torchtpu-omni-compiled`):**
   - Opens all local TPU cores inside a single PJRT process (`run_child_qwen3_omni_worker` in [`scripts/run_qwen3_omni_tpu_bench.py`](./scripts/run_qwen3_omni_tpu_bench.py)) to avoid `/dev/vfio` lock contention across `mp.Process` spawns.
   - **Full-depth speech path:** Executes all `20` `Talker` layers + `5` `CodePredictor` layers + `8` `Code2Wav` (`24 kHz`) vocoder blocks unscaled.
   - **Layer-scaled `Thinker` path (caveat):** Holds the real `.safetensors` shards in HBM while timing `6` (`<=16K` context) or `4` (`>=64K` context) `Thinker` MoE layers on local shards and multiplying by `48 / N` **without inter-chip `AllReduce` / `AllToAll` collectives**. Therefore, `Thinker` TTFT/TPOT columns in the `torchtpu-omni-*` tables below reflect local operator compute only, not full distributed serving latency.
   - **Cold vs. warm `TorchDynamo` shapes:** In `torchtpu-omni-compiled` on `v5p-16`, `4,096` (`B=1`, `B=4`) and `16,384` (`B=2`) rows (`513.8–589.4 ms` speech chunk) included worker-thread first-shape `TorchDynamo` compilation, whereas `16,384` (`B=1`), `65,536` (`B=1`, `B=2`), and `131,072` (`B=1`) executed warm (`1.25–3.64 ms` per `2 s` speech chunk).
3. **Disaggregated CPU offload progression (`v5p-16-baseline` vs. `v5p-16-keepalive`):**
   - Initial unpooled `urllib` HTTP calls (`v5p-16-baseline`) paid per-request TCP handshake + slow-start overhead (`16.70 ms` RTT for `5.38 ms` compute on `2× 640×480` frames; `5.04–6.81 ms` RTT on `0.1 ms` WAV packets) and software `fp8` dequantization overhead.
   - Switching to thread-local `HTTP/1.1 Keep-Alive + TCP_NODELAY` connections (`v5p-16-keepalive`), hardware `torch._scaled_mm(..., out_dtype=torch.bfloat16)`, and an adaptive `256 KB` payload threshold (`ADAPTIVE_OFFLOAD_MIN_BYTES=262144`) kept `<256 KB` `24 kHz` WAV framing in-memory (`0.068–0.445 ms`) while cutting remote `8× 720p` vision preprocessing RTT from `57.62 ms` to `41.91 ms` (`2.04×` faster than `85.57 ms` on-host CPU).

---

## 2. What Did Not Work / Runtime & Compiler Failures

| Platform | Engine & Configuration | Observed Failure | Root Cause |
| :--- | :--- | :--- | :--- |
| `v7x-4` & `v7x-8` | `torchtpu-omni-compiled` (`FP8_DYNAMIC`, `TP=4`, `4K–128K`) | `BackendCompilerFailed: RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index` | PyTorch/TorchTPU export bug when compiling graphs across non-zero device indices (`tpu:1..7`) inside a single multi-device process on `v7x`. |
| `v7x-4` & `v7x-8` | `torchtpu-omni-worker` (`BF16` `TP=2/4`, `FP8_DYNAMIC` `TP=2`) | `WORKER_EXIT_RC_1` | Child worker HBM/allocator exit when opening 8 `v7x` TensorCores under `TP=2` (`4` replicas/host) or unquantized `BF16` inside a single process after `vLLM` runs. |
| `v7x-4` & `v7x-8` | `vllm-torchtpu-ep` (`BF16`, `65,536` & `131,072` context) | `FAIL(HTTPError: HTTP Error 400: Bad Request)` | Requested prompt length exceeded the active `vLLM` server `max_model_len` budget (`16,384` tokens) configured for dual/quad per-host DP replicas. |
| `v5p-16` | `vllm-torchtpu` probes (`13` diagnostic probes) | Skipped / diagnostic | Single-host `v5p` (`4 × 95 GiB = 380 GiB` HBM2e) lacked headroom for multiple `65.68 GiB` MoE+Talker replicas alongside long-context KV caches without multi-host sharding. |
| `v5p-16` & `v7x` | HTTP offload of `2 s` `24 kHz` WAV (`96 KB`) and `2 × 640×480` on `v7x` | `0.12–0.38×` speedup (WAV), `0.81–0.82×` (`2`-cam on `v7x`) | Sub-millisecond host numpy WAV framing (`0.068–0.085 ms`) and fast `v7x` host CPU `2`-frame resize (`5.16–5.20 ms`) are faster than network serialization + HTTP RTT (`0.51 ms` and `6.33 ms`). |

---

## 3. Complete `v7x-4` (`1` Host = `8` TensorCores) & `v7x-8` (`2` Hosts = `16` TensorCores) Tables

Raw data: [`results/v7x-4/summary.json`](./results/v7x-4/summary.json) and [`results/v7x-8/summary.json`](./results/v7x-8/summary.json).

### A. End-to-End `vllm-torchtpu-ep` HTTP Serving (`BF16`)

| Slice | `TP × Cluster DP` | Context | Concurrency | TTFT `p50` (s) | TPOT `p50` (ms) | E2E `p50` (s) | Cluster Prefill (tok/s) | Cluster Output (tok/s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v7x-4` (`8 TC`) | `TP=4 × DP=2` | 4,096 | 1 | 0.342 | 18.58 | 2.540 | 11,981.2 | 50.4 |
| `v7x-4` (`8 TC`) | `TP=4 × DP=2` | 4,096 | 4 | **0.056** | **7.46** | 0.991 | 290,592.0 | 516.6 |
| `v7x-4` (`8 TC`) | `TP=4 × DP=2` | 4,096 | 8 | 0.506 | 9.43 | 1.456 | 64,786.2 | **703.2** |
| `v7x-4` (`8 TC`) | `TP=4 × DP=2` | 16,384 | 1 | 0.711 | 17.48 | 2.646 | 23,045.2 | 48.4 |
| `v7x-4` (`8 TC`) | `TP=4 × DP=2` | 16,384 | 4 | 0.209 | 7.98 | 1.134 | 313,470.9 | **451.6** |
| `v7x-4` (`8 TC`) | `TP=2 × DP=4` | 4,096 | 1 | 0.347 | 15.71 | 2.177 | 11,791.3 | 58.8 |
| `v7x-4` (`8 TC`) | `TP=2 × DP=4` | 4,096 | 4 | 0.661 | 16.40 | 2.427 | 24,769.0 | 210.9 |
| `v7x-4` (`8 TC`) | `TP=2 × DP=4` | 4,096 | 8 | 0.465 | 15.65 | 2.243 | 70,397.6 | 456.6 |
| `v7x-4` (`8 TC`) | `TP=2 × DP=4` | 16,384 | 1 | 0.920 | 14.74 | 2.337 | 17,815.3 | 54.8 |
| `v7x-4` (`8 TC`) | `TP=2 × DP=4` | 16,384 | 4 | 0.567 | 16.14 | 4.183 | 115,684.6 | 189.0 |
| `v7x-8` (`16 TC`) | `TP=4 × DP=4` | 4,096 | 1 | 0.311 | 18.67 | 5.092 | 26,299.6 | 100.9 |
| `v7x-8` (`16 TC`) | `TP=4 × DP=4` | 4,096 | 4 | 0.454 | 8.06 | 2.517 | 72,171.9 | 812.4 |
| `v7x-8` (`16 TC`) | `TP=4 × DP=4` | 4,096 | 8 | **0.085** | 10.11 | 2.674 | 770,800.1 | **1,516.5** |
| `v7x-8` (`16 TC`) | `TP=4 × DP=4` | 16,384 | 1 | 0.695 | 17.47 | 4.504 | 47,178.7 | 97.2 |
| `v7x-8` (`16 TC`) | `TP=4 × DP=4` | 16,384 | 4 | **0.105** | **7.60** | 2.050 | **1,249,834.8** | **995.2** |
| `v7x-8` (`16 TC`) | `TP=2 × DP=8` | 4,096 | 1 | 0.326 | 15.71 | 4.349 | 25,156.4 | 118.2 |
| `v7x-8` (`16 TC`) | `TP=2 × DP=8` | 4,096 | 4 | 0.774 | 16.13 | 4.903 | 42,362.9 | 418.5 |
| `v7x-8` (`16 TC`) | `TP=2 × DP=8` | 4,096 | 8 | 0.452 | 15.61 | 4.430 | 145,062.0 | 917.9 |
| `v7x-8` (`16 TC`) | `TP=2 × DP=8` | 16,384 | 1 | 0.919 | 14.74 | 4.691 | 35,661.2 | 109.6 |
| `v7x-8` (`16 TC`) | `TP=2 × DP=8` | 16,384 | 4 | 0.472 | 16.73 | 4.066 | 277,953.3 | 360.8 |

### B. `v7x-4` & `v7x-8` Eager `FP8_DYNAMIC` Operator & Full-Depth Speech Chunk Sweeps

| Slice | `TP × Cluster DP` | Context | Batch | Layer-Scaled TTFT (s) | Layer-Scaled TPOT (ms) | Full-Depth `2 s` Speech Chunk (ms) | Speech RTF | Cluster Speech Rate (audio-s/s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v7x-4` | `TP=4 × DP=2` | 4,096 | 1 | 1.488 | 1,141.56 | 251.35 | 0.1260 | 15.9 |
| `v7x-4` | `TP=4 × DP=2` | 4,096 | 4 | 1.449 | 1,046.52 | 285.33 | 0.1432 | 56.1 |
| `v7x-4` | `TP=4 × DP=2` | 16,384 | 1 | 0.360 | 1,498.45 | 37.65 | 0.0192 | 106.2 |
| `v7x-4` | `TP=4 × DP=2` | 16,384 | 2 | 0.405 | 1,537.37 | 49.51 | 0.0252 | 161.6 |
| `v7x-4` | `TP=4 × DP=2` | 65,536 | 1 | 0.531 | 63.98 | 17.54 | 0.0091 | 228.0 |
| `v7x-4` | `TP=4 × DP=2` | 65,536 | 2 | 0.701 | 5,933.95 | 70.91 | 0.0359 | 112.8 |
| `v7x-4` | `TP=4 × DP=2` | 131,072 | 1 | 0.572 | 50.17 | **15.27** | **0.0080** | **262.0** |
| `v7x-8` | `TP=4 × DP=4` | 4,096 | 1 | 1.072 | 843.09 | 235.67 | 0.1181 | 34.0 |
| `v7x-8` | `TP=4 × DP=4` | 4,096 | 4 | 1.749 | 834.24 | 275.42 | 0.1383 | 116.2 |
| `v7x-8` | `TP=4 × DP=4` | 16,384 | 1 | 4.909 | 1,191.84 | 196.95 | 0.0989 | 40.6 |
| `v7x-8` | `TP=4 × DP=4` | 16,384 | 2 | 2.043 | 919.14 | 49.47 | 0.0251 | 323.4 |
| `v7x-8` | `TP=4 × DP=4` | 65,536 | 1 | 0.204 | 81.45 | **16.09** | **0.0084** | **497.3** |
| `v7x-8` | `TP=4 × DP=4` | 65,536 | 2 | 0.728 | 6,052.93 | 278.04 | 0.1394 | 57.6 |
| `v7x-8` | `TP=4 × DP=4` | 131,072 | 1 | 0.578 | 7,584.93 | 62.54 | 0.0316 | 127.9 |

---

## 4. Complete `v5p-16` Operator & Speech Synthesis Tables (`Keep-Alive` vs. Unpooled Baseline)

Raw data: [`results/v5p-16-keepalive/summary.json`](./results/v5p-16-keepalive/summary.json) and [`results/v5p-16-baseline/summary.json`](./results/v5p-16-baseline/summary.json).

### A. `v5p-16` `TorchTPU` Operator & Full-Depth `Talker + Code2Wav` Sweep (`Keep-Alive` Run, 35 Points)

| Engine | Precision | `TP × Cluster DP` | Context | Batch | Layer-Scaled TTFT (s) | Layer-Scaled TPOT (ms) | Full-Depth `2 s` Speech Chunk (ms) | First Audio (ms) | Speech RTF | Cluster Speech Rate |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | 4,096 | 1 | 1.317 | 819.14 | 252.75 | 1,578.1 | 0.1274 | 63.3 audio-s/s |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | 4,096 | 4 | 1.617 | 1,067.70 | 267.00 | 1,892.8 | 0.1344 | 239.7 audio-s/s |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | 16,384 | 1 | 2.475 | 213.46 | 44.40 | 2,554.9 | 0.0229 | 360.4 audio-s/s |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | 16,384 | 2 | 0.254 | 44.40 | 14.87 | 299.7 | 0.0082 | 2,152.4 audio-s/s |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | 65,536 | 1 | 4.895 | 1,509.65 | 60.35 | 4,991.3 | 0.0309 | 265.1 audio-s/s |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | 65,536 | 2 | 0.513 | 88.18 | 15.87 | 559.8 | 0.0087 | 2,016.5 audio-s/s |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=2 × DP=8` | 131,072 | 1 | 18.510 | 2,356.86 | 44.61 | 18,590.0 | 0.0230 | 358.6 audio-s/s |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | 4,096 | 1 | 0.058 | 360.39 | 19.68 | 85.7 | 0.0106 | 406.5 audio-s/s |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | 4,096 | 4 | 0.155 | 65.16 | 26.37 | 188.7 | 0.0140 | 1,213.6 audio-s/s |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | 16,384 | 1 | 0.279 | 67.03 | 27.67 | 336.5 | 0.0145 | 289.1 audio-s/s |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | 16,384 | 2 | 0.542 | 67.96 | 27.83 | 599.3 | 0.0146 | 575.0 audio-s/s |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | 65,536 | 1 | 2.808 | 68.12 | 28.18 | 2,866.0 | 0.0147 | 283.8 audio-s/s |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | 65,536 | 2 | 5.754 | 54.68 | 23.34 | 5,807.0 | 0.0124 | 685.5 audio-s/s |
| `torchtpu-omni-eager` | `BF16` | `TP=4 × DP=4` | 131,072 | 1 | 10.276 | 50.75 | 25.67 | 10,331.6 | 0.0135 | 311.6 audio-s/s |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | 4,096 | 1 *(cold)* | 1.936 | 1,388.13 | 513.75 | 2,457.2 | 0.2576 | 15.6 audio-s/s |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | 4,096 | 4 *(cold)* | 2.007 | 750.14 | 528.73 | 2,542.9 | 0.2652 | 60.5 audio-s/s |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | 16,384 | 1 *(warm)* | 3.829 | 1,373.83 | **1.27** | 3,860.1 | **0.0013** | 6,314.0 audio-s/s |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | 16,384 | 2 *(cold)* | 2.559 | 63.51 | 530.33 | 3,118.9 | 0.2659 | 30.2 audio-s/s |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | 65,536 | 1 *(warm)* | 4.515 | 63.33 | **1.25** | 4,545.8 | **0.0013** | 6,377.6 audio-s/s |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | 65,536 | 2 *(warm)* | 6.931 | 57.97 | **1.49** | 6,961.4 | **0.0015** | **10,745.2 audio-s/s** |
| `torchtpu-omni-compiled` | `BF16` | `TP=4 × DP=4` | 131,072 | 1 *(warm)* | 10.273 | 64.26 | **1.26** | 10,303.9 | **0.0013** | 6,332.9 audio-s/s |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 4,096 | 1 | 0.097 | 375.96 | 25.25 | 130.0 | 0.0134 | 316.8 audio-s/s |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 4,096 | 4 | 0.258 | 74.74 | 28.82 | 294.3 | 0.0152 | 1,110.5 audio-s/s |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 16,384 | 1 | 0.400 | 75.41 | 30.89 | 460.2 | 0.0161 | 259.0 audio-s/s |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 16,384 | 2 | 0.730 | 76.41 | 30.44 | 789.6 | 0.0160 | 525.7 audio-s/s |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 65,536 | 1 | 3.237 | 78.98 | 31.75 | 3,298.9 | 0.0165 | 252.0 audio-s/s |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 65,536 | 2 | 6.463 | 73.93 | 31.01 | 6,523.5 | 0.0162 | 516.0 audio-s/s |
| `torchtpu-omni-eager` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 131,072 | 1 | 11.121 | 75.55 | 30.02 | 11,181.2 | 0.0156 | 266.5 audio-s/s |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 4,096 | 1 *(cold)* | 2.555 | 1,531.36 | 568.81 | 3,130.9 | 0.2852 | 14.1 audio-s/s |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 4,096 | 4 *(cold)* | 2.697 | 842.31 | 577.03 | 3,282.1 | 0.2893 | 55.4 audio-s/s |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 16,384 | 1 *(warm)* | 4.622 | 1,523.15 | 3.63 | 4,654.9 | 0.0025 | 2,201.5 audio-s/s |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 16,384 | 2 *(cold)* | 3.338 | 51.42 | 535.30 | 3,903.3 | 0.2684 | 29.9 audio-s/s |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 65,536 | 1 *(warm)* | 5.381 | 76.65 | 3.63 | 5,414.8 | 0.0025 | 2,206.6 audio-s/s |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 65,536 | 2 *(warm)* | 8.072 | 73.15 | 3.59 | 8,105.5 | 0.0025 | 4,461.4 audio-s/s |
| `torchtpu-omni-compiled` | `FP8_DYNAMIC` | `TP=4 × DP=4` | 131,072 | 1 *(warm)* | 11.120 | 74.20 | 3.63 | 11,153.2 | 0.0025 | 2,203.8 audio-s/s |

### B. Complete Disaggregated CPU Offload Ablation Across All Four Runs

| Stage | Batch / Frames | `v5p-16` Host CPU / Unpooled RTT / Keep-Alive RTT (ms) | `v7x-4` Host CPU / Keep-Alive RTT (ms) | `v7x-8` Host CPU / Keep-Alive RTT (ms) | Remote Xeon Compute (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `audio_wav_24khz_2s` | 1 | `0.085` / `5.041` / `0.549` | `0.068` / `0.513` | `0.072` / `0.509` | `0.084–0.108` |
| `audio_wav_24khz_2s` | 2 | `0.111` / `5.970` / `0.600` | `0.071` / `0.574` | `0.086` / `0.557` | `0.112–0.172` |
| `audio_wav_24khz_2s` | 4 | `0.166` / `6.806` / `0.733` | `0.139` / `0.818` | `0.133` / `0.694` | `0.168–0.237` |
| `audio_wav_24khz_2s` | 8 | `0.429` / `3.064` / `1.118` | `0.379` / `1.208` | `0.337` / `1.122` | `0.432–0.481` |
| `vision_preproc_640x480_to_256x256` | 2 | `7.17` / `16.70` / `6.35` (`1.13×`) | `5.16` / `6.34` (`0.81×`) | `5.20` / `6.33` (`0.82×`) | `5.31–5.40` |
| `vision_preproc_640x480_to_256x256` | 4 | `14.29` / `19.70` / `8.69` (`1.64×`) | `10.51` / `9.09` (`1.16×`) | `10.66` / `9.12` (`1.17×`) | `6.99–7.35` |
| `vision_preproc_1280x720_to_448x448` | 8 | `85.57` / `57.62` / **`41.91` (`2.04×`)** | `61.32` / **`42.63` (`1.44×`)** | `63.41` / **`49.99` (`1.27×`)** | `30.24–39.31` |
