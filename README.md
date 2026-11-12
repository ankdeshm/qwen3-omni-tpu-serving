# Qwen3-Omni-30B-A3B-Instruct Serving on Google Cloud TPU v5p & v7x

End-to-end `vLLM` MoE serving (`v7x-4` and `v7x-8`), full-depth `Talker + Code2Wav` speech synthesis (`v5p-16`), and disaggregated CPU audio/vision preprocessing benchmarks for [`Qwen/Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) on Google Cloud TPU v5p and v7x.

**Summary:** On `vllm-torchtpu-ep` (`bfloat16`), sharding `Qwen3-Omni-30B-A3B-Instruct` with `TP=4` and scaling from 1 `v7x` host (`v7x-4`, 8 TensorCores) to 2 hosts (`v7x-8`, 16 TensorCores) increases `4K`-context output throughput from **703.2 tok/s (`9.43 ms` TPOT) to 1,516.5 tok/s (`10.11 ms` TPOT, `2.16×` higher than 1 host and `1.65×` higher than `v7x-8` `TP=2 × DP=8`)**. For streaming speech synthesis on `v5p-16`, `torch.compile` (`bfloat16`, warm) lowers full-depth `2 s` `24 kHz` `Talker + CodePredictor + Code2Wav` synthesis from `20.0–28.2 ms` (eager) to **1.25–1.50 ms (`14.8–22.5×` faster)**, while persistent `HTTP/1.1 Keep-Alive + TCP_NODELAY` offloading cuts 8-frame `720p` CPU vision preprocessing to **41.91 ms on `v5p-16` (`2.04×` faster than on-host CPU and `1.37×` faster than unpooled HTTP)**.

---

## 1. Model, Framework, and Hardware

- **Model:** `Qwen/Qwen3-Omni-30B-A3B-Instruct` (`30B` total / `~3.5B` active parameters, `65.68 GiB` `bfloat16` across 15 `.safetensors` shards):
  - **Architecture:** `Thinker` (48-layer `d = 2048` sparse MoE, 128 experts top-8, `32` Q / `4` KV heads) + `Talker` (20-layer 128-expert MoE generating 16-group RVQ speech codes + 5-layer `CodePredictor`) + `Code2Wav` (8-layer `24 kHz` streaming audio vocoder).
- **Framework:** `vllm-torchtpu-omni:v0.32.0` (`PyTorch` + `torch_tpu`, single-host PJRT metadata isolation for `TP=2`/`TP=4` expert-parallel `vLLM` replicas, single-client multi-device `TorchTPU` child workers, and `HTTP/1.1 Keep-Alive + TCP_NODELAY` adaptive payload routing `ADAPTIVE_OFFLOAD_MIN_BYTES=262144` to `c3-standard-44` AVX-512 CPU pods).
- **Hardware:**
  - **Cloud TPU `v5p-16`:** `2x2x4` topology (`4` hosts × `4` chips = `16` chips, `95 GiB` HBM2e per chip = `1,520 GiB` total HBM2e).
  - **Cloud TPU `v7x-4` & `v7x-8` (Ironwood):** `192 GiB` HBM3e per physical chip (`2` TensorCores per chip = `96 GiB` per TensorCore): `v7x-4` (`1` host = `4` chips / `8` TensorCores, `768 GiB` HBM3e) and `v7x-8` (`2` hosts = `8` chips / `16` TensorCores, `1,536 GiB` HBM3e).

---

## 2. Main Results

### A. End-to-End `vllm-torchtpu-ep` HTTP Serving (`bfloat16`, `v7x-4` vs. `v7x-8`)

| Slice (`TensorCores`) | Sharding (`TP × Cluster DP`) | Context | Concurrency | TTFT `p50` (s) | TPOT `p50` (ms) | Cluster Prefill (tok/s) | Cluster Output (tok/s) | Comparison |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `v7x-4` (`8 TC`) | `TP=2 × DP=4` | 4,096 | 1 / 8 | 0.347 / 0.465 | 15.71 / 15.65 | 11,791 / 70,398 | 58.8 / 456.6 | 1-host `TP=2` baseline (`1.0×`) |
| `v7x-4` (`8 TC`) | `TP=4 × DP=2` | 4,096 | 4 / 8 | **0.056** / 0.506 | **7.46** / 9.43 | 290,592 / 64,786 | 516.6 / **703.2** | **1.54× higher** output tok/s & **1.66× faster** TPOT vs. `v7x-4` `TP=2` |
| `v7x-4` (`8 TC`) | `TP=4 × DP=2` | 16,384 | 1 / 4 | 0.711 / 0.209 | 17.48 / **7.98** | 23,045 / 313,471 | 48.4 / **451.6** | **2.39× higher** output tok/s vs. `v7x-4` `TP=2` (`189.0 tok/s`) |
| `v7x-8` (`16 TC`) | `TP=2 × DP=8` | 4,096 | 1 / 8 | 0.326 / 0.452 | 15.71 / 15.61 | 25,156 / 145,062 | 118.2 / 917.9 | **2.01×** over 1-host `v7x-4` `TP=2` |
| **`v7x-8` (`16 TC`)** | **`TP=4 × DP=4`** | **4,096** | **4 / 8** | 0.454 / **0.085** | **8.06** / 10.11 | 72,172 / **770,800** | 812.4 / **1,516.5** | **2.16×** over 1-host `v7x-4` `TP=4`; **1.65×** over `v7x-8` `TP=2` |
| **`v7x-8` (`16 TC`)** | **`TP=4 × DP=4`** | **16,384** | **1 / 4** | 0.695 / **0.105** | 17.47 / **7.60** | 47,179 / **1,249,835** | 97.2 / **995.2** | **2.20×** over 1-host `v7x-4` `TP=4`; **2.76×** over `v7x-8` `TP=2` |

*Notes:* Contexts `64K` and `128K` returned `HTTP 400` in `vllm-torchtpu-ep` due to configured `max_model_len` caps (see [`APPENDIX.md`](./APPENDIX.md)). Raw logs are in [`results/v7x-4/summary.json`](./results/v7x-4/summary.json) and [`results/v7x-8/summary.json`](./results/v7x-8/summary.json).

### B. Full-Depth `Talker + CodePredictor + Code2Wav` `2 s` (`24 kHz`) Speech Synthesis (`v5p-16`, `TP=4 × DP=4`)

| Engine Mode (`Precision`) | Batch (`Context`) | `2 s` Speech Chunk Latency | Speech RTF | Cluster Speech Rate | Comparison |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `torchtpu-omni-eager` (`BF16`) | `B=1` / `B=2` (`64K`) | 28.18 ms / 23.34 ms | 0.0147 / 0.0124 | 283.8 / 685.5 audio-s/s | Uncompiled eager baseline (`1.0×`) |
| `torchtpu-omni-eager` (`FP8_DYNAMIC`) | `B=1` / `B=2` (`64K`) | 23.25 ms / 17.82 ms | 0.0143 / 0.0120 | 344.1 / 897.8 audio-s/s | **1.31×** higher `B=2` speech rate vs. eager `BF16` |
| **`torchtpu-omni-compiled` (`BF16`, warm)** | **`B=1` / `B=2` (`64K`)** | **1.25 ms / 1.49 ms** | **0.0013 / 0.0015** | **6,377.6 / 10,745.2 audio-s/s** | **15.7–22.5× faster** per `2 s` chunk than eager `BF16` |
| `torchtpu-omni-compiled` (`FP8_DYNAMIC`, warm) | `B=1` / `B=2` (`64K`) | 3.63 ms / 3.59 ms | 0.0025 / 0.0025 | 2,206.6 / 4,461.4 audio-s/s | **4.97–6.40× faster** than eager `FP8`, `2.4–2.9×` slower than compiled `BF16` |

*Notes:* Warm compiled measurements exclude first-shape `TorchDynamo` compilation (`513–589 ms` on cold shapes). Associated `Thinker` timings in [`results/v5p-16-keepalive/`](./results/v5p-16-keepalive/) are layer-scaled local-shard operator microbenchmarks and are reported separately with full caveats in [`APPENDIX.md`](./APPENDIX.md).

### C. Disaggregated CPU Multimodal Preprocessing (`c3-standard-44` AVX-512 vs. On-Host CPU)

| Workload (`Payload Class`) | Platform | On-Host CPU | Unpooled HTTP RTT | Keep-Alive HTTP RTT (`Server Compute`) | Comparison & Routing Policy |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `8 × 1280×720 → 448×448` vision (`>= 256 KB`) | `v5p-16` | 85.57 ms | 57.62 ms | **41.91 ms** (`30.24 ms`) | **2.04× faster** than on-host CPU (`1.37×` vs. unpooled HTTP) — **Offload** |
| `8 × 1280×720 → 448×448` vision (`>= 256 KB`) | `v7x-4` / `v7x-8` | 61.32–63.41 ms | — | **42.63–49.99 ms** (`31.39–39.31 ms`) | **1.27–1.44× faster** than `v7x` host CPU — **Offload** |
| `4 × 640×480 → 256×256` vision (`>= 256 KB`) | `v5p-16` / `v7x-8` | 14.29 / 10.66 ms | 19.70 ms / — | **8.69 / 9.12 ms** (`6.99 / 7.35 ms`) | **1.64× (`v5p`) / 1.17× (`v7x`) faster** than host CPU — **Offload** |
| `2 s` `24 kHz` WAV framing (`B=1`, `96 KB < 256 KB`) | `v5p-16` / `v7x-8` | **0.085 / 0.072 ms** | 5.041 ms / — | 0.549 / 0.509 ms (`0.098 / 0.090 ms`) | HTTP RTT is `6.5–7.1×` slower than in-memory — **Keep on host** |

---

## 3. Reproducing the Main Results

Scripts: [`scripts/run_qwen3_omni_tpu_bench.py`](./scripts/run_qwen3_omni_tpu_bench.py), [`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py), and [`k8s/`](./k8s/):

```bash
# 1. Deploy the disaggregated C3 Xeon CPU offload service (after setting YOUR_CPU_NODEPOOL and YOUR_GCP_PROJECT_ID)
kubectl apply -f k8s/multimodal-cpu-offload-service.yaml

# 2. Run on GKE TPU v7x-8 (2 hosts = 16 TensorCores), v7x-4 (1 host = 8 TensorCores), or v5p-16 (4 hosts = 16 chips)
kubectl apply -f k8s/v7x-8-qwen3-omni.yaml
kubectl apply -f k8s/v7x-4-qwen3-omni.yaml
kubectl apply -f k8s/v5p-16-qwen3-omni.yaml

# Or invoke directly on a TPU host:
python3 scripts/run_qwen3_omni_tpu_bench.py --chip-type=v7x --slice-name=v7x-8 --num-hosts=2 --cores-per-host=8 --use-shm
```

---

## 4. Detailed Ablations, Engineering Notes, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Complete 35-point `v5p-16` unpooled vs. Keep-Alive tables, full `v7x-4` and `v7x-8` tables, methodology notes on layer-scaled `Thinker` microbenchmarks vs. end-to-end `vLLM`, and all failed/rejected runs (`v7x` multi-device `torch.compile` export generator bug, `vLLM` `64K/128K` `HTTP 400` context limits, and cold `TorchDynamo` compile spikes).
- **[`results/`](./results/):** Raw `summary.json` outputs under [`v5p-16-baseline/`](./results/v5p-16-baseline/), [`v5p-16-keepalive/`](./results/v5p-16-keepalive/), [`v7x-4/`](./results/v7x-4/), and [`v7x-8/`](./results/v7x-8/).
