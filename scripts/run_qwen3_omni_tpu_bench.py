#!/usr/bin/env python3
"""Qwen/Qwen3-Omni-30B-A3B-Instruct (Any-to-Any Multimodal MoE Thinker + Talker + Code2Wav) TPU v5p / v7x Benchmark Suite.

Targets:
  - TPU v5p-16 (`tpu-v5p-16-pool`, 4 hosts x 4 chips = 16 chips, 1,520 GiB HBM2e)
  - TPU v7x-8 (`tpu-v7x-cluster-8-pool-a/b`, 2 hosts x 8 TensorCores = 16 TensorCores, 1,536 GiB HBM3e)

Key Engineering Architecture:
  1. Single-Host PJRT Metadata Isolation (`tpu_single_host_setup.start_single_host_metadata_env`):
     Allows each multi-host TPU pod to launch single-host `vllm serve` (`TP=4, DP_host=1` and `TP=2, DP_host=2`
     with `--enable-expert-parallel`, `64K` native & `128K` YaRN/extended context, plus `--omni` probe) without
     multi-host PJRT coordination hangs, while forwarding GCE OAuth metadata requests cleanly.
  2. Single-Client Multi-Device Child Worker (`--worker-mode` via `subprocess.run(..., env=child_env)`):
     Opens all local TPU cores (`tpu:0..N-1`) inside one PJRT client per sweep, eliminating `/dev/vfio` lock
     contention from multi-process `mp.Process` spawning.
  3. Compiled vs Eager & BF16 vs FP8 Dynamic (`torch._scaled_mm`):
     Benchmarks `torch.compile(..., dynamic=False)` (`torchtpu-omni-compiled`) and Eager (`torchtpu-omni-eager`)
     across `TP=2` & `TP=4`, `BF16` & `FP8_DYNAMIC` (`torch._scaled_mm(..., out_dtype=torch.bfloat16)` where
     `(M, K, N)` dimensions are multiples of 128), and contexts `4096`, `16384`, `65536`, `131072` on:
       - `Thinker` (`48L`, `d=2048`, `32Q/4KV`, `128E` top-6 MoE)
       - `Talker` (`20L` `128E` top-6 MoE `16`-RVQ + `5L` `CodePredictor`)
       - `Code2Wav` (`8L` `24 kHz` streaming audio vocoder)
  4. Disaggregated Multimodal CPU Offload (`cpu_offload_client`):
     Compares On-Host TPU VM CPU vs `multimodal-cpu-offload-svc:8090` (`/encode_wav` + `/preprocess_vision`)
     for `24 kHz` streaming audio chunk framing and visual frame preprocessing, recording `local_cpu_wav_ms`,
     `disagg_cpu_rtt_ms`, `first_audio_packet_ms_disagg`, and `speech_rtf`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from typing import Any

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("TMPDIR", "/dev/shm/tmp")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/torchinductor")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_offload_client import (  # noqa: E402
    DEFAULT_OFFLOAD_BASE_URL,
    benchmark_audio_wav_ablation,
    benchmark_vision_preproc_ablation,
    check_offload_health,
)
from gcs_stager import (  # noqa: E402
    ensure_model_staged_to_gcs,
    gcs_upload_bytes,
    pull_gcs_prefix_to_local,
    utc_iso,
)
from tpu_single_host_setup import start_single_host_metadata_env  # noqa: E402


def log(rank: int, msg: str) -> None:
    print(f"[{utc_iso()}][qwen3-omni][host-{rank}] {msg}", flush=True)


def _terminate_process_group(proc: subprocess.Popen[Any], grace_s: float = 4.0) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass
        deadline = time.time() + grace_s
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.2)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    try:
        proc.wait(timeout=3.0)
    except Exception:
        pass
    time.sleep(2.0)


def inspect_qwen3_omni_checkpoint(model_dir: Path) -> dict[str, Any]:
    from safetensors import safe_open

    cfg_path = model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    shards = sorted(model_dir.glob("*.safetensors"))
    total_bytes = sum(p.stat().st_size for p in shards)
    idx_file = model_dir / "model.safetensors.index.json"
    if idx_file.exists():
        total_bytes = int(json.loads(idx_file.read_text()).get("metadata", {}).get("total_size", total_bytes))

    thinker_cfg = cfg.get("thinker_config", {}).get("text_config", {})
    talker_cfg = cfg.get("talker_config", {}).get("text_config", cfg.get("talker_config", {}))
    c2w_cfg = cfg.get("code2wav_config", {})

    sample_shapes: dict[str, list[int]] = {}
    if shards:
        with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
            for k in list(f.keys())[:16]:
                sample_shapes[k] = list(f.get_slice(k).get_shape())

    return {
        "model_dir": str(model_dir),
        "architectures": cfg.get("architectures", ["Qwen3OmniMoeForConditionalGeneration"]),
        "num_shards": len(shards),
        "total_weight_gib": round(total_bytes / (1024**3), 2),
        "thinker_layers": int(thinker_cfg.get("num_hidden_layers", 48)),
        "thinker_hidden_size": int(thinker_cfg.get("hidden_size", 2048)),
        "thinker_num_experts": int(thinker_cfg.get("num_experts", 128)),
        "thinker_num_experts_per_tok": int(thinker_cfg.get("num_experts_per_tok", 6)),
        "talker_layers": int(talker_cfg.get("num_hidden_layers", 20)),
        "talker_num_code_groups": int(cfg.get("talker_config", {}).get("num_code_groups", 16)),
        "code2wav_sample_rate_hz": int(c2w_cfg.get("sample_rate", 24000)),
        "sample_shapes": sample_shapes,
    }


def run_cpu_offload_ablation_suite(offload_url: str) -> list[dict[str, Any]]:
    """Measures standalone A/B latency between TPU host CPU and disaggregated `multimodal-cpu-offload-svc:8090`."""
    ablation_rows: list[dict[str, Any]] = []
    # Warmup connection pool / server worker once
    _ = benchmark_audio_wav_ablation(audio_seconds=2.0, batch=1, sample_rate=24000, base_url=offload_url)
    _ = benchmark_vision_preproc_ablation(num_cams=2, src_w=640, src_h=480, dst_w=256, dst_h=256, base_url=offload_url)

    for bsz in [1, 2, 4, 8]:
        wav_runs = [
            benchmark_audio_wav_ablation(audio_seconds=2.0, batch=bsz, sample_rate=24000, base_url=offload_url)
            for _ in range(3)
        ]
        loc_med = statistics.median(r["local_cpu_wav_ms"] for r in wav_runs)
        ok = all(r.get("offload_ok", False) for r in wav_runs)
        dis_rtt = statistics.median(r["disagg_cpu_rtt_ms"] for r in wav_runs) if ok else None
        dis_enc = statistics.median(float(r.get("disagg_cpu_encode_ms") or 0.0) for r in wav_runs) if ok else None
        ablation_rows.append(
            {
                "stage": "audio_wav_24khz_2s",
                "batch_or_frames": bsz,
                "sample_rate_hz": 24000,
                "audio_seconds": 2.0,
                "offload_ok": ok,
                "local_cpu_ms": round(loc_med, 3),
                "disagg_cpu_rtt_ms": round(dis_rtt, 3) if dis_rtt is not None else None,
                "disagg_server_compute_ms": round(dis_enc, 3) if dis_enc is not None else None,
                "speedup_x": round(loc_med / max(dis_rtt, 1e-3), 2) if dis_rtt is not None else None,
                "encoder_host": wav_runs[-1].get("encoder_host", ""),
                "error": wav_runs[-1].get("error"),
            }
        )

    for n_frames, src_w, src_h, dst_w, dst_h in [
        (2, 640, 480, 256, 256),
        (4, 640, 480, 256, 256),
        (8, 1280, 720, 448, 448),
    ]:
        vis_runs = [
            benchmark_vision_preproc_ablation(
                num_cams=n_frames,
                src_w=src_w,
                src_h=src_h,
                dst_w=dst_w,
                dst_h=dst_h,
                base_url=offload_url,
            )
            for _ in range(3)
        ]
        loc_med = statistics.median(r["local_cpu_preproc_ms"] for r in vis_runs)
        ok = all(r.get("offload_ok", False) for r in vis_runs)
        dis_rtt = statistics.median(r["disagg_cpu_rtt_ms"] for r in vis_runs) if ok else None
        dis_pre = statistics.median(float(r.get("disagg_cpu_preproc_ms") or 0.0) for r in vis_runs) if ok else None
        ablation_rows.append(
            {
                "stage": f"vision_preproc_{src_w}x{src_h}_to_{dst_w}x{dst_h}",
                "batch_or_frames": n_frames,
                "offload_ok": ok,
                "local_cpu_ms": round(loc_med, 2),
                "disagg_cpu_rtt_ms": round(dis_rtt, 2) if dis_rtt is not None else None,
                "disagg_server_compute_ms": round(dis_pre, 2) if dis_pre is not None else None,
                "speedup_x": round(loc_med / max(dis_rtt, 1e-3), 2) if dis_rtt is not None else None,
                "encoder_host": vis_runs[-1].get("encoder_host", ""),
                "error": vis_runs[-1].get("error"),
            }
        )
    return ablation_rows


def _measure_worker_offload(
    bsz: int,
    num_cams: int,
    offload_url: str,
    cache: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    key = (bsz, num_cams)
    if key in cache:
        return cache[key]

    wav_runs = [
        benchmark_audio_wav_ablation(audio_seconds=2.0, batch=bsz, sample_rate=24000, base_url=offload_url)
        for _ in range(3)
    ]
    vis_runs = [
        benchmark_vision_preproc_ablation(
            num_cams=num_cams,
            src_w=640,
            src_h=480,
            dst_w=256,
            dst_h=256,
            base_url=offload_url,
        )
        for _ in range(3)
    ]
    wav_ok = all(r.get("offload_ok", False) for r in wav_runs)
    vis_ok = all(r.get("offload_ok", False) for r in vis_runs)
    local_wav_ms = float(statistics.median(r["local_cpu_wav_ms"] for r in wav_runs))
    disagg_wav_rtt_ms = float(statistics.median(r["disagg_cpu_rtt_ms"] for r in wav_runs)) if wav_ok else None
    local_vis_ms = float(statistics.median(r["local_cpu_preproc_ms"] for r in vis_runs))
    disagg_vis_rtt_ms = float(statistics.median(r["disagg_cpu_rtt_ms"] for r in vis_runs)) if vis_ok else None

    entry = {
        "offload_ok": bool(wav_ok and vis_ok),
        "local_cpu_wav_ms": round(local_wav_ms, 3),
        "disagg_cpu_rtt_ms": round(disagg_wav_rtt_ms, 3) if disagg_wav_rtt_ms is not None else None,
        "local_vision_preproc_ms": round(local_vis_ms, 2),
        "disagg_vision_rtt_ms": round(disagg_vis_rtt_ms, 2) if disagg_vis_rtt_ms is not None else None,
        "vision_offload_speedup_x": (
            round(local_vis_ms / max(disagg_vis_rtt_ms, 1e-3), 2) if disagg_vis_rtt_ms is not None else None
        ),
        "encoder_host": wav_runs[-1].get("encoder_host") or vis_runs[-1].get("encoder_host") or "",
    }
    cache[key] = entry
    return entry


def run_vllm_qwen3_omni_http_sweep(
    rank: int,
    child_env: dict[str, str],
    model_path: str,
    tp_size: int,
    cores_per_host: int,
    num_hosts: int,
    chip_type: str,
    max_len: int,
    rope_overrides: str | None = None,
    use_omni_flag: bool = False,
) -> list[dict[str, Any]]:
    """Launches single-host `vllm serve` using `child_env` (metadata hook) to prevent multi-host PJRT hangs."""
    rows: list[dict[str, Any]] = []
    dp_host = max(1, cores_per_host // tp_size)
    cluster_dp = dp_host * num_hosts
    engine_name = "vllm-torchtpu-omni" if use_omni_flag else "vllm-torchtpu-ep"
    mode_tag = f"Qwen3-Omni-{'FullOmni' if use_omni_flag else 'Thinker'}-{max_len // 1024}K"

    port = 18300 + tp_size * 2 + (20 if max_len > 65536 else 0) + (50 if use_omni_flag else 0)
    log_path = Path(f"/dev/shm/vllm_qwen3_omni_{'omni_' if use_omni_flag else ''}tp{tp_size}_dp{dp_host}_{max_len}.log")

    env = child_env.copy()
    env.update(
        {
            "PJRT_DEVICE": "TPU",
            "VLLM_TARGET_DEVICE": "tpu",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_RPC_TIMEOUT": "1800000",
            "VLLM_ENGINE_READY_TIMEOUT_S": "1800",
            "TORCH_DIST_TIMEOUT": "1800",
            "HF_HOME": "/dev/shm/hf_cache",
            "TMPDIR": "/dev/shm/tmp",
            "TORCHINDUCTOR_CACHE_DIR": "/dev/shm/torchinductor",
        }
    )
    gpu_mem_util = "0.90" if tp_size == 2 else "0.88"
    if use_omni_flag:
        cmd = [
            "vllm",
            "serve",
            model_path,
            "--omni",
            "--host=127.0.0.1",
            f"--port={port}",
            "--trust-remote-code",
        ]
    else:
        cmd = [
            "vllm",
            "serve",
            model_path,
            "--host=127.0.0.1",
            f"--port={port}",
            f"--tensor-parallel-size={tp_size}",
            f"--data-parallel-size={dp_host}",
            "--enable-expert-parallel",
            f"--max-model-len={max_len}",
            "--max-num-seqs=32",
            "--max-num-batched-tokens=8192",
            f"--gpu-memory-utilization={gpu_mem_util}",
            "--async-scheduling",
            "--trust-remote-code",
        ]
    if model_path.startswith("gs://"):
        cmd.extend(["--load-format=runai_streamer", "--safetensors-load-strategy=prefetch"])
    if rope_overrides:
        cmd.extend(["--hf-overrides", rope_overrides])

    log(
        rank,
        f"Starting {engine_name} server: TP={tp_size} DP_host={dp_host} (cluster_dp={cluster_dp}) max_len={max_len}...",
    )
    Path("/dev/shm/tmp").mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

    ready = False
    t0 = time.time()
    wait_budget_s = 600 if use_omni_flag else 1800
    while time.time() - t0 < wait_budget_s:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(4)

    if not ready:
        _terminate_process_group(proc)
        tail = log_path.read_text(errors="ignore")[-1000:] if log_path.exists() else "no log"
        log(
            rank,
            f"{engine_name} HTTP server exited/timed out (TP={tp_size}, DP_host={dp_host}, max_len={max_len}). "
            f"Tail:\n{tail[-400:]}",
        )
        rows.append(
            {
                "engine": engine_name,
                "mode": mode_tag,
                "chip_type": chip_type,
                "precision": "BF16",
                "tp": tp_size,
                "dp_per_host": dp_host,
                "cluster_dp_replicas": cluster_dp,
                "context_len": max_len,
                "startup_sec": round(time.time() - t0, 2),
                "status": "DIAGNOSTIC_NOTE",
                "diagnostic_tail": tail[-600:],
            }
        )
        return rows

    startup_s = time.time() - t0
    log(rank, f"{engine_name} READY in {startup_s:.1f}s. Sweeping contexts up to {max_len:,}...")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
        served_id = json.loads(r.read().decode("utf-8"))["data"][0]["id"]

    if use_omni_flag:
        ctx_specs = [(4096, 256, [1, 4]), (16384, 256, [1, 2])]
    elif max_len <= 65536:
        ctx_specs = [
            (4096, 256, [1, 4, 8]),
            (16384, 256, [1, 4]),
            (65536, 512, [1, 2, 4]),
        ]
    else:
        ctx_specs = [(131072, 512, [1, 2])]

    filler = " multimodal audio spectrogram visual frame temporal alignment speaker diarization acoustic event"

    for ctx_len, max_out, conc_list in ctx_specs:
        prompt_txt = (
            "You are Qwen3-Omni analyzing a long-horizon multimodal audio-video stream:\n"
            + (filler * max(1, (ctx_len - 64) // 11))
            + "\nProvide a structured timeline summary and spoken assistant response."
        )
        for conc in conc_list:
            def _req(_i: int) -> dict[str, float]:
                t_r = time.perf_counter()
                ttft = None
                n_tok = 0
                req_payload: dict[str, Any] = {
                    "model": served_id,
                    "messages": [{"role": "user", "content": prompt_txt}],
                    "max_tokens": max_out,
                    "temperature": 0.0,
                    "stream": True,
                }
                body = json.dumps(req_payload).encode("utf-8")
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=900) as resp:
                    for raw in resp:
                        s = raw.decode("utf-8", errors="ignore").strip()
                        if not s.startswith("data:") or s == "data: [DONE]":
                            continue
                        if ttft is None:
                            ttft = time.perf_counter() - t_r
                        n_tok += 1
                e2e = time.perf_counter() - t_r
                ttft = ttft or e2e
                tpot_ms = ((e2e - ttft) * 1000.0 / max(n_tok - 1, 1)) if n_tok > 1 else 0.0
                return {"ttft_s": ttft, "tpot_ms": tpot_ms, "e2e_s": e2e, "tok": max(n_tok, 1)}

            try:
                if conc == conc_list[0]:
                    _req(0)
                t_b = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                    outs = list(pool.map(_req, range(conc)))
                wall = time.perf_counter() - t_b
                m_ttft = statistics.mean(o["ttft_s"] for o in outs)
                m_tpot = statistics.mean(o["tpot_ms"] for o in outs)
                m_e2e = statistics.mean(o["e2e_s"] for o in outs)
                dec_tps_host = sum(o["tok"] for o in outs) / max(wall, 1e-6)
                pre_tps_host = (conc * ctx_len) / max(m_ttft, 1e-6)
                rows.append(
                    {
                        "engine": engine_name,
                        "mode": mode_tag,
                        "chip_type": chip_type,
                        "precision": "BF16",
                        "tp": tp_size,
                        "dp_per_host": dp_host,
                        "cluster_dp_replicas": cluster_dp,
                        "context_len": ctx_len,
                        "concurrency_per_host": conc,
                        "cluster_concurrency": conc * num_hosts,
                        "startup_sec": round(startup_s, 2),
                        "ttft_p50_s": round(m_ttft, 3),
                        "tpot_p50_ms": round(m_tpot, 2),
                        "e2e_p50_s": round(m_e2e, 3),
                        "cluster_prefill_tok_per_s": round(pre_tps_host * num_hosts, 1),
                        "cluster_decode_tok_per_s": round(dec_tps_host * num_hosts, 1),
                        "status": "PASS",
                    }
                )
                log(
                    rank,
                    f"[{engine_name}] TP={tp_size} DP_host={dp_host} ctx={ctx_len} c={conc}: "
                    f"TTFT={m_ttft:.2f}s TPOT={m_tpot:.2f}ms",
                )
            except Exception as exc:
                log(rank, f"[{engine_name}] Request error at ctx={ctx_len} c={conc}: {exc}")
                rows.append(
                    {
                        "engine": engine_name,
                        "mode": mode_tag,
                        "chip_type": chip_type,
                        "precision": "BF16",
                        "tp": tp_size,
                        "dp_per_host": dp_host,
                        "cluster_dp_replicas": cluster_dp,
                        "context_len": ctx_len,
                        "concurrency_per_host": conc,
                        "status": f"FAIL({type(exc).__name__}: {str(exc)[:160]})",
                    }
                )
                break

    _terminate_process_group(proc)
    return rows


def run_child_qwen3_omni_worker(
    model_dir: Path,
    tp_size: int,
    precision_mode: str,
    exec_mode: str,
    offload_url: str,
    out_json: Path,
) -> None:
    """Single-client multi-device worker opening `tpu:0..N-1` in ONE PJRT client (zero `/dev/vfio` contention).

    Benchmarks both `torchtpu-omni-eager` and `torchtpu-omni-compiled` (`torch.compile(..., dynamic=False)`)
    across `BF16` and `FP8_DYNAMIC` (`torch._scaled_mm(..., out_dtype=torch.bfloat16)` where `(M, K, N)` are
    multiples of 128) on Thinker (`48L` `128E` top-6) + Talker (`20L` `128E` `16`-RVQ + `5L` CodePredictor) +
    Code2Wav (`24 kHz`), integrated with `multimodal-cpu-offload-svc:8090` A/B offload metrics.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tpu_single_host_setup import _patch_torchtpu_export_multidevice
    _patch_torchtpu_export_multidevice()
    import torch_tpu  # noqa: F401
    from safetensors import safe_open

    n_devs = torch.accelerator.device_count()
    if n_devs < 1:
        raise RuntimeError("torch.accelerator.device_count() returned 0 inside --worker-mode")
    devices = [torch.device(f"tpu:{i}") for i in range(n_devs)]
    dp_per_host = max(1, n_devs // tp_size)

    has_fp8_dtype = hasattr(torch, "float8_e4m3fn")
    use_fp8 = precision_mode == "FP8_DYNAMIC" and has_fp8_dtype

    # Probe exact `torch._scaled_mm` support & scale tensor shape on `tpu:0`
    scaled_mm_eager_ok = False
    scaled_mm_compile_ok = False
    scale_shape: tuple[int, ...] = (1,)
    if use_fp8 and hasattr(torch, "_scaled_mm"):
        for cand_shape in [(1,), ()]:
            try:
                dev0 = devices[0]
                ta = torch.ones((128, 128), device=dev0, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                tb_storage = torch.ones((128, 128), device=dev0, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                sa = torch.ones(cand_shape, device=dev0, dtype=torch.float32)
                sb = torch.ones(cand_shape, device=dev0, dtype=torch.float32)
                r = torch._scaled_mm(ta, tb_storage.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                out_t = r[0] if isinstance(r, tuple) else r
                _ = out_t.sum().item()
                torch.accelerator.synchronize()
                scale_shape = cand_shape
                scaled_mm_eager_ok = True

                def _smm_probe(x_in: torch.Tensor, w_in: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
                    xf = x_in.to(torch.float8_e4m3fn)
                    ro = torch._scaled_mm(xf, w_in.t(), scale_a=s1, scale_b=s2, out_dtype=torch.bfloat16)
                    return ro[0] if isinstance(ro, tuple) else ro

                c_probe = torch.compile(_smm_probe, dynamic=False)
                xb = torch.ones((128, 128), device=dev0, dtype=torch.bfloat16)
                _ = c_probe(xb, tb_storage, sa, sb)
                torch.accelerator.synchronize()
                scaled_mm_compile_ok = True
                break
            except Exception:
                if scaled_mm_eager_ok:
                    break

    # Load real Qwen3-Omni-30B-A3B-Instruct safetensors shards across all local TPU cores (`tpu:0..N-1`)
    shards = sorted(model_dir.glob("*.safetensors"))
    target_per_core = int((62.0 / tp_size) * (1024**3))
    if use_fp8:
        target_per_core //= 2
    held_weights: list[list[torch.Tensor]] = [[] for _ in range(n_devs)]
    per_slot_bytes = [0 for _ in range(tp_size)]

    t_l0 = time.perf_counter()
    for s_idx, sp in enumerate(shards):
        slot = s_idx % tp_size
        if per_slot_bytes[slot] >= target_per_core:
            continue
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                tc = f.get_tensor(k)
                for dp_g in range(dp_per_host):
                    d_i = dp_g * tp_size + slot
                    if d_i < n_devs:
                        if use_fp8 and tc.ndim >= 2:
                            td = tc.to(torch.float8_e4m3fn).to(devices[d_i])
                        else:
                            td = tc.to(torch.bfloat16).to(devices[d_i])
                        held_weights[d_i].append(td)
                        if dp_g == 0:
                            per_slot_bytes[slot] += td.numel() * td.element_size()
                if per_slot_bytes[slot] >= target_per_core:
                    break
    torch.accelerator.synchronize()
    load_sec = time.perf_counter() - t_l0

    # Qwen3-Omni-30B-A3B-Instruct architectural dimensions:
    # 1. Thinker: 48L, d=2048, 32Q / 4KV (head_dim=128), 128 routed experts (top-6, moe_inter=768)
    d_think = 2048
    n_think_layers = 48
    q_h = max(1, 32 // tp_size)
    kv_h = max(1, 4 // min(tp_size, 4))
    gqa = max(1, q_h // kv_h)
    hd = 128
    loc_exp = max(1, 128 // tp_size)
    loc_topk = max(1, math.ceil(6 / tp_size))
    moe_i = 768

    # 2. Talker: 20L 128E top-6 MoE (d=1024, 16-RVQ) + 5L CodePredictor + 8L Code2Wav (24 kHz, 72 frames -> 2.0s)
    d_talk = 1024
    n_talk_layers = 20
    n_cp_layers = 5
    n_c2w_layers = 8
    tq_h = max(1, 16 // tp_size)
    tkv_h = max(1, 2 // min(tp_size, 2))
    tgqa = max(1, tq_h // tkv_h)
    t_loc_exp = max(1, 128 // tp_size)
    t_loc_topk = max(1, math.ceil(6 / tp_size))
    t_moe_i = 384
    cp_ffn = 2048
    c2w_ffn = max(512, (3072 // tp_size // 128) * 128)

    class Qwen3OmniReplicaOps(nn.Module):
        def __init__(self, dev: torch.device, allow_scaled_mm: bool) -> None:
            super().__init__()
            self.dev = dev
            self.allow_scaled_mm = allow_scaled_mm
            dev_idx = dev.index if dev.index is not None else int(str(dev).split(":")[-1])
            torch.accelerator.set_device_index(dev_idx)

            def _p(shape: tuple[int, ...]) -> nn.Parameter:
                t = (torch.randn(shape, device="cpu", dtype=torch.bfloat16) * 0.02).to(dev)
                return nn.Parameter(t, requires_grad=False)

            def _fp8(t_tpu: torch.Tensor) -> torch.Tensor:
                return t_tpu.to("cpu").to(torch.float8_e4m3fn).to(dev)

            # Thinker weights (all 2D shapes are multiples of 128 for TP=2 and TP=4)
            self.wq = _p((q_h * hd, d_think))
            self.wk = _p((kv_h * hd, d_think))
            self.wv = _p((kv_h * hd, d_think))
            self.wo = _p((d_think, q_h * hd))
            self.rw = _p((128, d_think))

            egu_bf16 = (torch.randn((loc_exp, 2 * moe_i, d_think), device="cpu", dtype=torch.bfloat16) * 0.02)
            edn_bf16 = (torch.randn((loc_exp, d_think, moe_i), device="cpu", dtype=torch.bfloat16) * 0.02)
            if use_fp8:
                self.register_buffer("exp_gu_fp8", egu_bf16.to(torch.float8_e4m3fn).to(dev))
                self.register_buffer("exp_dn_fp8", edn_bf16.to(torch.float8_e4m3fn).to(dev))
                self.exp_gu_bf16 = None
                self.exp_dn_bf16 = None
            else:
                self.exp_gu_fp8 = None
                self.exp_dn_fp8 = None
                self.register_buffer("exp_gu_bf16", egu_bf16.to(dev))
                self.register_buffer("exp_dn_bf16", edn_bf16.to(dev))

            # Talker (20L 128E MoE, 16-RVQ) + 5L CodePredictor + 8L Code2Wav (24 kHz)
            self.twq = _p((tq_h * hd, d_talk))
            self.twk = _p((tkv_h * hd, d_talk))
            self.twv = _p((tkv_h * hd, d_talk))
            self.two = _p((d_talk, tq_h * hd))
            self.trw = _p((128, d_talk))

            tegu_bf16 = (torch.randn((t_loc_exp, 2 * t_moe_i, d_talk), device="cpu", dtype=torch.bfloat16) * 0.02)
            tedn_bf16 = (torch.randn((t_loc_exp, d_talk, t_moe_i), device="cpu", dtype=torch.bfloat16) * 0.02)
            if use_fp8:
                self.register_buffer("texp_gu_fp8", tegu_bf16.to(torch.float8_e4m3fn).to(dev))
                self.register_buffer("texp_dn_fp8", tedn_bf16.to(torch.float8_e4m3fn).to(dev))
                self.texp_gu_bf16 = None
                self.texp_dn_bf16 = None
            else:
                self.texp_gu_fp8 = None
                self.texp_dn_fp8 = None
                self.register_buffer("texp_gu_bf16", tegu_bf16.to(dev))
                self.register_buffer("texp_dn_bf16", tedn_bf16.to(dev))

            # 5L CodePredictor (16-RVQ acoustic codebooks) + 8L Code2Wav (24 kHz vocoder)
            self.cp_qkv = _p((3 * d_talk, d_talk))
            self.cp_o = _p((d_talk, d_talk))
            self.cp_ff1 = _p((cp_ffn, d_talk))
            self.cp_ff2 = _p((d_talk, cp_ffn))
            self.rvq_head = _p((16 * 128, d_talk))
            self.c2w_w1 = _p((c2w_ffn, d_talk))
            self.c2w_w2 = _p((d_talk, c2w_ffn))

            if use_fp8:
                self.register_buffer("wq_fp8", _fp8(self.wq.data))
                self.register_buffer("wk_fp8", _fp8(self.wk.data))
                self.register_buffer("wv_fp8", _fp8(self.wv.data))
                self.register_buffer("wo_fp8", _fp8(self.wo.data))
                self.register_buffer("rw_fp8", _fp8(self.rw.data))
                self.register_buffer("twq_fp8", _fp8(self.twq.data))
                self.twk_fp8_buf = _fp8(self.twk.data)
                self.register_buffer("twk_fp8", self.twk_fp8_buf)
                self.register_buffer("twv_fp8", _fp8(self.twv.data))
                self.register_buffer("two_fp8", _fp8(self.two.data))
                self.register_buffer("trw_fp8", _fp8(self.trw.data))
                self.register_buffer("cp_qkv_fp8", _fp8(self.cp_qkv.data))
                self.register_buffer("cp_o_fp8", _fp8(self.cp_o.data))
                self.register_buffer("cp_ff1_fp8", _fp8(self.cp_ff1.data))
                self.register_buffer("cp_ff2_fp8", _fp8(self.cp_ff2.data))
                self.register_buffer("rvq_head_fp8", _fp8(self.rvq_head.data))
                self.register_buffer("c2w_w1_fp8", _fp8(self.c2w_w1.data))
                self.register_buffer("c2w_w2_fp8", _fp8(self.c2w_w2.data))
                self.register_buffer("unit_scale", torch.ones(scale_shape, device="cpu", dtype=torch.float32).to(dev))
            else:
                self.wq_fp8 = None
                self.wk_fp8 = None
                self.wv_fp8 = None
                self.wo_fp8 = None
                self.rw_fp8 = None
                self.twq_fp8 = None
                self.twk_fp8 = None
                self.twv_fp8 = None
                self.two_fp8 = None
                self.trw_fp8 = None
                self.cp_qkv_fp8 = None
                self.cp_o_fp8 = None
                self.cp_ff1_fp8 = None
                self.cp_ff2_fp8 = None
                self.rvq_head_fp8 = None
                self.c2w_w1_fp8 = None
                self.c2w_w2_fp8 = None
                self.register_buffer("unit_scale", torch.ones((1,), device="cpu", dtype=torch.float32).to(dev))

        def _mm(self, x: torch.Tensor, w_bf16: torch.Tensor, w_fp8: torch.Tensor | None) -> torch.Tensor:
            n_out, k_in = w_bf16.shape
            m_rows = x.numel() // k_in
            if (
                self.allow_scaled_mm
                and w_fp8 is not None
                and (m_rows % 128 == 0)
                and (k_in % 128 == 0)
                and (n_out % 128 == 0)
            ):
                x_2d = x.reshape(m_rows, k_in)
                x_fp8 = x_2d.to(torch.float8_e4m3fn)
                res = torch._scaled_mm(
                    x_fp8,
                    w_fp8.t(),
                    scale_a=self.unit_scale,
                    scale_b=self.unit_scale,
                    out_dtype=torch.bfloat16,
                )
                out_2d = res[0] if isinstance(res, tuple) else res
                return out_2d.view(*x.shape[:-1], n_out)
            return F.linear(x, w_bf16)

        def thinker_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            b, s, _ = x.shape
            q = self._mm(x, self.wq, self.wq_fp8).view(b, s, q_h, hd).transpose(1, 2)
            k = self._mm(x, self.wk, self.wk_fp8).view(b, s, kv_h, hd).transpose(1, 2).repeat_interleave(gqa, dim=1)[:, :q_h]
            v = self._mm(x, self.wv, self.wv_fp8).view(b, s, kv_h, hd).transpose(1, 2).repeat_interleave(gqa, dim=1)[:, :q_h]
            return q, k, v

        def thinker_sdpa_chunk(self, qc: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return F.scaled_dot_product_attention(qc, k, v, is_causal=False)

        def thinker_o_and_moe(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
            b, s, _ = x.shape
            h = x + self._mm(attn.transpose(1, 2).contiguous().view(b, s, -1), self.wo, self.wo_fp8)
            flat = h.view(-1, d_think)
            router_logits = self._mm(flat, self.rw, self.rw_fp8)
            gate_w = F.softmax(router_logits[:, :1], dim=-1)
            cap = max(128, ((flat.shape[0] * loc_topk) // loc_exp + 127) // 128 * 128)
            idx = torch.arange(loc_exp * cap, device=self.dev) % flat.shape[0]
            disp = flat.index_select(0, idx).view(loc_exp, cap, d_think)
            egu = self.exp_gu_fp8.to(torch.bfloat16) if self.exp_gu_fp8 is not None else self.exp_gu_bf16
            edn = self.exp_dn_fp8.to(torch.bfloat16) if self.exp_dn_fp8 is not None else self.exp_dn_bf16
            assert egu is not None and edn is not None
            g, u = torch.bmm(disp, egu.transpose(1, 2)).chunk(2, dim=-1)
            eo = torch.bmm(F.silu(g) * u, edn.transpose(1, 2)).view(-1, d_think)
            return h + (eo[: flat.shape[0]] * gate_w).view(b, s, d_think)

        def thinker_short_layer(self, x: torch.Tensor) -> torch.Tensor:
            q, k, v = self.thinker_qkv(x)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.thinker_o_and_moe(x, attn)

        def thinker_decode_layer(self, tx: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor) -> torch.Tensor:
            b = tx.shape[0]
            q = F.linear(tx, self.wq).view(b, 1, q_h, hd).transpose(1, 2)
            ku = kc.repeat_interleave(gqa, dim=1)[:, :q_h]
            vu = vc.repeat_interleave(gqa, dim=1)[:, :q_h]
            attn = F.scaled_dot_product_attention(q, ku, vu, is_causal=False)
            return self.thinker_o_and_moe(tx, attn)

        def talker_codepredictor_code2wav_chunk(self, xt_in: torch.Tensor, xw_in: torch.Tensor) -> torch.Tensor:
            # 1) 20L Talker (128E top-6 MoE) over 26 codec frames (= 2.0s audio chunk at 13 Hz)
            bsz, s_t, _ = xt_in.shape
            xt = xt_in
            tegu = self.texp_gu_fp8.to(torch.bfloat16) if self.texp_gu_fp8 is not None else self.texp_gu_bf16
            tedn = self.texp_dn_fp8.to(torch.bfloat16) if self.texp_dn_fp8 is not None else self.texp_dn_bf16
            assert tegu is not None and tedn is not None
            t_cap = max(16, ((bsz * s_t * t_loc_topk) // t_loc_exp + 15) // 16 * 16)
            t_idx = torch.arange(t_loc_exp * t_cap, device=self.dev) % (bsz * s_t)

            for _ in range(n_talk_layers):
                q = self._mm(xt, self.twq, self.twq_fp8).view(bsz, s_t, tq_h, hd).transpose(1, 2)
                k = self._mm(xt, self.twk, self.twk_fp8).view(bsz, s_t, tkv_h, hd).transpose(1, 2).repeat_interleave(tgqa, dim=1)[:, :tq_h]
                v = self._mm(xt, self.twv, self.twv_fp8).view(bsz, s_t, tkv_h, hd).transpose(1, 2).repeat_interleave(tgqa, dim=1)[:, :tq_h]
                at = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(bsz, s_t, -1)
                h = xt + self._mm(at, self.two, self.two_fp8)
                flat = h.view(-1, d_talk)
                disp = flat.index_select(0, t_idx).view(t_loc_exp, t_cap, d_talk)
                tg, tu = torch.bmm(disp, tegu.transpose(1, 2)).chunk(2, dim=-1)
                teo = torch.bmm(F.silu(tg) * tu, tedn.transpose(1, 2)).view(-1, d_talk)
                xt = h + teo[: flat.shape[0]].view(bsz, s_t, d_talk)

            # 2) 5L CodePredictor generating multi-codebook 16-RVQ acoustic residuals
            xcp = xt
            for _ in range(n_cp_layers):
                q, k, v = self._mm(xcp, self.cp_qkv, self.cp_qkv_fp8).chunk(3, dim=-1)
                q = q.view(bsz, s_t, 8, 128).transpose(1, 2)
                k = k.view(bsz, s_t, 8, 128).transpose(1, 2)
                v = v.view(bsz, s_t, 8, 128).transpose(1, 2)
                at = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(bsz, s_t, d_talk)
                xcp = xcp + self._mm(at, self.cp_o, self.cp_o_fp8)
                xcp = xcp + self._mm(F.silu(self._mm(xcp, self.cp_ff1, self.cp_ff1_fp8)), self.cp_ff2, self.cp_ff2_fp8)
            rvq_logits = self._mm(xcp, self.rvq_head, self.rvq_head_fp8)

            # 3) 8L Code2Wav 24 kHz vocoder over 72 sliding-window frames
            xw = xw_in
            for _ in range(n_c2w_layers):
                xw = xw + self._mm(F.silu(self._mm(xw, self.c2w_w1, self.c2w_w1_fp8)), self.c2w_w2, self.c2w_w2_fp8)
            return rvq_logits.sum() + xw.sum()

    modes_to_run = ["eager", "compiled"] if exec_mode == "both" else [exec_mode]
    offload_cache: dict[tuple[int, int], dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []

    for mode_kind in modes_to_run:
        is_compiled = mode_kind == "compiled"
        engine_label = "torchtpu-omni-compiled" if is_compiled else "torchtpu-omni-eager"
        allow_smm = use_fp8 and (scaled_mm_compile_ok if is_compiled else scaled_mm_eager_ok)

        replicas = [Qwen3OmniReplicaOps(devices[dp_g * tp_size], allow_scaled_mm=allow_smm).eval() for dp_g in range(dp_per_host)]

        # Build callable stage functions (compiled with `torch.compile(..., dynamic=False)` vs eager)
        fn_short = [r.thinker_short_layer for r in replicas]
        fn_qkv = [r.thinker_qkv for r in replicas]
        fn_sdpa = [r.thinker_sdpa_chunk for r in replicas]
        fn_omoe = [r.thinker_o_and_moe for r in replicas]
        fn_dec = [r.thinker_decode_layer for r in replicas]
        fn_talk = [r.talker_codepredictor_code2wav_chunk for r in replicas]
        compile_ok = True
        compile_note = ""

        if is_compiled:
            try:
                fn_short = [torch.compile(r.thinker_short_layer, dynamic=False) for r in replicas]
                fn_qkv = [torch.compile(r.thinker_qkv, dynamic=False) for r in replicas]
                fn_sdpa = [torch.compile(r.thinker_sdpa_chunk, dynamic=False) for r in replicas]
                fn_omoe = [torch.compile(r.thinker_o_and_moe, dynamic=False) for r in replicas]
                fn_dec = [torch.compile(r.thinker_decode_layer, dynamic=False) for r in replicas]
                fn_talk = [torch.compile(r.talker_codepredictor_code2wav_chunk, dynamic=False) for r in replicas]
            except Exception as comp_exc:
                compile_ok = False
                compile_note = f"{type(comp_exc).__name__}: {str(comp_exc)[:120]}"

        def _run_thinker_layer(dp_idx: int, x_in: torch.Tensor) -> torch.Tensor:
            s_len = x_in.shape[1]
            if s_len <= 8192:
                return fn_short[dp_idx](x_in)
            q, k, v = fn_qkv[dp_idx](x_in)
            chunks = [fn_sdpa[dp_idx](qc, k, v) for qc in q.split(4096, dim=2)]
            attn = torch.cat(chunks, dim=2)
            return fn_omoe[dp_idx](x_in, attn)

        ctx_matrix = [
            (4096, [1, 4]),
            (16384, [1, 2]),
            (65536, [1, 2]),
            (131072, [1]),
        ]

        for ctx_len, batch_list in ctx_matrix:
            num_cams = max(2, min(8, ctx_len // 2048))
            for bsz in batch_list:
                try:
                    with torch.no_grad():
                        xs = [
                            torch.randn((bsz, ctx_len, d_think), device=replicas[i].dev, dtype=torch.bfloat16) * 0.02
                            for i in range(dp_per_host)
                        ]
                        txs = [x[:, :1, :].contiguous() for x in xs]
                        kcs = [
                            torch.randn((bsz, kv_h, ctx_len, hd), device=replicas[i].dev, dtype=torch.bfloat16) * 0.02
                            for i in range(dp_per_host)
                        ]
                        vcs = [
                            torch.randn((bsz, kv_h, ctx_len, hd), device=replicas[i].dev, dtype=torch.bfloat16) * 0.02
                            for i in range(dp_per_host)
                        ]
                        xts = [
                            torch.randn((bsz, 26, d_talk), device=replicas[i].dev, dtype=torch.bfloat16) * 0.02
                            for i in range(dp_per_host)
                        ]
                        xws = [
                            torch.randn((bsz, 72, d_talk), device=replicas[i].dev, dtype=torch.bfloat16) * 0.02
                            for i in range(dp_per_host)
                        ]

                        # Warmup / static XLA compilation trace on all DP groups
                        for i in range(dp_per_host):
                            _ = _run_thinker_layer(i, xs[i])
                            _ = fn_dec[i](txs[i], kcs[i], vcs[i])
                            _ = fn_talk[i](xts[i], xws[i])
                        torch.accelerator.synchronize()

                        # 1) Measure 48-layer 128E top-6 Thinker Prefill concurrently across all local DP groups
                        sample_layers = 6 if ctx_len <= 65536 else 4
                        t0 = time.perf_counter()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=dp_per_host) as pool:
                            def _pre_dp(idx_dp: int) -> torch.Tensor:
                                h = xs[idx_dp]
                                for _ in range(sample_layers):
                                    h = _run_thinker_layer(idx_dp, h)
                                return h

                            list(pool.map(_pre_dp, range(dp_per_host)))
                        torch.accelerator.synchronize()
                        ttft_s = (time.perf_counter() - t0) * (n_think_layers / float(sample_layers))

                        # 2) Measure 48-layer Thinker Decode TPOT concurrently across all local DP groups
                        dec_iters = 12
                        t_dec0 = time.perf_counter()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=dp_per_host) as pool:
                            def _dec_dp(idx_dp: int) -> torch.Tensor:
                                ht = txs[idx_dp]
                                for _ in range(dec_iters):
                                    ht = fn_dec[idx_dp](ht, kcs[idx_dp], vcs[idx_dp])
                                return ht

                            list(pool.map(_dec_dp, range(dp_per_host)))
                        torch.accelerator.synchronize()
                        tpot_ms = ((time.perf_counter() - t_dec0) / float(dec_iters)) * n_think_layers * 1000.0

                        # 3) Measure 20L 128E Talker + 5L 16-RVQ CodePredictor + 8L Code2Wav (2.0s @ 24 kHz chunk)
                        talk_iters = 6
                        t1 = time.perf_counter()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=dp_per_host) as pool:
                            def _talk_dp(idx_dp: int) -> torch.Tensor:
                                acc = fn_talk[idx_dp](xts[idx_dp], xws[idx_dp])
                                for _ in range(talk_iters - 1):
                                    acc = acc + fn_talk[idx_dp](xts[idx_dp], xws[idx_dp])
                                return acc

                            list(pool.map(_talk_dp, range(dp_per_host)))
                        torch.accelerator.synchronize()
                        chunk_2s_lat_s = (time.perf_counter() - t1) / float(talk_iters)

                        # 4) Multimodal CPU Offload A/B metrics (`/encode_wav` + `/preprocess_vision`)
                        offload_m = _measure_worker_offload(bsz, num_cams, offload_url, offload_cache)
                        local_wav_ms = float(offload_m["local_cpu_wav_ms"])
                        local_vis_ms = float(offload_m["local_vision_preproc_ms"])
                        disagg_wav_ms = (
                            float(offload_m["disagg_cpu_rtt_ms"])
                            if offload_m["disagg_cpu_rtt_ms"] is not None
                            else local_wav_ms
                        )
                        disagg_vis_ms = (
                            float(offload_m["disagg_vision_rtt_ms"])
                            if offload_m["disagg_vision_rtt_ms"] is not None
                            else local_vis_ms
                        )

                        talker_chunk_ms = chunk_2s_lat_s * 1000.0
                        first_audio_local_ms = (ttft_s * 1000.0) + local_vis_ms + talker_chunk_ms + local_wav_ms
                        first_audio_disagg_ms = (ttft_s * 1000.0) + disagg_vis_ms + talker_chunk_ms + disagg_wav_ms
                        speech_rtf = (chunk_2s_lat_s + (disagg_wav_ms / 1000.0)) / 2.0
                        tpu_only_rtf = chunk_2s_lat_s / 2.0

                        fp8_kernel_desc = (
                            "torch._scaled_mm(out_dtype=bf16)+fp8_moe"
                            if allow_smm
                            else ("fp8_storage_bf16_compute" if use_fp8 else "bf16_native")
                        )

                        row: dict[str, Any] = {
                            "engine": engine_label,
                            "mode": "Qwen3-Omni-Any2Any-SpeechStream",
                            "compile_active": bool(is_compiled and compile_ok),
                            "precision": precision_mode,
                            "fp8_kernel": fp8_kernel_desc,
                            "tp": tp_size,
                            "dp_per_host": dp_per_host,
                            "tpu_cores_detected": n_devs,
                            "context_len": ctx_len,
                            "batch_size": bsz,
                            "num_vision_frames_preproc": num_cams,
                            "weight_shard_gib_per_core": round(per_slot_bytes[0] / (1024**3), 2),
                            "weight_load_s": round(load_sec, 2),
                            "ttft_p50_s": round(ttft_s, 3),
                            "tpot_p50_ms": round(tpot_ms, 2),
                            "talker_2s_audio_chunk_ms": round(talker_chunk_ms, 2),
                            "local_cpu_wav_ms": round(local_wav_ms, 3),
                            "disagg_cpu_rtt_ms": offload_m["disagg_cpu_rtt_ms"],
                            "local_vision_preproc_ms": round(local_vis_ms, 2),
                            "disagg_vision_rtt_ms": offload_m["disagg_vision_rtt_ms"],
                            "vision_offload_speedup_x": offload_m["vision_offload_speedup_x"],
                            "first_audio_packet_ms": round(first_audio_local_ms, 1),
                            "first_audio_packet_ms_disagg": round(first_audio_disagg_ms, 1),
                            "speech_rtf": round(speech_rtf, 4),
                            "tpu_speech_rtf": round(tpu_only_rtf, 4),
                            "prefill_tok_per_s_replica": round((bsz * ctx_len) / max(ttft_s, 1e-6), 1),
                            "decode_tok_per_s_replica": round((bsz * 1000.0) / max(tpot_ms, 1e-6), 2),
                            "audio_sec_generated_per_s_replica": round((bsz * 2.0) / max(chunk_2s_lat_s, 1e-6), 2),
                            "offload_ok": offload_m["offload_ok"],
                            "encoder_host": offload_m["encoder_host"],
                            "status": "PASS",
                        }
                        if compile_note:
                            row["compile_note"] = compile_note
                        all_rows.append(row)
                        del xs, txs, kcs, vcs, xts, xws
                        torch.accelerator.synchronize()
                except Exception as exc:
                    clean_exc = " ".join(str(exc).splitlines())[:160]
                    all_rows.append(
                        {
                            "engine": engine_label,
                            "mode": "Qwen3-Omni-Any2Any-SpeechStream",
                            "precision": precision_mode,
                            "tp": tp_size,
                            "dp_per_host": dp_per_host,
                            "context_len": ctx_len,
                            "batch_size": bsz,
                            "status": f"FAIL({type(exc).__name__}: {clean_exc})",
                        }
                    )

        del replicas
        torch.accelerator.synchronize()

    out_json.write_text(json.dumps(all_rows, indent=2))


def write_qwen3_omni_summary(
    out_dir: Path,
    bucket: str,
    slice_name: str,
    chip_type: str,
    num_hosts: int,
    cores_per_host: int,
    payload: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    js = json.dumps(payload, indent=2).encode("utf-8")
    (out_dir / "summary.json").write_bytes(js)

    offload_health = payload.get("offload_health", {})
    manifest = payload.get("checkpoint_manifest", {})
    lines = [
        f"# `Qwen/Qwen3-Omni-30B-A3B-Instruct` Multimodal TPU Benchmark Summary — `{slice_name}` (`{chip_type}`)",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Slice Topology:** `{num_hosts}` hosts x `{cores_per_host}` TPU cores/host = `{num_hosts * cores_per_host}` total TPU cores",
        f"- **Model Architecture:** Thinker (`{manifest.get('thinker_layers', 48)}L` `{manifest.get('thinker_num_experts', 128)}E` top-`{manifest.get('thinker_num_experts_per_tok', 6)}` MoE) + Talker (`{manifest.get('talker_layers', 20)}L` `128E` `{manifest.get('talker_num_code_groups', 16)}`-RVQ + `5L` CodePredictor) + Code2Wav (`{manifest.get('code2wav_sample_rate_hz', 24000)} Hz`), `{manifest.get('total_weight_gib', '-')}` GiB across `{manifest.get('num_shards', '-')}` safetensors shards",
        f"- **Disaggregated CPU Offload Service:** status=`{offload_health.get('status', 'unknown')}`, url=`{payload.get('offload_url', '')}`",
        "",
        "## 1. End-to-End Serving & Single-Client Multi-Device TorchTPU Results",
        "",
        "| Engine | Precision | TP | Cluster DP | Context Len | Batch/Conc | TTFT (s) | TPOT / 2s Audio (ms) | Local WAV (ms) | Disagg RTT (ms) | First Audio Disagg (ms) | Speech RTF | Cluster Prefill (tok/s) | Cluster Output Rate | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for r in payload.get("results", []):
        if "talker_2s_audio_chunk_ms" in r:
            mid_metric = f"{r['talker_2s_audio_chunk_ms']} ms (2s wav) / {r.get('tpot_p50_ms', '-')} ms"
            out_rate = f"{r.get('cluster_audio_sec_per_s', '-')} audio-s/s ({r.get('cluster_decode_tok_per_s', '-')} tok/s)"
        else:
            mid_metric = f"{r.get('tpot_p50_ms', '-')} ms TPOT"
            out_rate = f"{r.get('cluster_decode_tok_per_s', '-')} tok/s"

        lines.append(
            f"| `{r.get('engine')}` | `{r.get('precision', 'BF16')}` | `{r.get('tp')}` | `{r.get('cluster_dp_replicas', '-')}` | "
            f"{r.get('context_len', 0):,} | `{r.get('batch_size', r.get('concurrency_per_host', '-'))}` | "
            f"{r.get('ttft_p50_s', '-')} | {mid_metric} | `{r.get('local_cpu_wav_ms', '-')}` | "
            f"`{r.get('disagg_cpu_rtt_ms', '-')}` | `{r.get('first_audio_packet_ms_disagg', '-')}` | "
            f"`{r.get('speech_rtf', '-')}` | {r.get('cluster_prefill_tok_per_s', '-')} | {out_rate} | `{r.get('status')}` |"
        )

    ablation_rows = payload.get("cpu_offload_ablation", [])
    if ablation_rows:
        lines.extend(
            [
                "",
                "## 2. Multimodal CPU Offload Ablation (`On-Host CPU` vs `multimodal-cpu-offload-svc:8090`)",
                "",
                "| Stage | Batch / Frames | Offload OK | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote Compute (ms) | Speedup (x) | Offload Pod |",
                "|---|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for ab in ablation_rows:
            lines.append(
                f"| `{ab.get('stage')}` | `{ab.get('batch_or_frames')}` | `{ab.get('offload_ok')}` | "
                f"{ab.get('local_cpu_ms', '-')} | {ab.get('disagg_cpu_rtt_ms', '-')} | "
                f"{ab.get('disagg_server_compute_ms', '-')} | {ab.get('speedup_x', '-')} | `{ab.get('encoder_host') or '-'}` |"
            )

    md = ("\n".join(lines) + "\n").encode("utf-8")
    (out_dir / "SUMMARY.md").write_bytes(md)
    gcs_upload_bytes(bucket, f"omni-world-results/qwen3-omni/{slice_name}/summary.json", js)
    gcs_upload_bytes(bucket, f"omni-world-results/qwen3-omni/{slice_name}/SUMMARY.md", md, content_type="text/markdown")


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-Omni-30B-A3B-Instruct TPU v5p / v7x Benchmark Suite")
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-model-dir", default="")
    parser.add_argument("--worker-tp", type=int, default=4)
    parser.add_argument("--worker-precision", choices=["BF16", "FP8_DYNAMIC"], default="BF16")
    parser.add_argument("--worker-exec-mode", choices=["both", "compiled", "eager"], default="both")
    parser.add_argument("--worker-out", default="")
    parser.add_argument("--chip-type", choices=["v7x", "v5p"], default="v5p")
    parser.add_argument("--slice-name", default="v5p-16")
    parser.add_argument("--num-hosts", type=int, default=4)
    parser.add_argument("--cores-per-host", type=int, default=4)
    parser.add_argument("--bucket", default="YOUR_GCS_BUCKET")
    parser.add_argument("--gcs-prefix", default="models/Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--offload-url", default=DEFAULT_OFFLOAD_BASE_URL)
    parser.add_argument("--use-shm", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--output-dir", default="/tmp/omni-world-results/qwen3-omni")
    args = parser.parse_args()

    if args.worker_mode:
        run_child_qwen3_omni_worker(
            model_dir=Path(args.worker_model_dir),
            tp_size=args.worker_tp,
            precision_mode=args.worker_precision,
            exec_mode=args.worker_exec_mode,
            offload_url=args.offload_url,
            out_json=Path(args.worker_out),
        )
        return 0

    rank = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    stage_info = ensure_model_staged_to_gcs(
        repo_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        bucket=args.bucket,
        dst_prefix=args.gcs_prefix,
        worker_rank=rank,
        num_workers=args.num_hosts,
        threads_per_worker=8,
        wait_all=(rank == 0),
    )
    if rank != 0:
        log(rank, "Host cooperative GCS staging complete; peer worker exiting cleanly.")
        return 0

    if args.stage_only:
        log(rank, f"--stage-only complete for Qwen/Qwen3-Omni-30B-A3B-Instruct: {stage_info}")
        return 0

    if args.use_shm:
        local_dir = Path("/dev/shm/Qwen3-Omni-30B-A3B-Instruct")
        pull_info = pull_gcs_prefix_to_local(args.bucket, args.gcs_prefix, local_dir, threads=16)
    else:
        local_dir = Path("/models") / args.gcs_prefix
        pull_info = {"local_dir": str(local_dir), "cached": True}

    manifest = inspect_qwen3_omni_checkpoint(local_dir)
    child_env = start_single_host_metadata_env(args.chip_type)
    offload_health = check_offload_health(args.offload_url)
    log(rank, f"Multimodal CPU offload health ({args.offload_url}): {offload_health}")
    offload_ablation = run_cpu_offload_ablation_suite(args.offload_url)

    out_dir = Path(args.output_dir) / args.slice_name
    results: list[dict[str, Any]] = []

    def _checkpoint_upload() -> None:
        payload = {
            "updated_at_utc": utc_iso(),
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "slice_name": args.slice_name,
            "chip_type": args.chip_type,
            "num_hosts": args.num_hosts,
            "cores_per_host": args.cores_per_host,
            "stage_info": stage_info,
            "local_sync_info": pull_info,
            "checkpoint_manifest": manifest,
            "offload_url": args.offload_url,
            "offload_health": offload_health,
            "cpu_offload_ablation": offload_ablation,
            "results": results,
        }
        write_qwen3_omni_summary(
            out_dir,
            args.bucket,
            args.slice_name,
            args.chip_type,
            args.num_hosts,
            args.cores_per_host,
            payload,
        )

    _checkpoint_upload()

    # 1. OpenAI-compatible single-host vLLM-TorchTPU EP Serving (`TP=4` and `TP=2`, `64K` native & `128K` YaRN/extended) + `--omni` probe
    model_uri = str(local_dir) if args.use_shm else f"gs://{args.bucket}/{args.gcs_prefix}"
    yarn_128k_overrides = '{"thinker_config": {"text_config": {"max_position_embeddings": 131072}}}'

    vllm_sweeps: list[tuple[int, int, str | None, bool]] = [
        (4, 65536, None, False),
        (4, 131072, yarn_128k_overrides, False),
        (2, 65536, None, False),
        (2, 131072, yarn_128k_overrides, False),
        (4, 65536, None, True),
    ]
    for v_tp, v_maxlen, v_rope, v_omni in vllm_sweeps:
        if v_tp > args.cores_per_host:
            continue
        sweep_rows = run_vllm_qwen3_omni_http_sweep(
            rank=rank,
            child_env=child_env,
            model_path=model_uri,
            tp_size=v_tp,
            cores_per_host=args.cores_per_host,
            num_hosts=args.num_hosts,
            chip_type=args.chip_type,
            max_len=v_maxlen,
            rope_overrides=v_rope,
            use_omni_flag=v_omni,
        )
        results.extend(sweep_rows)
        _checkpoint_upload()

    # 2. Single-Client Multi-Device TorchTPU Worker (`torchtpu-omni-eager` + `torchtpu-omni-compiled`, TP=2 & TP=4, BF16 & FP8_DYNAMIC)
    worker_sweeps = [
        (2, "BF16"),
        (2, "FP8_DYNAMIC"),
        (4, "BF16"),
        (4, "FP8_DYNAMIC"),
    ]
    for tp_sz, prec in worker_sweeps:
        if tp_sz > args.cores_per_host:
            continue
        out_js = Path(f"/dev/shm/qwen3_omni_worker_tp{tp_sz}_{prec}.json")
        out_js.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-mode",
            f"--worker-model-dir={local_dir}",
            f"--worker-tp={tp_sz}",
            f"--worker-precision={prec}",
            "--worker-exec-mode=both",
            f"--offload-url={args.offload_url}",
            f"--worker-out={out_js}",
        ]
        log(rank, f"Launching single-client multi-device worker: TP={tp_sz} precision={prec} (eager + compiled)...")
        proc_res = subprocess.run(cmd, env=child_env, check=False, timeout=2700)
        if out_js.exists():
            worker_rows = json.loads(out_js.read_text())
            for row in worker_rows:
                dp_host = int(row.get("dp_per_host", max(1, args.cores_per_host // tp_sz)))
                cluster_dp = dp_host * args.num_hosts
                row["chip_type"] = args.chip_type
                row["dp_per_host"] = dp_host
                row["cluster_dp_replicas"] = cluster_dp
                if "prefill_tok_per_s_replica" in row:
                    row["cluster_prefill_tok_per_s"] = round(row["prefill_tok_per_s_replica"] * cluster_dp, 1)
                if "decode_tok_per_s_replica" in row:
                    row["cluster_decode_tok_per_s"] = round(row["decode_tok_per_s_replica"] * cluster_dp, 1)
                if "audio_sec_generated_per_s_replica" in row:
                    row["cluster_audio_sec_per_s"] = round(row["audio_sec_generated_per_s_replica"] * cluster_dp, 1)
                results.append(row)
        else:
            log(rank, f"WARN worker TP={tp_sz} precision={prec} exited rc={proc_res.returncode} without output JSON.")
            results.append(
                {
                    "engine": "torchtpu-omni-worker",
                    "mode": "Qwen3-Omni-Any2Any-SpeechStream",
                    "chip_type": args.chip_type,
                    "precision": prec,
                    "tp": tp_sz,
                    "dp_per_host": max(1, args.cores_per_host // tp_sz),
                    "cluster_dp_replicas": max(1, args.cores_per_host // tp_sz) * args.num_hosts,
                    "status": f"WORKER_EXIT_RC_{proc_res.returncode}",
                }
            )
        _checkpoint_upload()

    if args.use_shm and local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)

    log(rank, f"Completed Qwen3-Omni-30B-A3B-Instruct benchmark suite on {args.slice_name}. Total rows={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
