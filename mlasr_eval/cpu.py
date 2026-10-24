"""Pin the process to N physical CPU cores and cap math-library threads.

Call `configure()` BEFORE importing numpy, torch, onnxruntime or ctranslate2, because
several libraries read their thread settings at import time.

Pinning with sched_setaffinity is what makes "2 cores" hold for every engine, including
ones whose thread pools we can't set directly (the ONNX Runtime sessions inside
IndicConformer's remote code): extra threads still only run on the pinned cores.
"""
from __future__ import annotations

import os
import platform

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def physical_core_map(cpus: set[int]) -> dict[tuple[str, str], list[int]]:
    """Group logical CPUs by (package, core) using sysfs topology."""
    cores: dict[tuple[str, str], list[int]] = {}
    for cpu in sorted(cpus):
        base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
        key = (_read(f"{base}/physical_package_id") or "0", _read(f"{base}/core_id") or str(cpu))
        cores.setdefault(key, []).append(cpu)
    return cores


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def configure(cores: int = 2, threads: int | None = None) -> dict:
    """Pin to `cores` physical cores (all their hyper-threads) and set `threads` math threads.

    Returns a dict describing what was applied, for the results file.
    """
    threads = threads or cores
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    info = {"cpu_model": cpu_model(), "logical_cpus_machine": os.cpu_count(),
            "requested_cores": cores, "threads": threads}
    if not hasattr(os, "sched_getaffinity"):
        info["affinity"] = "unsupported on this OS (threads capped only)"
        return info

    allowed = os.sched_getaffinity(0)
    core_map = physical_core_map(allowed)
    info["physical_cores_machine"] = len(core_map)
    chosen = list(core_map.values())[:cores]
    if len(core_map) < cores:
        info["warning"] = f"only {len(core_map)} physical cores available; using all of them"
    pinned = sorted(c for group in chosen for c in group)
    os.sched_setaffinity(0, pinned)
    info["pinned_logical_cpus"] = pinned
    info["pinned_physical_cores"] = len(chosen)
    return info


def torch_threads(threads: int) -> None:
    """Apply thread limits inside torch (call after `import torch`)."""
    import torch
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set once in this process


def peak_rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def isa_flags() -> list[str]:
    """SIMD flags that matter for int8/fp32 CPU kernels (speed differs a lot across them)."""
    flags: set[str] = set()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
                break
    except OSError:
        pass
    return sorted(f for f in ("avx", "avx2", "fma", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8") if f in flags)
