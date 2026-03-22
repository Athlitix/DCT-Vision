"""Timing and memory measurement utilities for benchmarking."""

import time
import tracemalloc
from typing import Callable, Any


class BenchmarkResult:
    """Container for a single benchmark measurement."""

    def __init__(
        self,
        name: str,
        elapsed_ms: float,
        peak_memory_kb: float,
    ):
        self.name = name
        self.elapsed_ms = elapsed_ms
        self.peak_memory_kb = peak_memory_kb

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "peak_memory_kb": round(self.peak_memory_kb, 1),
        }


def time_fn(fn: Callable[[], Any], warmup: int = 3, repeats: int = 20) -> dict:
    """Time a function with warmup and repetitions.

    Parameters
    ----------
    fn : callable
        Zero-argument function to benchmark.
    warmup : int
        Number of warmup iterations (discarded).
    repeats : int
        Number of timed iterations.

    Returns
    -------
    dict
        Keys: mean_ms, std_ms, min_ms, max_ms, repeats.
    """
    # Warmup
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        elapsed = (time.perf_counter_ns() - start) / 1e6  # ns -> ms
        times.append(elapsed)

    import numpy as np
    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "repeats": repeats,
    }


def measure_memory(fn: Callable[[], Any]) -> dict:
    """Measure peak memory usage of a function.

    Parameters
    ----------
    fn : callable
        Zero-argument function to profile.

    Returns
    -------
    dict
        Keys: peak_kb, current_kb.
    """
    tracemalloc.start()
    fn()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "peak_kb": peak / 1024,
        "current_kb": current / 1024,
    }


def psnr(original, reconstructed) -> float:
    """Compute PSNR between two images."""
    import numpy as np
    mse = float(np.mean(
        (original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2
    ))
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)
