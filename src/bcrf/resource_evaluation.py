"""
BCRF Experiment - Resource Evaluation
Measures CPU time and memory usage for recovery methods.
"""

import time
import psutil
import os
from typing import Callable, Dict, Any, Tuple


def measure_resource_usage(fn: Callable, *args, **kwargs) -> Tuple[Any, Dict]:
    """
    Execute a function and measure its resource usage.

    Returns:
        (result, resource_metrics)

    resource_metrics contains:
        cpu_time_ms, wall_time_ms, memory_delta_bytes, peak_memory_bytes
    """
    process = psutil.Process(os.getpid())

    # Baseline measurements
    mem_before = process.memory_info().rss
    cpu_before = process.cpu_times()
    t_wall_start = time.perf_counter()

    # Execute
    result = fn(*args, **kwargs)

    # Post-execution measurements
    t_wall_end = time.perf_counter()
    cpu_after = process.cpu_times()
    mem_after = process.memory_info().rss

    wall_time_ms = (t_wall_end - t_wall_start) * 1000
    cpu_user_ms = (cpu_after.user - cpu_before.user) * 1000
    cpu_sys_ms = (cpu_after.system - cpu_before.system) * 1000
    cpu_time_ms = cpu_user_ms + cpu_sys_ms
    memory_delta = mem_after - mem_before

    metrics = {
        "cpu_time_ms": round(cpu_time_ms, 3),
        "wall_time_ms": round(wall_time_ms, 3),
        "memory_delta_bytes": memory_delta,
        "peak_memory_bytes": mem_after,
        "cpu_user_ms": round(cpu_user_ms, 3),
        "cpu_sys_ms": round(cpu_sys_ms, 3),
    }

    return result, metrics
