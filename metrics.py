import time
import psutil
import os
import torch
from typing import Callable, Any

class QRHMetrics:
    """A utility class for profiling function performance (time and memory)."""
    def __init__(self):
        try:
            self.process = psutil.Process(os.getpid())
        except psutil.NoSuchProcess:
            print("Warning: Could not get current process for memory profiling.")
            self.process = None

    def profile_forward(self, fn: Callable, *args, **kwargs) -> tuple[Any, dict]:
        """
        Profiles a function call for execution time and memory usage delta.

        Args:
            fn: The function to profile.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            A tuple containing the result of the function call and a dictionary
            with performance metrics ('time_ms', 'memory_mb_delta', 'device').
        """
        start_time = time.perf_counter()
        start_mem = self.process.memory_info().rss / 1024**2 if self.process else 0

        result = fn(*args, **kwargs)

        end_time = time.perf_counter()
        end_mem = self.process.memory_info().rss / 1024**2 if self.process else 0

        # Determine device from result if it's a tensor
        device_type = 'unknown'
        if isinstance(result, torch.Tensor):
            device_type = result.device.type

        metrics = {
            'time_ms': (end_time - start_time) * 1000,
            'memory_mb_delta': end_mem - start_mem,
            'device': device_type
        }

        return result, metrics
