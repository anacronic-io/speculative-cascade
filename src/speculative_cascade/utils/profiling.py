"""TPU profiling utilities."""

import jax
import time
from typing import Dict, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class ProfileResult:
    """Result from profiling a code block."""

    name: str
    duration_ms: float
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    flops: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format profile result."""
        result = f"{self.name}:\n"
        result += f"  Duration: {self.duration_ms:.2f} ms\n"
        if self.memory_allocated_mb > 0:
            result += f"  Memory Allocated: {self.memory_allocated_mb:.1f} MB\n"
        if self.flops is not None:
            tflops = (self.flops / 1e12) / (self.duration_ms / 1000)
            result += f"  TFLOPS: {tflops:.2f}\n"
        return result


class TPUProfiler:
    """Profiler for TPU execution."""

    def __init__(self, enabled: bool = True):
        """Initialize profiler.

        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.results: Dict[str, ProfileResult] = {}

    @contextmanager
    def profile(
        self,
        name: str,
        flops: Optional[int] = None,
    ):
        """Profile a code block.

        Args:
            name: Name of the operation
            flops: Optional FLOP count for the operation

        Yields:
            None
        """
        if not self.enabled:
            yield
            return

        # Block until all previous operations complete
        jax.block_until_ready(jax.numpy.array(0))

        start_time = time.perf_counter()

        try:
            yield
        finally:
            # Block until this operation completes
            jax.block_until_ready(jax.numpy.array(0))

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Create result
            result = ProfileResult(
                name=name,
                duration_ms=duration_ms,
                flops=flops,
            )

            # Store result
            if name in self.results:
                # Average with previous results
                prev = self.results[name]
                result.duration_ms = (prev.duration_ms + duration_ms) / 2

            self.results[name] = result

    def get_result(self, name: str) -> Optional[ProfileResult]:
        """Get profiling result by name.

        Args:
            name: Operation name

        Returns:
            Profile result or None
        """
        return self.results.get(name)

    def print_summary(self):
        """Print summary of all profiling results."""
        if not self.results:
            print("No profiling results available")
            return

        print("\n=== Profiling Summary ===")
        total_time = sum(r.duration_ms for r in self.results.values())

        for name, result in sorted(
            self.results.items(),
            key=lambda x: x[1].duration_ms,
            reverse=True
        ):
            pct = 100 * result.duration_ms / total_time if total_time > 0 else 0
            print(f"\n{name}:")
            print(f"  Time: {result.duration_ms:.2f} ms ({pct:.1f}%)")

            if result.flops is not None:
                tflops = (result.flops / 1e12) / (result.duration_ms / 1000)
                print(f"  TFLOPS: {tflops:.2f}")

        print(f"\nTotal Time: {total_time:.2f} ms")

    def reset(self):
        """Reset all profiling results."""
        self.results.clear()


def profile_model_inference(
    model: Any,
    input_data: Any,
    num_warmup: int = 5,
    num_iterations: int = 100,
) -> Dict[str, float]:
    """Profile model inference performance.

    Args:
        model: Model to profile
        input_data: Input data
        num_warmup: Number of warmup iterations
        num_iterations: Number of profiling iterations

    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    for _ in range(num_warmup):
        _ = model(input_data)
        jax.block_until_ready(jax.numpy.array(0))

    # Profile
    timings = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        output = model(input_data)
        jax.block_until_ready(output)
        end = time.perf_counter()

        timings.append((end - start) * 1000)  # Convert to ms

    import numpy as np
    timings = np.array(timings)

    return {
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
    }


def estimate_mxu_utilization(
    compute_time_ms: float,
    memory_time_ms: float,
) -> float:
    """Estimate MXU (Matrix Multiply Unit) utilization.

    Args:
        compute_time_ms: Time spent computing
        memory_time_ms: Time spent on memory transfers

    Returns:
        MXU utilization percentage (0-100)
    """
    total_time = compute_time_ms + memory_time_ms

    if total_time == 0:
        return 0.0

    # MXU is utilized during compute time
    return 100 * compute_time_ms / total_time
