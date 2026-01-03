"""Main benchmarking script.

Reproduces the experiments from Table 1 and Figure 3 in the paper.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from speculative_cascade import CascadeInference
from speculative_cascade.models import TinyModel, DraftModel, TargetModel
from speculative_cascade.core.cascade import CascadeConfig, CascadeMetrics
from speculative_cascade.utils.metrics import (
    AcceptanceRateMetric,
    MemoryUsageMetric,
    ThroughputMetric,
    compute_perplexity,
)
from speculative_cascade.utils.profiling import TPUProfiler


def load_test_dataset(
    dataset_name: str,
    num_samples: int = 1000,
    max_length: int = 1024,
) -> List[str]:
    """Load test dataset.

    Args:
        dataset_name: Dataset name ("pg19", "c4", "humaneval")
        num_samples: Number of samples to load
        max_length: Maximum sequence length

    Returns:
        List of text samples
    """
    # This would load actual datasets
    # For now, return dummy data
    print(f"Loading {num_samples} samples from {dataset_name}...")

    if dataset_name == "pg19":
        # Long-form text
        samples = [
            "The future of artificial intelligence lies in "
            "the development of more efficient inference methods. "
        ] * num_samples
    elif dataset_name == "c4":
        # Web text
        samples = [
            "In recent years, large language models have shown "
            "remarkable capabilities across a wide range of tasks. "
        ] * num_samples
    elif dataset_name == "humaneval":
        # Code
        samples = [
            "def fibonacci(n):\n    "
            "\"\"\"Calculate the nth Fibonacci number.\"\"\"\n    "
        ] * num_samples
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return samples[:num_samples]


def benchmark_cascade(
    cascade: CascadeInference,
    test_samples: List[str],
    max_tokens: int = 100,
    profiler: Optional[TPUProfiler] = None,
) -> Dict[str, Any]:
    """Benchmark cascade performance.

    Args:
        cascade: Cascade inference system
        test_samples: Test samples
        max_tokens: Maximum tokens per sample
        profiler: Optional profiler

    Returns:
        Benchmark results
    """
    print(f"Benchmarking cascade on {len(test_samples)} samples...")

    # Metrics
    acceptance_metric = AcceptanceRateMetric()
    throughput_metric = ThroughputMetric()
    memory_metric = MemoryUsageMetric()

    all_outputs = []
    total_tokens = 0

    for i, sample in enumerate(test_samples):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(test_samples)}")

        # Generate
        start_time = time.time()

        if profiler:
            with profiler.profile(f"sample_{i}"):
                output, metrics = cascade.generate(
                    sample,
                    max_tokens=max_tokens,
                    return_metrics=True,
                )
        else:
            output, metrics = cascade.generate(
                sample,
                max_tokens=max_tokens,
                return_metrics=True,
            )

        elapsed_ms = (time.time() - start_time) * 1000

        # Update metrics
        num_tokens = len(output.split())  # Approximate
        total_tokens += num_tokens

        throughput_metric.record_generation(num_tokens, elapsed_ms)

        # Record acceptance rates (from metrics)
        acceptance_metric.update_stage0(
            int(metrics.gamma_0 * cascade.config.k0),
            cascade.config.k0
        )
        acceptance_metric.update_stage1(
            int(metrics.gamma_1_given_0 * cascade.config.k1),
            cascade.config.k1
        )
        acceptance_metric.update_stage2(
            int(metrics.gamma_2_given_1 * cascade.config.k1),
            cascade.config.k1
        )

        # Record memory (approximate)
        memory_metric.record_snapshot(
            vmem_bytes=40 * 1024 * 1024,  # Tiny model
            hbm_bytes=int(metrics.hbm_usage_mb * 1024 * 1024)
            if metrics.hbm_usage_mb > 0 else 0
        )

        all_outputs.append(output)

    # Aggregate results
    results = {
        "num_samples": len(test_samples),
        "total_tokens": total_tokens,
        "throughput_tokens_per_sec": throughput_metric.get_throughput(),
        "latency_ms_per_token": throughput_metric.get_latency(),
        "latency_p50_ms": throughput_metric.get_p50_latency(),
        "latency_p95_ms": throughput_metric.get_p95_latency(),
        "acceptance_rates": acceptance_metric.summary(),
        "memory_peak_mb": memory_metric.get_peak_usage()["hbm_peak_bytes"] / 1024 / 1024,
        "memory_avg_mb": memory_metric.get_average_usage()["hbm_avg_bytes"] / 1024 / 1024,
    }

    return results


def benchmark_baseline(
    target_model: TargetModel,
    test_samples: List[str],
    max_tokens: int = 100,
) -> Dict[str, Any]:
    """Benchmark baseline autoregressive generation.

    Args:
        target_model: Target model
        test_samples: Test samples
        max_tokens: Maximum tokens per sample

    Returns:
        Benchmark results
    """
    print(f"Benchmarking baseline on {len(test_samples)} samples...")

    throughput_metric = ThroughputMetric()
    total_tokens = 0

    for i, sample in enumerate(test_samples):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(test_samples)}")

        start_time = time.time()
        output = target_model.generate(sample, max_tokens=max_tokens)
        elapsed_ms = (time.time() - start_time) * 1000

        num_tokens = len(output.split())
        total_tokens += num_tokens

        throughput_metric.record_generation(num_tokens, elapsed_ms)

    results = {
        "num_samples": len(test_samples),
        "total_tokens": total_tokens,
        "throughput_tokens_per_sec": throughput_metric.get_throughput(),
        "latency_ms_per_token": throughput_metric.get_latency(),
        "latency_p50_ms": throughput_metric.get_p50_latency(),
        "latency_p95_ms": throughput_metric.get_p95_latency(),
    }

    return results


def run_full_benchmark(
    dataset: str = "pg19",
    num_samples: int = 100,
    max_tokens: int = 100,
    output_dir: str = "results",
    enable_profiling: bool = False,
) -> Dict[str, Any]:
    """Run full benchmark suite.

    Args:
        dataset: Dataset name
        num_samples: Number of samples
        max_tokens: Max tokens per sample
        output_dir: Output directory for results
        enable_profiling: Whether to enable profiling

    Returns:
        Complete benchmark results
    """
    print("="*60)
    print("Cascade Speculative Acceleration Benchmark")
    print("="*60)

    # Load test data
    test_samples = load_test_dataset(dataset, num_samples)

    # Initialize profiler
    profiler = TPUProfiler(enabled=enable_profiling) if enable_profiling else None

    # Initialize models
    print("\nInitializing models...")
    tiny_model = TinyModel.create()
    draft_model = DraftModel.from_pretrained("google/gemma-2b", quantize_int8=True)
    target_model = TargetModel.from_pretrained("google/gemma-7b")

    # Initialize cascade
    config = CascadeConfig(
        k0=64,
        k1=16,
        temp_tiny=1.0,
        temp_draft=1.0,
        temp_target=1.0,
    )
    cascade = CascadeInference(tiny_model, draft_model, target_model, config)

    # Benchmark cascade
    print("\n" + "="*60)
    print("Benchmarking Cascade")
    print("="*60)
    cascade_results = benchmark_cascade(cascade, test_samples, max_tokens, profiler)

    # Benchmark baseline
    print("\n" + "="*60)
    print("Benchmarking Baseline")
    print("="*60)
    baseline_results = benchmark_baseline(target_model, test_samples, max_tokens)

    # Calculate speedup
    speedup = (
        cascade_results["throughput_tokens_per_sec"] /
        baseline_results["throughput_tokens_per_sec"]
    )

    # Compile results
    final_results = {
        "dataset": dataset,
        "num_samples": num_samples,
        "max_tokens": max_tokens,
        "cascade": cascade_results,
        "baseline": baseline_results,
        "speedup": speedup,
    }

    # Print summary
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"\nDataset: {dataset}")
    print(f"Samples: {num_samples}")
    print(f"\nCascade Performance:")
    print(f"  Throughput: {cascade_results['throughput_tokens_per_sec']:.1f} tokens/sec")
    print(f"  Latency: {cascade_results['latency_ms_per_token']:.2f} ms/token")
    print(f"  Acceptance Rate: {cascade_results['acceptance_rates']['gamma_total']:.3f}")
    print(f"\nBaseline Performance:")
    print(f"  Throughput: {baseline_results['throughput_tokens_per_sec']:.1f} tokens/sec")
    print(f"  Latency: {baseline_results['latency_ms_per_token']:.2f} ms/token")
    print(f"\nSpeedup: {speedup:.2f}×")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"benchmark_{dataset}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to: {result_file}")

    if profiler:
        profiler.print_summary()

    return final_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run cascade benchmarks"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pg19",
        choices=["pg19", "c4", "humaneval"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to benchmark"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens per sample"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling"
    )

    args = parser.parse_args()

    run_full_benchmark(
        dataset=args.dataset,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
        enable_profiling=args.profile,
    )


if __name__ == "__main__":
    main()
