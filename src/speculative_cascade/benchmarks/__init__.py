"""Benchmarking suite for cascade evaluation."""

from speculative_cascade.benchmarks.run_benchmarks import run_full_benchmark
from speculative_cascade.benchmarks.compare_methods import compare_baseline_methods

__all__ = ["run_full_benchmark", "compare_baseline_methods"]
