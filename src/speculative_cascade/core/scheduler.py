"""Memory-Aware Scheduler for Cascade Models.

Implements HBM-aware scheduling and dynamic batch adjustment.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class MemoryLevel(Enum):
    """Memory hierarchy levels."""
    VMEM = "vmem"  # Vector Memory (fast, small)
    HBM = "hbm"    # High Bandwidth Memory (slower, large)
    DRAM = "dram"  # System DRAM (slowest, largest)


@dataclass
class MemoryHierarchy:
    """Memory hierarchy specification for TPU v5e."""

    # Capacities (in bytes)
    vmem_capacity: int = 16 * 1024 * 1024  # 16MB per core
    hbm_capacity: int = 16 * 1024 * 1024 * 1024  # 16GB per chip

    # Bandwidths (in GB/s)
    vmem_bandwidth: float = 2000.0  # Estimated for VMEM
    hbm_bandwidth: float = 819.0    # TPU v5e HBM bandwidth

    # Latencies (in nanoseconds)
    vmem_latency: float = 10.0     # Very fast
    hbm_latency: float = 100.0     # Slower

    def get_access_time(self, size_bytes: int, level: MemoryLevel) -> float:
        """Calculate memory access time.

        Args:
            size_bytes: Size of data to transfer
            level: Memory level

        Returns:
            Access time in milliseconds
        """
        if level == MemoryLevel.VMEM:
            latency_ms = self.vmem_latency / 1e6
            transfer_time_ms = (size_bytes / (self.vmem_bandwidth * 1e9)) * 1000
        elif level == MemoryLevel.HBM:
            latency_ms = self.hbm_latency / 1e6
            transfer_time_ms = (size_bytes / (self.hbm_bandwidth * 1e9)) * 1000
        else:
            # DRAM (much slower, used as fallback)
            latency_ms = 0.001  # 1us
            transfer_time_ms = (size_bytes / (50 * 1e9)) * 1000  # ~50 GB/s

        return latency_ms + transfer_time_ms


@dataclass
class ModelPlacement:
    """Model placement in memory hierarchy."""
    model_name: str
    size_bytes: int
    level: MemoryLevel
    pinned: bool = False  # Whether to keep in memory


class MemoryAwareScheduler:
    """Scheduler for optimal model placement and batch sizing.

    Implements the memory-aware scheduling strategy from the paper.
    """

    def __init__(
        self,
        vmem_capacity: int = 16 * 1024 * 1024,
        hbm_capacity: int = 16 * 1024 * 1024 * 1024,
        num_cores: int = 1,
    ):
        """Initialize scheduler.

        Args:
            vmem_capacity: VMEM capacity in bytes
            hbm_capacity: HBM capacity in bytes
            num_cores: Number of TPU cores
        """
        self.hierarchy = MemoryHierarchy(
            vmem_capacity=vmem_capacity * num_cores,
            hbm_capacity=hbm_capacity,
        )
        self.num_cores = num_cores

        # Track current placements
        self.placements: Dict[str, ModelPlacement] = {}
        self.vmem_used = 0
        self.hbm_used = 0

    def place_model(
        self,
        model_name: str,
        model_size: int,
        prefer_vmem: bool = False,
    ) -> MemoryLevel:
        """Determine optimal placement for a model.

        Args:
            model_name: Model identifier
            model_size: Model size in bytes
            prefer_vmem: Whether to prefer VMEM placement

        Returns:
            Assigned memory level
        """
        # Try VMEM first if preferred and fits
        if prefer_vmem and model_size <= self.hierarchy.vmem_capacity - self.vmem_used:
            level = MemoryLevel.VMEM
            self.vmem_used += model_size
        # Otherwise try HBM
        elif model_size <= self.hierarchy.hbm_capacity - self.hbm_used:
            level = MemoryLevel.HBM
            self.hbm_used += model_size
        else:
            # Fallback to DRAM (will be slow)
            level = MemoryLevel.DRAM
            print(f"Warning: {model_name} placed in DRAM (may be slow)")

        # Record placement
        self.placements[model_name] = ModelPlacement(
            model_name=model_name,
            size_bytes=model_size,
            level=level,
        )

        return level

    def adjust_batch_size_for_hbm_pressure(
        self,
        base_batch_size: int,
        model_sizes: List[int],
        kv_cache_per_token: int,
        max_seq_len: int,
    ) -> int:
        """Adjust batch size based on HBM memory pressure.

        Args:
            base_batch_size: Desired batch size
            model_sizes: Sizes of all models in bytes
            kv_cache_per_token: KV cache size per token in bytes
            max_seq_len: Maximum sequence length

        Returns:
            Adjusted batch size
        """
        # Calculate static memory usage (models)
        static_memory = sum(model_sizes)

        # Calculate dynamic memory usage (KV cache)
        dynamic_memory_per_sample = kv_cache_per_token * max_seq_len

        # Total memory needed
        total_needed = static_memory + (base_batch_size * dynamic_memory_per_sample)

        # Check if it fits
        if total_needed <= self.hierarchy.hbm_capacity:
            return base_batch_size

        # Reduce batch size to fit
        available_for_batch = self.hierarchy.hbm_capacity - static_memory
        adjusted_batch_size = max(1, int(available_for_batch / dynamic_memory_per_sample))

        print(f"Adjusted batch size: {base_batch_size} -> {adjusted_batch_size} (HBM pressure)")
        return adjusted_batch_size

    def schedule(
        self,
        models: Dict[str, int],  # model_name -> size_bytes
        batch_size: int,
        kv_cache_per_token: int = 1024,
        max_seq_len: int = 2048,
    ) -> Tuple[Dict[str, MemoryLevel], int]:
        """Schedule models and determine batch size.

        Args:
            models: Dictionary of model names to sizes
            batch_size: Desired batch size
            kv_cache_per_token: KV cache size per token
            max_seq_len: Maximum sequence length

        Returns:
            Tuple of (placements, adjusted_batch_size)
        """
        placements = {}

        # Place Tiny model in VMEM
        if "tiny" in models:
            placements["tiny"] = self.place_model("tiny", models["tiny"], prefer_vmem=True)

        # Place Draft and Target in HBM
        for name in ["draft", "target"]:
            if name in models:
                placements[name] = self.place_model(name, models[name], prefer_vmem=False)

        # Adjust batch size
        model_sizes = list(models.values())
        adjusted_batch_size = self.adjust_batch_size_for_hbm_pressure(
            batch_size,
            model_sizes,
            kv_cache_per_token,
            max_seq_len,
        )

        return placements, adjusted_batch_size

    def get_transfer_cost(
        self,
        model_name: str,
        target_level: Optional[MemoryLevel] = None,
    ) -> float:
        """Calculate cost of transferring model data.

        Args:
            model_name: Name of model
            target_level: Target memory level (use current if None)

        Returns:
            Transfer cost in milliseconds
        """
        if model_name not in self.placements:
            return 0.0

        placement = self.placements[model_name]
        level = target_level or placement.level

        return self.hierarchy.get_access_time(placement.size_bytes, level)

    def print_memory_layout(self):
        """Print current memory layout."""
        print("\n=== Memory Layout ===")
        print(f"VMEM: {self.vmem_used / 1024 / 1024:.1f} MB / "
              f"{self.hierarchy.vmem_capacity / 1024 / 1024:.1f} MB "
              f"({100 * self.vmem_used / self.hierarchy.vmem_capacity:.1f}%)")
        print(f"HBM:  {self.hbm_used / 1024 / 1024 / 1024:.1f} GB / "
              f"{self.hierarchy.hbm_capacity / 1024 / 1024 / 1024:.1f} GB "
              f"({100 * self.hbm_used / self.hierarchy.hbm_capacity:.1f}%)")

        print("\nModel Placements:")
        for name, placement in self.placements.items():
            size_mb = placement.size_bytes / 1024 / 1024
            if size_mb < 1024:
                size_str = f"{size_mb:.1f} MB"
            else:
                size_str = f"{size_mb / 1024:.2f} GB"
            print(f"  {name:10s}: {size_str:10s} in {placement.level.value.upper()}")


def estimate_hbm_stall_percentage(
    scheduler: MemoryAwareScheduler,
    compute_time_ms: float,
    model_accesses: List[Tuple[str, int]],  # (model_name, num_accesses)
) -> float:
    """Estimate percentage of time spent in HBM stalls.

    Args:
        scheduler: Memory scheduler
        compute_time_ms: Pure compute time in milliseconds
        model_accesses: List of (model_name, access_count) tuples

    Returns:
        Percentage of time in HBM stalls (0-100)
    """
    total_memory_time = 0.0

    for model_name, num_accesses in model_accesses:
        access_time = scheduler.get_transfer_cost(model_name)
        total_memory_time += access_time * num_accesses

    total_time = compute_time_ms + total_memory_time

    if total_time == 0:
        return 0.0

    return 100 * (total_memory_time / total_time)
