"""Analytical Cost Model for Cascade Optimization.

Implements the theoretical framework from Section 5 of the paper,
including the Vertical Synergy Principle and optimal configuration prediction.
"""

import jax.numpy as jnp
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass
class HardwareSpec:
    """Hardware specification for cost modeling."""

    # Memory hierarchy
    vmem_size: int = 16 * 1024 * 1024  # 16MB
    hbm_size: int = 16 * 1024 * 1024 * 1024  # 16GB
    vmem_bandwidth: float = 2000.0  # GB/s
    hbm_bandwidth: float = 819.0    # GB/s

    # Compute
    mxu_tflops: float = 197.0  # TPU v5e TFLOPS (bf16)
    num_cores: int = 1
    cores_per_chip: int = 256


@dataclass
class ModelSpec:
    """Model specification for cost modeling."""

    num_params: int
    num_layers: int
    hidden_dim: int
    dtype_bytes: int = 2  # BF16


class CascadeCostModel:
    """Analytical cost model for predicting cascade performance.

    Implements Equation (4-7) from the paper:
        C_total = Σᵢ (Wᵢ/BWᵢ + Tᵢᶜᵒᵐᵖᵘᵗᵉ) · E[Nᵢ]
    """

    def __init__(self, hardware: HardwareSpec):
        """Initialize cost model.

        Args:
            hardware: Hardware specification
        """
        self.hardware = hardware

    def compute_memory_cost(
        self,
        model_size: int,
        bandwidth: float,
    ) -> float:
        """Compute memory transfer cost.

        Args:
            model_size: Model size in bytes
            bandwidth: Memory bandwidth in GB/s

        Returns:
            Transfer time in milliseconds
        """
        # Time = Size / Bandwidth
        time_seconds = model_size / (bandwidth * 1e9)
        return time_seconds * 1000  # Convert to ms

    def compute_compute_cost(
        self,
        num_params: int,
        seq_len: int,
        batch_size: int = 1,
    ) -> float:
        """Compute computational cost.

        For transformer: FLOPs ≈ 2 · num_params · seq_len

        Args:
            num_params: Number of model parameters
            seq_len: Sequence length
            batch_size: Batch size

        Returns:
            Compute time in milliseconds
        """
        # Total FLOPs
        flops = 2 * num_params * seq_len * batch_size

        # Time = FLOPs / TFLOPS
        time_seconds = flops / (self.hardware.mxu_tflops * 1e12)
        return time_seconds * 1000  # Convert to ms

    def compute_stage_cost(
        self,
        model: ModelSpec,
        memory_level: str,  # "vmem" or "hbm"
        seq_len: int,
        num_tokens_processed: int,
    ) -> float:
        """Compute cost for a single cascade stage.

        Args:
            model: Model specification
            memory_level: Memory level ("vmem" or "hbm")
            seq_len: Sequence length
            num_tokens_processed: Expected number of tokens processed

        Returns:
            Stage cost in milliseconds
        """
        # Memory cost
        model_size = model.num_params * model.dtype_bytes
        if memory_level == "vmem":
            bandwidth = self.hardware.vmem_bandwidth
        else:
            bandwidth = self.hardware.hbm_bandwidth

        memory_cost = self.compute_memory_cost(model_size, bandwidth)

        # Compute cost
        compute_cost = self.compute_compute_cost(model.num_params, seq_len)

        # Total cost for processing N tokens
        total_cost = (memory_cost + compute_cost) * num_tokens_processed

        return total_cost

    def predict_cascade_cost(
        self,
        tiny_spec: ModelSpec,
        draft_spec: ModelSpec,
        target_spec: ModelSpec,
        gamma_0: float,
        gamma_1_given_0: float,
        gamma_2_given_1: float,
        seq_len: int,
        speculation_horizon: int,
    ) -> Dict[str, float]:
        """Predict total cascade cost using Equation (4).

        Args:
            tiny_spec: Tiny model specification
            draft_spec: Draft model specification
            target_spec: Target model specification
            gamma_0: Tiny acceptance rate
            gamma_1_given_0: Draft conditional acceptance rate
            gamma_2_given_1: Target conditional acceptance rate
            seq_len: Sequence length
            speculation_horizon: Number of speculated tokens

        Returns:
            Dictionary with cost breakdown
        """
        # Expected number of tokens processed at each stage
        E_N0 = speculation_horizon  # Tiny processes all candidates
        E_N1 = E_N0 * gamma_0  # Draft processes accepted from Tiny
        E_N2 = E_N1 * gamma_1_given_0  # Target processes accepted from Draft

        # Stage costs
        cost_tiny = self.compute_stage_cost(tiny_spec, "vmem", seq_len, E_N0)
        cost_draft = self.compute_stage_cost(draft_spec, "hbm", seq_len, E_N1)
        cost_target = self.compute_stage_cost(target_spec, "hbm", seq_len, E_N2)

        # Total cost
        total_cost = cost_tiny + cost_draft + cost_target

        # Expected tokens accepted
        expected_accepted = speculation_horizon * gamma_0 * gamma_1_given_0 * gamma_2_given_1

        return {
            "cost_tiny_ms": cost_tiny,
            "cost_draft_ms": cost_draft,
            "cost_target_ms": cost_target,
            "total_cost_ms": total_cost,
            "expected_accepted_tokens": expected_accepted,
            "cost_per_token_ms": total_cost / expected_accepted if expected_accepted > 0 else float('inf'),
        }

    def predict_baseline_cost(
        self,
        target_spec: ModelSpec,
        seq_len: int,
        num_tokens: int,
    ) -> float:
        """Predict baseline autoregressive cost.

        Args:
            target_spec: Target model specification
            seq_len: Sequence length
            num_tokens: Number of tokens to generate

        Returns:
            Total cost in milliseconds
        """
        # Each token requires full model forward pass
        cost_per_token = self.compute_stage_cost(target_spec, "hbm", seq_len, 1)
        return cost_per_token * num_tokens

    def predict_speedup(
        self,
        tiny_spec: ModelSpec,
        draft_spec: ModelSpec,
        target_spec: ModelSpec,
        gamma_0: float,
        gamma_1_given_0: float,
        gamma_2_given_1: float,
        seq_len: int,
        speculation_horizon: int,
    ) -> float:
        """Predict speedup vs baseline.

        Implements Theorem 1 from the paper.

        Args:
            tiny_spec: Tiny model specification
            draft_spec: Draft model specification
            target_spec: Target model specification
            gamma_0: Tiny acceptance rate
            gamma_1_given_0: Draft conditional acceptance rate
            gamma_2_given_1: Target conditional acceptance rate
            seq_len: Sequence length
            speculation_horizon: Number of speculated tokens

        Returns:
            Predicted speedup factor
        """
        # Cascade cost
        cascade_result = self.predict_cascade_cost(
            tiny_spec, draft_spec, target_spec,
            gamma_0, gamma_1_given_0, gamma_2_given_1,
            seq_len, speculation_horizon,
        )
        cascade_cost_per_token = cascade_result["cost_per_token_ms"]

        # Baseline cost
        baseline_cost_per_token = self.compute_stage_cost(target_spec, "hbm", seq_len, 1)

        # Speedup
        speedup = baseline_cost_per_token / cascade_cost_per_token

        return speedup

    def verify_vertical_synergy(
        self,
        cascade_cost: float,
        isolated_costs: List[float],
    ) -> bool:
        """Verify Vertical Synergy Principle (Definition 1).

        Cost_cascade < Σ Cost_isolated(Mi)

        Args:
            cascade_cost: Total cascade cost
            isolated_costs: List of isolated model costs

        Returns:
            True if vertical synergy holds
        """
        total_isolated = sum(isolated_costs)
        return cascade_cost < total_isolated

    def optimize_speculation_horizon(
        self,
        tiny_spec: ModelSpec,
        draft_spec: ModelSpec,
        target_spec: ModelSpec,
        gamma_0: float,
        gamma_1_given_0: float,
        gamma_2_given_1: float,
        seq_len: int,
        max_horizon: int = 128,
    ) -> Tuple[int, float]:
        """Find optimal speculation horizon K.

        Implements Theorem 1: optimal K* that maximizes speedup.

        Args:
            tiny_spec: Tiny model specification
            draft_spec: Draft model specification
            target_spec: Target model specification
            gamma_0: Tiny acceptance rate
            gamma_1_given_0: Draft conditional acceptance rate
            gamma_2_given_1: Target conditional acceptance rate
            seq_len: Sequence length
            max_horizon: Maximum horizon to consider

        Returns:
            Tuple of (optimal_horizon, max_speedup)
        """
        best_speedup = 0.0
        best_horizon = 1

        for k in range(1, max_horizon + 1):
            speedup = self.predict_speedup(
                tiny_spec, draft_spec, target_spec,
                gamma_0, gamma_1_given_0, gamma_2_given_1,
                seq_len, k,
            )

            if speedup > best_speedup:
                best_speedup = speedup
                best_horizon = k

        return best_horizon, best_speedup

    def compute_memory_hierarchy_utilization(
        self,
        cascade_cost_breakdown: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute memory hierarchy utilization breakdown.

        Returns percentages of time spent in different activities.

        Args:
            cascade_cost_breakdown: Cost breakdown from predict_cascade_cost

        Returns:
            Dictionary with utilization percentages
        """
        total = cascade_cost_breakdown["total_cost_ms"]

        if total == 0:
            return {
                "vmem_access_pct": 0.0,
                "hbm_access_pct": 0.0,
                "compute_pct": 0.0,
            }

        # Approximate breakdown (simplified)
        # Tiny is mostly VMEM
        # Draft and Target are mostly HBM
        vmem_time = cascade_cost_breakdown["cost_tiny_ms"] * 0.7
        hbm_time = (
            cascade_cost_breakdown["cost_draft_ms"] * 0.6 +
            cascade_cost_breakdown["cost_target_ms"] * 0.6
        )
        compute_time = total - vmem_time - hbm_time

        return {
            "vmem_access_pct": 100 * vmem_time / total,
            "hbm_access_pct": 100 * hbm_time / total,
            "compute_pct": 100 * compute_time / total,
        }


def create_default_model_specs() -> Tuple[ModelSpec, ModelSpec, ModelSpec]:
    """Create default model specifications for Gemma cascade.

    Returns:
        Tuple of (tiny_spec, draft_spec, target_spec)
    """
    tiny_spec = ModelSpec(
        num_params=10_000_000,  # 10M
        num_layers=2,
        hidden_dim=256,
        dtype_bytes=2,  # BF16
    )

    draft_spec = ModelSpec(
        num_params=2_000_000_000,  # 2B
        num_layers=18,
        hidden_dim=2048,
        dtype_bytes=1,  # INT8
    )

    target_spec = ModelSpec(
        num_params=7_000_000_000,  # 7B
        num_layers=28,
        hidden_dim=3072,
        dtype_bytes=2,  # BF16
    )

    return tiny_spec, draft_spec, target_spec


if __name__ == "__main__":
    # Example usage
    hardware = HardwareSpec()
    model = CascadeCostModel(hardware)

    tiny, draft, target = create_default_model_specs()

    # Predict speedup with empirical acceptance rates from paper
    speedup = model.predict_speedup(
        tiny, draft, target,
        gamma_0=0.95,
        gamma_1_given_0=0.85,
        gamma_2_given_1=0.90,
        seq_len=1024,
        speculation_horizon=16,
    )

    print(f"Predicted speedup: {speedup:.2f}×")

    # Optimize horizon
    opt_horizon, max_speedup = model.optimize_speculation_horizon(
        tiny, draft, target,
        gamma_0=0.95,
        gamma_1_given_0=0.85,
        gamma_2_given_1=0.90,
        seq_len=1024,
    )

    print(f"Optimal horizon: {opt_horizon} (speedup: {max_speedup:.2f}×)")
