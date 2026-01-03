"""Metrics for evaluating cascade performance."""

import jax.numpy as jnp
from typing import List, Dict, Any
import time
from dataclasses import dataclass, field


@dataclass
class AcceptanceRateMetric:
    """Track acceptance rates across cascade stages."""

    stage0_accepts: List[int] = field(default_factory=list)
    stage0_proposals: List[int] = field(default_factory=list)

    stage1_accepts: List[int] = field(default_factory=list)
    stage1_proposals: List[int] = field(default_factory=list)

    stage2_accepts: List[int] = field(default_factory=list)
    stage2_proposals: List[int] = field(default_factory=list)

    def update_stage0(self, num_accepted: int, num_proposed: int):
        """Update Stage 0 (Tiny) metrics."""
        self.stage0_accepts.append(num_accepted)
        self.stage0_proposals.append(num_proposed)

    def update_stage1(self, num_accepted: int, num_proposed: int):
        """Update Stage 1 (Draft) metrics."""
        self.stage1_accepts.append(num_accepted)
        self.stage1_proposals.append(num_proposed)

    def update_stage2(self, num_accepted: int, num_proposed: int):
        """Update Stage 2 (Target) metrics."""
        self.stage2_accepts.append(num_accepted)
        self.stage2_proposals.append(num_proposed)

    def get_stage_rate(self, stage: int) -> float:
        """Get acceptance rate for a stage.

        Args:
            stage: Stage number (0, 1, or 2)

        Returns:
            Acceptance rate (0-1)
        """
        if stage == 0:
            accepts, proposals = self.stage0_accepts, self.stage0_proposals
        elif stage == 1:
            accepts, proposals = self.stage1_accepts, self.stage1_proposals
        elif stage == 2:
            accepts, proposals = self.stage2_accepts, self.stage2_proposals
        else:
            raise ValueError(f"Invalid stage: {stage}")

        if not proposals or sum(proposals) == 0:
            return 0.0

        return sum(accepts) / sum(proposals)

    def get_total_rate(self) -> float:
        """Get overall cascade acceptance rate.

        Returns:
            γ_total = γ₀ · γ₁|₀ · γ₂|₁
        """
        gamma_0 = self.get_stage_rate(0)
        gamma_1 = self.get_stage_rate(1)
        gamma_2 = self.get_stage_rate(2)

        return gamma_0 * gamma_1 * gamma_2

    def summary(self) -> Dict[str, float]:
        """Get summary statistics.

        Returns:
            Dictionary with acceptance rates
        """
        return {
            "gamma_0": self.get_stage_rate(0),
            "gamma_1_given_0": self.get_stage_rate(1),
            "gamma_2_given_1": self.get_stage_rate(2),
            "gamma_total": self.get_total_rate(),
        }


@dataclass
class MemoryUsageMetric:
    """Track memory usage across hierarchy."""

    vmem_snapshots: List[int] = field(default_factory=list)
    hbm_snapshots: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    def record_snapshot(self, vmem_bytes: int, hbm_bytes: int):
        """Record memory usage snapshot.

        Args:
            vmem_bytes: VMEM usage in bytes
            hbm_bytes: HBM usage in bytes
        """
        self.vmem_snapshots.append(vmem_bytes)
        self.hbm_snapshots.append(hbm_bytes)
        self.timestamps.append(time.time())

    def get_peak_usage(self) -> Dict[str, int]:
        """Get peak memory usage.

        Returns:
            Dictionary with peak usage in bytes
        """
        return {
            "vmem_peak_bytes": max(self.vmem_snapshots) if self.vmem_snapshots else 0,
            "hbm_peak_bytes": max(self.hbm_snapshots) if self.hbm_snapshots else 0,
        }

    def get_average_usage(self) -> Dict[str, float]:
        """Get average memory usage.

        Returns:
            Dictionary with average usage in bytes
        """
        return {
            "vmem_avg_bytes": sum(self.vmem_snapshots) / len(self.vmem_snapshots)
            if self.vmem_snapshots else 0,
            "hbm_avg_bytes": sum(self.hbm_snapshots) / len(self.hbm_snapshots)
            if self.hbm_snapshots else 0,
        }


@dataclass
class ThroughputMetric:
    """Track throughput and latency."""

    token_counts: List[int] = field(default_factory=list)
    time_intervals_ms: List[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record_generation(self, num_tokens: int, time_ms: float):
        """Record a generation event.

        Args:
            num_tokens: Number of tokens generated
            time_ms: Time taken in milliseconds
        """
        self.token_counts.append(num_tokens)
        self.time_intervals_ms.append(time_ms)

    def get_throughput(self) -> float:
        """Get throughput in tokens/second.

        Returns:
            Tokens per second
        """
        if not self.token_counts or sum(self.time_intervals_ms) == 0:
            return 0.0

        total_tokens = sum(self.token_counts)
        total_time_sec = sum(self.time_intervals_ms) / 1000

        return total_tokens / total_time_sec

    def get_latency(self) -> float:
        """Get latency in ms/token.

        Returns:
            Milliseconds per token
        """
        if not self.token_counts or sum(self.token_counts) == 0:
            return 0.0

        total_tokens = sum(self.token_counts)
        total_time_ms = sum(self.time_intervals_ms)

        return total_time_ms / total_tokens

    def get_p50_latency(self) -> float:
        """Get median latency in ms/token.

        Returns:
            P50 latency
        """
        if not self.token_counts:
            return 0.0

        latencies = [
            time_ms / tokens if tokens > 0 else 0
            for tokens, time_ms in zip(self.token_counts, self.time_intervals_ms)
        ]

        return float(jnp.percentile(jnp.array(latencies), 50))

    def get_p95_latency(self) -> float:
        """Get P95 latency in ms/token.

        Returns:
            P95 latency
        """
        if not self.token_counts:
            return 0.0

        latencies = [
            time_ms / tokens if tokens > 0 else 0
            for tokens, time_ms in zip(self.token_counts, self.time_intervals_ms)
        ]

        return float(jnp.percentile(jnp.array(latencies), 95))


def compute_speedup(
    cascade_throughput: float,
    baseline_throughput: float,
) -> float:
    """Compute speedup vs baseline.

    Args:
        cascade_throughput: Cascade tokens/sec
        baseline_throughput: Baseline tokens/sec

    Returns:
        Speedup factor
    """
    if baseline_throughput == 0:
        return 0.0

    return cascade_throughput / baseline_throughput


def compute_perplexity(
    logits: jnp.ndarray,
    target_ids: jnp.ndarray,
) -> float:
    """Compute perplexity from logits and targets.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        target_ids: Target token IDs [batch_size, seq_len]

    Returns:
        Perplexity
    """
    # Cross-entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Gather log probs for targets
    target_log_probs = jnp.take_along_axis(
        log_probs,
        target_ids[..., None],
        axis=-1
    ).squeeze(-1)

    # Average negative log likelihood
    nll = -jnp.mean(target_log_probs)

    # Perplexity
    return float(jnp.exp(nll))
