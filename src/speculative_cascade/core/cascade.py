"""Cascading Speculative Decoding Coordinator.

Implements the 3-stage cascade architecture with hierarchical filtering.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any, List
import time
from dataclasses import dataclass

from speculative_cascade.models.tiny import TinyModel
from speculative_cascade.models.draft import DraftModel
from speculative_cascade.models.target import TargetModel
from speculative_cascade.core.scheduler import MemoryAwareScheduler


@dataclass
class CascadeConfig:
    """Configuration for cascade inference."""

    # Speculation horizons
    k0: int = 64  # Tiny model candidates
    k1: int = 16  # Draft model proposals
    alpha: float = 4.0  # Filtering ratio (k0 = k1 * alpha)

    # Temperatures
    temp_tiny: float = 1.0
    temp_draft: float = 1.0
    temp_target: float = 1.0

    # Memory management
    vmem_size: int = 16 * 1024 * 1024  # 16MB for TPU v5e
    hbm_size: int = 16 * 1024 * 1024 * 1024  # 16GB

    # Distributed settings
    use_distributed: bool = True
    num_devices: Optional[int] = None

    # Monitoring
    profile_memory: bool = True
    log_acceptance_rates: bool = True


@dataclass
class CascadeMetrics:
    """Metrics for cascade performance."""

    # Acceptance rates
    gamma_0: float = 0.0  # Tiny acceptance rate
    gamma_1_given_0: float = 0.0  # Draft given Tiny
    gamma_2_given_1: float = 0.0  # Target given Draft
    gamma_total: float = 0.0  # Overall

    # Throughput and latency
    tokens_per_second: float = 0.0
    latency_ms_per_token: float = 0.0

    # Memory usage
    vmem_usage_mb: float = 0.0
    hbm_usage_mb: float = 0.0
    total_memory_mb: float = 0.0

    # Stage timings
    time_tiny_ms: float = 0.0
    time_draft_ms: float = 0.0
    time_target_ms: float = 0.0
    time_total_ms: float = 0.0

    def __str__(self) -> str:
        """Format metrics for display."""
        return f"""
Cascade Metrics:
  Acceptance Rates:
    γ₀ (Tiny):          {self.gamma_0:.3f}
    γ₁|₀ (Draft|Tiny):  {self.gamma_1_given_0:.3f}
    γ₂|₁ (Target|Draft):{self.gamma_2_given_1:.3f}
    γ_total:            {self.gamma_total:.3f}

  Performance:
    Throughput:         {self.tokens_per_second:.1f} tokens/sec
    Latency:            {self.latency_ms_per_token:.2f} ms/token

  Memory Usage:
    VMEM:               {self.vmem_usage_mb:.1f} MB
    HBM:                {self.hbm_usage_mb:.1f} MB
    Total:              {self.total_memory_mb:.1f} MB

  Timing Breakdown:
    Tiny:               {self.time_tiny_ms:.2f} ms
    Draft:              {self.time_draft_ms:.2f} ms
    Target:             {self.time_target_ms:.2f} ms
    Total:              {self.time_total_ms:.2f} ms
"""


class CascadeInference:
    """3-Stage Cascading Speculative Decoding System.

    Implements the algorithm from the paper:
        1. Tiny model generates K₀ candidates (VMEM-cached)
        2. Draft model filters to K₁ proposals (INT8, HBM)
        3. Target model verifies proposals (BF16, HBM, distributed)
    """

    def __init__(
        self,
        tiny_model: TinyModel,
        draft_model: DraftModel,
        target_model: TargetModel,
        config: Optional[CascadeConfig] = None,
    ):
        """Initialize cascade system.

        Args:
            tiny_model: Stage 0 tiny projection model
            draft_model: Stage 1 draft model (quantized)
            target_model: Stage 2 target model
            config: Cascade configuration
        """
        self.tiny_model = tiny_model
        self.draft_model = draft_model
        self.target_model = target_model
        self.config = config or CascadeConfig()

        # Initialize scheduler
        self.scheduler = MemoryAwareScheduler(
            vmem_capacity=self.config.vmem_size,
            hbm_capacity=self.config.hbm_size,
        )

        # Verify memory constraints
        self._verify_memory_constraints()

        # Metrics
        self.metrics = CascadeMetrics()

    def _verify_memory_constraints(self):
        """Verify that models fit in memory hierarchy."""
        # Check Tiny model fits in VMEM
        tiny_size = self.tiny_model.get_memory_footprint()
        if tiny_size > self.config.vmem_size:
            raise ValueError(
                f"Tiny model ({tiny_size / 1024 / 1024:.1f} MB) "
                f"exceeds VMEM capacity ({self.config.vmem_size / 1024 / 1024:.1f} MB)"
            )

        # Check all models fit in HBM
        total_size = (
            tiny_size +
            self.draft_model.get_memory_footprint() +
            self.target_model.get_memory_footprint()
        )
        if total_size > self.config.hbm_size:
            raise ValueError(
                f"Total model size ({total_size / 1024 / 1024 / 1024:.1f} GB) "
                f"exceeds HBM capacity ({self.config.hbm_size / 1024 / 1024 / 1024:.1f} GB)"
            )

        print(f"✓ Memory constraints verified")
        print(f"  Tiny: {tiny_size / 1024 / 1024:.1f} MB (VMEM)")
        print(f"  Draft: {self.draft_model.get_memory_footprint() / 1024 / 1024:.1f} MB (HBM)")
        print(f"  Target: {self.target_model.get_memory_footprint() / 1024 / 1024 / 1024:.1f} GB (HBM)")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        return_metrics: bool = False,
    ) -> str:
        """Generate text using cascaded speculative decoding.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            return_metrics: Whether to return detailed metrics

        Returns:
            Generated text (and optionally metrics)
        """
        # Tokenize prompt
        input_ids = self.target_model.tokenizer(prompt, return_tensors="jax").input_ids

        # Track metrics
        total_tokens_generated = 0
        total_time = 0.0
        acceptance_history = {"tiny": [], "draft": [], "target": []}

        generated_tokens = []

        while total_tokens_generated < max_tokens:
            start_time = time.time()

            # Run cascade step
            accepted_tokens, step_metrics = self._cascade_step(input_ids)

            step_time = time.time() - start_time
            total_time += step_time

            # Update context
            if accepted_tokens.shape[1] > 0:
                input_ids = jnp.concatenate([input_ids, accepted_tokens], axis=1)
                generated_tokens.extend(accepted_tokens[0].tolist())
                total_tokens_generated += accepted_tokens.shape[1]

                # Track acceptance rates
                acceptance_history["tiny"].append(step_metrics.get("gamma_0", 0))
                acceptance_history["draft"].append(step_metrics.get("gamma_1", 0))
                acceptance_history["target"].append(step_metrics.get("gamma_2", 0))
            else:
                # Fallback: generate one token with target model
                outputs = self.target_model(input_ids)
                logits = outputs.logits[:, -1, :] / self.config.temp_target
                next_token = jax.random.categorical(
                    jax.random.PRNGKey(int(time.time() * 1000)),
                    logits,
                    axis=-1
                )
                input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)
                generated_tokens.append(int(next_token[0]))
                total_tokens_generated += 1

            # Check for EOS
            if generated_tokens and generated_tokens[-1] == self.target_model.tokenizer.eos_token_id:
                break

        # Decode
        output_text = self.target_model.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Update final metrics
        if total_tokens_generated > 0:
            self.metrics.tokens_per_second = total_tokens_generated / total_time
            self.metrics.latency_ms_per_token = (total_time / total_tokens_generated) * 1000

            if acceptance_history["tiny"]:
                self.metrics.gamma_0 = sum(acceptance_history["tiny"]) / len(acceptance_history["tiny"])
            if acceptance_history["draft"]:
                self.metrics.gamma_1_given_0 = sum(acceptance_history["draft"]) / len(acceptance_history["draft"])
            if acceptance_history["target"]:
                self.metrics.gamma_2_given_1 = sum(acceptance_history["target"]) / len(acceptance_history["target"])

            self.metrics.gamma_total = (
                self.metrics.gamma_0 * self.metrics.gamma_1_given_0 * self.metrics.gamma_2_given_1
            )

        if return_metrics:
            return output_text, self.metrics
        return output_text

    def _cascade_step(
        self,
        input_ids: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Single cascade step with all 3 stages.

        Args:
            input_ids: Current context [batch_size, seq_len]

        Returns:
            Tuple of (accepted_tokens, step_metrics)
        """
        metrics = {}

        # Stage 0: Tiny model candidate generation
        t0 = time.time()
        embeddings = self.draft_model.get_embeddings(input_ids)
        tiny_candidates, tiny_probs = self.tiny_model.generate_candidates(
            embeddings,
            k=self.config.k0,
            temperature=self.config.temp_tiny,
        )
        metrics["time_tiny"] = (time.time() - t0) * 1000

        # Stage 1: Draft model semantic filtering
        t1 = time.time()
        draft_proposals, draft_probs = self.draft_model.generate_proposals(
            input_ids,
            tiny_candidates,
            num_proposals=self.config.k1,
            temperature=self.config.temp_draft,
        )
        metrics["time_draft"] = (time.time() - t1) * 1000

        # Stage 2: Target model verification
        t2 = time.time()
        accepted_tokens, num_accepted = self.target_model.verify_proposals(
            input_ids,
            draft_proposals,
            draft_probs,
            use_distributed=self.config.use_distributed,
        )
        metrics["time_target"] = (time.time() - t2) * 1000

        # Calculate stage acceptance rates
        metrics["gamma_0"] = min(self.config.k1 / self.config.k0, 1.0)  # Approximation
        metrics["gamma_1"] = num_accepted / self.config.k1 if self.config.k1 > 0 else 0
        metrics["gamma_2"] = num_accepted / self.config.k1 if self.config.k1 > 0 else 0

        return accepted_tokens, metrics

    def benchmark(
        self,
        test_prompts: List[str],
        max_tokens_per_prompt: int = 100,
    ) -> CascadeMetrics:
        """Run benchmark on test prompts.

        Args:
            test_prompts: List of test prompts
            max_tokens_per_prompt: Maximum tokens per prompt

        Returns:
            Aggregated metrics
        """
        all_metrics = []

        for prompt in test_prompts:
            _, metrics = self.generate(
                prompt,
                max_tokens=max_tokens_per_prompt,
                return_metrics=True,
            )
            all_metrics.append(metrics)

        # Aggregate metrics
        avg_metrics = CascadeMetrics()
        avg_metrics.gamma_0 = sum(m.gamma_0 for m in all_metrics) / len(all_metrics)
        avg_metrics.gamma_1_given_0 = sum(m.gamma_1_given_0 for m in all_metrics) / len(all_metrics)
        avg_metrics.gamma_2_given_1 = sum(m.gamma_2_given_1 for m in all_metrics) / len(all_metrics)
        avg_metrics.gamma_total = sum(m.gamma_total for m in all_metrics) / len(all_metrics)
        avg_metrics.tokens_per_second = sum(m.tokens_per_second for m in all_metrics) / len(all_metrics)
        avg_metrics.latency_ms_per_token = sum(m.latency_ms_per_token for m in all_metrics) / len(all_metrics)

        self.metrics = avg_metrics
        return avg_metrics

    def get_speedup(self, baseline_throughput: float) -> float:
        """Calculate speedup vs baseline.

        Args:
            baseline_throughput: Baseline tokens/sec

        Returns:
            Speedup factor
        """
        return self.metrics.tokens_per_second / baseline_throughput if baseline_throughput > 0 else 0.0
