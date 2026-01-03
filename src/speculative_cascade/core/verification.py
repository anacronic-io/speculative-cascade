"""Distributed Verification System.

Implements Algorithm 1: Distributed Cascade Verification from the paper.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional
from functools import partial


class DistributedVerifier:
    """Distributed verification using JAX pmap for parallel token verification."""

    def __init__(
        self,
        target_model: Any,
        num_devices: Optional[int] = None,
    ):
        """Initialize distributed verifier.

        Args:
            target_model: Target model for verification
            num_devices: Number of devices (auto-detected if None)
        """
        self.target_model = target_model
        self.num_devices = num_devices or jax.device_count()
        self.devices = jax.devices()[:self.num_devices]

        print(f"Initialized DistributedVerifier with {self.num_devices} devices")

    @partial(jax.pmap, in_axes=(0, None), out_axes=0, static_broadcasted_argnums=(1,))
    def _verify_chunk_pmap(
        self,
        chunk_data: Dict[str, jnp.ndarray],
        model_apply: Any,
    ) -> jnp.ndarray:
        """Verify a chunk of proposals on a single device (pmapped).

        Args:
            chunk_data: Dictionary with 'input_ids' and 'params'
            model_apply: Model apply function

        Returns:
            Verification probabilities
        """
        outputs = model_apply(
            chunk_data['params'],
            chunk_data['input_ids'],
        )
        logits = outputs.logits[:, -1, :]  # Last token
        probs = jax.nn.softmax(logits, axis=-1)
        return probs

    def verify_parallel(
        self,
        context: jnp.ndarray,
        draft_outputs: jnp.ndarray,
        draft_probs: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, int]:
        """Verify draft outputs in parallel across devices.

        This implements Algorithm 1 from the paper.

        Args:
            context: Context tokens [batch_size, seq_len]
            draft_outputs: Draft model outputs [batch_size, num_proposals]
            draft_probs: Draft probabilities [batch_size, num_proposals]

        Returns:
            Tuple of (accepted_tokens, num_accepted)
        """
        batch_size, num_proposals = draft_outputs.shape

        # Create candidate sequences for each proposal
        candidate_seqs = []
        for i in range(num_proposals):
            seq = jnp.concatenate([context, draft_outputs[:, i:i+1]], axis=1)
            candidate_seqs.append(seq)

        all_seqs = jnp.stack(candidate_seqs, axis=0)  # [num_proposals, batch_size, seq_len+1]

        # Split across devices
        chunks = self._split_for_devices(all_seqs)

        # Prepare replicated parameters
        params_replicated = jax.device_put_replicated(
            self.target_model.params,
            self.devices
        )

        # Create chunk data
        chunk_datas = [
            {'input_ids': chunk, 'params': params_replicated[i]}
            for i, chunk in enumerate(chunks)
        ]

        # Parallel verification
        chunk_probs = self._verify_chunk_pmap(
            chunk_datas,
            self.target_model.apply,
        )

        # Gather results
        all_probs = jnp.concatenate(list(chunk_probs), axis=0)[:num_proposals]

        # Extract probabilities for proposed tokens
        target_probs = jnp.take_along_axis(
            all_probs,
            draft_outputs.T[..., None],
            axis=-1
        ).squeeze(-1).T  # [batch_size, num_proposals]

        # Speculative acceptance
        accepted_tokens, num_accepted = self._speculative_acceptance(
            draft_outputs,
            draft_probs,
            target_probs,
        )

        return accepted_tokens, num_accepted

    def _split_for_devices(
        self,
        data: jnp.ndarray,
    ) -> list:
        """Split data evenly across devices.

        Args:
            data: Data to split [num_items, ...]

        Returns:
            List of chunks for each device
        """
        num_items = data.shape[0]
        chunk_size = (num_items + self.num_devices - 1) // self.num_devices

        chunks = []
        for i in range(self.num_devices):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_items)

            if start_idx < num_items:
                chunk = data[start_idx:end_idx]

                # Pad if necessary
                if chunk.shape[0] < chunk_size and i < self.num_devices - 1:
                    padding_shape = (chunk_size - chunk.shape[0], *chunk.shape[1:])
                    padding = jnp.zeros(padding_shape, dtype=chunk.dtype)
                    chunk = jnp.concatenate([chunk, padding], axis=0)

                chunks.append(chunk)
            else:
                # Empty chunk
                empty_shape = (chunk_size, *data.shape[1:])
                chunks.append(jnp.zeros(empty_shape, dtype=data.dtype))

        return chunks

    def _speculative_acceptance(
        self,
        draft_tokens: jnp.ndarray,
        draft_probs: jnp.ndarray,
        target_probs: jnp.ndarray,
        rng_seed: int = 42,
    ) -> Tuple[jnp.ndarray, int]:
        """Apply speculative acceptance criterion.

        Accepts token if: random < min(1, target_prob / draft_prob)

        Args:
            draft_tokens: Proposed tokens [batch_size, num_proposals]
            draft_probs: Draft probabilities [batch_size, num_proposals]
            target_probs: Target probabilities [batch_size, num_proposals]
            rng_seed: Random seed

        Returns:
            Tuple of (accepted_tokens, num_accepted)
        """
        # Acceptance probability
        accept_prob = jnp.minimum(target_probs / (draft_probs + 1e-10), 1.0)

        # Random sampling
        rng = jax.random.PRNGKey(rng_seed)
        random_vals = jax.random.uniform(rng, shape=accept_prob.shape)

        # Accept mask
        accepted_mask = random_vals < accept_prob

        # Find longest accepted prefix
        batch_size, num_proposals = draft_tokens.shape
        accepted_tokens = []
        num_accepted = 0

        for i in range(num_proposals):
            if jnp.all(accepted_mask[:, i]):
                accepted_tokens.append(draft_tokens[:, i])
                num_accepted = i + 1
            else:
                # Stop at first rejection
                break

        if accepted_tokens:
            accepted_tokens = jnp.stack(accepted_tokens, axis=1)
        else:
            accepted_tokens = jnp.zeros((batch_size, 0), dtype=draft_tokens.dtype)

        return accepted_tokens, num_accepted

    def get_device_utilization(self) -> Dict[int, float]:
        """Get utilization for each device.

        Returns:
            Dictionary mapping device_id to utilization percentage
        """
        # Placeholder - would need actual profiling
        return {i: 0.0 for i in range(self.num_devices)}
