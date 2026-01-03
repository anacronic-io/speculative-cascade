"""Stage 2: Target Model with BF16 Precision.

Implements Gemma-7B with parallel verification using JAX pmap
for distributed execution across TPU cores.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any, List
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
from functools import partial


class TargetModel:
    """Target model (Gemma-7B) for Stage 2 verification."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        num_devices: Optional[int] = None,
    ):
        """Initialize Target model.

        Args:
            model: The base model (FlaxAutoModelForCausalLM)
            tokenizer: Tokenizer
            num_devices: Number of TPU/GPU devices (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.num_devices = num_devices or jax.device_count()

        # Replicate parameters across devices for pmap
        self.replicated_params = jax.device_put_replicated(
            model.params, jax.devices()[:self.num_devices]
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "google/gemma-7b",
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> "TargetModel":
        """Load pretrained Target model.

        Args:
            model_name: HuggingFace model name
            dtype: Data type for model weights

        Returns:
            Initialized TargetModel instance
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model in JAX/Flax
        model = FlaxAutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        )

        return cls(model, tokenizer)

    def verify_proposals(
        self,
        input_ids: jnp.ndarray,
        proposal_ids: jnp.ndarray,
        proposal_probs: jnp.ndarray,
        use_distributed: bool = True,
    ) -> Tuple[jnp.ndarray, int]:
        """Verify token proposals from Draft model.

        Implements the parallel verification algorithm from the paper.

        Args:
            input_ids: Context token IDs [batch_size, seq_len]
            proposal_ids: Proposed token IDs [batch_size, num_proposals]
            proposal_probs: Proposal probabilities [batch_size, num_proposals]
            use_distributed: Whether to use distributed verification

        Returns:
            Tuple of (accepted_tokens, num_accepted)
        """
        if use_distributed and self.num_devices > 1:
            return self._verify_distributed(input_ids, proposal_ids, proposal_probs)
        else:
            return self._verify_sequential(input_ids, proposal_ids, proposal_probs)

    def _verify_sequential(
        self,
        input_ids: jnp.ndarray,
        proposal_ids: jnp.ndarray,
        proposal_probs: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, int]:
        """Sequential verification on single device.

        Args:
            input_ids: Context [batch_size, seq_len]
            proposal_ids: Proposals [batch_size, num_proposals]
            proposal_probs: Probabilities [batch_size, num_proposals]

        Returns:
            Tuple of (accepted_tokens, num_accepted)
        """
        batch_size, num_proposals = proposal_ids.shape

        # Construct candidate sequences
        # For each proposal, append it to the context
        candidate_seqs = []
        for i in range(num_proposals):
            seq = jnp.concatenate([input_ids, proposal_ids[:, i:i+1]], axis=1)
            candidate_seqs.append(seq)

        # Stack into single batch
        all_seqs = jnp.concatenate(candidate_seqs, axis=0)

        # Get target model probabilities
        outputs = self.model(all_seqs)
        logits = outputs.logits[:, -1, :]  # Last position logits
        target_probs = jax.nn.softmax(logits, axis=-1)

        # Reshape back to [batch_size, num_proposals, vocab_size]
        target_probs = target_probs.reshape(batch_size, num_proposals, -1)

        # Gather probabilities for proposed tokens
        proposal_idx_expanded = proposal_ids[..., None]
        target_proposal_probs = jnp.take_along_axis(
            target_probs, proposal_idx_expanded, axis=-1
        ).squeeze(-1)

        # Apply speculative acceptance criterion
        # Accept if target_prob >= draft_prob (with random sampling)
        accept_threshold = proposal_probs
        random_values = jax.random.uniform(
            jax.random.PRNGKey(0),
            shape=target_proposal_probs.shape
        )

        # Accept token if: random < min(1, target_prob / draft_prob)
        accept_prob = jnp.minimum(target_proposal_probs / (proposal_probs + 1e-10), 1.0)
        accepted_mask = random_values < accept_prob

        # Find longest accepted prefix
        accepted_tokens = []
        num_accepted = 0

        for i in range(num_proposals):
            if jnp.all(accepted_mask[:, i]):
                accepted_tokens.append(proposal_ids[:, i])
                num_accepted = i + 1
            else:
                break

        if accepted_tokens:
            accepted_tokens = jnp.stack(accepted_tokens, axis=1)
        else:
            accepted_tokens = jnp.zeros((batch_size, 0), dtype=jnp.int32)

        return accepted_tokens, num_accepted

    @partial(jax.pmap, in_axes=(None, 0, None), out_axes=0, static_broadcasted_argnums=(0,))
    def _verify_chunk(
        self,
        chunk_ids: jnp.ndarray,
        params: Dict,
    ) -> jnp.ndarray:
        """Verify a chunk of proposals on a single device.

        This function is mapped across TPU cores using pmap.

        Args:
            chunk_ids: Token IDs for this chunk [chunk_size, seq_len]
            params: Model parameters (replicated)

        Returns:
            Logits [chunk_size, seq_len, vocab_size]
        """
        outputs = self.model.apply(params, chunk_ids)
        return outputs.logits

    def _verify_distributed(
        self,
        input_ids: jnp.ndarray,
        proposal_ids: jnp.ndarray,
        proposal_probs: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, int]:
        """Distributed verification across multiple TPU cores.

        Implements Algorithm 1 from the paper using jax.pmap.

        Args:
            input_ids: Context [batch_size, seq_len]
            proposal_ids: Proposals [batch_size, num_proposals]
            proposal_probs: Probabilities [batch_size, num_proposals]

        Returns:
            Tuple of (accepted_tokens, num_accepted)
        """
        batch_size, num_proposals = proposal_ids.shape

        # Construct candidate sequences
        candidate_seqs = []
        for i in range(num_proposals):
            seq = jnp.concatenate([input_ids, proposal_ids[:, i:i+1]], axis=1)
            candidate_seqs.append(seq)

        all_seqs = jnp.stack(candidate_seqs, axis=0)  # [num_proposals, batch_size, seq_len]

        # Split across devices
        num_devices = self.num_devices
        chunks = jnp.array_split(all_seqs, num_devices, axis=0)

        # Pad chunks to same size for pmap
        max_chunk_size = max(chunk.shape[0] for chunk in chunks)
        padded_chunks = []
        for chunk in chunks:
            if chunk.shape[0] < max_chunk_size:
                padding = jnp.zeros(
                    (max_chunk_size - chunk.shape[0], *chunk.shape[1:]),
                    dtype=chunk.dtype
                )
                chunk = jnp.concatenate([chunk, padding], axis=0)
            padded_chunks.append(chunk)

        # Stack chunks for pmap [num_devices, chunk_size, batch_size, seq_len]
        device_chunks = jnp.stack(padded_chunks, axis=0)

        # Distributed verification
        device_logits = self._verify_chunk(device_chunks, self.replicated_params)

        # Concatenate results from all devices
        all_logits = jnp.concatenate(list(device_logits), axis=0)[:num_proposals]

        # Get probabilities for last position
        target_probs = jax.nn.softmax(all_logits[:, :, -1, :], axis=-1)

        # Reshape to [batch_size, num_proposals, vocab_size]
        target_probs = target_probs.transpose(1, 0, 2)

        # Rest of verification logic (same as sequential)
        proposal_idx_expanded = proposal_ids[..., None]
        target_proposal_probs = jnp.take_along_axis(
            target_probs, proposal_idx_expanded, axis=-1
        ).squeeze(-1)

        accept_prob = jnp.minimum(target_proposal_probs / (proposal_probs + 1e-10), 1.0)
        random_values = jax.random.uniform(
            jax.random.PRNGKey(0),
            shape=accept_prob.shape
        )
        accepted_mask = random_values < accept_prob

        # Find longest accepted prefix
        accepted_tokens = []
        num_accepted = 0

        for i in range(num_proposals):
            if jnp.all(accepted_mask[:, i]):
                accepted_tokens.append(proposal_ids[:, i])
                num_accepted = i + 1
            else:
                break

        if accepted_tokens:
            accepted_tokens = jnp.stack(accepted_tokens, axis=1)
        else:
            accepted_tokens = jnp.zeros((batch_size, 0), dtype=jnp.int32)

        return accepted_tokens, num_accepted

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
    ) -> str:
        """Standard autoregressive generation (baseline).

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="jax").input_ids

        # Generate
        for _ in range(max_tokens):
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature
            probs = jax.nn.softmax(logits, axis=-1)

            # Sample next token
            next_token = jax.random.categorical(
                jax.random.PRNGKey(0),
                logits,
                axis=-1
            )

            # Append to sequence
            input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)

            # Check for EOS
            if next_token[0] == self.tokenizer.eos_token_id:
                break

        # Decode
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def __call__(self, input_ids: jnp.ndarray, **kwargs) -> Any:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments

        Returns:
            Model outputs
        """
        return self.model(input_ids, **kwargs)

    def get_memory_footprint(self) -> int:
        """Calculate memory footprint in bytes.

        Returns:
            Memory usage in bytes (BF16: 2 bytes per parameter)
        """
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(self.model.params))
        return num_params * 2  # BF16

    def estimate_acceptance_rate(
        self,
        test_proposals: List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    ) -> float:
        """Estimate acceptance rate γ₂|₁ for verification.

        Args:
            test_proposals: List of (input_ids, proposal_ids, proposal_probs) tuples

        Returns:
            Average acceptance rate
        """
        total_proposed = 0
        total_accepted = 0

        for input_ids, proposal_ids, proposal_probs in test_proposals:
            _, num_accepted = self.verify_proposals(input_ids, proposal_ids, proposal_probs)
            total_accepted += num_accepted
            total_proposed += proposal_ids.shape[1]

        return total_accepted / total_proposed if total_proposed > 0 else 0.0
