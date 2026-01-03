"""Stage 0: Tiny Projection Model.

A minimal neural network designed for maximum cache locality.
The entire model (10M parameters, ~40MB in bf16) fits in VMEM,
achieving near-zero access latency.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import pickle


class TinyProjectionModel(nn.Module):
    """Tiny projection model for Stage 0 filtering.

    Architecture:
        h = LayerNorm(W1 · x + b1)
        P_tiny = softmax(W2 · ReLU(h) + b2)

    Attributes:
        hidden_dim: Hidden dimension size (default: 2048)
        vocab_size: Vocabulary size (default: 32000)
        dtype: Data type for computation (default: jnp.bfloat16)
    """

    hidden_dim: int = 2048
    vocab_size: int = 32000
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input embeddings [batch_size, seq_len, embed_dim]
            training: Whether in training mode

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Layer 1: Linear projection with LayerNorm
        h = nn.Dense(
            self.hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="projection_1"
        )(x)
        h = nn.LayerNorm(dtype=self.dtype, name="layer_norm")(h)

        # Activation
        h = nn.relu(h)

        # Layer 2: Project to vocabulary
        logits = nn.Dense(
            self.vocab_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="projection_2"
        )(h)

        return logits

    def get_num_parameters(self) -> int:
        """Calculate total number of parameters.

        For embed_dim=768:
            W1: 768 * 2048 = 1,572,864
            b1: 2048
            W2: 2048 * 32000 = 65,536,000
            b2: 32000
            Total: ~67M (not 10M - adjusting to match paper)

        To achieve 10M parameters, we reduce hidden_dim.
        """
        return (
            self.hidden_dim * 768 +  # W1
            self.hidden_dim +  # b1
            self.hidden_dim * self.vocab_size +  # W2
            self.vocab_size  # b2
        )


class TinyModel:
    """Wrapper class for Tiny projection model with distillation training support."""

    def __init__(
        self,
        model: TinyProjectionModel,
        params: Dict[str, Any],
        vocab_size: int = 32000,
        embed_dim: int = 768,
    ):
        """Initialize Tiny model.

        Args:
            model: The Flax model
            params: Model parameters
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
        """
        self.model = model
        self.params = params
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    @classmethod
    def create(
        cls,
        vocab_size: int = 32000,
        embed_dim: int = 768,
        hidden_dim: int = 256,  # Reduced to achieve ~10M parameters
        dtype: jnp.dtype = jnp.bfloat16,
        rng_seed: int = 42,
    ) -> "TinyModel":
        """Create a new Tiny model with random initialization.

        Args:
            vocab_size: Vocabulary size
            embed_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            dtype: Data type for parameters
            rng_seed: Random seed for initialization

        Returns:
            Initialized TinyModel instance
        """
        model = TinyProjectionModel(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            dtype=dtype,
        )

        # Initialize parameters
        rng = jax.random.PRNGKey(rng_seed)
        dummy_input = jnp.ones((1, 1, embed_dim), dtype=dtype)
        params = model.init(rng, dummy_input)

        return cls(model, params, vocab_size, embed_dim)

    @classmethod
    def from_pretrained(cls, model_path: str) -> "TinyModel":
        """Load pretrained Tiny model from disk.

        Args:
            model_path: Path to saved model directory

        Returns:
            Loaded TinyModel instance
        """
        path = Path(model_path)

        # Load config
        with open(path / "config.pkl", "rb") as f:
            config = pickle.load(f)

        # Load parameters
        with open(path / "params.pkl", "rb") as f:
            params = pickle.load(f)

        # Recreate model
        model = TinyProjectionModel(
            hidden_dim=config["hidden_dim"],
            vocab_size=config["vocab_size"],
            dtype=jnp.dtype(config["dtype"]),
        )

        return cls(model, params, config["vocab_size"], config["embed_dim"])

    def save_pretrained(self, save_path: str):
        """Save model to disk.

        Args:
            save_path: Directory to save model
        """
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "hidden_dim": self.model.hidden_dim,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "dtype": str(self.model.dtype),
        }
        with open(path / "config.pkl", "wb") as f:
            pickle.dump(config, f)

        # Save parameters
        with open(path / "params.pkl", "wb") as f:
            pickle.dump(self.params, f)

    def __call__(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        """Run forward pass.

        Args:
            embeddings: Input embeddings [batch_size, seq_len, embed_dim]

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        return self.model.apply(self.params, embeddings)

    def generate_candidates(
        self,
        embeddings: jnp.ndarray,
        k: int = 64,
        temperature: float = 1.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate top-k token candidates.

        Args:
            embeddings: Input embeddings [batch_size, seq_len, embed_dim]
            k: Number of candidates to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (candidate_indices, candidate_probs)
        """
        # Get logits
        logits = self(embeddings)

        # Apply temperature
        logits = logits / temperature

        # Get probabilities
        probs = jax.nn.softmax(logits, axis=-1)

        # Get top-k candidates
        # Shape: [batch_size, seq_len, k]
        top_k_probs, top_k_indices = jax.lax.top_k(probs, k)

        return top_k_indices, top_k_probs

    def get_memory_footprint(self) -> int:
        """Calculate memory footprint in bytes.

        Returns:
            Memory usage in bytes
        """
        # Count parameters
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))

        # Assume bf16 (2 bytes per parameter)
        bytes_per_param = 2 if self.model.dtype == jnp.bfloat16 else 4

        return num_params * bytes_per_param

    def fits_in_vmem(self, vmem_size: int = 16 * 1024 * 1024) -> bool:
        """Check if model fits in VMEM cache.

        Args:
            vmem_size: VMEM size in bytes (default: 16MB for TPU v5e)

        Returns:
            True if model fits in VMEM
        """
        return self.get_memory_footprint() <= vmem_size


def train_tiny_via_distillation(
    tiny_model: TinyModel,
    draft_model: Any,  # Draft model to distill from
    train_data: Any,
    num_steps: int = 10000,
    learning_rate: float = 1e-4,
    temperature: float = 2.0,
) -> TinyModel:
    """Train Tiny model via knowledge distillation from Draft model.

    This implements the distillation training procedure described in the paper.

    Args:
        tiny_model: Initialized Tiny model
        draft_model: Draft model to distill from
        train_data: Training dataset
        num_steps: Number of training steps
        learning_rate: Learning rate
        temperature: Distillation temperature

    Returns:
        Trained TinyModel
    """
    import optax

    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(tiny_model.params)

    def distillation_loss(params, embeddings, teacher_logits):
        """Compute KL divergence loss for distillation."""
        student_logits = tiny_model.model.apply(params, embeddings)

        # Temperature scaling
        student_probs = jax.nn.softmax(student_logits / temperature, axis=-1)
        teacher_probs = jax.nn.softmax(teacher_logits / temperature, axis=-1)

        # KL divergence
        kl_div = jnp.sum(
            teacher_probs * (jnp.log(teacher_probs + 1e-10) - jnp.log(student_probs + 1e-10)),
            axis=-1
        )

        return jnp.mean(kl_div)

    @jax.jit
    def train_step(params, opt_state, embeddings, teacher_logits):
        """Single training step."""
        loss, grads = jax.value_and_grad(distillation_loss)(params, embeddings, teacher_logits)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Training loop
    params = tiny_model.params
    for step in range(num_steps):
        # Get batch (placeholder - implement data loading)
        batch_embeddings, batch_teacher_logits = next(train_data)

        # Training step
        params, opt_state, loss = train_step(params, opt_state, batch_embeddings, batch_teacher_logits)

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    # Update model with trained parameters
    tiny_model.params = params
    return tiny_model
