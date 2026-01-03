"""Tests for Tiny projection model."""

import pytest
import jax
import jax.numpy as jnp
from speculative_cascade.models.tiny import TinyModel, TinyProjectionModel


def test_tiny_model_creation():
    """Test creating a new Tiny model."""
    model = TinyModel.create(
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=64,
    )

    assert model is not None
    assert model.vocab_size == 1000
    assert model.embed_dim == 128


def test_tiny_model_forward():
    """Test forward pass through Tiny model."""
    model = TinyModel.create(
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=64,
    )

    # Create dummy input
    batch_size = 2
    seq_len = 10
    embeddings = jnp.ones((batch_size, seq_len, 128))

    # Forward pass
    logits = model(embeddings)

    # Check output shape
    assert logits.shape == (batch_size, seq_len, 1000)


def test_tiny_model_candidate_generation():
    """Test candidate generation."""
    model = TinyModel.create(
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=64,
    )

    embeddings = jnp.ones((1, 5, 128))

    # Generate candidates
    candidates, probs = model.generate_candidates(embeddings, k=10)

    # Check shapes
    assert candidates.shape == (1, 5, 10)
    assert probs.shape == (1, 5, 10)

    # Check probabilities sum to approximately 1 (for top-k)
    assert jnp.all(probs >= 0)
    assert jnp.all(probs <= 1)


def test_tiny_model_memory_footprint():
    """Test memory footprint calculation."""
    model = TinyModel.create(
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=64,
    )

    footprint = model.get_memory_footprint()

    # Should be reasonable size
    assert footprint > 0
    assert footprint < 100 * 1024 * 1024  # Less than 100MB


def test_tiny_model_vmem_fit():
    """Test checking if model fits in VMEM."""
    model = TinyModel.create(
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=64,
    )

    # Should fit in 16MB VMEM
    assert model.fits_in_vmem(16 * 1024 * 1024)


def test_tiny_model_save_load():
    """Test saving and loading model."""
    import tempfile
    import shutil

    model = TinyModel.create(
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=64,
    )

    # Create temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Save model
        model.save_pretrained(temp_dir)

        # Load model
        loaded_model = TinyModel.from_pretrained(temp_dir)

        # Check loaded model
        assert loaded_model.vocab_size == model.vocab_size
        assert loaded_model.embed_dim == model.embed_dim

        # Test forward pass
        embeddings = jnp.ones((1, 5, 128))
        logits1 = model(embeddings)
        logits2 = loaded_model(embeddings)

        # Outputs should be identical
        assert jnp.allclose(logits1, logits2, atol=1e-5)

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
