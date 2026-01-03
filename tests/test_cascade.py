"""Tests for cascade inference system."""

import pytest
import jax.numpy as jnp
from unittest.mock import Mock, MagicMock

from speculative_cascade.core.cascade import CascadeInference, CascadeConfig
from speculative_cascade.models.tiny import TinyModel


def create_mock_models():
    """Create mock models for testing."""
    # Mock Tiny model
    tiny_model = Mock()
    tiny_model.generate_candidates = Mock(return_value=(
        jnp.array([[[1, 2, 3, 4, 5]]]),  # candidates
        jnp.array([[[0.2, 0.2, 0.2, 0.2, 0.2]]])  # probs
    ))
    tiny_model.get_memory_footprint = Mock(return_value=10 * 1024 * 1024)

    # Mock Draft model
    draft_model = Mock()
    draft_model.generate_proposals = Mock(return_value=(
        jnp.array([[[1, 2, 3]]]),  # proposals
        jnp.array([[[0.3, 0.3, 0.4]]])  # probs
    ))
    draft_model.get_embeddings = Mock(return_value=jnp.ones((1, 5, 768)))
    draft_model.get_memory_footprint = Mock(return_value=2 * 1024 * 1024 * 1024)

    # Mock Target model
    target_model = Mock()
    target_model.verify_proposals = Mock(return_value=(
        jnp.array([[[1, 2]]]),  # accepted
        2  # num_accepted
    ))
    target_model.get_memory_footprint = Mock(return_value=7 * 1024 * 1024 * 1024)
    target_model.tokenizer = Mock()
    target_model.tokenizer.return_value = Mock(input_ids=jnp.array([[1, 2, 3]]))
    target_model.tokenizer.eos_token_id = 2
    target_model.tokenizer.decode = Mock(return_value="generated text")

    return tiny_model, draft_model, target_model


def test_cascade_initialization():
    """Test cascade initialization."""
    tiny, draft, target = create_mock_models()

    config = CascadeConfig(k0=64, k1=16)
    cascade = CascadeInference(tiny, draft, target, config)

    assert cascade is not None
    assert cascade.config.k0 == 64
    assert cascade.config.k1 == 16


def test_cascade_step():
    """Test single cascade step."""
    tiny, draft, target = create_mock_models()

    config = CascadeConfig(k0=5, k1=3)
    cascade = CascadeInference(tiny, draft, target, config)

    # Run cascade step
    input_ids = jnp.array([[1, 2, 3]])
    accepted_tokens, metrics = cascade._cascade_step(input_ids)

    # Check that all stages were called
    assert tiny.generate_candidates.called
    assert draft.generate_proposals.called
    assert target.verify_proposals.called

    # Check metrics
    assert "gamma_0" in metrics
    assert "gamma_1" in metrics
    assert "gamma_2" in metrics


def test_cascade_config():
    """Test cascade configuration."""
    config = CascadeConfig(
        k0=64,
        k1=16,
        alpha=4.0,
        temp_tiny=0.8,
        temp_draft=1.0,
        temp_target=1.2,
    )

    assert config.k0 == 64
    assert config.k1 == 16
    assert config.alpha == 4.0
    assert config.temp_tiny == 0.8
    assert config.temp_draft == 1.0
    assert config.temp_target == 1.2


def test_cascade_metrics():
    """Test metrics tracking."""
    tiny, draft, target = create_mock_models()

    config = CascadeConfig(k0=5, k1=3)
    cascade = CascadeInference(tiny, draft, target, config)

    # Check initial metrics
    assert cascade.metrics.gamma_0 == 0.0
    assert cascade.metrics.tokens_per_second == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
