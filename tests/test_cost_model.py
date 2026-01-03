"""Tests for analytical cost model."""

import pytest
from speculative_cascade.core.cost_model import (
    CascadeCostModel,
    HardwareSpec,
    ModelSpec,
    create_default_model_specs,
)


def test_cost_model_initialization():
    """Test cost model initialization."""
    hardware = HardwareSpec()
    model = CascadeCostModel(hardware)

    assert model is not None
    assert model.hardware.vmem_bandwidth > 0
    assert model.hardware.hbm_bandwidth > 0


def test_memory_cost_calculation():
    """Test memory transfer cost calculation."""
    hardware = HardwareSpec(hbm_bandwidth=819.0)
    model = CascadeCostModel(hardware)

    # 1GB transfer
    cost_ms = model.compute_memory_cost(
        model_size=1024 * 1024 * 1024,
        bandwidth=819.0,
    )

    # Should take approximately 1.22 ms (1GB / 819GB/s)
    assert cost_ms > 1.0
    assert cost_ms < 2.0


def test_compute_cost_calculation():
    """Test computational cost calculation."""
    hardware = HardwareSpec(mxu_tflops=197.0)
    model = CascadeCostModel(hardware)

    # 1B parameter model, 1024 seq_len
    cost_ms = model.compute_compute_cost(
        num_params=1_000_000_000,
        seq_len=1024,
    )

    # Should be reasonable
    assert cost_ms > 0


def test_cascade_cost_prediction():
    """Test cascade cost prediction."""
    hardware = HardwareSpec()
    cost_model = CascadeCostModel(hardware)

    tiny_spec, draft_spec, target_spec = create_default_model_specs()

    result = cost_model.predict_cascade_cost(
        tiny_spec=tiny_spec,
        draft_spec=draft_spec,
        target_spec=target_spec,
        gamma_0=0.95,
        gamma_1_given_0=0.85,
        gamma_2_given_1=0.90,
        seq_len=1024,
        speculation_horizon=16,
    )

    # Check result structure
    assert "cost_tiny_ms" in result
    assert "cost_draft_ms" in result
    assert "cost_target_ms" in result
    assert "total_cost_ms" in result
    assert "expected_accepted_tokens" in result

    # Check values are reasonable
    assert result["total_cost_ms"] > 0
    assert result["expected_accepted_tokens"] > 0


def test_speedup_prediction():
    """Test speedup prediction."""
    hardware = HardwareSpec()
    cost_model = CascadeCostModel(hardware)

    tiny_spec, draft_spec, target_spec = create_default_model_specs()

    speedup = cost_model.predict_speedup(
        tiny_spec=tiny_spec,
        draft_spec=draft_spec,
        target_spec=target_spec,
        gamma_0=0.95,
        gamma_1_given_0=0.85,
        gamma_2_given_1=0.90,
        seq_len=1024,
        speculation_horizon=16,
    )

    # Speedup should be > 1
    assert speedup > 1.0


def test_optimize_speculation_horizon():
    """Test speculation horizon optimization."""
    hardware = HardwareSpec()
    cost_model = CascadeCostModel(hardware)

    tiny_spec, draft_spec, target_spec = create_default_model_specs()

    opt_horizon, max_speedup = cost_model.optimize_speculation_horizon(
        tiny_spec=tiny_spec,
        draft_spec=draft_spec,
        target_spec=target_spec,
        gamma_0=0.95,
        gamma_1_given_0=0.85,
        gamma_2_given_1=0.90,
        seq_len=1024,
        max_horizon=32,
    )

    # Optimal horizon should be reasonable
    assert opt_horizon > 0
    assert opt_horizon <= 32
    assert max_speedup > 1.0


def test_vertical_synergy():
    """Test vertical synergy verification."""
    hardware = HardwareSpec()
    cost_model = CascadeCostModel(hardware)

    # Cascade should have lower cost than sum of isolated
    cascade_cost = 100.0
    isolated_costs = [50.0, 60.0, 70.0]

    synergy = cost_model.verify_vertical_synergy(cascade_cost, isolated_costs)

    assert synergy == True

    # Test case where synergy doesn't hold
    cascade_cost = 200.0
    synergy = cost_model.verify_vertical_synergy(cascade_cost, isolated_costs)

    assert synergy == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
