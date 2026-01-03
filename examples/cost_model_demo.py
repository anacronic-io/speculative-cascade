"""Demonstrate analytical cost model predictions."""

from speculative_cascade.core.cost_model import (
    CascadeCostModel,
    HardwareSpec,
    ModelSpec,
    create_default_model_specs,
)


def main():
    """Cost model demonstration."""
    print("Analytical Cost Model Demonstration")
    print("="*60)

    # Create hardware spec (TPU v5e)
    hardware = HardwareSpec(
        vmem_size=16 * 1024 * 1024,  # 16MB
        hbm_size=16 * 1024 * 1024 * 1024,  # 16GB
        vmem_bandwidth=2000.0,  # GB/s
        hbm_bandwidth=819.0,  # GB/s
        mxu_tflops=197.0,
    )

    # Create cost model
    cost_model = CascadeCostModel(hardware)

    # Get default model specs
    tiny_spec, draft_spec, target_spec = create_default_model_specs()

    print("\nModel Specifications:")
    print(f"  Tiny:   {tiny_spec.num_params / 1e6:.1f}M parameters")
    print(f"  Draft:  {draft_spec.num_params / 1e9:.1f}B parameters (INT8)")
    print(f"  Target: {target_spec.num_params / 1e9:.1f}B parameters (BF16)")

    # Predict cascade performance
    print("\n" + "="*60)
    print("Cascade Cost Prediction")
    print("="*60)

    cascade_cost = cost_model.predict_cascade_cost(
        tiny_spec=tiny_spec,
        draft_spec=draft_spec,
        target_spec=target_spec,
        gamma_0=0.95,
        gamma_1_given_0=0.85,
        gamma_2_given_1=0.90,
        seq_len=1024,
        speculation_horizon=16,
    )

    print("\nCost Breakdown:")
    print(f"  Tiny stage:   {cascade_cost['cost_tiny_ms']:.2f} ms")
    print(f"  Draft stage:  {cascade_cost['cost_draft_ms']:.2f} ms")
    print(f"  Target stage: {cascade_cost['cost_target_ms']:.2f} ms")
    print(f"  Total:        {cascade_cost['total_cost_ms']:.2f} ms")
    print(f"\nExpected accepted tokens: {cascade_cost['expected_accepted_tokens']:.1f}")
    print(f"Cost per token: {cascade_cost['cost_per_token_ms']:.2f} ms")

    # Predict baseline
    print("\n" + "="*60)
    print("Baseline Cost Prediction")
    print("="*60)

    baseline_cost = cost_model.predict_baseline_cost(
        target_spec=target_spec,
        seq_len=1024,
        num_tokens=1,
    )

    print(f"Baseline cost per token: {baseline_cost:.2f} ms")

    # Predict speedup
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

    print(f"\nPredicted speedup: {speedup:.2f}×")

    # Optimize speculation horizon
    print("\n" + "="*60)
    print("Optimizing Speculation Horizon")
    print("="*60)

    opt_horizon, max_speedup = cost_model.optimize_speculation_horizon(
        tiny_spec=tiny_spec,
        draft_spec=draft_spec,
        target_spec=target_spec,
        gamma_0=0.95,
        gamma_1_given_0=0.85,
        gamma_2_given_1=0.90,
        seq_len=1024,
        max_horizon=64,
    )

    print(f"Optimal speculation horizon: {opt_horizon}")
    print(f"Maximum speedup: {max_speedup:.2f}×")

    # Memory hierarchy utilization
    print("\n" + "="*60)
    print("Memory Hierarchy Utilization")
    print("="*60)

    utilization = cost_model.compute_memory_hierarchy_utilization(cascade_cost)

    print(f"VMEM access: {utilization['vmem_access_pct']:.1f}%")
    print(f"HBM access:  {utilization['hbm_access_pct']:.1f}%")
    print(f"Compute:     {utilization['compute_pct']:.1f}%")

    # Vertical synergy verification
    print("\n" + "="*60)
    print("Vertical Synergy Verification")
    print("="*60)

    # Compute isolated costs
    isolated_tiny = cost_model.compute_stage_cost(tiny_spec, "vmem", 1024, 16)
    isolated_draft = cost_model.compute_stage_cost(draft_spec, "hbm", 1024, 16)
    isolated_target = cost_model.compute_stage_cost(target_spec, "hbm", 1024, 16)

    synergy_holds = cost_model.verify_vertical_synergy(
        cascade_cost['total_cost_ms'],
        [isolated_tiny, isolated_draft, isolated_target],
    )

    print(f"Cascade cost: {cascade_cost['total_cost_ms']:.2f} ms")
    print(f"Sum of isolated costs: {isolated_tiny + isolated_draft + isolated_target:.2f} ms")
    print(f"Vertical synergy holds: {synergy_holds}")


if __name__ == "__main__":
    main()
