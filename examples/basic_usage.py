"""Basic usage example for Cascade Speculative Acceleration."""

from speculative_cascade import CascadeInference
from speculative_cascade.models import TinyModel, DraftModel, TargetModel
from speculative_cascade.core.cascade import CascadeConfig


def main():
    """Basic usage example."""
    print("Cascade Speculative Acceleration - Basic Example")
    print("="*60)

    # Step 1: Initialize models
    print("\n1. Initializing models...")

    # Tiny model (Stage 0)
    print("  - Creating Tiny model (10M parameters)...")
    tiny_model = TinyModel.create(
        vocab_size=32000,
        embed_dim=768,
        hidden_dim=256,
    )

    # Draft model (Stage 1)
    print("  - Loading Draft model (Gemma-2B with INT8 quantization)...")
    draft_model = DraftModel.from_pretrained(
        "google/gemma-2b",
        quantize_int8=True,
    )

    # Target model (Stage 2)
    print("  - Loading Target model (Gemma-7B)...")
    target_model = TargetModel.from_pretrained(
        "google/gemma-7b",
    )

    # Step 2: Configure cascade
    print("\n2. Configuring cascade...")
    config = CascadeConfig(
        k0=64,  # Tiny model candidates
        k1=16,  # Draft model proposals
        alpha=4.0,  # k0 = k1 * alpha
        temp_tiny=1.0,
        temp_draft=1.0,
        temp_target=1.0,
        use_distributed=True,  # Use distributed verification on TPU
    )

    # Step 3: Create cascade
    print("\n3. Creating cascade inference system...")
    cascade = CascadeInference(
        tiny_model=tiny_model,
        draft_model=draft_model,
        target_model=target_model,
        config=config,
    )

    # Step 4: Generate text
    print("\n4. Generating text...")
    prompt = "The future of artificial intelligence is"

    print(f"\nPrompt: {prompt}")
    print("\nGenerating (this may take a moment)...")

    output, metrics = cascade.generate(
        prompt,
        max_tokens=50,
        return_metrics=True,
    )

    print(f"\nGenerated: {output}")

    # Step 5: Display metrics
    print("\n" + "="*60)
    print("Performance Metrics:")
    print("="*60)
    print(metrics)

    # Step 6: Compare with baseline
    print("\n5. Comparing with baseline...")
    baseline_output = target_model.generate(prompt, max_tokens=50)
    print(f"Baseline output: {baseline_output}")

    # Calculate speedup
    baseline_throughput = 35.2  # From paper
    speedup = cascade.get_speedup(baseline_throughput)
    print(f"\nSpeedup vs baseline: {speedup:.2f}×")


if __name__ == "__main__":
    main()
