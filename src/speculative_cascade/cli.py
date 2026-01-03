"""Command-line interface for cascade inference."""

import argparse
import sys
from pathlib import Path

from speculative_cascade import CascadeInference
from speculative_cascade.models import TinyModel, DraftModel, TargetModel
from speculative_cascade.core.cascade import CascadeConfig


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cascade Speculative Acceleration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate text with cascade
  cascade-infer --prompt "The future of AI is" --max-tokens 100

  # Use custom models
  cascade-infer --draft-model google/gemma-2b --target-model google/gemma-7b

  # Show metrics
  cascade-infer --prompt "Hello world" --show-metrics
        """
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for generation"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)"
    )

    parser.add_argument(
        "--draft-model",
        type=str,
        default="google/gemma-2b",
        help="Draft model name (default: google/gemma-2b)"
    )

    parser.add_argument(
        "--target-model",
        type=str,
        default="google/gemma-7b",
        help="Target model name (default: google/gemma-7b)"
    )

    parser.add_argument(
        "--k0",
        type=int,
        default=64,
        help="Tiny model candidates (default: 64)"
    )

    parser.add_argument(
        "--k1",
        type=int,
        default=16,
        help="Draft model proposals (default: 16)"
    )

    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Display performance metrics"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file (default: stdout)"
    )

    args = parser.parse_args()

    # Initialize models
    print("Initializing models...", file=sys.stderr)

    tiny_model = TinyModel.create()
    draft_model = DraftModel.from_pretrained(args.draft_model, quantize_int8=True)
    target_model = TargetModel.from_pretrained(args.target_model)

    # Create cascade
    config = CascadeConfig(k0=args.k0, k1=args.k1)
    cascade = CascadeInference(tiny_model, draft_model, target_model, config)

    # Generate
    print("Generating...", file=sys.stderr)

    output, metrics = cascade.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        return_metrics=True,
    )

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Output saved to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Show metrics
    if args.show_metrics:
        print("\n" + "="*60, file=sys.stderr)
        print("Performance Metrics:", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(metrics, file=sys.stderr)


if __name__ == "__main__":
    main()
