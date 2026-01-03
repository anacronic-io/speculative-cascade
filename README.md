# Cascading Speculative Acceleration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.26-orange.svg)](https://github.com/google/jax)

**Breaking the Memory Hierarchy Bottleneck in Multi-Model LLM Inference**

Official implementation of "Cascading Speculative Acceleration: Breaking the Memory Hierarchy Bottleneck in Multi-Model LLM Inference" by Marco Durán Cabobianco and the Distributed Systems Research Team at Anachroni s.coop.

## Overview

This repository contains a 3-stage speculative decoding system optimized for Google Cloud TPU v5e Pods that achieves **4.2×** speedup over baseline autoregressive decoding and **1.8×** over standard speculative decoding.

### Key Features

- **3-Stage Cascade Architecture**: Hierarchical filtering with Tiny (10M) → Draft (2B-int8) → Target (7B-bf16) models
- **Vertical Synergy Principle**: Cooperative filtering that reduces memory hierarchy pressure
- **TPU-Optimized**: Distributed verification using `jax.pmap` with HBM-aware scheduling
- **Analytical Cost Model**: Mathematical framework for optimal configuration
- **High Performance**: 147.2 tokens/sec throughput, 6.8 ms/token latency

## Performance

| Method | Throughput (tok/s) | Latency (ms/tok) | Speedup |
|--------|-------------------|------------------|---------|
| Autoregressive | 35.2 | 28.4 | 1.00× |
| Standard Speculative | 66.8 | 14.9 | 1.90× |
| EAGLE | 94.7 | 10.6 | 2.69× |
| **Cascade (Ours)** | **147.2** | **6.8** | **4.18×** |

## Installation

```bash
# Clone the repository
git clone https://github.com/anacronic-io/speculative-cascade.git
cd speculative-cascade

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Requirements

- Python 3.10+
- JAX 0.4.26+ with TPU support
- Transformers 4.38+
- NumPy, SciPy

For TPU access, you'll need Google Cloud credentials with TPU v5e access.

## Quick Start

```python
from speculative_cascade import CascadeInference
from speculative_cascade.models import TinyModel, DraftModel, TargetModel

# Initialize models
tiny_model = TinyModel.from_pretrained("anachroni/gemma-tiny-10m")
draft_model = DraftModel.from_pretrained("google/gemma-2b", quantize_int8=True)
target_model = TargetModel.from_pretrained("google/gemma-7b")

# Create cascade
cascade = CascadeInference(
    tiny_model=tiny_model,
    draft_model=draft_model,
    target_model=target_model,
    speculation_horizon=8
)

# Run inference
prompt = "The future of AI is"
outputs = cascade.generate(prompt, max_tokens=100)
print(outputs)
```

## Architecture

```
Context → [Stage 0: Tiny (VMEM)]  → K₀ candidates
       ↓                              ↓
       → [Stage 1: Draft (HBM)]    → K₁ filtered
       ↓                              ↓
       → [Stage 2: Target (HBM)]   → Verified tokens
```

### Components

- **Stage 0 (Tiny)**: 10M parameter projection model, fits in VMEM cache
- **Stage 1 (Draft)**: Gemma-2B quantized to INT8, semantic filtering
- **Stage 2 (Target)**: Gemma-7B in BF16, parallel verification

## Benchmarking

```bash
# Run full benchmark suite
python -m speculative_cascade.benchmarks.run_benchmarks \
    --dataset pg19 \
    --hardware tpu-v5e \
    --output results/

# Compare against baselines
python -m speculative_cascade.benchmarks.compare_methods \
    --methods cascade,standard,medusa,eagle \
    --sequences 1000
```

## Documentation

- [Architecture Details](docs/architecture.md)
- [TPU Optimization Guide](docs/tpu_optimization.md)
- [API Reference](docs/api_reference.md)
- [Benchmarking Guide](docs/benchmarking.md)

## Citation

```bibtex
@article{duran2025cascading,
  title={Cascading Speculative Acceleration: Breaking the Memory Hierarchy Bottleneck in Multi-Model LLM Inference},
  author={Durán Cabobianco, Marco and Distributed Systems Research Team},
  journal={arXiv preprint},
  year={2025},
  organization={Anachroni s.coop}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Cloud TRC program for TPU v5e access
- JAX/XLA development teams
- Anachroni s.coop R&D department

## Contact

- Marco Durán Cabobianco: marco@anachroni.co
- General inquiries: info@anachroni.co
- GitHub Issues: [Report bugs or request features](https://github.com/anacronic-io/speculative-cascade/issues)
