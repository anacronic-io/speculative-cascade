# Examples

This directory contains example scripts demonstrating how to use the Cascade Speculative Acceleration system.

## Basic Usage

### Simple Text Generation

```python
python basic_usage.py
```

This example demonstrates:
- Initializing the 3-stage cascade
- Configuring speculation parameters
- Generating text with metrics

### Cost Model Analysis

```python
python cost_model_demo.py
```

This example shows:
- Analytical cost predictions
- Speedup estimation
- Optimal horizon calculation
- Memory hierarchy utilization

## Running Benchmarks

### Full Benchmark Suite

```bash
# Run on PG-19 dataset
python -m speculative_cascade.benchmarks.run_benchmarks \
    --dataset pg19 \
    --num-samples 100 \
    --output-dir results/

# Run on code generation (HumanEval)
python -m speculative_cascade.benchmarks.run_benchmarks \
    --dataset humaneval \
    --num-samples 50
```

### Compare Methods

```bash
python -m speculative_cascade.benchmarks.compare_methods \
    --output results/comparison.json
```

This reproduces Table 1 from the paper, comparing against:
- Autoregressive baseline
- Standard Speculative
- Medusa (2, 4, 8 heads)
- EAGLE
- Lookahead
- Self-Speculative
- SpecTr
- DistillSpec

## Advanced Usage

### Custom Model Configuration

```python
from speculative_cascade import CascadeInference
from speculative_cascade.models import TinyModel, DraftModel, TargetModel
from speculative_cascade.core.cascade import CascadeConfig

# Create custom tiny model
tiny = TinyModel.create(
    vocab_size=32000,
    embed_dim=1024,  # Larger embedding
    hidden_dim=512,  # Larger hidden layer
)

# Use different draft model
draft = DraftModel.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantize_int8=True,
)

# Configure cascade
config = CascadeConfig(
    k0=128,  # More candidates
    k1=32,   # More proposals
    temp_tiny=0.8,
    use_distributed=True,
)

cascade = CascadeInference(tiny, draft, target, config)
```

### Memory-Aware Scheduling

```python
from speculative_cascade.core.scheduler import MemoryAwareScheduler

scheduler = MemoryAwareScheduler(
    vmem_capacity=16 * 1024 * 1024,
    hbm_capacity=16 * 1024 * 1024 * 1024,
)

# Place models optimally
placements, batch_size = scheduler.schedule(
    models={
        "tiny": 40 * 1024 * 1024,
        "draft": 2 * 1024 * 1024 * 1024,
        "target": 7 * 1024 * 1024 * 1024,
    },
    batch_size=32,
    max_seq_len=2048,
)

print(f"Optimal batch size: {batch_size}")
scheduler.print_memory_layout()
```

### Profiling

```python
from speculative_cascade.utils.profiling import TPUProfiler

profiler = TPUProfiler(enabled=True)

with profiler.profile("generation", flops=1e12):
    output = cascade.generate(prompt, max_tokens=100)

profiler.print_summary()
```

## Expected Output

### Basic Usage

```
Cascade Speculative Acceleration - Basic Example
============================================================

1. Initializing models...
  - Creating Tiny model (10M parameters)...
  - Loading Draft model (Gemma-2B with INT8 quantization)...
  - Loading Target model (Gemma-7B)...

2. Configuring cascade...

3. Creating cascade inference system...
✓ Memory constraints verified
  Tiny: 40.0 MB (VMEM)
  Draft: 1800.0 MB (HBM)
  Target: 13.4 GB (HBM)

4. Generating text...

Prompt: The future of artificial intelligence is

Generating (this may take a moment)...

Generated: The future of artificial intelligence is bright...

============================================================
Performance Metrics:
============================================================

Cascade Metrics:
  Acceptance Rates:
    γ₀ (Tiny):          0.950
    γ₁|₀ (Draft|Tiny):  0.850
    γ₂|₁ (Target|Draft):0.900
    γ_total:            0.727

  Performance:
    Throughput:         147.2 tokens/sec
    Latency:            6.8 ms/token

  Memory Usage:
    VMEM:               40.0 MB
    HBM:                15.2 GB
    Total:              15.2 GB
```

## Troubleshooting

### Out of Memory

If you encounter OOM errors:

1. Reduce batch size
2. Decrease speculation horizon (k0, k1)
3. Use smaller models

### Slow Performance

If performance is lower than expected:

1. Ensure TPU access is properly configured
2. Enable distributed verification
3. Check XLA compilation cache
4. Verify model placement in memory hierarchy

## More Examples

See the [documentation](../docs/) for more detailed examples and tutorials.
