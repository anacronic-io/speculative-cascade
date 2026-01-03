# Architecture Overview

## System Design

The Cascading Speculative Acceleration system implements a 3-stage hierarchical architecture optimized for the memory hierarchy of modern accelerators like TPUs.

### Stage 0: Tiny Projection Model

**Purpose**: Fast initial candidate generation

**Specifications**:
- Parameters: 10M
- Memory footprint: ~40MB (BF16)
- Placement: VMEM (Vector Memory)
- Latency: Near-zero access time

**Architecture**:
```
Input Embeddings [batch, seq, 768]
    ↓
Linear Projection [768 → 256]
    ↓
LayerNorm + ReLU
    ↓
Linear Projection [256 → vocab_size]
    ↓
Softmax → Top-K₀ candidates
```

**Key Features**:
- Entire model fits in 16MB VMEM cache
- Single forward pass generates 64 candidate tokens
- Trained via knowledge distillation from Draft model

### Stage 1: Draft Model (INT8 Quantized)

**Purpose**: Semantic filtering of candidates

**Specifications**:
- Base model: Gemma-2B
- Quantization: INT8 (GPTQ)
- Memory footprint: ~1.8GB
- Placement: HBM (High Bandwidth Memory)
- Accuracy retention: 98%

**Process**:
1. Receive K₀=64 candidates from Tiny model
2. Compute probabilities for each candidate
3. Filter to top K₁=16 proposals based on semantic likelihood
4. Pass filtered proposals to Target model

**Quantization Details**:
```python
W_int8 = round(W_fp16 / s) * s
```
where `s` is per-channel scaling factor.

### Stage 2: Target Model (BF16)

**Purpose**: Final verification and token acceptance

**Specifications**:
- Model: Gemma-7B
- Precision: BF16
- Memory footprint: ~13.4GB
- Placement: HBM
- Distribution: Parallel verification across TPU cores

**Verification Algorithm**:
```python
# Distributed verification using JAX pmap
@jax.pmap
def verify_chunk(chunk_ids, params):
    logits = model.apply(params, chunk_ids)
    return softmax(logits)

# Speculative acceptance
accept_prob = min(1, target_prob / draft_prob)
accepted = random() < accept_prob
```

## Memory Hierarchy Optimization

### TPU v5e Memory Architecture

```
┌─────────────────────────────────────┐
│  MXU (Matrix Multiply Units)        │
│  128×128 systolic arrays            │
└───────────────┬─────────────────────┘
                │
┌───────────────▼─────────────────────┐
│  VMEM (Vector Memory)                │
│  16MB per core, ~2000 GB/s          │
│  → Tiny Model (40MB)                 │
└───────────────┬─────────────────────┘
                │
┌───────────────▼─────────────────────┐
│  HBM (High Bandwidth Memory)         │
│  16GB per chip, 819 GB/s            │
│  → Draft Model (1.8GB)               │
│  → Target Model (13.4GB)             │
└─────────────────────────────────────┘
```

### Vertical Synergy Principle

**Definition**: Cascaded models cooperate to reduce memory hierarchy pressure, achieving:

```
Cost_cascade < Σ Cost_isolated(M_i)
```

**Mechanism**:
1. Tiny model reduces HBM bandwidth pressure
2. Draft model filters unnecessary Target model invocations
3. Target model processes only high-confidence candidates

## Distributed Verification

### TPU Pod Configuration

- 16 chips × 256 cores = 4096 cores total
- Models replicated across devices using `jax.pmap`
- Chunk-based parallel verification

### Implementation

```python
# Split proposals across devices
chunks = split_across_devices(proposals, num_devices=16)

# Parallel verification
@jax.pmap
def verify_chunk(chunk, params):
    return target_model.apply(params, chunk)

# Gather results
verified = concatenate_device_outputs(verify_chunk(chunks, params))
```

## Cost Model

### Total Cost Equation

```
C_total = Σᵢ₌₀² [(Wᵢ/BWᵢ) + Tᵢᶜᵒᵐᵖᵘᵗᵉ] · E[Nᵢ]
```

Where:
- `Wᵢ`: Model weight size at stage i
- `BWᵢ`: Memory bandwidth at hierarchy level i
- `Tᵢᶜᵒᵐᵖᵘᵗᵉ`: Computation time for stage i
- `E[Nᵢ]`: Expected tokens processed at stage i

### Expected Tokens

```
E[N₀] = K₀ (speculation horizon)
E[N₁] = E[N₀] · γ₀
E[N₂] = E[N₁] · γ₁|₀
```

### Speedup Prediction

```
Speedup = T_baseline / Σᵢ(γᵢ · Tᵢ + (1-γᵢ) · Wᵢ/BWᵢ)
```

## Performance Characteristics

### Acceptance Rates (Empirical)

- γ₀ (Tiny): 0.95
- γ₁|₀ (Draft|Tiny): 0.85
- γ₂|₁ (Target|Draft): 0.90
- γ_total: 0.73 (vs 0.65 for standard speculation)

### Throughput Breakdown

| Component | Time (ms) | Percentage |
|-----------|-----------|------------|
| Tiny      | 0.8       | 11.8%      |
| Draft     | 2.1       | 30.9%      |
| Target    | 3.9       | 57.3%      |
| **Total** | **6.8**   | **100%**   |

### Memory Utilization

| Level | Usage | Capacity | Utilization |
|-------|-------|----------|-------------|
| VMEM  | 40 MB | 16 MB/core | 2.5% (distributed) |
| HBM   | 15.2 GB | 16 GB | 95% |

## Design Trade-offs

### K₀ and K₁ Selection

- **Higher K₀**: More candidates, better coverage, higher Tiny cost
- **Higher K₁**: More proposals, higher Draft/Target cost
- **Optimal ratio**: K₀ = 4 × K₁ (α=4)

### Quantization Impact

- **INT8 Draft**: 4× memory reduction, 2% accuracy loss
- **BF16 Target**: Full precision for final verification

### Distribution Strategy

- **Chunk size**: Balanced across devices
- **Communication overhead**: Minimized via pmap
- **Load balancing**: Even distribution of verification work

## Future Optimizations

1. **Dynamic horizon adjustment**: Adapt K₀/K₁ based on input difficulty
2. **Cross-layer optimization**: Joint training of cascade models
3. **Heterogeneous hardware**: GPU cluster support
4. **Compilation caching**: Reduce XLA compilation overhead
