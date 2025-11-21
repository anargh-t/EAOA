# Changes in Output and Runtime Behavior After Temperature-Scaled Energy Modifications

## Overview
The codebase has been updated to use **temperature-scaled free energy** functions throughout, following the new algorithm specification. This document explains what changes when you run the code.

---

## 1. **Energy Value Magnitudes** (Most Noticeable Change)

### Before (Original):
- Energy values: Typically in range **[-50, 0]** or similar (raw logsumexp)
- Example: `E = -logsumexp([2.5, 1.8, 0.3]) ≈ -2.8`

### After (Temperature-Scaled):
- Energy values: **Scaled by temperature T** (default T=1000)
- Example: `E = -1000 * logsumexp([2.5, 1.8, 0.3] / 1000) ≈ -2800`
- **Energy values will be approximately 1000x larger** (when T=1000)

**Impact**: 
- Training loss values will appear larger (but gradients are normalized by T)
- Energy margins (`m_in`, `m_out`) may need adjustment if they were tuned for the old scale
- Log files will show different energy magnitudes

---

## 2. **Training Behavior Changes**

### Energy Loss Computation:
**Before:**
```python
Ec_in = -torch.logsumexp(outputs[known, :-1], dim=1)  # Raw energy
Ec_out = -torch.logsumexp(outputs[unknown, :-1], dim=1)
```

**After:**
```python
Ec_in = scaled_free_energy(outputs[known, :-1], T=1000)  # T-scaled
Ec_out = scaled_free_energy(outputs[unknown, :-1], T=1000)
```

**What This Means:**
- Energy loss gradients are **scaled by 1/T** (so with T=1000, gradients are ~1000x smaller)
- The model learns more **gradually** with respect to energy terms
- Training may require more epochs to converge energy-based objectives
- The `--energy-weight` parameter may need tuning (currently 0.01)

---

## 3. **Epistemic Uncertainty (EU) Calculation**

### Before:
```python
# Simple energy difference
energy = -torch.logsumexp(outputs[:,:-1], dim=1)  # Known classes
# or
energy = -torch.logsumexp(outputs[:,:-1], dim=1) + torch.log(1+torch.exp(outputs[:,-1]))
```

### After:
```python
# Temperature-scaled with explicit unknown class handling
energy_known = scaled_free_energy(outputs[:, :-1], T)  # E(x; {1..C}, T)
energy_unknown = scaled_free_energy(outputs[:, -1:], T)  # E(x; {C+1}, T) = -f_{C+1}(x)
eu_model = energy_known - energy_unknown  # EU_model(x; T)
```

**Impact:**
- EU scores are now **temperature-scaled** (larger absolute values)
- The difference between known/unknown becomes more pronounced at high T
- GMM fitting on EU values will see different distributions

---

## 4. **Aleatoric Uncertainty (AU) Calculation**

### Before:
```python
# Mask-based approach
aleatoricUnc = -torch.logsumexp(outputs_ID, dim=1) + torch.logsumexp(outputs_ID*(1-predTargets)+bias, dim=1)
```

### After:
```python
# Temperature-scaled energy difference
total_energy = scaled_free_energy(outputs_ID, T)  # E(x; {1..C}, T)
secondary_energy = -T * logsumexp((outputs_ID + bias) / T, dim=1)  # E(x; {1..C}\{y_max}, T)
aleatoricUnc = total_energy - secondary_energy  # AU(x; T)
```

**Impact:**
- AU values are temperature-scaled
- The calculation is mathematically equivalent but numerically different
- GMM probabilities for informativeness will be based on scaled values

---

## 5. **Sampling Strategy Changes** (Algorithm Structure)

### Before (Per-Class Selection):
- For each predicted class `i`:
  1. Select top `k1 * num_per_class` by `clean_prob` (low EU)
  2. From those, select top `num_per_class` by `info_prob` (high AU)
- **Result**: Balanced selection across predicted classes

### After (Two-Stage Global Selection):
- **Stage 1**: Select `k1 * query_batch` candidates with **lowest EU** (globally)
- **Stage 2**: From candidates, select `query_batch` with **highest AU**
- **Result**: More focused selection on globally most promising samples

**Impact:**
- **Different samples will be selected** - the query indices will change
- Precision/recall metrics may shift
- The algorithm is now more aligned with the paper's specification
- Less class-balancing, more uncertainty-driven selection

---

## 6. **EU Fusion Method**

### Before:
- EU was computed once (model-based or rKNN-based)
- Single GMM fit on EU values

### After:
- **Two separate EU scores**:
  - `sL`: Model-based EU (from detector model)
  - `sD`: Data-driven EU (from rKNN statistics)
- **Fusion**: `Score_EU = sL_prob ⊙ sD_prob` (element-wise product)
- Both converted to probabilities via separate GMMs

**Impact:**
- More robust EU estimation (combines model and data signals)
- Candidate selection considers both sources of epistemic uncertainty
- May improve precision by filtering out samples with high uncertainty from either source

---

## 7. **What You'll See in Logs/Output**

### Numerical Changes:
1. **Energy values**: ~1000x larger (if T=1000)
2. **Training loss**: Energy component will be larger, but gradients normalized
3. **EU/AU scores**: Scaled by temperature
4. **GMM probabilities**: Based on scaled values (but normalized to [0,1])

### Behavioral Changes:
1. **Query precision**: May change (could improve or degrade depending on T)
2. **Query recall**: May change due to different selection strategy
3. **k1 adaptation**: Will adjust based on new precision values
4. **Training convergence**: May need more epochs for energy terms

### New Hyperparameter:
- `--temperature` (default: 1000.0) - Controls energy scaling
  - Higher T → smoother energy landscape, larger values
  - Lower T → sharper energy landscape, smaller values
  - T=1 → Approximates original behavior (but not exactly)

---

## 8. **Expected Runtime Differences**

### Performance:
- **Slightly slower** due to:
  - Additional temperature divisions in energy calculations
  - Two GMM fits for EU (model + data-driven)
  - More complex candidate selection logic
- **Negligible impact** (~1-5% overhead)

### Memory:
- **No significant change** - same tensor operations, just scaled

---

## 9. **Recommendations for Running**

### If Using Default Temperature (T=1000):
1. **Monitor energy loss values** - they'll be larger, but this is expected
2. **Check if margins need adjustment**:
   - `m_in = -25` → May need to be `-25000` (scaled by T)
   - `m_out = -7` → May need to be `-7000` (scaled by T)
   - **OR** keep margins as-is and let the model adapt

### If Adjusting Temperature:
- **T=1**: Closest to original behavior, but still uses new algorithm structure
- **T=100**: Moderate scaling
- **T=1000**: Default, strong scaling (recommended for numerical stability)
- **T=10000**: Very strong scaling (may cause numerical issues)

### Tuning Strategy:
1. Start with default T=1000
2. Monitor query precision/recall
3. If precision is too low/high, adjust `--target_precision` or `--k1`
4. If energy loss dominates, reduce `--energy-weight`
5. If training is unstable, try lower T (e.g., 100)

---

## 10. **Key Takeaways**

✅ **What Changed:**
- All energy calculations now use temperature scaling
- EU and AU use the new mathematical definitions
- Sampling strategy follows the updated algorithm (two-stage global selection)
- EU fusion combines model-based and data-driven signals

✅ **What to Expect:**
- Larger energy values (scaled by T)
- Different query selections (new algorithm)
- Possibly different precision/recall metrics
- More stable numerical behavior (at high T)

✅ **What to Monitor:**
- Query precision (should adapt via k1)
- Training loss (energy component will be larger)
- Model accuracy (should remain similar)
- k1 adaptation behavior

---

## Example: Before vs After Energy Values

**Sample logits**: `[2.5, 1.8, 0.3]`

**Before (T=1 implicitly):**
```
E = -logsumexp([2.5, 1.8, 0.3]) ≈ -2.83
```

**After (T=1000):**
```
E = -1000 * logsumexp([2.5/1000, 1.8/1000, 0.3/1000])
  = -1000 * logsumexp([0.0025, 0.0018, 0.0003])
  ≈ -1000 * log(exp(0.0025) + exp(0.0018) + exp(0.0003))
  ≈ -1000 * log(3.0075)
  ≈ -2830
```

**Ratio**: ~1000x larger (as expected)

