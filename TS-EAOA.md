## Temperature-Scaled EAOA (TS-EAOA)

This note summarizes the temperature-scaled variant of the Energy-based Active Open-set Annotation strategy that we just integrated into the codebase (`--query-strategy ts_eaoa_sampling`). TS-EAOA keeps the two-head architecture (C-way classifier + (C+1)-way detector) but replaces the GMM/rKNN heuristics with analytical, temperature-controlled free-energy scores.

### 1. Inputs

- `D_U`: unlabeled pool with `N_U` samples `{x_1, …, x_{N_U}}`
- `M_det`: trained `(C + 1)`-class detector (C known + 1 unknown)
- `M_cls`: trained `C`-class classifier
- `b`: query budget per round (`--query-batch`)
- `k`: candidate expansion factor (`--k1`)
- `T > 0`: temperature (`--temperature`)

### 2. Step-by-step procedure

#### Step 1 – Scaled Epistemic Uncertainty (EU)
For each unlabeled sample `x_i`:
```
q_i = M_det(x_i)
E_kno(x_i; T) = -T * log( Σ_{c=1}^{C} exp(q_{i,c} / T) )
E_unk(x_i; T) = -q_{i,C+1}
S_EU(x_i) = E_kno(x_i; T) - E_unk(x_i; T)
```
Lower `S_EU` means the sample is more likely closed-set. We sort all samples in ascending `S_EU` and keep the top `M = floor(k * b)` as the candidate set `C_candidate`.

#### Step 2 – Scaled Aleatoric Uncertainty (AU)
For every candidate `x_j ∈ C_candidate`:
```
p_j = M_cls(x_j)
E_total(x_j; T) = -T * log( Σ_{c=1}^{C} exp(p_{j,c} / T) )
ŷ = argmax_c p_{j,c}
E_sec(x_j; T) = -T * log( Σ_{c ≠ ŷ} exp(p_{j,c} / T) )
S_AU(x_j) = E_total(x_j; T) - E_sec(x_j; T)
```
`S_AU` measures how informative the sample is for the predicted class.

#### Step 3 – Final query set
Sort `C_candidate` by `S_AU` in descending order and select the top `b` samples. Those indices become `X_query`.

### 3. Controlled temperature sweep

To reproduce the controlled experiment from the note, sweep the temperature after each round and measure precision:
```
T ∈ {1.0, 0.5, 2.0}
Precision(T) = (# known samples in X_query(T)) / b
```

You can run each setting by changing `--temperature`:
```
python main.py --query-strategy ts_eaoa_sampling --temperature 0.5 ...
python main.py --query-strategy ts_eaoa_sampling --temperature 1.0 ...
python main.py --query-strategy ts_eaoa_sampling --temperature 2.0 ...
```

### 4. Practical tips

- Keep `k1` adaptation enabled (`--target_precision`, `--a`, `--z`) to automatically widen or shrink the candidate set when precision drifts.
- When `T` is small (<1), free-energy differences sharpen, favoring confident predictions; higher `T` smooths the scores and broadens exploration.
- TS-EAOA does not require rKNN features or GMM fitting, so it is faster per query round than the original EAOA sampler.

