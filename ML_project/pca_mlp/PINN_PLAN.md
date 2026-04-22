# PINN + Inverse Design Plan for PcaMLP

Reference document for upgrading the geometry → PCA-coefficient surrogate into a physics-informed model that supports inverse design (find geometries that produce spectra with target resonance features, e.g. shifted peak wavelength).

---

## 1. Goal

**End goal:** use the surrogate as `geom → PCA coeffs → spectrum` to search the geometry space — including regions outside the training distribution — for spectra with desired resonance features (initially: push the dominant Lorentzian peak to a target wavelength).

**Why this changes the model design:**
- Pure spectrum-MSE / PC-MSE is a fine training objective for *interpolation*, but it is misaligned with inverse design and brittle under *extrapolation*.
- PCA coefficients are basis-fit to the training spectra. Outside the training geometry distribution, real spectra may need basis components the PCA never saw — the MLP will confidently extrapolate PC coefficients into nonsense.
- Spectrum MSE penalizes "the curve looks wrong everywhere a little." The design objective only cares about *one feature* (peak location). Loss alignment matters more once the model is being used as an objective function.

---

## 2. Inspiration: PNAS 2025 hBN polariton PINN

The paper trains a multi-output network with a **physics-consistency loss** between its outputs:
- Network predicts `(c, g, γ_avg)` from a coupled-mode spectrum.
- Standard MSE losses on each output.
- Physics term: `L_phy = (ĉ − 2ĝ / γ̂_avg)²` — penalizes any prediction triple that violates the known physical relation.
- Result: model generalizes well enough to act as a synthetic-label generator for a related (but unlabeled) dataset.

**What does NOT transfer:**
- Their two-stage transfer / pseudo-labeling setup. We have one dataset, fully labeled.
- Direction is reversed (they: spectrum → scalar; us: geometry → spectrum).

**What DOES transfer:**
- The pattern of *multiple output heads with a soft physics-consistency loss tying them together*. This is the actual reusable idea.

---

## 3. Proposed Architecture Changes

### 3.1 Multi-headed PcaMLP

Extend [PcaMLP.py](../util/classes/PcaMLP.py) to output two things from a shared trunk:

```
geom (4 or 8) → shared MLP trunk (hidden_dim × n_layers)
                ├── PCA head    → K coefficients          (existing)
                └── Peak head   → (λ_res, γ, amplitude)   (new)
```

Implementation notes:
- Keep the existing trunk (Linear → ReLU → Dropout stack) unchanged.
- Add a second output head: `nn.Linear(hidden_dim, 3)` for `(λ_res, γ, amplitude)`. Could expand to N peaks later (mirrors `Phase2Dataset`), but start with the dominant peak.
- Normalize peak head outputs the same way `Phase1Dataset` already normalizes its labels (zero-mean, unit-variance per channel).
- Forward returns a dict `{"pca": ..., "peak": ...}` so downstream code is explicit about which head is being used.

### 3.2 Loss function

Three terms, weighted:

```
L = λ_pca · L_pca + λ_peak · L_peak + λ_consistency · L_consistency
```

| Term | Definition | Source of truth |
|---|---|---|
| `L_pca` | MSE on K PC coefficients | existing PCA features from `pca_4D.py` / `pca_8D.py` |
| `L_peak` | MSE on `(λ_res, γ, amplitude)` directly from peak head | Lorentzian fit labels from `highest_Q()` (already computed in `Phase1Dataset`) |
| `L_consistency` | MSE between peak head's `λ_res` and the *peak location of the reconstructed spectrum from the PCA head* | self-consistency — no new data needed |

**Why the consistency term is the real "physics-informed" piece:**
Two output paths predict the same physical quantity (peak wavelength). They must agree. This:
1. Anchors PCA-coefficient predictions to a physically meaningful feature, even outside the training distribution.
2. Gives the model an inductive bias that holds globally, not just within the training data.
3. Costs no new labels.

### 3.3 Implementation gotchas

**Differentiable peak finding** for `L_consistency`:
- `argmax` is non-differentiable. Use a soft-argmax over the reconstructed spectrum:
  ```
  weights = softmax(τ · spectrum)              # τ controls sharpness, e.g. 50–200
  λ_peak = sum(weights · wavelength_grid)
  ```
- Alternative: Gaussian-weighted center-of-mass around the discrete argmax. Soft-argmax is simpler.
- Reconstruction: `spectrum = pca_artifacts.inverse_transform(pca_coeffs)`. Already implemented in `PCADataset.reconstruct_spectrum()` — but that uses numpy. Need a torch-native version (just a matmul against the stored PCA components matrix and adding the mean).

**Loss weighting** (`λ_pca`, `λ_peak`, `λ_consistency`):
- Start equal (1.0 each) but expect to tune. The three losses live on very different scales — normalize each by its initial value or use uncertainty-weighted multi-task loss (Kendall et al.) if hand-tuning is painful.
- Optuna sweep over the three weights as Phase 3 of [pca_mlp_optimization.py](pca_mlp_optimization.py).

**Don't replace, augment:** keep `L_pca` as the dominant term initially. Peak loss is a regularizer, not a replacement — without `L_pca` the rest of the spectrum (off-resonance baseline, secondary features) goes unconstrained.

---

## 4. Inverse Design Scheme

Once the model has a peak-location head, inverse design becomes nearly trivial because we no longer need to differentiate through peak-finding.

### 4.1 Phase 1: gradient descent on geometry (start here)

```python
geom = torch.tensor(initial_guess, requires_grad=True)
optimizer = torch.optim.Adam([geom], lr=1e-2)

for step in range(n_steps):
    out = model(normalize(geom))
    loss = (out["peak"][0] - λ_target)**2          # peak head's λ_res
    # Optional: add penalties for going outside physical bounds
    loss += boundary_penalty(geom, geom_bounds)
    loss.backward()
    optimizer.step()
```

Notes:
- Run from many random initial geometries (parallel batch) → keep the best.
- Project `geom` back into physical bounds each step (or use a sigmoid reparameterization).
- The model is microseconds per forward pass. Thousands of restarts are free.

### 4.2 Phase 2: if gradient descent gets stuck

- **CMA-ES** over the surrogate. Black-box, handles multi-modal objectives, no gradients needed. `cma` library is one-line install.
- **Dense grid search** on subspaces (mirroring the paper's `100 × 100` grids per parameter pair). Cheap because the surrogate is fast.
- **Bayesian optimization** as a higher-level loop, especially if the objective involves multiple resonance features.

### 4.3 What NOT to do (yet)

- **Don't train a separate spectrum → geometry inverse network.** The mapping is one-to-many (different geometries can produce similar spectra), so it requires a conditional generative model (cVAE, conditional diffusion). Big jump in complexity for unclear gain.
- **Don't write analytic priors like the grating equation.** SiC grating resonances depend on multiple geometric parameters in coupled ways; a clean closed-form prior is hard, and the peak-consistency loss already does most of the regularization work.

---

## 5. The OOD Trust Region (critical)

This is what the paper's NN-MTGP buys with Gaussian processes, and what the current setup lacks. Without it, inverse design will happily propose geometries where the surrogate is just hallucinating.

### 5.1 Free uncertainty: ensemble disagreement

The training script already produces 4 fold models per run (and the memory note says ensemble inference is the recommended mode). Reuse them:

- For each candidate geometry during inverse design, evaluate all 4 fold models.
- Compute disagreement: `σ_λ = std([fold_i.predict(geom)["peak"][0] for i in 4])`.
- Define a trust threshold (e.g. `σ_λ < 0.005 µm`). Reject candidates above the threshold or downweight them in the inverse design objective:
  ```
  loss = (predicted_peak - target)² + λ_trust · σ_λ²
  ```
- This is essentially deep-ensembles uncertainty quantification — well-validated, almost free to add.

### 5.2 Active learning loop (the actual generalization fix)

Surrogate-only inverse design will eventually run out of trustworthy region. The fix is to expand the training set with the proposed designs:

1. Inverse design proposes top-K candidate geometries (favoring high-confidence ones first, then exploratory ones).
2. Run those K geometries through COMSOL.
3. Add to `data/batch2/`, rerun preprocessing, retrain.
4. Repeat.

Even 5–10 iterations of this dramatically expand the reliable design region. This is more important than any architectural tweak.

### 5.3 Optional: explicit uncertainty head

If ensemble inference is too slow, add an aleatoric-uncertainty head (predict mean + log-variance, train with Gaussian NLL loss). Cheaper than ensembling but only captures aleatoric, not epistemic uncertainty. Skip unless ensembling becomes a bottleneck.

---

## 6. Implementation Order

Recommended order, smallest changes first:

1. **Add peak head + `L_peak`** to [PcaMLP.py](../util/classes/PcaMLP.py) and [training_ex_abs_pca.py](training_ex_abs_pca.py). Verify peak head accuracy matches the standalone `RegMLP` from Phase 1. (No new behavior yet — just confirms the multi-task setup works.)
2. **Add `L_consistency`** with soft-argmax peak finding. Verify training still converges and test recon RMSE doesn't regress significantly from the 0.01435 ensemble baseline.
3. **Sweep loss weights** via Optuna (extend [pca_mlp_optimization.py](pca_mlp_optimization.py) Phase 3).
4. **Build gradient-based inverse design script.** Start with single-target peak-location objective, batch random restarts, ensemble-disagreement gate. New file: `pca_mlp/inverse_design.py`.
5. **Validate against COMSOL.** Take top inverse-design proposals, run in COMSOL, compare predicted vs. simulated peak location. This is the real test.
6. **Active learning loop** (only if inverse design is producing useful designs that don't validate well — meaning the trust region needs to grow).

---

## 7. Success Criteria

- Phase 1–3 should not regress test recon RMSE much below the current ensemble baseline (test_recon_rmse ≈ 0.01435, peak_loc_MAE ≈ 0.00151 µm). Some regression is acceptable if peak_loc_MAE improves and OOD behavior is better.
- Inverse design proposals should validate in COMSOL with peak-location error within ~2× of the test-set peak_loc_MAE. If COMSOL says the peak is somewhere completely different, the trust-region machinery isn't working.
- Active learning iterations should monotonically improve OOD validation accuracy. If they don't, something is wrong with the labeling pipeline or model capacity.

---

## 8. Open Questions to Revisit

- Single dominant peak vs. N peaks: is the design objective always single-peak, or do we want to control multiple resonances? If multi-peak, switch peak head to mirror `Phase2Dataset`'s N-peak structure.
- Soft-argmax temperature τ: too low (smoothing) → biased peak location estimate; too high (sharp) → vanishing gradients. May need annealing.
- Does `L_consistency` actually help on OOD samples, or just on in-distribution? Only way to know is to construct a held-out OOD test set (e.g., hold out the highest-`s` quintile from training).
- Is 4-fold ensemble disagreement well-calibrated as an uncertainty estimate? May need to compare against a deeper ensemble (10+ models) or MC dropout.
