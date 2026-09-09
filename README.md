# Segreg

Cell segmentation aware regression (and other tools for assessing missegmentation risk).

## The drift model

Missegmentation does not merely add noise to a differential-expression fit: it adds a
**design-correlated** shift, which is indistinguishable from real regulation. Segreg
treats it as an omitted variable. The mean model fits

    λ = exp(design @ (β + κ·u) + gene_bias + log_size)

where `u[k, g]` is the exposure slope of gene `g`'s predicted foreign share along design
column `k`, and `κ` is one fitted scalar per column. Reported coefficients are **β** —
what is left after the contamination-attributable part is taken out.

`u` is built from the segmenter's own output — the segmented counts and its cell↔cell
transition matrix — with no reference, no annotation and no truth, and **it is computed
inside the constructor**, so the correction needs no preprocessing step:

```python
from segreg import Regression

reg = Regression(adata, "~ 0 + C(cell_type) + C(cell_type):exposure",
                 include_drift=True)
reg.fit(nepochs=800, device="cuda")
coefs = reg.get_regression_coefficients()   # β, de-biased
reg.get_drift_scales()                      # the fitted κ, one per design column
reg.get_drift_covariate()                   # u itself, [design column x gene]
```

`adata` needs `obsp["state_transitions"]` and `obs["to_bg_trans_count"]` — proseg's
`--record-state-transitions` output.

Knobs, all optional: `drift_clusters=` supplies the pooling partition instead of running
Leiden; `drift_weights=` picks the projection's weight tier (`none` / `size` / `irls`);
`drift=` supplies a covariate built some other way, which is the seam an oracle-built or
deliberately perturbed `u` goes through. `src/segreg/drift.py` documents the construction
and the reasons behind each default.

> **The likelihood does not choose the β/κ split.** λ depends on them only through
> `β + κ·u`, so it is exactly invariant along `(β + c·u, κ − c)` — one flat direction per
> design column. Correction strength is a declared regularization choice (set by β's
> prior, and optionally a κ prior), not a fitted quantity. Say so when reporting it.
