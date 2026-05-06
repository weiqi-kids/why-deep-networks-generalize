# paper-d metadata

## Title
A Cross-Entropy Noise Propagation Operator and the Effective Rank at the Neural Collapse Fixed Point

## Author
Lightman Chang (Independent Researcher), lightman.chang@gmail.com

## Type
Type A (theory, no experiments). Companion to paper-a (M-framework + benign overfitting trichotomy) and paper-c (feature-learning dimension collapse + architecture-specific critical depth in regression).

## Provenance
Originally Section 10 ("Theorem 3: Cross-Entropy Extension via Neural Collapse") of paper-c. Split out into a self-contained companion paper because the cross-entropy / Neural Collapse content is logically independent of the regression results in paper-c (Theorem 1: feature learning dimension collapse, Theorem 2: architecture-specific critical depth). The split addresses reviewer R3's blocker B3.

## Summary

The paper extends the noise-propagation framework of paper-a (originally formulated for MSE regression) to multi-class classification under cross-entropy loss, and computes the effective rank at the Neural Collapse (NC) fixed point.

1. **Derivation of M_CE from the CE Hessian.** Starting from the CE parameter Hessian ∇²ℓ_CE = J^T H J with H(x) = diag(p) − pp^T (the multinomial Fisher matrix), define the Fisher-weighted Jacobian J̃ = H^(1/2) J_CE with SVD J̃ = ŨΣ̃Ṽ^T, and define M_CE := Σ̃^(-1) Γ̃ Σ̃^(-1) with Γ̃ the cross-correlation in the Ṽ basis. This replaces the postulated form M_CE = F^(-1) Γ F^(-1) with F = H^(1/2) Σ_CE in the original paper-c §10 by a direct derivation from the CE Hessian.

2. **CE trichotomy (Theorem 3.1 / Theorem 3 in paper-c §10).** Under power-law decay of the singular values of J̃ and a uniform Fisher lower bound h_min > 0, tr(M_CE) < ∞ ⟺ β > α + 1/2, identical to the MSE trichotomy of paper-a.

3. **Effective rank at NC (Theorem 4.3).** At the NC fixed point with class-balanced features, the limiting M_CE on the zero-sum subspace has effective rank exactly K − 1, where K is the number of classes. The proof is now careful about the strict-one-hot limit: at a literal NC limit, H → 0 and M_CE degenerates; we work in the finite-logit-scale regime where H is invertible on the (K−1)-dimensional zero-sum subspace Z_K = {u ∈ R^K : 1^T u = 0}. The corrected derivation uses Lemma 4.1 and Remark 4.2 to make this precise, replacing the wrong identification "H = (1/K)(I − K^(-1) 11^T) at the NC fixed point" in the original paper-c §10.

## Key result
Cross-entropy noise propagation reproduces the MSE trichotomy verbatim, and at the Neural Collapse fixed point the effective rank of the noise propagation operator equals K − 1 (number of classes minus one), independent of input dimension.

## Target journals (ranked)

| Rank | Journal | Acceptance estimate | Rationale |
|------|---------|---------------------|-----------|
| 1 | JMLR | 25–35% | Same as paper-c; companion submission. |
| 2 | Information and Inference (IMA) | 30–40% | Strong fit; shorter paper, focused result. |
| 3 | NeurIPS / COLT (conference) | 20–30% | The K−1 effective rank at NC is a clean standalone NC contribution. |
| 4 | SIMODS | 30–40% | Mathematical-data-science framing. |

## MSC 2020 codes
- **Primary**: 68T07 (Artificial neural networks and deep learning)
- **Secondary**: 62G05 (Nonparametric inference), 41A25 (Rate of convergence), 60J60 (Diffusion processes / SDE-style limits)

## Submission strategy
1. Submit alongside paper-a, paper-b, paper-c as a coordinated package.
2. The Theorem 4.3 effective-rank-(K−1) result is the headline; the trichotomy preservation is a useful structural observation but the NC effective-rank computation is the technical contribution.

## What user needs to fill in

### Critical (block submission)
- [ ] **Citation conversion.** `paperA`, `paperC` are placeholder bibitems; replace with arXiv numbers on submission.
- [ ] **Affiliation/address line.** Confirm whether JMLR requires postal address.

### Suggested
- [ ] Quantify the off-fixed-point gap r_eff(M_CE)(t) − (K−1) along the NC convergence trajectory.
- [ ] Treat class-imbalanced data (Fang et al. 2021 minority collapse).
- [ ] Add a short experimental verification on MNIST/CIFAR-10 confirming r_eff ≈ K − 1 at the NC plateau.

## File structure
```
paper-d/
├── paper.tex
└── meta.md          # this file
```

## Verification
- `pdflatex paper.tex` compiles cleanly.
- All theorems are self-contained; cite paper-a only for the underlying noise-propagation framework.
- All claims are properly conditioned on NC.
