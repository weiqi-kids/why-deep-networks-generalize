# Paper A: Noise Propagation Operators and a Two-Parameter Characterization of Benign Overfitting

## Status
- LaTeX: compiles with 0 errors (tectonic)
- PDF: ~158 KB
- QC Round 1: FAIL (5 MAJOR fixed) -> Round 2 pending user review

## Target Journals (ranked)
1. **JMLR** (Journal of Machine Learning Research) -- best fit for framework paper
2. **NeurIPS 2026** (if results strengthened with delta_lin quantification)
3. **Annals of Statistics** (if expanded with finite-sample Gamma analysis)

## MSC 2020
- 62J05 (Linear regression; mixed models)
- 68T07 (Artificial neural networks and deep learning)
- 62G20 (Asymptotic properties of nonparametric inference)

## Submission Strategy
1. Post to **arXiv** first (stat.ML / cs.LG)
2. Seek feedback from community (consider MathOverflow "Is this known?" post)
3. Submit to JMLR after incorporating feedback

## Key Novelty Claim
Two-parameter (alpha, beta) benign overfitting condition via M = Sigma^{-1} Gamma Sigma^{-1}, refining Mallinar et al. (2022) single-parameter taxonomy.

## Known Limitations (must address before submission)
- delta_lin (linearization residual) not quantified
- B^2_signal (signal bias) not analyzed
- Gamma defined at population level only
- Reviewer risk: "reformulation of kernel regression theory"

## User TODO
- [ ] Fill in author name, institution, email
- [ ] Add acknowledgements
- [ ] Consider adding numerical experiments (strengthens empirical support)
- [ ] Consider posting "Is this formulation known?" on MathOverflow
- [ ] Add missing references suggested by R3: Rakhlin & Zhai, Belkin et al. (2019)
