# paper-c metadata

## Title
Feature Learning, Effective Dimension, and Architecture-Specific Critical Depth

## Author
Lightman Chang (Independent Researcher), lightman.chang@gmail.com

## Type
Type A (theory, no experiments). Companion to paper-a (M-framework + benign overfitting trichotomy), paper-b (alignment via landscape), and paper-d (cross-entropy / Neural Collapse extension, split out from this paper after reviewer feedback).

## Summary
The paper develops two main results from the readme.md research notebook:

1. **Theorem 1 (Feature Learning Dimension Collapse).** In the NTK regime, two-layer ReLU networks attain the kernel rate Θ(n^(-2s/(2s+d))) on single-index targets in d dimensions; in the feature-learning regime, conditional on the alignment hypothesis from paper-b, the rate becomes Θ(n^(-2s/(2s+1))). The proof attributes the curse of dimensionality to the spherical-harmonic multiplicity N(d,l) ≍ l^(d-2). The kernel-collapse lemma (Lemma 5.1) is now stated for the *population* equivalent kernel (averaged over the perpendicular components of inputs), not the pointwise kernel; this fixes a critical flaw in the original draft that conflated random and constant quantities.

2. **Theorem 2 (Architecture-Specific Critical Depth).** Define L*_A as the smallest depth at which r_eff(M_A) ≤ n^(2s/(2s+1)). The paper derives:
   - **L*_FC = ⌈(d-1)/2⌉** for fully-connected ReLU stacks. The per-layer singular-value exponent is α₀ = (d+1)/(2(d-1)) (corrected from the original draft's α₀ = 1/(d-1)), derived in Lemma 6.1 via the spherical-harmonic NTK eigenvalues λ_l ≍ l^(-(d+1)) and the multiplicity-flattening conversion l ↔ j ≍ l^(d-1). The cross-correlation exponent β₀ = (d+3)/(2(d-1)) is derived in Lemma 6.2 from one-step gradient analysis (citing Damian-Lee-Soltanolkotabi 2022 and Ba-Mei-Misiakiewicz-Montanari).  The depth-dependent constant L^(Lα₀) in the Weyl-product chain is now tracked explicitly (Proposition 6.3, Remark 6.4).
   - **L*_ResNet ∈ {1, ∞}** decoupled from L by the identity-bypass structure.
   - **L*_Transformer**: stated as a heuristic Conjecture (downgraded from a Lemma), not a Theorem. The original [1, 4] for d ≤ 512 has been removed since it had no derivation.

3. **Cross-entropy / Neural Collapse extension** is split out into companion paper-d.

## Provenance / changes from previous version

This is a major revision. Headline changes following 4 reviewer reports:

- **Theorem 3 (CE/NC) split to paper-d.** The original §10 cross-entropy / Neural Collapse extension has been moved to a separate companion paper. This addresses reviewer R3 blocker B3.
- **Title shortened.** Dropped "in Two-Layer ReLU Networks" since the body covers FC arbitrary depth + ResNet (R4).
- **Lemma 5.1 (Kernel collapse) rewritten.** Now states a population-kernel result, with explicit decomposition of the perpendicular components and a remark distinguishing pointwise vs. population (R1 CRITICAL 1).
- **α₀ corrected to (d+1)/(2(d-1))**, β₀ corrected to (d+3)/(2(d-1)) (R1 CRITICAL 2). Lemma 6.1 (per-layer α₀) and Lemma 6.2 (per-layer β₀) added with explicit derivations.
- **Weyl chain depth-dependent constant tracked** as L^(Lα₀) (R1 CRITICAL 3, MAJOR 3).
- **Transformer Lemma 8.1 → Conjecture 8.1.** Removed [1,4] / d≤512 specific numbers since they had no derivation (R1 CRITICAL 4, R3 B5, R4).
- **Hermite source-condition decay corrected** to c_l² = O(l^(-2s-1/2)), citing Atkinson-Han 2012 (R1 MAJOR 1).
- **β₀ = 2/(d-1) one-step claim** now cited to Damian-Lee-Soltanolkotabi 2022 / Ba-Mei-Misiakiewicz-Montanari with brief sketch (R1 MAJOR 2; R4 ba2022 author mismatch fixed by replacing with the correct ba2022mei = Ba-Erdogdu-Suzuki-Wu-Zhang spiked-RMT paper that contains the relevant statement, plus DLS 2022).
- **2x2 misalignment counterexample replaced** with a valid construction using J₂ = Q diag(1, ε) Q^T (rotated singular directions); the original J₁ = diag(1, ε), J₂ = R_{π/4} diag(1, ε) R_{-π/4} did not actually exhibit the claimed σ₂ = Θ(ε) behavior (R4).
- **Bach 2017 description corrected** from "mean-field" to "convex/variational" (R4).
- **DLS 2022 / BenArous 2021 descriptions disambiguated**: DLS gives the post-detection rate, BenArous gives the signal-detection complexity (R4).
- **"post-alignment" → "alignment"** throughout (R2).
- **MSC codes corrected** to 68T07 (primary); 62G05, 41A25, 60J60 (R2).
- **Symbol clash on ρ resolved**: feature norm renamed to varrho_j (\\varrho_j), keeping ρ for the link function (R2).
- **Bibliography**:
  - Added: Vaswani 2017 (vaswani2017), Caponnetto-De Vito 2007 (capDeVito2007), Mei-Misiakiewicz-Montanari 2022 (MMM2022), Abbe-Adsera-Misiakiewicz 2022 (abbe2022), Atkinson-Han 2012 (AtkinsonHan2012), Horn-Johnson 2013 (HornJohnson2013), Ba-Erdogdu-Suzuki-Wu-Zhang 2023 (ba2022mei), paperD.
  - Removed: ba2022 (replaced by ba2022mei since author mismatch), HMRT2022, RV2013, TB2023, paperA-readme (Working notebook), lu2022, han2022, mixon2022 (all moved to paper-d's bibliography).
- **"Delta from prior work" subsection** added in introduction (R3 B1).
- **NTK regime terminology standardized** (was: "lazy", "kernel", "fixed-kernel"; now: "NTK regime") (R2).
- **Keywords capitalized** ("Neural Collapse" removed since moved to paper-d; remaining keywords) (R2).
- **paperA-readme citations replaced** with self-contained text or references to paper-a/paper-b proper (R2).

## Key result

The dimension-free rate n^(-2s/(2s+1)) is attained by two-layer ReLU networks under the alignment hypothesis. The architectural critical depth determines how easily the benign regime is reached: L*_FC = ⌈(d-1)/2⌉ scales as d/2, and L*_ResNet is depth-decoupled. The transformer case is conjectured but not proven.

## Target journals
JMLR (primary), Information and Inference, Annals of Statistics, NeurIPS/COLT, SIMODS.

## File structure
```
paper-c/
├── paper.tex      # 22-page PDF
└── meta.md        # this file
```

## Verification
- `pdflatex paper.tex` compiles cleanly (22 pages, ~430 KB PDF).
- All internal references and bibliographic citations resolve on the second pass.
- All theorems have full proofs at the level of mathematical statement; gaps are explicitly stated as conjectures (Conjecture 8.1) or open problems (Section 11).
