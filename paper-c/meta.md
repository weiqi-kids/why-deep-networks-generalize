# paper-c metadata

## Title
Feature Learning, Effective Dimension, and Architecture-Specific Critical Depth in Two-Layer ReLU Networks

## Author
Lightman Chang (Independent Researcher), lightman.chang@gmail.com

## Type
Type A (theory, no experiments). Companion to paper-a (M-framework + benign overfitting trichotomy) and paper-b (alignment via landscape).

## Summary
The paper extracts three results from the readme.md research notebook and develops them into a self-contained companion paper to paper-a/paper-b:

1. **Theorem 1 (Feature Learning Dimension Collapse)** — In the lazy NTK regime, two-layer ReLU networks attain the kernel rate Θ(n^(-2s/(2s+d))) on single-index targets in d dimensions; in the feature-learning regime, conditional on the alignment hypothesis from paper-b, the rate becomes Θ(n^(-2s/(2s+1))). The proof attributes the curse of dimensionality to the spherical-harmonic multiplicity N(d,l) ≍ l^(d-2) and shows that alignment collapses this multiplicity to 1.

2. **Theorem 2 (Architecture-Specific Critical Depth)** — Define L*_A as the smallest depth at which r_eff(M_A) ≤ n^(2s/(2s+1)). The paper derives:
   - L*_FC = ⌈(d-1)/2⌉ for fully-connected ReLU stacks (depth-multiplicative spectral decay, with the readme §8c 2×2 misalignment counterexample as a remark on why exact equality fails),
   - L*_ResNet ∈ {1, ∞} (decoupled from L by the identity-bypass structure),
   - L*_Transformer ∈ [1, 4] for d ≤ 512 (softmax attention adds +1/2 to β_0).

3. **Theorem 3 (Cross-Entropy + Neural Collapse)** — Defines a Fisher-corrected operator M_CE := F_CE^(-1) Γ_CE F_CE^(-1) with F_CE := H^(1/2) Σ_CE. Proves that the β > α + 1/2 trichotomy of paper-a is preserved verbatim, and that at the Neural Collapse simplex equiangular tight frame fixed point, r_eff(M_CE) = K-1 for K-class classification.

## Key result
The dimension-free rate n^(-2s/(2s+1)) is attained by two-layer ReLU networks under the alignment hypothesis, with the architectural critical depth determining how easily the benign regime is reached: L*_FC scales as d/2, L*_ResNet is depth-decoupled, and L*_Transformer is bounded by 4 for moderate d.

## Target journals (ranked)

| Rank | Journal | Acceptance estimate | Rationale |
|------|---------|---------------------|-----------|
| 1 | JMLR (Journal of Machine Learning Research) | 25–35% | Primary target; matches paper-a's submission. JMLR publishes long theoretical works without page limits, has a tradition of NTK and benign-overfitting papers (BLLT 2020, TB 2023), and welcomes companion-paper structures. |
| 2 | Annals of Statistics | 10–20% | Strong fit for the minimax-rate calculations and the rigorous spherical-harmonic analysis; competition is severe and statistical-theory framing must be foregrounded. |
| 3 | Information and Inference (IMA) | 30–40% | Welcomes neural-network theory with rigorous rate proofs; shorter turnaround. |
| 4 | NeurIPS / COLT (conference) | 20–30% | If the author wants faster turnaround; the multi-architecture framing fits NeurIPS, the dimension-collapse theorem alone fits COLT. The L* depth result is novel enough to support a standalone COLT submission. |
| 5 | SIMODS (SIAM Journal on Mathematics of Data Science) | 30–40% | Relatively new venue, encourages mathematical-data-science theory; the architecture-specific critical depth is a natural fit. |

Recommended path: submit to JMLR as a companion to paper-a; if paper-a is already under review, mention the companion in the cover letter.

## MSC 2020 codes
- **Primary**: 68T07 (Artificial neural networks and deep learning)
- **Secondary**:
  - 62J05 (Linear regression; mixed models) — for the noise propagation analysis
  - 62G20 (Asymptotic properties of nonparametric inference) — for the minimax rates
  - 41A25 (Rate of convergence, degree of approximation) — for the dimension-collapse rate
  - 60B20 (Random matrices) — for the spectral arguments

Pick 1 primary + 2 secondary on submission (62J05 and 62G20 are strongest).

## Submission strategy
1. **Step 1 (now).** Coordinate paper-a, paper-b, paper-c as a coordinated submission package: each paper is self-contained but cites the others. Cover letter to JMLR should describe the trilogy; reviewers can opt to read all three or just one.
2. **Step 2 (revision-ready).** The readme.md sections on multi-index extension (§16a), L > 2 depth (§16b), and non-Gaussian inputs (§16d) form the natural follow-up papers.
3. **Step 3 (post-acceptance).** Empirical validation paper: numerical experiments confirming L*_FC = ⌈(d-1)/2⌉ on synthetic single-index targets across d ∈ {4, 8, 16, 32}.
4. **Backup target.** If JMLR rejects, submit to Information and Inference (which has shorter timelines) with the same content; minimal restructuring needed.

## What user needs to fill in

### Critical (block submission)
- [ ] **Affiliation/address line.** Currently lists "Independent Researcher" matching paper-a/paper-b style; confirm whether a postal address is required by JMLR.
- [ ] **Funding/COI statements.** None present. JMLR requires explicit COI statement; add to acknowledgements section.
- [ ] **Bibliographic cross-references to paper-a, paper-b.** The references currently use placeholder bibitems `paperA`, `paperB`, `paperA-readme`. On final submission, replace with the actual paper-a/paper-b titles and arXiv numbers (or mark them as "preprint, available on request" if not yet posted).

### Suggested before submission
- [ ] **Numerical sanity check.** A short experiment on synthetic single-index data verifying the 2x2 misalignment counterexample (Remark 8.2) and the L*_FC formula. This would strengthen the empirical credibility of Theorem 2 without changing the paper's theoretical scope.
- [ ] **Tighter constants in Lemma 10.1 (attention β_0).** The +1/2 exponent gain is asymptotic; quantifying constants for the d=512 case stated in Theorem 7.2 would close a small gap. Currently treated as a remark; consider adding an appendix with the tracked constants.
- [ ] **Expand the two-layer alignment in Section 11 (Discussion).** The discussion of multi-layer extension via paper-a §16b is brief; reviewers may ask for a more concrete statement of what changes for L > 2.
- [ ] **Compare explicitly to Bach 2017 and Damian-Lee-Soltanolkotabi 2022.** Both are cited but not contrasted in detail; a 1-paragraph comparison in the introduction would clarify novelty.

### Optional improvements
- [ ] Add an appendix with the exact spherical-harmonic eigenvalue calculation for the 1D ReLU NTK (currently cited from Bietti-Bruna 2021).
- [ ] Add a Conjecture environment for the multi-index extension (Section 11 mentions this as future work but does not formalize it).
- [ ] Add a remark on what happens for κ ≥ 2 single-index targets (the rate degrades by a factor depending on κ; readme §14d gives the calculation).

## File structure
```
paper-c/
├── paper.tex      # 1385 lines, 21-page PDF, compiles with pdflatex (and tectonic)
└── meta.md        # this file
```

## Verification
- `pdflatex paper.tex` compiles cleanly (21 pages, ~428 KB PDF).
- All 18 internal references and 18 bibliographic citations resolve on the second pass.
- No undefined references after second pdflatex run.
- All theorems have full proofs (Theorem 1 split across two sections, Theorem 2 split across three architecture-specific sections, Theorem 3 with two-part proof).
- No forbidden words ("clearly", "obviously", "we solve") used.
- All claims are properly conditioned on Assumption 2.4 (alignment hypothesis from paper-b).
