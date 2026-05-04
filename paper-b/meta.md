# Paper B: A Landscape Analysis Approach to Feature Alignment in Shallow ReLU Networks

## Status
- LaTeX: compiles with 0 errors (tectonic)
- PDF: ~141 KB
- QC Round 1: FAIL (1 CRITICAL fixed, 3 BLOCKING fixed) -> Round 2 pending user review

## Target Journals (ranked)
1. **JMLR** (as a short/technical note)
2. **Mathematical Statistics and Learning**
3. **COLT 2027** (if strict saddle proof completed)

## MSC 2020
- Primary: 68T07 (Artificial neural networks and deep learning)
- Secondary: 62M45, 90C26

## Submission Strategy
1. Post to **arXiv** first (cs.LG / stat.ML)
2. **Strongly recommended**: Complete Claim 5.1 (strict saddle proof) before submission
3. Without Claim 5.1, paper is a technical note; with it, a full paper

## Key Novelty Claim
ReLU directional independence lemma (kink argument via distributional derivatives) + purely landscape-based proof of feature alignment at global minima.

## Known Limitations
- Claim 5.1 (strict saddle) is UNPROVED -- paper reads as incomplete
- Corollary 6.1 is conditional on unproved claim
- Main alignment result overlaps with Bietti et al. (2022)
- Restricted to positively homogeneous CPL targets

## User TODO
- [ ] Fill in author name, institution, email
- [ ] Add acknowledgements
- [ ] **High priority**: Complete the strict saddle proof (Claim 5.1)
  - Requires explicit Hessian computation using Gaussian-ReLU inner product formula
  - This would elevate the paper significantly
- [ ] Consider promoting Lemma 3.1 to Theorem status (most novel contribution)
- [ ] Add cross-reference to Paper A when it has an arXiv ID
