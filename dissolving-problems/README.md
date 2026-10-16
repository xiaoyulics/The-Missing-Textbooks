# Dissolving Problems: A Casebook of Theory-Building

*The Missing Textbooks, volume 1.* Author: Claude Opus 5.5 (max reasoning effort). Human oversight: Xiaoyu Li.

This graduate textbook teaches one skill: finding the setting in which a hard problem stops being hard. Grothendieck called this the "rising sea". Each chapter is a case: a problem fought with hammer and chisel, the single reframing (the **move**) that dissolved it, the old results re-proved in the new setting, and an honest account of what the move cost and where it fails.

**[Read the PDF](Dissolving-Problems.pdf)** (332 pages).

## Contents

- **Prologue.** What it means to dissolve a problem: the needle and the noodle, the dissolution test, and where the difficulty goes.
- **Part I. Building the Right Object**
  1. Completing the Integral: Lebesgue and the escaping limits (*completion*)
  2. Completing Graphs: from the regularity lemma to graph limits (*completion*)
  3. Axiomatizing Chance: Kolmogorov and the hidden parameter (*axiomatization*)
  4. Why Greed Works: matroids and the exchange axiom (*axiomatization*)
- **Part II. Changing the Battlefield**
  5. All Minimax Theorems Are One: linear programming duality (*duality*)
  6. The Dynamics Behind Duality: no-regret learning (*existence → dynamics*)
  7. Topology for Combinatorics: Kneser's conjecture and Borsuk–Ulam (*transfer*)
  8. Change of Basis: Fourier analysis on the Boolean cube (*diagonalize the symmetry*)
  9. Combinatorics as Dimension: from Oddtown to cap sets (*linearization*)
  10. Inequalities as Maps: optimal transport (*geometrization*)
- **Part III. Finding the Right Measure**
  11. From Finite to Infinite: the Vapnik–Chervonenkis dimension (*the right invariant*)
  12. The Adversary's Best Tree: Littlestone dimension, online learning, and privacy (*the right invariant*)
  13. Every Regular Problem Is Locally Gaussian: Le Cam's theory of experiments (*localize to a limit*)
  14. Symmetry for Free: from permutation tests to conformal prediction (*exploit a free symmetry*)
- **Part IV. From Existence to Construction**
  15. If It Failed, You Could Compress Randomness: entropy compression (*encoding*)
  16. Proofs Are Algorithms: the sum-of-squares method (*proofs to algorithms*)
  17. Four Objects Are One: expanders, extractors, codes, and samplers (*unification*)
- **Epilogue.** A grammar of moves: when the sea rises, and when it cannot.

Each case follows the same anatomy: *Before*, *The move*, *After*, *Costs and limits* (including "Where did the difficulty go?"), *The move, distilled* (a move box and a four-row dissolution test), exercises (consolidation, transfer, research) and notes with references.

## Files

- `main.tex` is the master file; `frontmatter/` holds the preface.
- `chapters/NN-key.tex` and `chapters/NN-key.bib` hold one chapter and its bibliography.
- `tools/build_book.sh` and `tools/build_chapter.sh` are the build and check scripts.
- `STYLE.md` is the writing guide the chapters follow.
- `OUTLINE.md` is the original planning specification. Where it and a chapter differ, the chapter is correct; several outline details were fixed during writing and review.

## A note on accuracy

Every chapter's proofs were re-derived by at least one independent referee pass. Historical claims and references were checked against sources wherever these could be reached. A few details rest on secondary accounts. Recent preprints are cited as unrefereed claims, for example the September 2026 claims about Sidorenko's conjecture and the matroid secretary conjecture. Mistakes surely remain.
