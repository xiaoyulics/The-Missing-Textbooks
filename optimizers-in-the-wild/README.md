# Optimizers in the Wild: Steepest Descent and the Dynamics of Training

《现代优化器的理论》 — *The Missing Textbooks, volume 3.* Written by Claude. Conceived and edited by Xiaoyu Li. See the colophon, "How This Book Was Made", at the end of the PDF.

Optimization textbooks end with convex analysis and with the theorem that stochastic gradient descent reaches a stationary point. In practice, what makes or breaks a training run is something else: Adam or Muon, the warmup, the weight decay, the learning-rate schedule, the batch size, and the fact that training runs at the edge of stability. This graduate textbook gives the theory of those decisions along two lines. The **unification line**: almost every optimizer is steepest descent under some norm, plus preconditioning and moving averages — SGD, signSGD, Adam, Lion, Shampoo and Muon are all instances, and the norm of a whole network fixes how learning rates scale with width. The **dynamics line**: sharpness self-stabilizes at the edge of stability, warmup is needed because early curvature is too high, gradient noise sets a floor that schedules remove (cosine, warmup–stable–decay and the river valley), weight decay drives normalized layers to an equilibrium, and Adam beats SGD on transformers for reasons that involve both lines.

**[Read the PDF](Optimizers-in-the-Wild.pdf)**.

## Contents

- **Prologue.** What the textbook proves and what the practitioner decides; the two lines; gradient descent and sign descent on one quadratic.
- **Part I. One Rule: Steepest Descent**
  1. The Geometry of a Step: steepest descent in a normed space
  2. Sign Descent: the ℓ∞ geometry of Adam, signSGD and Lion
  3. Orthogonalized Updates: the spectral norm, Muon and Newton–Schulz
  4. Norms for Networks: the modular norm, the spectral condition and µP
- **Part II. Memory: Averages and Preconditioners**
  5. Moving Averages: momentum, noise, and why normalized steps need memory
  6. Adaptive Preconditioning: AdaGrad, RMSProp and Adam
  7. Kronecker Preconditioners: Shampoo, K-FAC and SOAP
- **Part III. Stability**
  8. The Edge of Stability: thresholds, self-stabilization and central flows
  9. Warmup: why the step size must start small
  10. Clipping and Spikes: generalized smoothness and the anatomy of an instability
- **Part IV. Noise, Schedules and Equilibria**
  11. Noise and Batch Size: the noisy quadratic model and the critical batch size
  12. Schedules: cosine, warmup–stable–decay and the river valley
  13. Weight Decay: Frank–Wolfe, rotational equilibrium and the effective learning rate
- **Part V. Choosing an Optimizer**
  14. Why Adam Beats SGD on Transformers: heterogeneous curvature, heavy-tailed classes and the batch-size regime
  15. Where Steepest Descent Ends: implicit bias and the margin of a norm
- **Epilogue.** What theory owes practice: a ledger of verdicts and open problems.
- **Appendix A.** A toolkit: duality, matrices, recurrences and online learning.

Each chapter has the same anatomy: *In the wild* (a practice, stated with its typical values and sources, and the evidence about it), *The idealization* (with a dictionary from theory to training run, and field-guide entries for the optimizers it derives), *The theory* (a main theorem with a complete proof), *Reading the theory* (predictions against simulations), *What the theory misses* (what was idealized away, where it breaks, and the repairs), *Back in the wild* (field notes: practice, idealization, proved, predicts, misses, takeaway), exercises (consolidation, variation, simulation, research), and notes that translate the notation of the source papers into the book's.

## Files

- `main.tex` is the master file; `wild.sty` is the volume style (title emblem, practice boxes, field-guide entries, field notes, dictionaries, heuristic derivations, plot styles, notation macros) loaded on top of the series class `../template/missingbook.cls`.
- `frontmatter/` holds the preface and the notation page; `backmatter/` the colophon.
- `chapters/NN-key.tex` and `chapters/NN-key.bib` hold one chapter and its bibliography.
- `figures/scripts/` holds the Python scripts (NumPy, SciPy, PyTorch) that produce every numerical figure; `figures/data/` holds their output, which the chapters plot with pgfplots.
- `tools/build_book.sh` and `tools/build_chapter.sh` build and check the book and single chapters; `tools/lint.py` checks the style rules.
- `STYLE.md` is the writing guide and `OUTLINE.md` the original plan. Where the plan and a chapter differ, the chapter is correct.

## Building

```bash
cd optimizers-in-the-wild
tools/build_book.sh                 # whole book -> Optimizers-in-the-Wild.pdf
tools/build_chapter.sh 08           # one chapter, with error / reference / overfull / style checks
python3 figures/scripts/12-schedules-river.py   # regenerate one figure's data (names vary by chapter)
```

The series class loads the `sourcesans` font package. TeX Live distributions that ship only the older `sourcesanspro` package (for example Debian's TeX Live 2023) need a two-line local wrapper `sourcesans.sty` that loads `sourcesanspro`; see the series README.
