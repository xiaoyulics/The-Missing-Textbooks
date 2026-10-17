# Deep Learning as an Experimental Science: A Laboratory Manual

*The Missing Textbooks, volume 4* (complete draft, 352 pp.; the independent referee pass of the earlier volumes is still to come). Written by Claude. Conceived and edited by Xiaoyu Li. See the colophon, "How This Book Was Made", at the end of the PDF.

Physicists learn error analysis and biologists learn controls before their experiments matter; deep learning researchers learn experimental method by word of mouth. This graduate textbook teaches it through the field's famous wrong conclusions. Each chapter opens with a conclusion that the field once believed, reconstructs the experiment behind it, reports what later work found, and names the methodological failure. It then teaches the method that would have caught it, reproduces the failure and its fix in a small experiment that runs on a laptop in minutes, and ends with a protocol for the next experiment of the same kind.

**[Read the PDF](Deep-Learning-as-an-Experimental-Science.pdf)** (352 pages).

## Contents

| | Chapter | The claim | Verdict |
|---|---|---|---|
| | **Prologue.** Ten runs of the same algorithm | | |
| | **Part I. Noise** | | |
| 1 | Seeds — the spread of a single method | One run of each method is enough to rank them | vanished |
| 2 | Instability — why a fixed seed does not fix a run | Fixing the seed makes a run reproducible | narrowed |
| 3 | Many Tasks, Few Runs — aggregates, intervals and multiplicity | A higher median score over a few runs per game identifies the better algorithm | narrowed |
| | **Part II. Fair Comparison** | | |
| 4 | Tuning Budgets — how an unequal search decides a comparison | Adaptive methods generalize worse than SGD | reversed |
| 5 | Controls — random search as the placebo arm | Architecture search beats random search | vanished |
| 6 | Confounding — the architecture and the recipe | Vision transformers beat convolutional networks | confounded |
| 7 | What Was Held Fixed — batch size, steps and the generalization gap | Large batches generalize worse because of sharp minima | confounded |
| | **Part III. Causes** | | |
| 8 | Ablations — interactions and the home-field advantage | A winning ticket needs its original initialization | narrowed |
| 9 | Mechanisms — intervening on the explanation | Batch normalization reduces internal covariate shift | misattributed |
| 10 | Correlation in a Sweep — flat minima and measures of generalization | Flat minima generalize better | narrowed |
| | **Part IV. Scale** | | |
| 11 | Fitting a Scaling Law — Chinchilla's third approach and its replication | L(N,D) = 1.69 + 406.4/N^0.34 + 410.7/D^0.28 | unreplicated |
| 12 | The Protocol Behind the Exponent — Kaplan, Chinchilla and the schedule | Model size should grow as C^0.73 | reversed |
| 13 | Extrapolation — predicting the large from the small | On these tasks, larger models do worse | narrowed |
| | **Part V. Measurement** | | |
| 14 | The Metric Makes the Phenomenon — emergent abilities | Abilities emerge abruptly and unpredictably | artifact |
| 15 | The Instrument's Settings — prompts, formats and demonstrations | Replacing demonstration labels with random ones barely hurts in-context learning | narrowed |
| | **Epilogue.** The laboratory notebook | | |
| A | A Statistical Toolkit | | |

Each chapter has the same anatomy: *The claim*, *The experiment* (a record of the original design), *The post-mortem* (ending with a verdict, stamped in the margin), *The method* (with at least one complete proof), *In the laboratory* (lab sheets and figures from real runs), *Protocol* (a checklist), exercises (consolidation, lab, design, critique, research), and notes that translate the notation of the sources.

## Design

The volume style `laboratory.sty` re-skins the series class as a laboratory notebook: vermilion rubrics for structure, blue-black ink for measurements, cream record cards, a millimetre ruler under each chapter title, lab sheets on engineering paper, verdicts stamped in the margin. The title page is log-log paper carrying a fitted scaling law and its extrapolation band; the foot of each chapter opening is millimetre paper on which a running mean of repeated measurements converges, one run per chapter. The numbers of both ornaments come from `figures/scripts/00-emblem.py`.

## Files

- `main.tex` is the master file (chapters are included when they exist); `laboratory.sty` is the volume style on top of `../template/missingbook.cls`.
- `frontmatter/` holds the preface and the notation; `backmatter/` the colophon.
- `chapters/NN-key.tex` and `chapters/NN-key.bib` hold one chapter and its bibliography.
- `figures/scripts/` holds the laboratory: `labkit.py` (the shared kit: data, models, seeded runs, statistics) and one script per experiment; `figures/data/` holds their output, which the chapters plot with pgfplots.
- `tools/build_book.sh` and `tools/build_chapter.sh` build and check the book and single chapters.
- `STYLE.md` is the writing guide and `OUTLINE.md` the plan. Where the plan and a chapter differ, the chapter is correct.

## Building and running

```bash
cd experimental-science
tools/build_book.sh                       # whole book -> Deep-Learning-as-an-Experimental-Science.pdf
tools/build_chapter.sh 04                 # one chapter, with error / reference / overfull / style checks
python3 figures/scripts/labkit.py         # self-test of the laboratory kit
python3 figures/scripts/00-prologue-split.py
```

The scripts need Python 3 with NumPy, SciPy, scikit-learn and PyTorch, and run on a laptop CPU.
