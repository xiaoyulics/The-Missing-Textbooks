# Deep Learning as an Experimental Science — Book Plan and Chapter Specifications

*The Missing Textbooks, volume 4.* Working title: **Deep Learning as an Experimental Science: A
Laboratory Manual** (Chinese working title 深度学习实验室手册). Written by Claude; conceived and
edited by Xiaoyu Li, from the editor's proposal "ML-1".

## The thesis

Most of what the field of deep learning knows, it knows by experiment: which optimizer to use, how to
allocate compute between parameters and data, which architectural changes matter, what large models can
do. Theorems are rare and usually arrive after the fact. Deep learning is, in practice, an experimental
science, and a young one.

Older experimental sciences teach their method. A physics student takes a laboratory course and learns
error analysis before running anything that matters; a biology student learns controls, blinding and
replication; agricultural and industrial statisticians built the design of experiments (Fisher, Box)
around the problem of drawing conclusions from noisy, expensive runs. Deep learning researchers learn
experimental method by word of mouth: how much seed variance is enough to matter, how an unequal tuning
budget can flip a comparison, how to fit a scaling law and how far to trust its extrapolation, how to
design an ablation that does not confound. The field has written down pieces of this in reality-check
papers, but nobody has written the laboratory manual.

This book is that manual. **Each chapter opens with a famous conclusion that turned out to be wrong, or
much narrower than claimed, and performs its post-mortem**: it reconstructs the original experiment,
shows what later work found, names the methodological failure precisely, teaches the method that would
have caught it (with mathematics where mathematics helps), reproduces the failure and the fix in a small
laboratory experiment that runs on a laptop in minutes, and ends with a protocol: a checklist for the
next experiment of the same kind.

The post-mortems are **blameless** in the engineering sense: the aim is the mechanism of the failure, not
the fault of the people. Most of the original papers were careful by the standards of their time; several
of their authors wrote or joined the post-mortems. The book states each claim as its authors stated it,
separates it from the claim as the field received it, and credits what the original work got right.

## Positioning (verified 2026-10-17)

- Moritz Hardt, *The Emerging Science of Machine Learning Benchmarks* (Princeton University Press, US
  publication 6 October 2026). It argues that model rankings, not absolute evaluations, are the scientific
  product of benchmarks, and treats the science of benchmarks and rankings. It does not teach how to run
  and analyze training experiments. This book cites it for benchmark questions and stays with training
  experiments: seeds, budgets, controls, ablations, mechanisms, scaling fits, metrics.
- Google Research, *Deep Learning Tuning Playbook* (Godbole, Dahl, Gilmer, Shallue, Nado; version 1.0,
  2023; github.com/google-research/tuning_playbook). Practical guidance on hyperparameter tuning, written
  as advice rather than as a course in method.
- Paul R. Cohen, *Empirical Methods for Artificial Intelligence* (MIT Press, 1995). The closest ancestor:
  experiment design, hypothesis testing, resampling and performance assessment for AI programs, written
  before deep learning. Its preface makes the same complaint this book makes: computer science has no
  research-methods curriculum of the kind other sciences have.
- Classic method texts from other sciences that the book draws on: Taylor, *An Introduction to Error
  Analysis*; Box, Hunter and Hunter, *Statistics for Experimenters*; Montgomery, *Design and Analysis of
  Experiments*; Efron and Tibshirani, *An Introduction to the Bootstrap*.
- Papers and tutorials: Colas, Sigaud and Oudeyer (2018), "How Many Random Seeds?"; Agarwal et al. (2021),
  "Deep Reinforcement Learning at the Edge of the Statistical Precipice"; Bouthillier et al. (2021),
  "Accounting for Variance in Machine Learning Benchmarks"; Dodge et al. (2019), "Show Your Work"; Lipton
  and Steinhardt (2018), "Troubling Trends in Machine Learning Scholarship"; Sculley et al. (2018),
  "Winner's Curse?". These are the raw material; none is a textbook.

## Verdicts

Every chapter's opening states the claim and a one-line verdict, and its post-mortem ends with a
`verdict` box whose word is stamped in the margin. The vocabulary is fixed:

| Verdict | Meaning |
|---|---|
| **reversed** | under a fair protocol the comparison or conclusion flips |
| **vanished** | the effect is within noise, or disappears under a fair protocol |
| **confounded** | the effect is real but belongs to another variable that changed with the treatment |
| **artifact** | the effect is produced by the measurement (metric, fit, evaluation procedure) |
| **narrowed** | the claim holds only in a smaller regime than claimed |
| **misattributed** | the effect is real; the explanation offered for it is not |
| **unreplicated** | an independent reanalysis or rerun does not reproduce the reported result |
| **open** | the evidence does not yet settle the question |

A chapter may qualify its verdict in the box text ("narrowed, and in part reversed"), but the stamp
carries one word.

## Structure, keys, and files

| # | key | file | Title — Subtitle | Verdict |
|---|-----|------|------------------|---------|
| – | prologue | 00-prologue.tex | Prologue — Ten runs of the same algorithm | (special chapter) |
| | | | **Part I. Noise** | |
| 1 | seeds | 01-seeds.tex | Seeds — The spread of a single method | vanished |
| 2 | instability | 02-instability.tex | Instability — Why a fixed seed does not fix a run | narrowed |
| 3 | aggregates | 03-aggregates.tex | Many Tasks, Few Runs — Aggregates, intervals and multiplicity | narrowed |
| | | | **Part II. Fair Comparison** | |
| 4 | budgets | 04-budgets.tex | Tuning Budgets — How an unequal search decides a comparison | reversed |
| 5 | controls | 05-controls.tex | Controls — Random search as the placebo arm | vanished |
| 6 | confounding | 06-confounding.tex | Confounding — The architecture and the recipe | confounded |
| 7 | heldfixed | 07-heldfixed.tex | What Was Held Fixed — Batch size, steps and the generalization gap | confounded |
| | | | **Part III. Causes** | |
| 8 | ablations | 08-ablations.tex | Ablations — Interactions and the home-field advantage | narrowed |
| 9 | mechanisms | 09-mechanisms.tex | Mechanisms — Intervening on the explanation | misattributed |
| 10 | sharpness | 10-sharpness.tex | Correlation in a Sweep — Flat minima and the measures of generalization | narrowed |
| | | | **Part IV. Scale** | |
| 11 | fitting | 11-fitting.tex | Fitting a Scaling Law — Chinchilla's third approach and its replication | unreplicated |
| 12 | protocol | 12-protocol.tex | The Protocol Behind the Exponent — Kaplan, Chinchilla and the learning-rate schedule | reversed |
| 13 | extrapolation | 13-extrapolation.tex | Extrapolation — Predicting the large from the small | narrowed |
| | | | **Part V. Measurement** | |
| 14 | emergence | 14-emergence.tex | The Metric Makes the Phenomenon — Emergent abilities and the smooth change beneath them | artifact |
| 15 | instruments | 15-instruments.tex | The Instrument's Settings — Prompts, formats and demonstrations as nuisance factors | narrowed |
| – | epilogue | 16-epilogue.tex | Epilogue — The laboratory notebook | (special chapter) |
| A | toolkit | 17-toolkit.tex | A Statistical Toolkit (appendix) | |

Files: `chapters/NN-key.tex` and `chapters/NN-key.bib`; scripts `figures/scripts/NN-key-name.py`; data
`figures/data/NN-key-name.dat`. Chapter labels `ch:key`. The opening of each chapter is

```
\labchapter{Title}{Subtitle}{The claim}{Its source}{The verdict, one sentence}{ch:key}
```

The first line of each chapter specification below gives proposed values for these arguments. The claim
must be verified against the source before it is printed, and may be reworded to be accurate.

## Global notation

All chapters use this notation. A chapter that needs a clashing local symbol keeps the global meaning
and chooses another symbol; a source paper's notation is translated in "Translating the literature".

**Runs and statistics.**
- A *run* is one training of one network; its *outcome* is a scalar $Y$ (test error, accuracy, loss).
  Runs are indexed by $i$ or $j$; the number of runs (replicates, seeds) per arm is $n$.
- Two arms of a comparison are indexed 1 and 2 (in prose: "method A" and "method B", "treatment" and
  "control" or "baseline"). Population means $\mu_1,\mu_2$, standard deviations $\sigma_1,\sigma_2$ (or
  $\sigma$), effect $\Delta=\mu_1-\mu_2$.
- Sample mean $\bar Y$, sample standard deviation $S$ (divisor $n-1$), standard error
  $\SE=S/\sqrt n$ (`\SE`), standard deviation operator `\SD`. Probability $\Prob$ (`\Prob`),
  expectation $\E$, variance $\Var$, covariance $\Cov$, correlation $\Corr$ (`\Corr`).
- $\Phi$ is the standard normal distribution function, $z_q$ its $q$-quantile; $t_{\nu,q}$ is the
  $q$-quantile of Student's $t$ with $\nu$ degrees of freedom. Significance level $\alpha$; confidence
  level $1-\alpha$, usually 95\%. Power is written in words ("power $0.8$"), never as $1-\beta$.
- Order statistics $Y_{(1)}\le\dots\le Y_{(n)}$. The best of $k$ draws is $M_k$; $e_k=\E[\max$ of $k$
  standard normals$]$.
- Rank correlation: Kendall's $\tau$. Interquartile mean `\IQM`, median `\med`.
- Asymptotics: $a\lesssim b$, $a\asymp b$, $O,\Omega,\Theta,o$ as in volume 2; say what is fixed.

**Training.**
- Parameters $\theta$; loss $\Loss(\theta)$ (population, `\Loss`) and $\Lhat(\theta)$ (training, `\Lhat`).
- Learning rate $\eta$; batch size $B$; number of optimizer steps $T$; epochs in words; weight decay
  $\lambda$; momentum and Adam parameters $\beta_1,\beta_2,\epsilon$ (Adam's $\epsilon$ is written
  `\epsilon`; the class's `\eps` is reserved for small quantities in proofs).
- Hyperparameter configuration $h$, search space $\cH$; a tuning budget of $k$ trials.
- Hessian $\nabla^2\Lhat(\theta)$, top eigenvalue $\lambda_{\max}$ (always subscripted).

**Scale.**
- Model size $N$ (parameters), data $D$ (training tokens or examples seen), compute $C$ (FLOPs;
  $C\approx 6ND$ for dense networks).
- The parametric law is written
  $\Loss(N,D)=E+A_N N^{-\alpha_N}+A_D D^{-\alpha_D}$, with irreducible loss $E$. Hoffmann et al.'s
  $(A,B,\alpha,\beta)$ are our $(A_N,A_D,\alpha_N,\alpha_D)$; Kaplan et al.'s $\alpha_N,\alpha_D$ keep
  their names. Compute-optimal allocation $N_{\mathrm{opt}}\propto C^{a}$, $D_{\mathrm{opt}}\propto C^{b}$.

**Evaluation.**
- Number of test items $m$; accuracy estimate $\hat p$; number of tasks in a suite $J$, task index $j$.

## Laboratory conventions

- Every lab runs on a laptop CPU with numpy, scipy, scikit-learn and torch, using the shared kit
  `figures/scripts/labkit.py` (read its docstring). Each script runs in at most about five minutes with
  two threads; a chapter's scripts together in at most about fifteen.
- Built-in data: `digits` (scikit-learn's 8x8 digits, 1797 images, no download; a 20-epoch MLP run takes
  about 0.1 s) and `teacher` (a synthetic regression task whose test loss falls smoothly with model width
  and data over about two decades each, with irreducible loss 0.01; a run of 2000 steps takes about 0.3 s).
  `labkit.run(cfg, init, order, split, noise)` separates the four sources of randomness.
- For scaling labs, the "online" protocol mirrors language-model pretraining: `Config(data="teacher",
  train_size=D, epochs=1, schedule="cosine")` trains for one pass over $D$ fresh examples with a schedule
  matched to the horizon. Model size $N$ is the parameter count returned by `run`.
- A lab may define what `labkit` lacks (a tiny CNN, a pruning routine, noise injection after batch
  normalization) in its own script. It must not modify `labkit.py`; if a shared feature is needed, the
  chapter report says so.
- The miniature is honest: if the laptop experiment does not reproduce the famous effect, the chapter
  says so, explains why, and adjusts the setting only in ways it reports. A miniature that fails to
  reproduce is a finding.

## Chapter specifications

Each specification gives the opening, the story, the method with its main results and proof plans, the
lab, the protocol, exercise ideas, cross-links and key references. "Verify" marks a fact that the chapter
author must check against the source before printing. The author may improve every proposal here; where
the chapter and the plan differ, the chapter is correct.

---

### Prologue — Ten runs of the same algorithm (key: prologue; special chapter)

Written by the coordinating author. Opens with Henderson et al. (2018): the same reinforcement-learning
algorithm (TRPO on HalfCheetah), ten seeds split into two groups of five, two mean learning curves that a
t-test would call different. Then: why deep learning is an experimental science; what older sciences
teach their students (error analysis, controls, replication, design of experiments) and what deep
learning does not; the anatomy of a training experiment (the run as the experimental unit; treatment,
factors and levels; nuisance variables; held-fixed variables; response; budget); the four questions
that organize the book (How much does a number move when nothing changes? Is the comparison fair? Is the
cause identified? Does the result extend beyond the scales and settings measured?) plus the fifth, about
what the metric measures; the anatomy of a chapter and the verdict vocabulary; how to use the labs; the
blameless post-mortem; the ornaments (the running mean at the foot of each opening page).

---

### Part I. Noise

*A training run is a random experiment. Before two methods can be compared, the experimenter has to know
how far one method moves when nothing but its seed changes, and how far a summary over many tasks moves
when nothing but the runs change.*

---

### Chapter 1. Seeds — The spread of a single method (key: seeds)

`\labchapter{Seeds}{The spread of a single method}{One run per method is enough to rank them.}{Common practice; audited for sequence tagging by Reimers and Gurevych (2017) and for BERT fine-tuning by Dodge et al.\ (2020)}{Vanished: changing only the seed moves scores by as much as the gaps between published systems.}{ch:seeds}`

**The claim.** Leaderboard practice in 2015–2019 compared single runs, or the best of a few restarts.
Two documented cases: sequence tagging on CoNLL-2003 named-entity recognition, where successive published
systems differed by tenths of an F1 point (verify the published numbers used); and BERT fine-tuning on
small GLUE tasks, where Devlin et al. (2019) report restarting fine-tuning of BERT-Large several times
and keeping the best model on the development set (verify the wording and paraphrase it).

**The experiment.** An `experiment` record of a typical single-run comparison: treatment (architecture),
response (test F1), replicates (one), selection (best dev run).

**The post-mortem.** Reimers and Gurevych (2017): the same BiLSTM-CNN-CRF trained with different seeds
spreads by more than the published gaps, and the seed alone produces "significant" differences between
runs of one system (verify their numbers). Dodge et al. (2020): for BERT fine-tuning, weight
initialization and data order contribute comparably to variance; many runs fail outright on small tasks;
the best of many seeds beats previously reported results for BERT and approaches later models (verify).
Picard (2021): seed-to-seed range on CIFAR-10 and ImageNet (verify: about 1.8 and 0.5 points).
Bouthillier et al. (2021): the data split is a larger source of variance than initialization, and
randomizing only the seed underestimates it. Mosbach et al. (2021) and Zhang et al. (2021): failed BERT
runs come from optimization (no bias correction in BERT's Adam, too few steps), not from small data — a
mechanism finding, cross-referenced to `\cref{ch:mechanisms}`. Verdict box: **vanished**.

**The method.**
- The run distribution; sources of randomness (initialization, data order, augmentation, dropout, data
  split, nondeterministic hardware: `\cref{ch:instability}`).
- Point and interval estimates: mean, standard deviation, standard error, $t$-interval, bootstrap
  interval (toolkit). The two noise sources of a measured accuracy: the run and the finite test set
  ($\hat p$ has standard error $\sqrt{p(1-p)/m}$); paired evaluation on the same test items.
- Variance components: a crossed design (initialization seed $\times$ order seed) and the two-way random
  effects model $Y_{ij}=\mu+a_i+b_j+e_{ij}$.
- Best-of-$k$ reporting and its bias. Failed runs, bimodality, heavy tails; reporting all runs.
- How many seeds: the two-sample power calculation.

*Main results.*
1. **Proposition (best of $k$).** If $Y_1,\dots,Y_k\iid\cN(\mu,\sigma^2)$ then
   $\E[\max_i Y_i]=\mu+\sigma e_k$ and $e_k\le\sqrt{2\ln k}$. *Proof plan:* scale and shift; Jensen on
   $e^{t\max}$, bound by the sum of moment generating functions, optimize $t=\sqrt{2\ln k}$.
   Table of $e_k$ for $k=2,\dots,10$ (e.g. $e_2=1/\sqrt\pi\approx0.564$, $e_5\approx1.163$,
   $e_{10}\approx1.539$; verify numerically).
2. **Proposition (two runs of one method).** For two independent runs of the same method,
   $\Prob(\abs{Y_1-Y_2}>\delta)=2\Phi(-\delta/(\sigma\sqrt2))$ under normality; with $\sigma$ equal to
   the published gap, two runs disagree by more than the gap about half the time.
3. **Theorem (variance components, complete proof).** In the balanced crossed design with $I$
   initialization seeds and $J$ order seeds, one run per cell, and the additive random-effects model,
   the ANOVA estimators $\hat\sigma_a^2=(\mathrm{MS}_A-\mathrm{MS}_E)/J$,
   $\hat\sigma_b^2=(\mathrm{MS}_B-\mathrm{MS}_E)/I$, $\hat\sigma_e^2=\mathrm{MS}_E$ are unbiased.
   *Proof plan:* compute the expected mean squares by expanding the sums of squares; the cross terms
   vanish by independence. Say what is lost when the model has an interaction (it is confounded with
   $e$ with one run per cell).
4. **Corollary (number of seeds).** For a two-sided level-$\alpha$ comparison with power $0.8$,
   $n\approx2\sigma^2(z_{1-\alpha/2}+z_{0.8})^2/\Delta^2$ per arm; paired seeds replace $2\sigma^2$ by
   $\Var(Y_1-Y_2)$.

**In the laboratory.**
- Lab 1.1 (`01-seeds-spread`): 200 runs of the digits MLP with all seeds varied; the distribution of test
  error; the running mean with its $t$-interval as runs accumulate (this is the ornament at the foot of
  every chapter opening, now with real data).
- Lab 1.2 (`01-seeds-components`): a crossed $20\times20$ design (initialization $\times$ order) plus
  split seeds; estimate the variance components; compare with the test-set binomial noise.
- Lab 1.3 (`01-seeds-bestof`): the inflation of best-of-$k$ reporting against the prediction
  $\mu+\sigma e_k$; with label noise (e.g. 20\%) to make the effect larger.

**Protocol.** "Measuring one method": decide which sources to randomize; fix $n$ before looking;
report mean, standard deviation, interval and every run; never report a best-of-$k$ number without $k$;
use paired seeds and the same test items when comparing.

**Exercises.** Compute $e_k$ by numerical integration; prove $e_k\sim\sqrt{2\ln k}$; the expected
maximum under a uniform or heavy-tailed run distribution; seeds needed to detect a 0.3-point gain at a
seed standard deviation of 0.2; the lab with label noise; design a variance-components study for a
fine-tuning pipeline; critique a paper that reports its method's best of five seeds against baselines'
published single numbers; research: interval estimates for bimodal run distributions.

**Cross-links.** `\cref{ch:instability}` (where variance comes from), `\cref{ch:aggregates}` (many
tasks), `\cref{ch:budgets}` (the best of $k$ hyperparameter trials is the same order statistic).

**Key references.** Reimers & Gurevych 2017 (EMNLP); Dodge et al. 2020 (arXiv 2002.06305); Devlin et al.
2019 (NAACL); Picard 2021 (arXiv 2109.08203); Bouthillier et al. 2021 (MLSys); Bouthillier, Laurent &
Vincent 2019 (ICML); Mosbach, Andriushchenko & Klakow 2021 (ICLR); Zhang et al. 2021 (ICLR, "Revisiting
few-sample BERT fine-tuning"); Colas, Sigaud & Oudeyer 2018 (arXiv 1806.08295); Henderson et al. 2018
(AAAI); Madhyastha & Jain 2019 (CoNLL); Gundersen et al. 2022 (sources of irreproducibility).

---

### Chapter 2. Instability — Why a fixed seed does not fix a run (key: instability)

`\labchapter{Instability}{Why a fixed seed does not fix a run}{Fixing the random seed makes a training run reproducible.}{Common practice, and the advice of framework documentation}{Narrowed: bitwise repetition needs deterministic kernels, and one changed bit still gives a new draw from the run distribution.}{ch:instability}`

**The claim.** Setting every seed is the standard advice for reproducibility. On parallel hardware
nondeterministic kernels (atomic additions, autotuned convolution algorithms) break bitwise repetition;
and where repetition holds, it is fragile.

**The post-mortem.** Summers & Dinneen (2021, ICML): each source of nondeterminism (initialization,
shuffling, augmentation, nondeterministic GPU kernels) produces variability of similar size, and a
change of a single bit in the initial weights produces about as much as all of them together; they call
this instability (verify the exact claims and settings). Zhuang et al. (2022, MLSys): tooling alone
(nondeterministic GPU operations) produces variance comparable to seeds; deterministic execution has a
cost in time (verify the numbers). Pham et al. (2020, ASE): measured variance from nondeterminism and a
survey of practitioners' awareness (verify). Fort et al. (2020) and Frankle et al. (2020): the early
phase of training is chaotic; later, runs from a shared checkpoint stay close ("stability to SGD noise",
cross-reference `\cref{ch:ablations}`). Verdict box: **narrowed**.

**The method.**
- Three levels of reproducibility: bitwise (same numbers), distributional (a rerun is a draw from the same
  run distribution), conclusion-level (the same comparison comes out the same). What a fixed seed buys:
  debugging and auditability, not evidence.
- Floating-point arithmetic: non-associative addition; reductions whose order depends on scheduling.
- Training as a dynamical system: sensitive dependence on initial conditions; a perturbation of one ulp
  grows to the size of seed-to-seed differences in a number of steps that depends logarithmically on its
  size.
- Testing distributional reproducibility: two-sample tests on run outcomes (permutation test,
  Kolmogorov–Smirnov; toolkit).
- What to record: seeds, library versions, hardware, determinism flags, data order.

*Main results.*
1. **Proposition (error of a reordered sum, complete proof).** Recursive summation of $x_1,\dots,x_n$ in
   floating point with unit roundoff $u$ has error at most $(n-1)u\sum_i\abs{x_i}+O(u^2)$, and two orders
   can differ by twice that (Higham 2002, Ch.~4). *Proof plan:* induction on the partial sums with the
   standard model $\mathrm{fl}(a+b)=(a+b)(1+\delta)$, $\abs\delta\le u$.
2. **Proposition (growth of a perturbation under gradient descent, complete proof).** For gradient
   descent on $f(\theta)=\tfrac12\theta^\top H\theta$ with step $\eta$, two runs started $\delta$ apart
   differ by $(I-\eta H)^t\delta$; the difference grows like $\abs{1-\eta\lambda}^t$ along each
   eigendirection with $\abs{1-\eta\lambda}>1$ (a negative eigenvalue at a saddle, or a step above the
   stability threshold $2/\lambda$). A perturbation of relative size $10^{-7}$ (one bit in single
   precision) reaches order one after about $16/\ln\abs{1-\eta\lambda}$ steps.
3. **Heuristic derivation.** For a nonlinear network, a positive finite-time Lyapunov exponent
   $\ell$ gives a divergence time $t^\star\approx\ln(\Delta/\delta)/\ell$; after $t^\star$ the twin runs
   are as different as runs with different seeds.

**In the laboratory.**
- Lab 2.1 (`02-instability-twins`): twin runs of the digits MLP that differ by one bit in one initial
  weight (`torch.nextafter`), or only in the order of examples inside each minibatch (the mean gradient
  is mathematically identical); the distance between their parameter vectors over training (log scale:
  growth, then saturation).
- Lab 2.2 (`02-instability-distribution`): final test errors of 100 one-bit twins against 100 new
  seeds; a two-sample test; the two distributions agree.

**Protocol.** "Making a run repeatable and an experiment reproducible".

**Exercises.** Compute a reordered sum that changes the last bit; the growth rate for a quadratic with
a negative eigenvalue; repeat Lab 2.1 with a smaller learning rate (does divergence slow?); design a test
for distributional reproducibility across two machines; critique "we fixed all seeds, so our results
are reproducible"; research: when does a training run forget its initial perturbation?

**Key references.** Summers & Dinneen 2021 (ICML); Zhuang, Zhang, Song & Hooker 2022 (MLSys); Pham et al.
2020 (ASE); Nagarajan, Warnell & Stone 2019 (deterministic implementations in deep RL; verify venue);
Higham 2002 (*Accuracy and Stability of Numerical Algorithms*); Goldberg 1991; Fort et al. 2020
(NeurIPS); Frankle et al. 2020 (ICML); Gundersen & Kjensmo 2018 (AAAI); Pineau et al. 2021 (JMLR,
NeurIPS 2019 reproducibility program); PyTorch reproducibility documentation.

---

### Chapter 3. Many Tasks, Few Runs — Aggregates, intervals and multiplicity (key: aggregates)

`\labchapter{Many Tasks, Few Runs}{Aggregates, intervals and multiplicity}{Algorithm X is the new state of the art on Atari 100k.}{Point estimates over a few runs per game, 2019--2021, as audited by Agarwal et al.\ (2021)}{Narrowed: with interval estimates over many runs, several claimed improvements shrink or reverse.}{ch:aggregates}`

**The claim.** Sample-efficient Atari (26 games, 100k interactions) papers reported the median
human-normalized score over 3–10 runs and claimed state of the art by its point estimate.

**The post-mortem.** Agarwal, Schwarzer, Castro, Courville & Bellemare (2021, NeurIPS outstanding paper):
with 100 runs per algorithm, the uncertainty of the median from few runs is large; interval estimates
overlap; conclusions drawn from point estimates differ from those of a careful analysis; the evaluation
protocols of some papers differed (verify the specific findings, e.g. about DER, OTRainbow, DrQ and
DrQ($\epsilon$), SPR, CURL). Their recommendations: stratified bootstrap intervals, the interquartile
mean, performance profiles, probability of improvement. Also Henderson et al. (2018), Colas et al.
(2018, 2019), Jordan et al. (2020), and Demšar (2006) for comparisons of classifiers over many datasets.
Verdict box: **narrowed**.

**The method.**
- Two levels of variation: across tasks and across runs within a task. What an aggregate estimates.
- Aggregates: mean (outlier-sensitive), median (high variance with few tasks), interquartile mean,
  optimality gap; performance profiles (score distributions); probability of improvement
  $\Prob(X>Y)$ and its Mann–Whitney estimate.
- Stratified bootstrap: resample runs within each task.
- Multiplicity: family-wise error (Bonferroni, Holm), false discovery rate (Benjamini–Hochberg); Demšar's
  Friedman test and post-hoc procedures for many datasets; the garden of forking paths (Gelman & Loken):
  the choice of aggregate, of tasks and of runs as forking paths.

*Main results.*
1. **Theorem (Holm's procedure, complete proof).** Holm's step-down procedure controls the family-wise
   error rate at $\alpha$ under arbitrary dependence. *Proof plan:* let $I_0$ be the true nulls,
   $m_0=\abs{I_0}$; on the event that every true-null $p$-value exceeds $\alpha/m_0$ (probability at
   least $1-\alpha$ by the union bound) the procedure stops before rejecting any true null.
2. **Proposition.** With $m$ independent true-null comparisons at level $\alpha$, the expected number of
   false "wins" is $m\alpha$ and the probability of at least one is $1-(1-\alpha)^m$.
3. **Proposition (sketch allowed).** Asymptotic variances of the mean, the median and the
   interquartile mean under a location family; why the IQM is a good compromise with few runs.

**In the laboratory.**
- Lab 3.1 (`03-aggregates-suite`): a suite of about 12 tasks built from digits (for instance subsets of
  classes, training-set sizes and label-noise levels), two methods (e.g. two optimizers at default
  settings, or with and without augmentation); 3 versus 50 runs per task; mean, median and IQM with
  stratified bootstrap intervals.
- Lab 3.2 (`03-aggregates-replications`): re-run the whole three-runs-per-task "paper" 200 times with
  fresh seeds (reuse a bank of runs to keep the cost low) and count how often the winner by median
  changes; compare with the IQM.

**Protocol.** "Reporting results over a suite of tasks".

**Exercises.** Prove the Bonferroni bound; Benjamini–Hochberg on a list of $p$-values; the median's
variance for normal scores; the lab with IQM vs median; design an evaluation for 26 tasks with a fixed
compute budget; critique a table that bolds the best mean in every column; research: aggregates that are
robust to task selection.

**Key references.** Agarwal et al. 2021 (NeurIPS); Henderson et al. 2018; Colas, Sigaud & Oudeyer 2018
and 2019; Jordan et al. 2020 (ICML, "Evaluating the performance of reinforcement learning algorithms");
Demšar 2006 (JMLR); Holm 1979; Benjamini & Hochberg 1995; Gelman & Loken 2013/2014; Dehghani et al.
2021 ("The benchmark lottery"); Kaiser et al. 2020 (Atari 100k, SimPLe).

---

### Part II. Fair Comparison

*A comparison of two methods is a comparison of two procedures, each with its tuning, its pipeline and its
budget. An experiment is fair when the arms differ only in what the claim is about.*

---

### Chapter 4. Tuning Budgets — How an unequal search decides a comparison (key: budgets)

`\labchapter{Tuning Budgets}{How an unequal search decides a comparison}{Adaptive gradient methods generalize worse than SGD.}{The received reading of Wilson et al.\ (2017)}{Reversed: when every optimizer is tuned over all its hyperparameters with the same budget, adaptive methods match or beat SGD.}{ch:budgets}`

**The claim.** Wilson, Roelofs, Stern, Srebro & Recht (2017, NeurIPS): on several workloads (CIFAR-10
with VGG, character-level language modeling, parsing; verify the list) adaptive methods (AdaGrad,
RMSProp, Adam) reached worse test performance than SGD with or without momentum, even when they reached
lower training loss; plus a constructed linearly separable problem on which adaptive methods generalize
worse. The paper tuned the initial learning rate (and decay scheme) on a grid; the received reading
became "adaptive methods generalize worse".

**The post-mortem.** Choi, Shallue, Nado, Lee, Maddison & Dahl (2019, arXiv 1910.05446): inclusion
relationships among optimizers (e.g. Adam with large $\epsilon$ approximates momentum), so a fully tuned
Adam can do anything momentum can; when all hyperparameters including $\epsilon$ are tuned, adaptive
methods never underperform momentum or SGD; rankings depend on the tuning protocol (verify). Sivaprasad
et al. (2020, ICML): benchmarking that accounts for tuning cost favors Adam (verify). Schmidt, Schneider &
Hennig (2021, ICML): fifteen optimizers, no clear winner, default Adam competitive. Nado et al. (2021):
LARS and LAMB versus tuned standard optimizers at large batch (verify). Wen, Hall, Ma & Liang (ICLR
2026, arXiv 2509.02046): with fair tuning, claimed 1.4–2x speedups over AdamW shrink, to about 1.1x at
1.2B parameters; intermediate checkpoints mislead. Verdict box: **reversed**.

**The method.**
- A method is an algorithm plus a search space plus a budget plus a protocol.
- The budget curve $k\mapsto\E[M_k]$; why curves cross; the inclusion argument and its limits at finite
  budget; equal budgets in trials or in compute; time-to-result (AlgoPerf: Dahl et al. 2023, Kasimbeg et
  al. 2025).
- Estimating budget curves from $N\ge k$ trials: the unbiased estimator (Dodge et al. 2019 used a
  with-replacement estimator; Tang et al. 2020 discussed its bias — verify).

*Main results.*
1. **Theorem (unbiased expected best, complete proof).** For $N$ i.i.d. trial scores sorted increasingly,
   $\hat M_k=\sum_{i=1}^N\binom{i-1}{k-1}\binom Nk^{-1}Y_{(i)}$ is the average of the maximum over all
   $k$-subsets and is unbiased for $\E[M_k]$. *Proof plan:* symmetry of i.i.d. samples; counting the
   subsets in which $Y_{(i)}$ is the maximum. (`labkit.expected_best` implements it.)
2. **Proposition (budget curves cross, complete proof).** For random search with score distributions
   $F_1,F_2$, $\E[M_k^{(1)}]-\E[M_k^{(2)}]=\int\bigl(F_2(x)^k-F_1(x)^k\bigr)\dd x$ (scores to be
   maximized). If space 1 contains space 2 diluted by a region of poor configurations, then arm 2 wins at
   small $k$ and arm 1 at large $k$; give an explicit two-point example.
3. **Corollary.** Comparing two methods at their single best configurations found with unequal $k$
   compares the budgets.

**In the laboratory.**
- Lab 4.1 (`04-budgets-protocols`): momentum versus Adam on digits (choose a setting where the
  generalization difference is visible, e.g. small training set, label noise or longer training; report
  the choice): (a) the 2017 protocol, learning rate on a grid with $\epsilon=10^{-8}$; (b) the 2019
  protocol, random search over learning rate, $\epsilon$, $\beta_1$, $1-\beta_2$ and weight decay, with
  the same number of trials.
- Lab 4.2 (`04-budgets-curves`): budget curves with intervals from the unbiased estimator and a
  bootstrap; where they cross.

**Protocol.** "Comparing two training methods".

**Exercises.** Prove monotonicity and concavity-type properties of $k\mapsto\E[M_k]$ for a given $F$;
show Adam with $\epsilon\to\infty$ and learning rate $\eta\epsilon$ tends to momentum SGD; compute the
budget curve of a uniform distribution; lab variant with $\epsilon$ fixed; design an equal-compute
comparison when one method's trials are twice as expensive; critique "we used the hyperparameters
recommended by the authors for each method"; research: budget-aware comparison when trials are chosen
adaptively (Bayesian optimization).

**Key references.** Wilson et al. 2017; Choi et al. 2019; Sivaprasad et al. 2020; Schmidt, Schneider &
Hennig 2021; Dodge et al. 2019 (EMNLP, "Show your work"); Tang et al. 2020 (ACL, "Showing your work
doesn't always work"); Nado et al. 2021; Dahl et al. 2023 (AlgoPerf); Kasimbeg et al. 2025 (ICLR);
Wen et al. 2026 (ICLR); Bergstra & Bengio 2012 (random search); Godbole et al. 2023 (tuning playbook);
Kingma & Ba 2015; Melis, Dyer & Blunsom 2018 (cross-link to controls).

---

### Chapter 5. Controls — Random search as the placebo arm (key: controls)

`\labchapter{Controls}{Random search as the placebo arm}{Neural architecture search finds better architectures than random search.}{As reported for ENAS, DARTS and their successors, 2018--2019}{Vanished: trained with the same pipeline, random architectures from the same search space do about as well.}{ch:controls}`

**The claim.** ENAS (Pham et al. 2018) and DARTS (Liu, Simonyan & Yang 2019) reported architectures found
by search with strong CIFAR-10 and Penn Treebank results at low search cost.

**The post-mortem.** Li & Talwalkar (2019, UAI): random search with early stopping, and random search
with weight sharing, are competitive with the published methods (verify the numbers); reproducibility
problems in the published code. Sciuto, Yu, Jaggi, Musat & Salzmann (2020, ICLR): on average the search
policies of ENAS, DARTS and NAO are no better than random; weight sharing degrades the ranking of
architectures. Yang, Esperança & Carlucci (2020, ICLR): across eight methods and five datasets, randomly
sampled architectures perform about as well; the training protocol accounts for most of the gains; the
search spaces are narrow by design. Lindauer & Hutter (2020, JMLR): best-practice checklist. Further
reality checks without a placebo arm: Melis et al. (2018, LSTMs), Lucic et al. (2018, GANs), Musgrave et
al. (2020, metric learning). Verdict box: **vanished**.

**The method.**
- Controls in experimental science: negative control (no effect expected), positive control (a known
  effect must appear), sham or placebo (everything but the active ingredient). For a search method the
  placebo is random sampling from the same space, trained by the same pipeline with the same budget.
- The spread of the search space: how much any search could gain.
- Proxies: a search ranks candidates by a cheap proxy (short training, shared weights); the quality of the
  search is bounded by the proxy's correlation with the truth (Kendall's $\tau$).

*Main results.*
1. **Proposition (random search, complete proof).** The best of $k$ uniform samples lies in the top
   $q$-quantile with probability $1-(1-q)^k$; $k\ge\ln(1/\delta)/q$ suffices for probability
   $1-\delta$.
2. **Theorem (selection on a noisy proxy, complete proof).** If true quality $T\sim\cN(\mu,\sigma^2)$ and
   proxy $P=T+\xi$ with independent $\xi\sim\cN(0,s^2)$, choosing the best of $k$ candidates by proxy gives
   $\E[T_{\text{chosen}}]=\mu+\rho\sigma e_k$ with $\rho=\Corr(T,P)=\sigma/\sqrt{\sigma^2+s^2}$.
   *Proof plan:* condition on all proxies; Gaussian regression $\E[T\mid P]$; the maximum of the proxies.
   Consequence: a search guided by a proxy with small $\rho$ is close to random.
3. **Remark.** No search can beat random by more than the spread of the space.

**In the laboratory.**
- Lab 5.1 (`05-controls-space`): an architecture space for digits MLPs (depth, width, activation,
  dropout, batch normalization, ...); the distribution of true quality (mean over seeds) of about 100
  random architectures.
- Lab 5.2 (`05-controls-search`): an evolutionary search guided by a short-training proxy versus random
  search with the same compute; the true quality of what each finds, with intervals over repeated
  searches; the proxy's Kendall $\tau$.

**Protocol.** "Controls for a search method".

**Exercises.** Expected rank of the best of $k$; the proxy theorem with non-Gaussian noise (what
survives?); lab: make the proxy better or worse; design positive and negative controls for a data-
pruning method (random pruning as the placebo); critique a NAS paper table without a random-search row;
research: when can a search beat random by more than noise?

**Key references.** Pham et al. 2018 (ICML); Liu, Simonyan & Yang 2019 (ICLR); Li & Talwalkar 2019 (UAI);
Sciuto et al. 2020 (ICLR); Yang, Esperança & Carlucci 2020 (ICLR); Lindauer & Hutter 2020 (JMLR); Ying et
al. 2019 (NAS-Bench-101); Dong & Yang 2020 (NAS-Bench-201); Melis, Dyer & Blunsom 2018 (ICLR); Lucic et
al. 2018 (NeurIPS); Musgrave, Belongie & Lim 2020 (ECCV); Bergstra & Bengio 2012.

---

### Chapter 6. Confounding — The architecture and the recipe (key: confounding)

`\labchapter{Confounding}{The architecture and the recipe}{Vision transformers are better image classifiers than convolutional networks.}{The received reading of Dosovitskiy et al.\ (2021) and Touvron et al.\ (2021)}{Confounded: modern training recipes close most of the gap, and at matched data and compute the two families scale alike.}{ch:confounding}`

**The claim.** ViT (Dosovitskiy et al. 2021) outperformed strong ResNet baselines when pre-trained on
large data (JFT-300M) with less pre-training compute, while underperforming on ImageNet-1k alone; DeiT
(Touvron et al. 2021) trained ViTs on ImageNet-1k with strong augmentation, regularization and
distillation. The received reading: attention beats convolution. State the original, narrower claim
accurately (verify).

**The post-mortem.** Wightman, Touvron & Jégou (2021, "ResNet strikes back"): a modern recipe raises
ResNet-50 to about 80.4\% top-1 on ImageNet (verify). Bello et al. (2021, "Revisiting ResNets"): training
and scaling strategies matter more than architectural changes. Liu et al. (2022, ConvNeXt): modernizing
a ResNet step by step matches Swin transformers; the steps form a path whose gains are attributed in
order. Smith, Brock, Berrada & De (2023): NFNets pre-trained on JFT-4B match ViTs at comparable compute.
Narang et al. (2021, EMNLP): most proposed transformer modifications do not transfer across
implementations. Steiner et al. (2022) and Beyer et al. (2022) on ViT recipes. Verdict box:
**confounded**.

**The method.**
- Confounding: the treatment (architecture) arrives bundled with co-treatments (augmentation, optimizer,
  schedule, epochs, regularization). Potential outcomes for a $2\times2$ factorial (architecture $\times$
  recipe).
- Two legitimate questions: the effect at a fixed recipe, and the effect when each arm gets its own best
  recipe. Compute-, parameter- and throughput-matched comparisons.
- Sequential attribution along a path (ConvNeXt) and its dependence on the order; the Shapley value as
  the average over orders.

*Main results.*
1. **Proposition (complete proof).** In a $2\times2$ design with cell means $\mu_{ar}$, the naive contrast
   $\mu_{\text{new},\text{new}}-\mu_{\text{old},\text{old}}$ equals the architecture effect at the old
   recipe plus the recipe effect at the new architecture, and the two orders of attribution differ by
   exactly the interaction contrast.
2. **Theorem (complete proof for $k$ factors in a $2^k$ design).** Sequential attributions are independent
   of the order if and only if all interaction contrasts involving the factor vanish; the Shapley
   attribution averages over orders and sums to the total effect.

**In the laboratory.**
- Lab 6.1 (`06-confounding-factorial`): on digits, an MLP versus a small convolutional network (defined
  in the script), crossed with an old recipe (SGD, constant learning rate, no augmentation, few epochs)
  and a new recipe (AdamW, cosine schedule, label smoothing, shift augmentation, longer training); 20
  seeds per cell; the naive gap, main effects, interaction, and the two path attributions.

**Protocol.** "Attributing an improvement".

**Exercises.** Shapley values for three factors; when does matching parameter count mislead (compute,
throughput)?; lab: add a third factor; design a fair architecture comparison under a fixed GPU-hour
budget; critique an ablation path that changes eight things in a fixed order; research: confounding by
hyperparameters tuned on the newer architecture.

**Key references.** Dosovitskiy et al. 2021 (ICLR); Touvron et al. 2021 (ICML); Wightman, Touvron & Jégou
2021 (arXiv 2110.00476); Bello et al. 2021 (NeurIPS); Liu et al. 2022 (CVPR); Smith et al. 2023 (arXiv
2310.16764); Narang et al. 2021 (EMNLP); Steiner et al. 2022 (TMLR); Beyer, Zhai & Kolesnikov 2022;
He et al. 2019 ("Bag of tricks"); Shapley 1953; Pearl 2009; Imbens & Rubin 2015.

---

### Chapter 7. What Was Held Fixed — Batch size, steps and the generalization gap (key: heldfixed)

`\labchapter{What Was Held Fixed}{Batch size, steps and the generalization gap}{Large-batch training generalizes worse because it converges to sharp minima.}{Keskar et al.\ (2017)}{Confounded: matched in steps and learning rate, most of the gap disappears; the sharpness explanation is examined in Chapter 10.}{ch:heldfixed}`

**The claim.** Keskar, Mudigere, Nocedal, Smelyanskiy & Tang (2017, ICLR): large-batch training reaches
worse test accuracy than small-batch training, and converges to sharp minimizers (verify settings:
batch as a fraction of the data, fixed epochs).

**The post-mortem.** Hoffer, Hubara & Soudry (2017, NeurIPS): the gap comes from fewer updates; with a
learning rate adapted to the batch size, "ghost" batch normalization and more steps it closes. Goyal et
al. (2017): the linear scaling rule with warm-up trains ResNet-50 on ImageNet with batch 8192 without
loss of accuracy. Shallue et al. (2019, JMLR): steps-to-result versus batch size across workloads —
perfect scaling, then diminishing returns, then a maximum useful batch size; with metaparameters tuned,
no evidence that large batches inherently hurt generalization (verify the exact statement). McCandlish
et al. (2018): the gradient noise scale predicts the critical batch size. Zhang et al. (2019, NeurIPS):
the noisy quadratic model. Kaddour et al. (2023, NeurIPS, "No train no gain"): efficient-training methods
compared at fixed budgets against a baseline whose schedule is decayed for that budget. Verdict box:
**confounded**.

**The method.**
- The budget axis defines the question: epochs, steps, examples seen, FLOPs, wall-clock, tuning trials.
  "What was held fixed" analysis of an experiment.
- Steps-to-target curves; matched comparisons with schedules matched to each budget; scaling rules
  (linear for SGD in the noise-dominated regime; square-root for Adam, Malladi et al. 2022); the critical
  batch size.

*Main results.*
1. **Theorem (noisy quadratic, complete proof in one dimension).** SGD on $f(\theta)=\tfrac12h\theta^2$
   with gradient noise of variance $c/B$ satisfies
   $\E\theta_{t+1}^2=(1-\eta h)^2\E\theta_t^2+\eta^2c/B$, with stationary value
   $\eta c/(Bh(2-\eta h))$. Derive the number of steps to reach a target loss with the best learning
   rate for each $B$: proportional to $1/B$ below a critical batch size, saturating above it.
2. **Corollary.** At a fixed number of epochs the number of steps is $T=\text{epochs}\cdot n/B$; above
   the critical batch size the final loss grows with $B$ — a gap that needs no sharp minima.
3. **Heuristic derivation.** The diagonal (many-eigenvalue) case and the critical batch size
   $B_{\mathrm{crit}}$ as a ratio of noise to curvature (McCandlish et al.).

**In the laboratory.**
- Lab 7.1 (`07-heldfixed-gap`): digits MLP, batch sizes 8 to 1024 (the training set has about 1257
  examples): (i) fixed epochs and fixed learning rate; (ii) fixed epochs with the learning rate tuned for
  each batch size (with warm-up); (iii) fixed steps with tuned learning rate. Test error versus batch size
  under the three protocols.
- Lab 7.2 (`07-heldfixed-steps`): steps to reach a target test error versus batch size (log-log), with
  the noisy-quadratic prediction overlaid.

**Protocol.** "Choosing what to hold fixed".

**Exercises.** The noisy quadratic with momentum; derive the linear scaling rule from the stationary
variance; lab with Adam (square-root rule?); design a batch-size study with a fixed GPU budget; critique
"method X reaches the same accuracy in half the epochs"; research: what replaces the critical batch size
when the noise is not isotropic?

**Key references.** Keskar et al. 2017; Hoffer, Hubara & Soudry 2017; Goyal et al. 2017 (arXiv
1706.02677); Smith & Le 2018 (ICLR); Smith, Kindermans, Ying & Le 2018 (ICLR); Shallue et al. 2019
(JMLR); McCandlish et al. 2018 (arXiv 1812.06162); Zhang et al. 2019 (NeurIPS); Masters & Luschi 2018;
Malladi et al. 2022 (NeurIPS, SDEs for adaptive methods); Kaddour et al. 2023 (NeurIPS); Dinh et al.
2017 (pointer to Chapter 10).

---

### Part III. Causes

*Knowing that a component helps is not knowing why. An ablation removes a part and watches what changes;
a mechanism claim names what the part does; a correlation across many runs suggests a cause without
isolating it. Each of these has a design that makes it informative and a way of going wrong.*

---

### Chapter 8. Ablations — Interactions and the home-field advantage (key: ablations)

`\labchapter{Ablations}{Interactions and the home-field advantage}{A winning ticket trains well because of its original initialization.}{Frankle and Carbin (2019)}{Narrowed: the original initialization matters at small learning rates; at standard ones it matters only for tickets rewound to early in training.}{ch:ablations}`

**The claim.** Frankle & Carbin (2019, ICLR best paper): iterative magnitude pruning finds sparse
subnetworks that, reset to their original initialization, train to the accuracy of the full network;
randomly re-initialized, they train worse. For deeper networks the paper used warm-up or a lower
learning rate (verify).

**The post-mortem.** Liu, Sun, Zhou, Huang & Darrell (2019, ICLR, "Rethinking the value of network
pruning"): with the standard (larger) learning rate, re-initialized subnetworks match the original-
initialization tickets in their settings; the effect appears at small learning rates (verify the exact
settings and numbers). Frankle, Dziugaite, Roy & Carbin (2020, ICML): for larger networks, matching
subnetworks exist only when rewound to an early iteration, which coincides with stability to SGD noise.
Gale, Elsen & Hooker (2019). Frankle et al. (2021, ICLR, "Pruning neural networks at initialization: why
are we missing the mark?"): randomly shuffling the masks of pruning-at-initialization methods within each
layer does not hurt — a sham ablation showing they learn only layerwise sparsity. Blalock et al. (2020,
MLSys): the state of pruning research. Verdict box: **narrowed**.

**The method.**
- An ablation is an experiment with factors; one-factor-at-a-time designs versus factorial designs;
  interactions (here: initialization $\times$ learning rate).
- The home-field advantage: hyperparameters tuned for the full system favor it in every ablation.
- Re-tuning the ablated system; ablating at each arm's optimum; reporting ablation tables with intervals.
- Sham ablations (shuffled masks) as controls (`\cref{ch:controls}`).

*Main results.*
1. **Proposition (home-field advantage, complete proof).** If $h^\star$ maximizes the full system's score,
   the fixed-hyperparameter ablation effect exceeds the re-tuned effect by exactly the ablated system's
   regret at $h^\star$, which is nonnegative. With $h^\star$ chosen on a noisy validation score the full
   system's own score is also biased upward (`\cref{ch:seeds}`).
2. **Theorem (efficiency of factorial designs, complete proof).** In a $2^k$ factorial with $N$ runs, each
   main effect is estimated with variance $4\sigma^2/N$; one-factor-at-a-time with the same $N$ runs gives
   $2(k+1)\sigma^2/N$, and estimates no interaction.

**In the laboratory.**
- Lab 8.1 (`08-ablations-tickets`): magnitude pruning of a digits MLP to a few percent density (one-shot
  and iterative); original-initialization ticket, random re-initialization, and a shuffled-mask sham, at a
  small and a large learning rate; 10 or more seeds per cell. The interaction between initialization and
  learning rate. (The effect may be small on tiny MLPs: find the regime where it shows, and report.)

**Protocol.** "Designing an ablation".

**Exercises.** Fractional factorial designs and aliasing; the regret of a fixed $h^\star$ for a quadratic
score; lab with rewinding; design an ablation of a four-component training recipe within 64 runs;
critique an ablation table without intervals or re-tuning; research: ablations under strong interaction
(when is "the contribution of a component" undefined?).

**Key references.** Frankle & Carbin 2019; Liu et al. 2019 (ICLR); Frankle et al. 2020 (ICML); Frankle et
al. 2021 (ICLR); Gale, Elsen & Hooker 2019; Blalock et al. 2020 (MLSys); Zhou et al. 2019 (NeurIPS,
"Deconstructing lottery tickets"); Box, Hunter & Hunter 2005; Fisher 1935 (*The Design of Experiments*);
Montgomery (*Design and Analysis of Experiments*).

---

### Chapter 9. Mechanisms — Intervening on the explanation (key: mechanisms)

`\labchapter{Mechanisms}{Intervening on the explanation}{Batch normalization works by reducing internal covariate shift.}{Ioffe and Szegedy (2015)}{Misattributed: covariate shift injected after normalization does not hurt training; the benefit comes from elsewhere.}{ch:mechanisms}`

**The claim.** Ioffe & Szegedy (2015, ICML) introduced batch normalization to reduce internal covariate
shift, the change in the distribution of a layer's inputs during training.

**The post-mortem.** Santurkar, Tsipras, Ilyas & Madry (2018, NeurIPS): adding random, time-varying shifts
and scalings after each batch-normalization layer (more covariate shift) leaves the benefit intact;
measured by a gradient-based definition, batch normalization need not reduce covariate shift; they
propose a smoother loss landscape instead (verify). Bjorck, Gomes, Selman & Weinberger (2018, NeurIPS):
batch normalization allows larger learning rates; at small learning rates its benefit largely
disappears. Arora, Li & Lyu (2019, ICLR): scale invariance and automatic rate tuning. Yang et al. (2019,
ICLR): batch normalization causes gradient explosion at initialization in deep networks — the smoothing
story is not the whole story either. Fixup (Zhang, Dauphin & Ma 2019) and NFNets (Brock et al. 2021):
normalization-free training as a test of necessity. Verdict box: **misattributed**; the replacement
mechanism is itself contested, which the chapter says.

**The method.**
- A mechanism claim is a mediation claim: treatment $\to$ mediator $\to$ outcome. Tests: operationalize
  the mediator (the definition of covariate shift matters); check that the treatment moves it; intervene
  on the mediator without the treatment (knockout) and supply it without the treatment (rescue);
  dose–response; rival mediators and the experiment that separates them.
- Necessity versus sufficiency; the burden of a mechanism claim is a new prediction.

*Main results.*
1. **Theorem (the batch-normalization Jacobian, complete proof).** For one unit over a batch of size $m$
   with $\hat y=(y-\mu\mathbf 1)/\sigma$ and $z=\gamma\hat y+\beta$, writing $g=\partial\Loss/\partial z$,
   $\norm{\partial\Loss/\partial y}^2=(\gamma^2/\sigma^2)\bigl(\norm g^2-\tfrac1m\ip{\mathbf 1,g}^2-\tfrac1m\ip{\hat y,g}^2\bigr)$,
   because $I-\tfrac1m\mathbf 1\mathbf 1^\top-\tfrac1m\hat y\hat y^\top$ is an orthogonal projection.
   (Compare with the statement in Santurkar et al.; check the factor of the last term — the published
   statement may print $1/\sqrt m$.)
2. **Proposition (scale invariance, complete proof).** If $\Loss(cw)=\Loss(w)$ for all $c>0$ then
   $\ip{\nabla\Loss(w),w}=0$ and $\nabla\Loss(cw)=\nabla\Loss(w)/c$; gradient descent increases $\norm w$
   and has effective learning rate $\eta/\norm w^2$.

**In the laboratory.**
- Lab 9.1 (`09-mechanisms-noisybn`): a deeper digits MLP (e.g. 6 layers) without normalization, with
  batch normalization, and with batch normalization followed by injected per-step random shifts and
  scalings; training curves and final error over seeds; a covariate-shift measure for each arm.
- Lab 9.2 (`09-mechanisms-lr`): the batch-normalization advantage as a function of the learning rate.

**Protocol.** "Testing a mechanism claim".

**Exercises.** Derive the batch-normalization Jacobian; scale invariance for layer normalization; lab:
inject shift before rather than after normalization; design a knockout-and-rescue experiment for "warm-up
helps because it reduces Adam's variance early"; critique a mechanism claim supported only by a
correlation; research: which mediator explains batch normalization at small batch sizes?

**Key references.** Ioffe & Szegedy 2015; Santurkar et al. 2018; Bjorck et al. 2018; Arora, Li & Lyu 2019;
Yang et al. 2019 (ICLR, mean-field theory of batch normalization); Zhang, Dauphin & Ma 2019 (ICLR, Fixup);
Brock et al. 2021 (ICML, NFNets); De & Smith 2020 (NeurIPS); van Laarhoven 2017; Kohler et al. 2019
(AISTATS); Baron & Kenny 1986 and Pearl 2001 on mediation; Liu et al. 2020 (RAdam) and Ma & Yarats 2021
for the warm-up exercise (verify).

---

### Chapter 10. Correlation in a Sweep — Flat minima and the measures of generalization (key: sharpness)

`\labchapter{Correlation in a Sweep}{Flat minima and the measures of generalization}{Flat minima generalize better.}{Hochreiter and Schmidhuber (1997); Keskar et al.\ (2017)}{Narrowed: sharpness depends on the parametrization, and its correlation with generalization does not survive controlled sweeps of modern networks.}{ch:sharpness}`

**The claim.** Hochreiter & Schmidhuber (1997, Neural Computation): flat minima, by a minimum-description-
length argument, generalize. Keskar et al. (2017) linked the large-batch gap to sharpness
(`\cref{ch:heldfixed}`). Sharpness became a standard explanation and a target (SAM, Foret et al. 2021).

**The post-mortem.** Dinh, Pascanu, Bengio & Bengio (2017, ICML): for ReLU networks, function-preserving
reparametrizations make minima arbitrarily sharp. Jiang, Neyshabur, Mobahi, Krishnan & Bengio (2020,
ICLR): over a large controlled sweep (verify size), with rank correlation, granulated correlation and
conditional independence tests, some sharpness-based measures did best among more than 40 measures and
many norm-based measures correlated negatively (verify). Dziugaite et al. (2020, NeurIPS): judged by the
worst case over environments, every measure fails somewhere. Andriushchenko, Croce, Müller, Hein &
Flammarion (2023, ICML): even reparametrization-invariant (adaptive) sharpness correlates weakly or
negatively with generalization in modern settings (verify). Wen, Ma & Li (2023) and Andriushchenko &
Flammarion (2022) on why SAM works. Verdict box: **narrowed**.

**The method.**
- Observational analysis of hyperparameter sweeps; hyperparameters as confounders of a measure and the
  gap; Simpson's paradox; the law of total covariance; granulated and conditional rank correlation;
  worst-case evaluation over environments.
- Validity requirements for a measure: invariance under function-preserving reparametrization.
- From correlation to intervention: what a SAM experiment can and cannot show.

*Main results.*
1. **Theorem (reparametrization, complete proof).** For $f(x)=W_2\relu(W_1x)$ and $T_c(W_1,W_2)=(cW_1,c^{-1}W_2)$,
   $f$ and the loss are unchanged, and the Hessian transforms as
   $\nabla^2\Loss(T_c\theta)=J_c^{-1}\nabla^2\Loss(\theta)J_c^{-1}$ with $J_c=\diag(cI,c^{-1}I)$; hence
   $\lambda_{\max}$ can be made arbitrarily large without changing the function (assuming the $W_1$
   block of the Hessian is nonzero).
2. **Proposition (Simpson, complete proof).** $\Cov(X,Y)=\E[\Cov(X,Y\mid H)]+\Cov(\E[X\mid H],\E[Y\mid H])$;
   an explicit sweep in which every within-stratum correlation is negative and the pooled one positive.

**In the laboratory.**
- Lab 10.1 (`10-sharpness-sweep`): a sweep of the digits MLP over learning rate, batch size, width and
  weight decay (on the order of 150 runs); sharpness as the top Hessian eigenvalue of the training loss
  (power iteration on Hessian–vector products) and as a random-perturbation sharpness; pooled versus
  within-stratum Kendall $\tau$ with the generalization gap.
- Lab 10.2 (`10-sharpness-reparam`): rescale the layers of a trained network: $\lambda_{\max}$ changes by
  orders of magnitude, the test error not at all.

**Protocol.** "Reading a correlation in a sweep".

**Exercises.** Reparametrization for deeper networks and with biases; normalized sharpness that is
invariant; lab: condition on learning rate; design a study that tests whether sharpness causes
generalization; critique "measure X correlates with generalization ($\tau=0.6$) across our sweep";
research: an invariant sharpness that predicts generalization under interventions.

**Key references.** Hochreiter & Schmidhuber 1997; Keskar et al. 2017; Dinh et al. 2017; Neyshabur et al.
2017 (NeurIPS); Jiang et al. 2020 (ICLR); Dziugaite et al. 2020 (NeurIPS); Andriushchenko et al. 2023
(ICML); Foret et al. 2021 (ICLR); Kwon et al. 2021 (ICML, ASAM); Andriushchenko & Flammarion 2022 (ICML);
Wen, Ma & Li 2023 (NeurIPS); Simpson 1951; Pearl 2009.

---

### Part IV. Scale

*Scaling laws turned the planning of large training runs into an exercise in fitting and extrapolation.
A fitted curve is a measurement with a protocol, an uncertainty and a domain of validity, and each of the
three has been the source of a famous error.*

---

### Chapter 11. Fitting a Scaling Law — Chinchilla's third approach and its replication (key: fitting)

`\labchapter{Fitting a Scaling Law}{Chinchilla's third approach and its replication}{$L(N,D)=1.69+406.4/N^{0.34}+410.7/D^{0.28}$.}{Hoffmann et al.\ (2022), Approach 3}{Unreplicated: a refit of the same data gives different exponents, consistent with the paper's other two approaches.}{ch:fitting}`

**The claim.** Hoffmann et al. (2022, NeurIPS) estimated compute-optimal allocation by three approaches;
the third fitted the parametric law above to about 400 runs with a Huber loss on log-loss and L-BFGS,
reporting $\alpha=0.34$, $\beta=0.28$ and narrow confidence intervals, implying an allocation exponent
near $0.46$, versus about $0.5$ from Approaches 1 and 2 (verify all numbers).

**The post-mortem.** Besiroglu, Erdil, Barnett & You (2024, arXiv 2404.10102): reconstructing the data
from the paper's figure and refitting gives estimates (verify: $E\approx1.82$, $A\approx482$,
$B\approx2085$, $\alpha\approx0.35$, $\beta\approx0.37$) consistent with Approaches 1 and 2; the reported
intervals would require on the order of $10^5$–$10^6$ experiments (verify); the original fit appears not
to have converged. Verify whether the original authors responded or corrected, and say so accurately.
Also Choshen, Zhang & Andreas (2024), "A hitchhiker's guide to scaling law estimation". Verdict box:
**unreplicated**; the qualitative conclusion (scale parameters and tokens about equally) holds.

**The method.**
- Fitting nonlinear laws: least squares in log space versus in loss; Huber loss; nonconvexity, grids of
  initializations and convergence checks; identifiability ($E$ trades off against $A_N,\alpha_N$ over a
  short range).
- Uncertainty: bootstrap over runs; profile likelihood; derived quantities (the allocation exponent,
  tokens per parameter) by the delta method and by bootstrap.
- Cross-checks: isoFLOP parabolas (Approach 2) and envelopes (Approach 1); publishing the data and code.

*Main results.*
1. **Theorem (compute-optimal allocation, complete proof).** Minimizing
   $E+A_NN^{-\alpha_N}+A_DD^{-\alpha_D}$ subject to $6ND=C$ gives
   $N_{\mathrm{opt}}=G(C/6)^{a}$, $D_{\mathrm{opt}}=G^{-1}(C/6)^{b}$ with $a=\alpha_D/(\alpha_N+\alpha_D)$,
   $b=\alpha_N/(\alpha_N+\alpha_D)$, $G=(\alpha_NA_N/(\alpha_DA_D))^{1/(\alpha_N+\alpha_D)}$.
2. **Proposition (delta method, complete proof).** The asymptotic variance of $\hat a$ from the covariance
   of $(\hat\alpha_N,\hat\alpha_D)$.
3. **Proposition (identifiability, complete proof of the linearization).** For $L(N)=E+AN^{-\alpha}$ on
   $N\in[N_0,rN_0]$, the Fisher information for $(E,A,\alpha)$ degenerates as $r\to1$: to first order in
   $\ln r$ the three columns of the Jacobian span only two dimensions.

**In the laboratory.**
- Lab 11.1 (`11-fitting-grid`): the online teacher protocol (one pass over $D$ fresh examples, cosine
  schedule matched to $D$) on a grid of widths and data sizes with 2–3 seeds; save the runs.
- Lab 11.2 (`11-fitting-fits`): fit the parametric law with Hoffmann's recipe (Huber on log-loss, grid
  initialization, L-BFGS) and with variants (one initialization, different $\delta$, early stopping of the
  optimizer); bootstrap intervals; the allocation exponent and its interval; compare with an isoFLOP
  analysis of the same runs.

**Protocol.** "Fitting and reporting a scaling law".

**Exercises.** Derive the allocation with a different FLOP model; the delta-method interval for tokens
per parameter; lab: drop the largest models from the fit; design the cheapest grid that pins down $a$ to
$\pm0.05$; critique a scaling-law paper that reports exponents without data or intervals; research:
scaling laws with an unknown irreducible loss.

**Key references.** Hoffmann et al. 2022; Besiroglu et al. 2024; Kaplan et al. 2020; Choshen, Zhang &
Andreas 2024; Hestness et al. 2017; Rosenfeld et al. 2020 (ICLR); Muennighoff et al. 2023 (NeurIPS,
data-constrained scaling); Caballero et al. 2023 (ICLR); Huber 1964; Efron & Tibshirani 1993.

---

### Chapter 12. The Protocol Behind the Exponent — Kaplan, Chinchilla and the learning-rate schedule (key: protocol)

`\labchapter{The Protocol Behind the Exponent}{Kaplan, Chinchilla and the learning-rate schedule}{Model size should grow much faster than data: $N_{\mathrm{opt}}\propto C^{0.73}$.}{Kaplan et al.\ (2020)}{Reversed: with schedules matched to each budget and parameters counted consistently, parameters and data grow at about the same rate.}{ch:protocol}`

**The claim.** Kaplan et al. (2020): for a fixed compute budget, most of the increase should go to model
size ($N_{\mathrm{opt}}\propto C^{0.73}$, data $\propto C^{0.27}$; verify), so large models trained on
modest data are compute-optimal; GPT-3-era models followed this.

**The post-mortem.** Hoffmann et al. (2022): with cosine schedules whose length matches each run's number
of tokens, parameters and data scale equally ($a\approx b\approx0.5$); Kaplan's fixed schedule makes
shorter runs look worse. Porian, Wortsman, Jitsev, Schmidt & Carmon (2024, NeurIPS): three factors —
the last layer's computational cost, warm-up duration, and scale-dependent optimizer tuning — explain
the discrepancy; correcting them recovers Chinchilla; AdamW $\beta_2$ at small batch sizes matters
(verify). Pearce & Song (2024): counting non-embedding parameters and working at small scale reproduces
Kaplan's exponent from a Chinchilla-type law (verify). Hägele et al. (2024, NeurIPS): constant learning
rate with cooldown (warmup–stable–decay) matches cosine and allows one run per model size. Verdict box:
**reversed**.

**The method.**
- The protocol defines the curve: the schedule must match the horizon (intermediate checkpoints of a
  long schedule overstate the loss at small $D$); counting $N$ and $C$ (embeddings, last layer,
  attention); hyperparameters tuned per scale; the range of scales and the curvature of the law.
- Local exponents; the protocol card of a scaling study.

*Main results.*
1. **Proposition (local exponent, complete proof).** For $L=E+AN^{-\alpha}$,
   $-\dd\ln L/\dd\ln N=\alpha(1-E/L)<\alpha$: a power law fitted without the irreducible term
   underestimates $\alpha$, increasingly as $L$ approaches $E$.
2. **Proposition (counting, complete proof).** If the loss depends on $N_{\mathrm{ne}}$ but is plotted
   against $N_{\mathrm{tot}}=N_{\mathrm{ne}}+N_{\mathrm{emb}}$, local slopes are multiplied by
   $N_{\mathrm{ne}}/N_{\mathrm{tot}}$; derive the effect on the fitted allocation exponent at small scale.
3. **Heuristic derivation.** A model of unfinished schedules: the loss of a checkpoint at $D$ inside a
   schedule of length $D_{\max}$ exceeds the matched-schedule loss by $g(D/D_{\max})$, $g(1)=0$; show how
   this tilts the fitted allocation toward larger models.

**In the laboratory.**
- Lab 12.1 (`12-protocol-schedules`): on the teacher task, protocol K (one long cosine run per width,
  intermediate checkpoints read as smaller $D$) versus protocol C (a separate run per $(N,D)$ with a
  matched schedule); isoFLOP curves and fitted allocation exponents with intervals under both.
- Lab 12.2 (`12-protocol-counting`): count parameters with and without a fixed overhead (e.g. the input
  layer as an "embedding") and refit.

**Protocol.** "A protocol card for a scaling study".

**Exercises.** Local exponents for the two-term law in $N$ and $D$; the bias of a fixed warm-up length
across scales; lab with a warmup–stable–decay schedule; design a scaling study with ten runs; critique
"we read the loss of smaller budgets from our longest run's checkpoints"; research: scaling laws for the
learning rate and batch size (Bi et al. 2024; verify).

**Key references.** Kaplan et al. 2020 (arXiv 2001.08361); Hoffmann et al. 2022; Porian et al. 2024
(NeurIPS); Pearce & Song 2024; Hägele et al. 2024 (NeurIPS); Besiroglu et al. 2024; Bi et al. 2024
(DeepSeek LLM); Yang et al. 2022 (μP, Tensor Programs V); Brown et al. 2020 (GPT-3).

---

### Chapter 13. Extrapolation — Predicting the large from the small (key: extrapolation)

`\labchapter{Extrapolation}{Predicting the large from the small}{On these tasks, larger language models do worse.}{The Inverse Scaling Prize (McKenzie et al.\ 2023)}{Narrowed: at larger scale several of these curves turn upward; a trend over a short range is not a law.}{ch:extrapolation}`

**The claim.** McKenzie et al. (2023, TMLR): eleven prize-winning tasks on which performance falls with
scale, grouped into four causes (verify).

**The post-mortem.** Wei, Kim, Tay & Le (2023, "Inverse scaling can become U-shaped"; verify venue):
evaluated at larger scale (PaLM up to 540B), several of the tasks become U-shaped (verify the count), and
one-shot or chain-of-thought prompting changes others. Lourie, Hu & Cho (2025, Findings of EMNLP;
arXiv 2507.00885): in a meta-analysis, downstream scaling is predictable in only 39\% of cases, and small
changes of setup change the trend. Caballero et al. (2023): broken power laws. Counterpoint: successful
pre-registered extrapolations (GPT-4 technical report: final loss and a coding pass rate predicted from
runs with much less compute; verify the wording). Verdict box: **narrowed**.

**The method.**
- Interpolation versus extrapolation; three sources of extrapolation error: parameter uncertainty
  (grows with distance), misspecification (the functional form), regime change (a new mechanism).
- Backtesting: hold out the largest scales; prediction intervals and their coverage; fitting several
  forms and reporting the spread; the design of a scale ladder; stating the domain of validity.

*Main results.*
1. **Theorem (prediction variance, complete proof).** For least squares $y=a+bx+\varepsilon$ on
   $x_1,\dots,x_n$, $\Var\hat y(x_0)=\sigma^2\bigl(1/n+(x_0-\bar x)^2/S_{xx}\bigr)$; the slope variance is
   minimized by putting the points at the ends of the range.
2. **Proposition (misspecification, complete proof).** If the truth is $y=a+bx+cx^2$, a fitted line's
   extrapolation error at $x_0$ grows like $c(x_0-\bar x)^2$ plus terms fixed by the design.
3. **Remark.** Finitely many points on a power law are consistent with a law that breaks beyond them.

**In the laboratory.**
- Lab 13.1 (`13-extrapolation-backtest`): from the teacher grid of `\cref{ch:fitting}` (or a new one), fit
  on the smallest models and predict the largest; intervals and coverage over many backtests.
- Lab 13.2 (`13-extrapolation-ushape`): a non-monotone scaling curve on digits: test error versus width
  with label noise shows a peak (double descent); a trend fitted before the peak extrapolates wrongly.

**Protocol.** "Before you extrapolate".

**Exercises.** Prediction intervals in log-log coordinates; optimal placement of three scales; lab with
a different noise level; design a pre-registered extrapolation; critique a plot that draws a fitted
line two decades past the data; research: detecting a break from data before it.

**Key references.** McKenzie et al. 2023 (TMLR); Wei, Kim, Tay & Le 2023; Lourie, Hu & Cho 2025; Caballero
et al. 2023; OpenAI 2023 (GPT-4 technical report); Gadre et al. 2024; Ruan, Maddison & Hashimoto 2024
(observational scaling laws); Nakkiran et al. 2020 (deep double descent, ICLR); Belkin et al. 2019
(PNAS); volume 2, Chapter 1 (double descent) may be mentioned in a side note.

---

### Part V. Measurement

*A measurement is a number produced by an instrument. Before a curve can be read as a fact about models,
the reader has to know what the instrument does to it: how the metric transforms the quantity underneath,
how many items it rests on, and how its settings move it.*

---

### Chapter 14. The Metric Makes the Phenomenon — Emergent abilities and the smooth change beneath them (key: emergence)

`\labchapter{The Metric Makes the Phenomenon}{Emergent abilities and the smooth change beneath them}{Some abilities of large language models emerge abruptly and unpredictably with scale.}{Wei et al.\ (2022)}{Artifact, in large part: with continuous metrics and enough test items, most of these curves are smooth.}{ch:emergence}`

**The claim.** Wei et al. (2022, TMLR): abilities "not present in smaller models but present in larger
models", which cannot be predicted by extrapolating smaller models (quote at most a short phrase;
verify).

**The post-mortem.** Schaeffer, Miranda & Koyejo (2023, NeurIPS): emergence appears under nonlinear or
discontinuous metrics (exact match, multiple-choice grade); continuous metrics (token edit distance,
Brier score) and larger test sets give smooth curves; emergence can be induced in vision models by the
choice of metric (verify the share of BIG-bench cases tied to two metrics). Hu et al. (2024, ICLR,
PassUntil): with "infinite-resolution" evaluation, tiny pass rates scale smoothly. Counterpoints, stated
fairly: Du, Zeng, Dong & Tang (2024, NeurIPS): abilities appear when the pre-training loss crosses a
threshold, even under continuous metrics; genuinely sharp transitions in training (induction heads,
Olsson et al. 2022; grokking; sparse parities with hidden progress, Barak et al. 2022). Verdict box:
**artifact**, "in large part", with the counterexamples named.

**The method.**
- Measurement validity: a metric as a transformation of a latent quantity; nonlinear (products),
  discontinuous (thresholds) and bounded metrics; floor and ceiling effects; resolution and the number of
  test items; scales of measurement (Stevens 1946).
- Practice: report a continuous metric alongside the headline one; predict from the latent; separate
  genuine phase transitions from metric-induced ones.

*Main results.*
1. **Proposition (exact match, complete derivation).** If per-token correctness is $p(N)=\exp(-cN^{-\alpha})$,
   exact match on $L$ tokens is $\exp(-LcN^{-\alpha})$, a Gumbel distribution function in $\ln N$ with
   location $\ln(Lc)/\alpha$ and scale $1/\alpha$: longer answers move the transition to larger models
   without changing its width in $\ln N$.
2. **Proposition (rule of three, complete proof).** With $m$ test items and no success, the one-sided 95\%
   upper confidence bound on the success rate is $1-0.05^{1/m}\approx3/m$.
3. **Remark.** Proper scoring rules (Brier, log score) are linear in the underlying probabilities.

**In the laboratory.**
- Lab 14.1 (`14-emergence-sequences`): digits as tokens: MLPs of increasing width; per-digit accuracy and
  log-likelihood (smooth) versus exact match on sequences of $L=1,3,10$ test digits (sharp), with the
  Gumbel prediction.
- Lab 14.2 (`14-emergence-resolution`): the same curves measured with $m=50$, $500$ and $5000$ items.

**Protocol.** "Before announcing an emergent ability".

**Exercises.** Exact match with heterogeneous per-token rates; multiple-choice grade as a threshold;
Clopper–Pearson intervals; lab with a different $L$; design an evaluation that could detect true
emergence; critique an "emergent" plot on 100 items with exact match; research: when is a transition in
a continuous metric genuine?

**Key references.** Wei et al. 2022 (TMLR); Schaeffer, Miranda & Koyejo 2023 (NeurIPS); Srivastava et al.
2023 (BIG-bench, TMLR); Ganguli et al. 2022 (FAccT); Hu et al. 2024 (ICLR); Du et al. 2024 (NeurIPS);
Olsson et al. 2022; Barak et al. 2022 (NeurIPS); Power et al. 2022 (grokking); Stevens 1946 (Science);
Clopper & Pearson 1934; Hanley & Lippman-Hand 1983 (rule of three).

---

### Chapter 15. The Instrument's Settings — Prompts, formats and demonstrations as nuisance factors (key: instruments)

`\labchapter{The Instrument's Settings}{Prompts, formats and demonstrations as nuisance factors}{In-context learning barely depends on whether the demonstrations' labels are correct.}{Min et al.\ (2022)}{Narrowed: the effect depends on model scale, task and format; larger models can follow flipped labels.}{ch:instruments}`

**The claim.** Min et al. (2022, EMNLP): replacing the labels of in-context demonstrations by random labels
barely hurts performance across many models and tasks; the label space, the input distribution and the
format matter more (verify the scope and numbers).

**The post-mortem.** Yoo et al. (2022, EMNLP): ground-truth labels matter more than reported, depending on
task and on how sensitivity is measured. Wei et al. (2023, "Larger language models do in-context learning
differently"): with flipped labels, small models keep their semantic priors while large models follow
the flipped labels; instruction tuning strengthens priors (verify). Sclar, Choi, Tsvetkov & Suhr (2024,
ICLR): meaning-preserving format changes move accuracy by up to 76 points for one model and reverse
model rankings (verify). Zhao et al. (2021, ICML) and Lu et al. (2022, ACL): demonstration order and
majority-label biases. Zheng et al. (2023) and Wang et al. (2023): position and verbosity biases of model
judges. Mizrahi et al. (2024, TACL): multi-prompt evaluation. Verdict box: **narrowed**.

**The method.**
- The evaluation pipeline as an instrument with settings: prompt wording and format, choice and order of
  demonstrations, decoding, the judge.
- Settings as random nuisance factors: sample them, report the distribution; crossed random effects
  (model $\times$ setting) and variance components (`\cref{ch:seeds}`); blocking (compare models on the
  same settings); reliability of judges (Cohen's $\kappa$, test–retest); the instrument card.

*Main results.*
1. **Theorem (ranking under random settings, complete proof).** In
   $Y_{mf}=\mu_m+u_f+v_{mf}+\varepsilon_{mf}$ with independent random effects, a comparison of two models on
   one random setting reverses their order with probability
   $\Phi\bigl(-\abs{\Delta}/\sqrt{2\sigma_v^2+2\sigma_\varepsilon^2}\bigr)$; averaging over $F$ settings
   divides the interaction variance by $F$; the setting's main effect $u_f$ cancels when both models see
   the same settings.
2. **Proposition.** Cohen's $\kappa$ and its relation to the accuracy of a judge against a reference.

**In the laboratory.**
- Lab 15.1 (`15-instruments-settings`): two digits classifiers evaluated under random test-time
  "formats" (brightness and contrast, one-pixel shifts, noise); variance components and ranking
  reversals on single formats versus averages.
- Optional Lab 15.2 (`15-instruments-icl`): a small transformer trained for in-context classification on
  sequences of digit images and labels (in the spirit of Chan et al. 2022), evaluated with correct and
  with random labels in context — only if it runs within the laptop budget.

**Protocol.** "Evaluating with an instrument that has settings".

**Exercises.** The variance of a format-averaged comparison; Cohen's $\kappa$ for a biased judge; lab
with more formats; design a multi-prompt evaluation for 10 models with 1000 calls each; critique a
leaderboard that uses one prompt template; research: choosing the distribution of settings.

**Key references.** Min et al. 2022; Yoo et al. 2022; Wei et al. 2023 (arXiv 2303.03846); Sclar et al.
2024; Zhao et al. 2021; Lu et al. 2022; Zheng et al. 2023 (NeurIPS); Wang et al. 2023; Mizrahi et al.
2024 (TACL); Chan et al. 2022 (NeurIPS); Cohen 1960; Miller 2024 ("Adding error bars to evals").

---

### Epilogue — The laboratory notebook (key: epilogue; special chapter)

Written by the coordinating author after the chapters. A complete worked experiment from question to
report on a fresh claim (e.g. "label smoothing helps on digits"), applying the protocols in order:
prediction table, experimental design record, seeds and budget, controls, ablation design, analysis,
report; the protocols of all chapters collected on facing pages; what the book does not cover (benchmark
science, causal inference from observational data, human evaluation); the culture of experiments
(incentives, pre-registration, publishing negative results, blameless post-mortems).

### Appendix A — A Statistical Toolkit (key: toolkit)

Statements with proofs or proof sketches, each with a fixed label that chapters cite:
`thm:toolkit:tinterval` ($t$-intervals and Welch's test), `thm:toolkit:bootstrap` (percentile and
stratified bootstrap; consistency sketched), `thm:toolkit:permutation` (exactness of permutation tests),
`thm:toolkit:mcnemar` (paired binary outcomes), `thm:toolkit:bonferroni`, `thm:toolkit:holm`,
`thm:toolkit:bh` (multiple testing), `thm:toolkit:orderstat` (expected maxima, the Gumbel limit),
`thm:toolkit:delta` (delta method), `thm:toolkit:nls` (nonlinear least squares asymptotics),
`thm:toolkit:anova` (two-way random effects and variance components), `thm:toolkit:factorial` (contrasts
in $2^k$ designs), `thm:toolkit:power` (sample sizes), `thm:toolkit:kendall` (Kendall's $\tau$ and its
variance), `thm:toolkit:clopper` (exact binomial intervals and the rule of three),
`thm:toolkit:mannwhitney` (probability of improvement). Section labels `sec:toolkit:...` as needed.
