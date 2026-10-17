# Optimizers in the Wild — Book Plan and Chapter Specifications

*Optimizers in the Wild: Steepest Descent and the Dynamics of Training* (《现代优化器的理论》) is volume 3 of
*The Missing Textbooks*. This file is the editor's plan. Where the plan and a finished chapter differ, the
chapter is correct and its author says why in the report.

## The thesis

Optimization textbooks end with convex analysis and with the theorem that stochastic gradient descent
reaches an approximate stationary point. The decisions that make or break a training run in practice are
elsewhere: the choice of Adam or Muon, the length of the warmup, the weight decay, the shape of the
learning-rate schedule, the batch size, and the fact that training runs at the edge of stability. This book
gives the theory of those decisions along two lines.

1. **The unification line (Parts I–II).** Almost every optimizer in use is *steepest descent under a norm*,
   combined with *preconditioning* (a norm that adapts to the gradients seen so far) and *moving averages*
   (of gradients, of squared gradients, of iterates). SGD is steepest descent in the Euclidean norm;
   signSGD, Lion and Adam without memory are steepest descent in the $\ell_\infty$ norm; Shampoo without
   accumulation and Muon are steepest descent in the spectral norm; a whole network gets a norm built from
   its layers' norms, which fixes how the learning rate must scale with width. Momentum and Adam's
   $\beta_1$ are moving averages of the gradient; AdaGrad, Adam's $\beta_2$, Shampoo, K-FAC and SOAP are
   preconditioners.
2. **The dynamics line (Parts III–IV).** What happens when these optimizers run with the step sizes that
   practice uses: the sharpness rises to the stability threshold and stabilizes itself there; warmup is
   needed because the curvature that the early steps meet is high; gradient clipping and normalized
   updates tame a curvature that grows with the gradient; gradient noise sets a floor that only a schedule
   removes, which explains cosine and warmup–stable–decay schedules and the "river valley" picture; weight
   decay drives normalized layers to an equilibrium that sets an effective learning rate.
3. **Part V brings the lines together**: why Adam beats SGD on transformers (curvature heterogeneity,
   heavy-tailed classes, $\ell_\infty$ geometry, and the batch-size regime), and where steepest descent ends
   up (the implicit bias of a norm).

Each chapter starts from a practice "in the wild" (a rule of thumb, a default, an observation from training
runs), builds the smallest idealization in which the practice can be analysed, proves one main theorem
completely, compares its prediction with a simulation, and then says precisely what the idealization
misses and what the theory does and does not tell a practitioner to do.

**Positioning (verified 2026-10-17; for the preface).** Laker Newhouse, *Duality, Weight Decay, and
Metrized Deep Learning* (MIT graduate thesis, May 2025, advised by Phillip Isola) presents the unification
line: duality links SGD, Adam, Shampoo and Muon as steepest descent under norms, and norm constraints give
Lipschitz guarantees for transformers. The underlying papers are Bernstein and Newhouse, "Old optimizer, new
norm: an anthology" (arXiv:2409.20325, 2024) and "Modular duality in deep learning" (ICML 2025), and Large et
al., "Scalable optimization in the modular norm" (NeurIPS 2024). Jeremy Bernstein's own PhD is from Caltech
(2022, Computation and Neural Systems); he did this work as a postdoc at MIT. (The editor's request said
"Bernstein's MIT thesis"; the MIT thesis that gives the unification line is Newhouse's. Write the preface
accordingly and credit both.) The thesis is a thesis: it does not cover the dynamics half. Frederik
Kunstner's UBC PhD thesis (2024; CAIAC best dissertation award 2025) treats why Adam outperforms gradient
descent on language models. Jeremy Cohen and coauthors' papers (ICLR 2021; arXiv 2022; ICLR 2025, central
flows) treat the edge of stability. Classical textbooks (Nesterov, *Lectures on Convex Optimization*, 2018;
Bubeck, "Convex optimization: algorithms and complexity", 2015; Beck, *First-Order Methods in Optimization*,
2017; Lan, *First-order and Stochastic Optimization Methods for Machine Learning*, 2020; Garrigos and Gower,
"Handbook of convergence theorems for (stochastic) gradient methods", arXiv 2023; Bottou, Curtis and Nocedal,
SIAM Review 2018) stop before any of this. Surveys: Sun, "Optimization for deep learning: theory and
algorithms" (2019); Nagwekar, *Towards Guided Descent* (UPenn master's thesis, arXiv:2512.18373, 2025). The
practical counterpart is Google's *Deep Learning Tuning Playbook* (Godbole et al. 2023). Companion volume:
*The Hydrogen Atoms of Deep Learning* (volume 2), whose Chapter 15 solves the two-parameter product model
and the catapult; this volume cites it and does not repeat it.

## Structure, keys, and files

| # | key | file | Title — Subtitle | "In the wild" tag (3rd arg of `\casechapter`) |
|---|-----|------|------------------|------|
| – | prologue | 00-prologue.tex | Optimizers in the Wild — What the textbook proves and what the practitioner decides | — (special chapter) |
| | | | **Part I. One Rule: Steepest Descent** | |
| 1 | steepest | 01-steepest.tex | The Geometry of a Step — Steepest descent in a normed space | Normalized updates |
| 2 | sign | 02-sign.tex | Sign Descent — The $\ell_\infty$ geometry of Adam, signSGD and Lion | signSGD, Lion, Adam without memory |
| 3 | spectral | 03-spectral.tex | Orthogonalized Updates — The spectral norm, Muon and Newton–Schulz | Muon and Shampoo |
| 4 | modular | 04-modular.tex | Norms for Networks — The modular norm, the spectral condition and $\mu$P | Hyperparameter transfer across width |
| | | | **Part II. Memory: Averages and Preconditioners** | |
| 5 | momentum | 05-momentum.tex | Moving Averages — Momentum, noise, and why normalized steps need memory | Momentum $0.9$ |
| 6 | adaptive | 06-adaptive.tex | Adaptive Preconditioning — AdaGrad, RMSProp and Adam | Adam's $\beta_2$ and $\epsilon$ |
| 7 | kronecker | 07-kronecker.tex | Kronecker Preconditioners — Shampoo, K-FAC and SOAP | Shampoo, K-FAC, SOAP |
| | | | **Part III. Stability** | |
| 8 | eos | 08-eos.tex | The Edge of Stability — Thresholds, self-stabilization and central flows | Learning rates at the stability limit |
| 9 | warmup | 09-warmup.tex | Warmup — Why the step size must start small | Linear warmup |
| 10 | spikes | 10-spikes.tex | Clipping and Spikes — Generalized smoothness and the anatomy of an instability | Gradient clipping, z-loss, QK-norm |
| | | | **Part IV. Noise, Schedules and Equilibria** | |
| 11 | noise | 11-noise.tex | Noise and Batch Size — The noisy quadratic model and the critical batch size | Batch-size scaling rules |
| 12 | schedules | 12-schedules.tex | Schedules — Cosine, warmup–stable–decay and the river valley | Cosine and WSD schedules |
| 13 | decay | 13-decay.tex | Weight Decay — Frank–Wolfe, rotational equilibrium and the effective learning rate | AdamW's decoupled weight decay |
| | | | **Part V. Choosing an Optimizer** | |
| 14 | transformers | 14-transformers.tex | Why Adam Beats SGD on Transformers — Heterogeneous curvature, heavy-tailed classes and the batch-size regime | Adam for transformers, SGD for convnets |
| 15 | bias | 15-bias.tex | Where Steepest Descent Ends — Implicit bias and the margin of a norm | Adaptive methods and generalization |
| – | epilogue | 16-epilogue.tex | What Theory Owes Practice — Open problems and a ledger of verdicts | — (special chapter) |
| A | toolkit | 17-toolkit.tex | A Toolkit — Duality, matrices, recurrences and online learning | — (appendix chapter, empty tag) |

Chapter labels are `ch:<key>`; `\cref{ch:<key>}` prints "Chapter N" (the appendix prints "Appendix A").
Each chapter starts with `\casechapter{Title}{Subtitle}{Tag}{ch:key}` using exactly the strings in the
table (write $\ell_\infty$ as `$\ell_\infty$`, $\mu$P as `$\mu$P`, $\beta_2$ as `$\beta_2$`). The appendix
uses `\casechapter{A Toolkit}{Duality, matrices, recurrences and online learning}{}{ch:toolkit}`. The
companion volume is cited in prose as "volume 2 of this series, *The Hydrogen Atoms of Deep Learning*,
Chapter 15" (no `\cref` and no bib entry).

### Fixed appendix labels (chapters may cite these; the appendix must define exactly these)

| label | content |
|---|---|
| `lem:toolkit:dualnorm` | dual norms: definition, $\norm{x}=\max_{\norm{z}_*\le1}\ip{z,x}$ (bidual), Hölder; duals of $\ell_p$ ($1/p+1/q=1$) |
| `lem:toolkit:subdiff` | the subdifferential of a norm is $\{z:\norm{z}_*\le1,\ \ip{z,x}=\norm{x}\}$; hence $\lmo(g)$ characterization |
| `lem:toolkit:conjugate` | the convex conjugate of $\tfrac12\norm{\cdot}^2$ is $\tfrac12\norm{\cdot}_*^2$; the sharp operator $g^\sharp=\argmax_d\{\ip{g,d}-\tfrac12\norm{d}^2\}=-\norm{g}_*\lmo(g)$ |
| `thm:toolkit:vonneumann` | von Neumann's trace inequality $\abs{\tr(A^\top B)}\le\sum_i\sigma_i(A)\sigma_i(B)$; duality of Schatten norms (spectral ↔ nuclear, Frobenius self-dual) |
| `thm:toolkit:polar` | polar decomposition $G=\polar(G)\,(G^\top G)^{1/2}$; $\polar(G)=UV^\top$ is the nearest semi-orthogonal matrix in every unitarily invariant norm (Frobenius proof complete) |
| `thm:toolkit:loewner` | Löwner–Heinz: $t\mapsto t^r$ is operator monotone on PSD matrices for $0\le r\le1$ (complete proof for $r=1/2$, statement for general $r$) |
| `lem:toolkit:kron` | Kronecker products: mixed product, $\mathrm{vec}(AXB)=(B^\top\otimes A)\mathrm{vec}(X)$, eigenvalues, $(A\otimes B)^r=A^r\otimes B^r$ for PSD, and $A\preceq A',B\preceq B'\Rightarrow A\otimes B\preceq A'\otimes B'$ (PSD) |
| `thm:toolkit:geomean` | matrix geometric mean $A\#B$: the variational characterization $A\#B=\max\{X\succeq0:\bigl(\begin{smallmatrix}A&X\\X&B\end{smallmatrix}\bigr)\succeq0\}$, joint concavity, and $A\#B=(AB)^{1/2}$ for commuting $A,B$ |
| `lem:toolkit:recurrence` | stability of $x_{t+1}=Ax_t$ (spectral radius); the Schur–Cohn (Jury) test: $z^2+a_1z+a_0$ has both roots in the open unit disk iff $\abs{a_0}<1$ and $\abs{a_1}<1+a_0$ |
| `lem:toolkit:lyapunov` | linear stochastic recursions $x_{t+1}=Ax_t+\xi_t$: stationary covariance solves $\Sigma_\infty=A\Sigma_\infty A^\top+Q$; scalar and $2\times2$ closed forms |
| `lem:toolkit:descent` | descent lemma for $L$-smooth functions w.r.t. a general norm, and its converse under convexity (co-coercivity in the Euclidean case only) |
| `lem:toolkit:pl` | the Polyak–Łojasiewicz inequality and the linear rate it gives |
| `lem:toolkit:adagrad` | for $a_t\ge0$, $\sum_{t\le T}a_t/\sqrt{\sum_{s\le t}a_s}\le2\sqrt{\sum_{t\le T}a_t}$ |
| `thm:toolkit:ogd` | online gradient descent regret bound $DG\sqrt T$-type, with a general step-size sequence |
| `thm:toolkit:azuma` | Azuma–Hoeffding, and Doob's $L^2$ maximal inequality |
| `thm:toolkit:gronwall` | Grönwall's inequality, continuous and discrete |
| `lem:toolkit:gauss` | Gauss's inequality for unimodal distributions, and the Chebyshev/Markov bounds used for sign errors |

### Cross-chapter threads (point them out with `\cref` where natural)

- **The three ingredients** — norm (1–4), moving average (5; Lion in 2; Adam's $\beta_1$ in 6), preconditioner
  (6, 7). The prologue's table and the epilogue's ledger list every optimizer as (norm, preconditioner,
  averages).
- **Normalized steps never converge at a fixed step size** — the prologue's sign-descent example ↔ momentum
  removes the noise bias (5) ↔ a decaying schedule removes the oscillation and noise floors (12) ↔ the
  schedule is the Frank–Wolfe step size (13).
- **Stability thresholds** — $\eta\lambda<2$ (prologue) ↔ the heavy-ball region $\eta\lambda<2+2\beta$ and the
  EMA region $\eta\lambda<(2+2\beta)/(1-\beta)$ (5) ↔ GD, momentum and Adam at the edge of stability (8) ↔
  warmup (9) ↔ spikes (10) ↔ curvature-limited versus noise-limited regimes (11, 14).
- **Noise floors** — momentum's stationary variance (5) ↔ the noisy quadratic model (11) ↔ the river valley
  (12) ↔ the equilibrium of weight decay (13).
- **Heterogeneity** — $\ell_\infty$ smoothness and effective rank (2) ↔ nuclear rank versus stable rank (3) ↔
  Kronecker conditioning (7) ↔ preconditioning raises the critical batch size (11) ↔ block heterogeneity and
  Zipf frequencies (14).
- **Norm control** — per-layer learning rates (4) ↔ the norm ball of decoupled weight decay (13) ↔ z-loss and
  QK-norm (10) ↔ the margin of a norm (15).
- **Effective learning rate** — per-layer rates in $\mu$P (4) ↔ angular updates during warmup (9) ↔ the
  rotational equilibrium $\sqrt{2\eta\lambda}$ (13) ↔ schedules acting on the effective rate (12).
- **Rotation dependence** — sign descent and Adam depend on the coordinate system (2) ↔ Shampoo/SOAP rotate
  into an eigenbasis (7) ↔ rotating a transformer's parameters slows Adam (14).

## Notation (global; summarized for readers in `frontmatter/notation.tex`)

- **Sizes.** $d$ number of parameters (the dimension of the optimization variable); a weight matrix
  $W\in\R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$; a generic matrix $G\in\R^{m\times n}$; a network has $N$
  layers $W_1,\dots,W_N$ with widths $d_0,\dots,d_N$ and layer index $l$; $n$ samples; $B$ batch size; $K$
  classes; $T$ number of steps; $t$ the step (continuous time for flows).
- **Objective.** $\Loss(\theta)$ the objective (training or population loss; the chapter says which);
  $\ell(\theta;\xi)$ the loss on one example or batch $\xi$; $\Loss^\star=\inf\Loss$; $\theta^\star$ a minimizer;
  $\Delta_t=\Loss(\theta_t)-\Loss^\star$. A superscript star marks an optimal value; a subscript star a target or
  teacher ($\theta_\star$), as in volume 2.
- **Gradients and curvature.** $\nabla\Loss(\theta)$; stochastic gradient $g_t$ at $\theta_t$, noise
  $\zeta_t=g_t-\nabla\Loss(\theta_t)$, per-example covariance $\Sigma(\theta)$ (so $\Cov(g_t)=\Sigma/B$);
  Hessian $H=\nabla^2\Loss$; sharpness $S(\theta)=\lmax(\nabla^2\Loss(\theta))$; preconditioned sharpness
  $S_P=\lmax(P^{-1}\nabla^2\Loss)$.
- **Optimizer.** Step size (learning rate) $\eta$, schedule $\eta_t$, peak $\eta_{\max}$, warmup length
  $T_{\mathrm w}$; momentum $\beta$; Adam $\beta_1,\beta_2,\epsilon$ (use `\epsilon` for Adam's epsilon and keep
  `\eps` for generic small numbers), moments $m_t,v_t$ and their bias-corrected versions $\hat m_t,\hat v_t$;
  decoupled weight decay $\lambda$ (update $\theta\leftarrow(1-\eta\lambda)\theta-\eta u$); eigenvalues always
  carry a subscript ($\lambda_i$, $\lmax$, $\lmin$), never a bare $\lambda$.
- **Norms.** $\norm{\cdot}$ a generic norm, $\norm{\cdot}_*$ its dual; $\norm{\cdot}_2$, $\norm{\cdot}_1$,
  $\norm{\cdot}_\infty$ on vectors; $\norm{\cdot}_{\op}$ spectral (operator $\ell_2\to\ell_2$), $\norm{\cdot}_F$
  Frobenius, $\norm{\cdot}_{\nuc}$ nuclear, $\norm{\cdot}_{p\to q}$ induced; RMS norm
  $\norm{x}_{\rms}=\norm{x}_2/\sqrt{\dim x}$, and $\norm{W}_{\rms\to\rms}=\sqrt{d_{\mathrm{in}}/d_{\mathrm{out}}}\,\norm{W}_{\op}$;
  $\norm{x}_P=\sqrt{x^\top Px}$ for $P\succ0$. In this book an unadorned $\norm{\cdot}$ is never silently
  Euclidean.
- **Steepest descent.** $\lmo(g)\in\argmin_{\norm{u}\le1}\ip{g,u}$ (linear minimization oracle; so
  $\ip{g,\lmo(g)}=-\norm{g}_*$); the sharp operator $g^\sharp=\argmax_u\{\ip{g,u}-\tfrac12\norm{u}^2\}=-\norm{g}_*\lmo(g)$.
  *Normalized* steepest descent: $\theta_{t+1}=\theta_t+\eta\,\lmo(g_t)$. *Unnormalized*:
  $\theta_{t+1}=\theta_t-\eta\,g_t^\sharp$. Coordinatewise $\sign$ with $\sign(0)=0$; $\polar(G)=UV^\top$ for a
  compact SVD $G=U\diag(\sigma)V^\top$ with singular values $\sigma_i(G)$.
- **Smoothness.** $\Loss$ is $L$-smooth w.r.t. $\norm{\cdot}$ if
  $\norm{\nabla\Loss(\theta)-\nabla\Loss(\theta')}_*\le L\norm{\theta-\theta'}$; $(L_0,L_1)$-smoothness
  $\norm{\nabla^2\Loss(\theta)}\le L_0+L_1\norm{\nabla\Loss(\theta)}$ (Chapter 10); loss-dependent smoothness
  $\norm{\nabla^2\Loss(\theta)}\le K_0+K_1(\Loss(\theta)-\Loss^\star)$ (Chapter 9).
- **Matrices.** $\otimes$ Kronecker, $\mathrm{vec}$ (`\vect`), Löwner order $\preceq$, $\tr$, $\diag$.
- **Probability and asymptotics** as in volume 2: $\cN$, $\E$, $\Var$, $\Cov$; $\lesssim,\asymp,\tilde O$
  with what is fixed stated.

Macros provided by `wild.sty`: `\Loss \Lhat \lmo \sign \polar \op \nuc \rms \vect \lmax \lmin \diag \softmax
\relu \Unif \eqd \Sph`. Chapter-local macros must start with the chapter key.

---

## Chapter specifications

Facts marked **[verify]** must be checked against sources before they go into the text. Theorem statements are
the editor's plan: re-derive each, fix constants, check numerically, and correct the plan where it is wrong
(say so in your report). "Complete proof" is mandatory where marked. Several plans below were checked
numerically by the editor (noted). Web access in the writing environment is limited to web search (direct
fetching of arXiv and publisher pages is blocked); verify bibliographic facts by search, and give arXiv
numbers or page ranges only when a search result confirms them.

### Prologue — Optimizers in the Wild (key `prologue`) — written by the editor

What the textbooks prove (the Ghadimi–Lan bound for SGD) versus what a practitioner decides (a modern
pretraining recipe: AdamW with $\beta=(0.9,0.95)$, $\epsilon=10^{-8}$, weight decay $0.1$, warmup, cosine or
WSD decay, gradient clipping at $1$, a batch size of millions of tokens, and increasingly Muon for hidden
matrices); for each ingredient, the chapter that explains it. The two lines. The first example, which
contains both lines: one quadratic, two steps. Gradient descent on $\frac12h x^2$ converges iff $\eta h<2$
(stability); sign descent $x\leftarrow x-\eta\,\sign(x)$ never diverges and never converges, ending on a
two-cycle with loss at most $h\eta^2/2$ (a floor that only a schedule removes); the update rule of every
optimizer as (norm, preconditioner, moving averages) in a table. The anatomy of a chapter, the field guide
entries and field notes, the status markers, a full-width map of the book, and how to read it.

### Chapter 1 — The Geometry of a Step (key `steepest`)

**In the wild.** Every optimizer turns a gradient into a step, and the step is not the gradient. Adam's
update has a roughly fixed per-coordinate size; Muon's has a fixed spectral norm; Lion's is a vector of signs;
practitioners compare optimizers by the RMS of their updates (Liu et al. 2025, "Muon is scalable for LLM
training", match Muon's update RMS to AdamW's typical $0.2$–$0.4$ **[verify]**). Normalized updates are the
rule, not the exception.

**Idealization.** A differentiable $\Loss$ on a finite-dimensional real vector space with a norm. The gradient
is a linear functional (a dual vector); to take a step one needs a map from the dual space back to the space
— the norm supplies it. Fix the vocabulary: dual norm, $\lmo$, sharp operator, normalized and unnormalized
steepest descent, smoothness with respect to a norm.

**Theory.**
- *Proposition (duality maps of the norms that optimizers use; complete proofs).* $\lmo$ and dual norm for:
  $\ell_2$ ($-g/\norm g_2$; dual $\ell_2$); $\ell_\infty$ ($-\sign(g)$; dual $\ell_1$; the set of maximizers when
  some $g_i=0$); $\ell_1$ ($-\sign(g_i)e_i$ at $i\in\argmax\abs{g_i}$; dual $\ell_\infty$: greedy coordinate
  descent); spectral norm on matrices ($-UV^\top$; dual nuclear — via `\cref{thm:toolkit:vonneumann}`, details
  in Chapter 3); Frobenius; the induced norms $\ell_1\to\ell_2$ (max column norm: normalize columns) and
  $\ell_2\to\ell_\infty$ (max row norm: normalize rows); and a product (max) norm over blocks
  $\norm{\theta}=\max_k\norm{\theta_k}_{(k)}$ (dual: sum of dual norms; $\lmo$ blockwise). Use
  `\cref{lem:toolkit:subdiff,lem:toolkit:conjugate}`.
- *Theorem (one-step decrease and rates in a general norm; complete proof).* If $\Loss$ is $L$-smooth w.r.t.
  $\norm\cdot$ then $\Loss(\theta+u)\le\Loss(\theta)+\ip{\nabla\Loss(\theta),u}+\frac L2\norm u^2$
  (`\cref{lem:toolkit:descent}`); the minimizer of the right side is $u=-g^\sharp/L$, with decrease
  $\norm{g}_*^2/(2L)$. Corollaries: (i) nonconvex, unnormalized: $\min_{t<T}\norm{\nabla\Loss(\theta_t)}_*^2\le
  2L\Delta_0/T$; (ii) nonconvex, normalized with step $\eta$: $\frac1T\sum_{t<T}\norm{\nabla\Loss(\theta_t)}_*\le
  \Delta_0/(\eta T)+L\eta/2$, which is $\sqrt{2L\Delta_0/T}$ at the best $\eta$; (iii) convex with
  $R=\sup\{\norm{\theta-\theta^\star}:\Loss(\theta)\le\Loss(\theta_0)\}$: $\Delta_T\le2LR^2/(T+\dots)$ (fix the
  constant; Nesterov's argument $\Delta_{t+1}\le\Delta_t-\Delta_t^2/(2LR^2)$). Note that these are worst-case
  statements whose *only* dependence on the norm is through $L$ and $\norm\cdot_*$: choosing a norm is
  choosing which smoothness constant and which gradient norm you pay.
- *Proposition (which norm? complete proof).* For a quadratic with Hessian $H\succeq0$: the smoothness
  constant w.r.t. $\ell_2$ is $\lmax(H)$; w.r.t. $\ell_\infty$ it is $\norm{H}_{\infty\to1}=\max_{s\in\{\pm1\}^d}
  s^\top Hs$, which is at least $\tr H$ (average over random signs), equals $\tr H$ for diagonal $H$, and is at
  most $d\,\lmax(H)$. Hence one step of normalized $\ell_\infty$ descent beats $\ell_2$ descent on the bound iff
  $\norm{g}_1^2/\norm{g}_2^2>\norm{H}_{\infty\to1}/\lmax(H)\ (\ge\tr H/\lmax(H))$: a dense gradient against a
  curvature with low effective rank. (Kelner, Lee, Orecchia, Sidford, SODA 2014 for $\ell_\infty$ steepest
  descent in max flow; Balles, Pedregosa, Le Roux 2020 **[verify]**.) This proposition is reused in
  Chapters 2 and 14; state it here, give the machine-learning reading in Chapter 2.
- *Remark: steepest descent is not mirror descent.* Mirror descent needs a strongly convex mirror map and moves
  in the dual; steepest descent in a non-Euclidean norm uses the non-smooth $\frac12\norm{\cdot}^2$; the two
  coincide for Euclidean norms only (Nemirovski–Yudin 1983; Beck–Teboulle 2003). One paragraph; no theorem.
- *Remark: normalized steepest descent with a fixed step is a trust-region method* with radius $\eta$ in the
  chosen norm; the "one formula" of the book.

**Reading.** Figure 1 (TikZ schematic computed by a script): unit balls of $\ell_1,\ell_2,\ell_\infty$ with
one gradient and the three steepest directions. Figure 2 (theory vs simulation): on a $d=1000$ diagonal
quadratic with heavy-tailed curvatures $h_i\propto i^{-a}$, the per-step decrease of normalized $\ell_2$ and
$\ell_\infty$ descent compared with the bound $\eta\norm g_*-L\eta^2/2$; or the iteration count to a target
versus the predicted ratio $\norm{g}_1^2\lmax/(\norm g_2^2\tr H)$ across $a$. Explain when the bound is tight.

**Misses.** Idealized away: a fixed norm and a global smoothness constant (deep networks are not $L$-smooth
with a useful $L$, and the sharpness moves — Chapter 8); deterministic gradients; no memory (Chapters 5–7).
Where it breaks: the worst-case rates do not rank optimizers in practice; the right norm is a modelling choice
(Chapter 4) and is not identifiable from the theorem; non-smooth norms make $\lmo$ discontinuous, which matters
with noise (Chapter 5). Repairs: local or anisotropic smoothness (Xie et al., ICLR 2025, "Adam exploits
$\ell_\infty$-geometry"), generalized smoothness (Chapters 9–10), trust-region and LMO views (Pethick et al.,
ICML 2025, "Training deep learning models with norm-constrained LMOs" **[verify]**; Kovalev 2025
**[verify]**).

**Exercises (ideas).** Duality maps for $\ell_p$, $1<p<\infty$; the dual of a max-of-norms and of a weighted
sum; $\ell_1$ steepest descent is greedy coordinate descent; the $\ell_{\infty\to1}$ norm is NP-hard to compute
in general but equals $\tr H$ for diagonal $H$; $L$-smoothness constants of least squares in $\ell_2$ and
$\ell_\infty$; normalized descent on a quadratic oscillates at amplitude $\eta$ (prologue); simulation:
steepest descent in three norms on a 2-D quadratic; research: a principled choice of norm from data.

### Chapter 2 — Sign Descent (key `sign`)

**In the wild.** Adam is the default optimizer for transformers; signSGD compresses communication to one bit
per coordinate (Bernstein, Wang, Azizzadenesheli, Anandkumar, ICML 2018); Lion, found by program search,
updates with the sign of an interpolated momentum (Chen et al., NeurIPS 2023, "Symbolic discovery of
optimization algorithms"). Observations: sign descent with momentum closes most of the gap between gradient
descent and Adam on transformers in full batch (Kunstner, Chen, Lavington, Schmidt, ICLR 2023); Adam's update
is sign-like when gradients are consistent (Balles and Hennig, ICML 2018, "Dissecting Adam").

**Idealization.** Steepest descent in $\ell_\infty$ on $\R^d$, deterministic and stochastic; coordinatewise
smoothness $\Loss(\theta+u)\le\Loss(\theta)+\ip{\nabla\Loss,u}+\frac12\sum_iL_iu_i^2$.

**Theory.**
- *Proposition (Adam without memory is sign descent; complete).* With $\beta_1=\beta_2=0$ and $\epsilon=0$
  Adam's step is $-\eta\,\sign(g_t)$; RMSProp with $\beta=0$ likewise; with bias correction the *first* step of
  Adam is $-\eta\,\sign(g_0)$ for every $\beta_1,\beta_2$.
- *Proposition (the size of Adam's step; complete proof; editor checked numerically).* Without bias correction
  and with $\epsilon=0$, $\abs{m_t/\sqrt{v_t}}\le(1-\beta_1)\big/\sqrt{(1-\beta_2)(1-\beta_1^2/\beta_2)}$ for
  $\beta_1^2<\beta_2$, by Cauchy–Schwarz, and the bound is attained (by $g_{t-k}\propto(\beta_1/\beta_2)^k$).
  Values: $7.27$ for $(0.9,0.999)$, $2.35$ for $(0.9,0.99)$, $1.17$ for $(0.9,0.95)$, exactly $1$ when
  $\beta_1=\beta_2$. So Adam is "sign descent with a bounded multiplier", and smaller $\beta_2$ keeps it closer
  to sign descent. (Reused in Chapters 6 and 10.)
- *Theorem (sign descent under coordinatewise smoothness; complete proof).* Deterministic:
  $\Loss(\theta_{t+1})\le\Loss(\theta_t)-\eta\norm{\nabla\Loss(\theta_t)}_1+\frac{\eta^2}2\norm{\vec L}_1$, hence
  $\frac1T\sum_t\norm{\nabla\Loss(\theta_t)}_1\le\Delta_0/(\eta T)+\eta\norm{\vec L}_1/2$. Compare gradient
  descent's $\min_t\norm{\nabla\Loss}_2^2\le2L_{\max}\Delta_0/T$: sign descent wins when gradients are dense
  ($\norm g_1^2\approx d\norm g_2^2$) and curvature is concentrated ($\norm{\vec L}_1\ll dL_{\max}$) (Bernstein
  et al. 2018). Connect to Chapter 1's proposition ($\norm{\vec L}_1=\tr H$ for diagonal quadratics).
- *Theorem (signSGD with minibatches; complete proof).* With coordinatewise noise standard deviations
  $\sigma_i$ and batch size $B$: $\Pr[\sign(g_{t,i})\ne\sign(\partial_i\Loss)]\le\sigma_i/(\sqrt B\abs{\partial_i\Loss})$
  (Markov/Chebyshev; `\cref{lem:toolkit:gauss}` gives the sharper unimodal constant), so the expected
  progress is at least $\eta(\norm{\nabla\Loss}_1-2\norm{\vec\sigma}_1/\sqrt B)-\frac{\eta^2}2\norm{\vec L}_1$;
  with $B\propto T$ the rate matches SGD's (Bernstein et al. 2018, Theorem 1 — check their constants). The bias
  term $2\norm{\vec\sigma}_1/\sqrt B$ does not vanish with small steps: sign descent needs large batches or
  momentum (Chapter 5).
- *Proposition (Adam as signal-to-noise-scaled sign descent; heuristic ◇).* For stationary gradients with
  coordinate means $\mu_i$ and variances $s_i^2$, long memories give $\hat m_i/\sqrt{\hat v_i}\approx
  \mu_i/\sqrt{\mu_i^2+s_i^2}$: a sign step shrunk by the coordinate's signal-to-noise ratio (Balles and
  Hennig 2018).
- *Proposition (sign descent is not rotation invariant; complete).* For $H$ diagonal with $h_1=\lmax$,
  $\norm{H}_{\infty\to1}=\tr H$; for a rotation $Q$, taking $s=\sign(Q^\top e_1)$ gives
  $\norm{Q^\top HQ}_{\infty\to1}\ge s^\top Q^\top HQs\ge\lmax\ip{s,Q^\top e_1}^2=\lmax\norm{Q^\top e_1}_1^2$, and for
  a uniformly random rotation $\norm{Q^\top e_1}_1^2\approx\frac2\pi d$. Rotation can multiply the $\ell_\infty$
  smoothness constant by about $\frac2\pi d\lmax/\tr H$. (Xie, Mohamadi, Li, ICLR 2025: rotating GPT-2's parameters slows
  Adam but not SGD **[verify]**.)

**Field guide entries.** signSGD (with majority vote), Lion, Adam (pointing to Chapter 6 for the full entry).

**Reading.** Figure: GD vs sign descent on a diagonal quadratic with heavy-tailed $h_i$ and on its random
rotation: predicted per-step bounds (curves) vs simulated losses (points); the rotated problem erases the
advantage. Figure (optional): sign-error probability vs batch size, Chebyshev and Gauss bounds vs Monte Carlo.

**Misses.** Idealized away: memory ($\beta_1,\beta_2$), $\epsilon$, coordinatewise smoothness as a global
assumption, symmetric noise. Where it breaks: signSGD without momentum does not converge at small batch;
Adam is not sign descent in the long-memory regime; worst-case rates do not explain the size of the Adam–SGD
gap (Chapter 14); the $\ell_\infty$ advantage depends on the coordinate system. Repairs: momentum and error
feedback (Karimireddy et al., ICML 2019 "Error feedback fixes SignSGD"), Lion's two betas, $\ell_\infty$
geometry analyses (Xie et al. 2025), Adam's implicit bias toward $\ell_\infty$ margins (Chapter 15).

**Exercises (ideas).** The Adam bound with bias correction; $\beta_1=\beta_2$ gives $\abs{m/\sqrt v}\le1$;
majority vote over $M$ workers; Lion with $\beta_1=\beta_2$ is signSGD with momentum; sign descent on a
separable quadratic reaches a two-cycle in $\lceil\abs{\theta_{0,i}}/\eta\rceil$ steps per coordinate; the
rotation bound; simulation: signSGD noise floor vs batch size; research: a rotation-invariant measure of
when sign descent helps.

### Chapter 3 — Orthogonalized Updates (key `spectral`)

**In the wild.** Muon (Jordan et al. 2024, blog "Muon: an optimizer for hidden layers in neural networks")
orthogonalizes the momentum of each hidden weight matrix with five Newton–Schulz iterations with
coefficients $(3.4445,-4.7750,2.0315)$ and is used with AdamW for embeddings, output layer and gains; it set
speed records on the modded-nanoGPT benchmark; Moonshot AI scaled it (Liu et al. 2025, arXiv:2502.16982) and
used a variant (MuonClip) for Kimi K2 **[verify]**; Shampoo won the external-tuning track of AlgoPerf
(Kasimbeg et al., ICLR 2025 **[verify]**). Observation: weight-matrix gradients have low stable rank;
orthogonalized updates have all singular values near one. Counter-observation: at 1B scale the measured
speedups of matrix optimizers over well-tuned AdamW shrink to about $1.1\times$ (Wen, Hall, Ma, Liang,
"Fantastic pretraining optimizers and where to find them", ICLR 2026).

**Idealization.** Steepest descent on matrices $G\in\R^{m\times n}$ under the spectral norm; the RMS→RMS
operator norm of a layer.

**Theory.**
- *Theorem (spectral steepest descent; complete proof).* The dual of $\norm\cdot_{\op}$ is $\norm\cdot_{\nuc}$;
  $\lmo_{\op}(G)\ni-UV^\top$ with equality of the set when $G$ has full rank (describe the maximizer set
  otherwise); $G^\sharp=-\norm{G}_{\nuc}\lmo$. (`\cref{thm:toolkit:vonneumann,thm:toolkit:polar}`.)
- *Proposition (Shampoo without accumulation is Muon without momentum; complete).*
  $(GG^\top)^{-1/4}G(G^\top G)^{-1/4}=UV^\top$ (pseudo-inverse powers) (Bernstein and Newhouse 2024).
- *Theorem (Newton–Schulz; complete proof).* For an odd polynomial $p(s)=as+bs^3+cs^5$, the matrix iteration
  $X\leftarrow aX+b(XX^\top)X+c(XX^\top)^2X$ acts on singular values by $p$ and keeps singular vectors
  (complete). The cubic $p(s)=\frac32s-\frac12s^3$ maps $(0,\sqrt3)$ into $(0,1]$ and converges to $1$,
  quadratically near $1$; with $X_0=G/\norm G_F$ it converges to $\polar(G)$. Muon's quintic is not
  convergent: it trades accuracy for speed by a large slope $a=3.4445$ at $0$. *Editor's numerics:* after five
  iterations every normalized singular value $\ge3\cdot10^{-3}$ lands in $[0.68,1.20]$, and every one $\ge10^{-2}$
  in $[0.68,1.13]$; the cubic, whose slope at $0$ is $3/2$, needs about ten iterations to lift $10^{-2}$ above
  $1/2$, the quintic about four. Polar
  Express (Amsel, Persson, Musco, Gower 2025 **[verify]**) chooses minimax-optimal polynomials per iteration.
- *Theorem (when spectral steps beat Frobenius steps; complete proof).* For a layer loss
  $\Loss(W)=\frac1{2n}\norm{WX-Y}_F^2$ with $A=XX^\top/n$: $L_F=\lmax(A)$ and $L_{\op}\le\tr A$ (equality when
  $d_{\mathrm{out}}\ge d_{\mathrm{in}}$). One-step guaranteed decrease $\norm{G}_{\nuc}^2/(2L_{\op})$ versus
  $\norm G_F^2/(2L_F)$, so spectral steepest descent wins iff the gradient's *nuclear rank*
  $\norm G_{\nuc}^2/\norm G_F^2$ exceeds the *stable rank* $\tr A/\lmax(A)$ of the layer's inputs (Davis and
  Drusvyatskiy 2025, "When do spectral gradient updates help in deep learning?" **[verify]**).
- *Proposition (why the spectral norm: the RMS→RMS norm; complete).* For $y=Wx$:
  $\norm{\Delta W\,x}_{\rms}\le\norm{\Delta W}_{\rms\to\rms}\norm x_{\rms}$, with equality for rank-one $\Delta W$
  whose right singular vector is $x/\norm x_2$. So a normalized spectral step of size $\eta$ in the RMS→RMS
  norm changes the layer's output by at most $\eta$ in RMS, whatever the widths (leads into Chapter 4).
- *Convergence.* Muon is normalized steepest descent with momentum in the spectral norm; its rates follow from
  Chapter 1 (deterministic) and Chapter 5 (stochastic, with momentum). Cite Li and Hong 2025, Kovalev 2025,
  Shen et al. 2025 **[verify all]** for published analyses; do not re-prove.

**Field guide entries.** Muon; Shampoo (pointing to Chapter 7); spectral descent (Carlson, Collins, Hsieh,
Carin, Cevher, AISTATS 2015 and NeurIPS 2015, the first spectral descent for deep learning **[verify]**).

**Reading.** Figure 1 (theory vs simulation): the singular-value map after $k=1,\dots,5$ iterations for the
cubic, Muon's quintic and Polar Express (exact polynomial composition, curves) against singular values of
iterated random matrices (points). Figure 2: spectral vs Frobenius descent on matrix least squares with inputs
of controlled stable rank: predicted crossover at nuclear rank = stable rank versus measured iterations to
target.

**Misses.** Idealized away: exact orthogonalization (Newton–Schulz is approximate and in low precision), one
matrix at a time (no cross-layer norm — Chapter 4), embeddings and gains (Adam handles them), stochasticity
(Chapter 5). Where it breaks: speedups shrink with scale and with longer training (Wen et al. 2026: SOAP
overtakes Muon at $8\times$ Chinchilla **[verify]**); Muon needs weight decay and update-RMS matching at scale
(Liu et al. 2025); attention logits can blow up (MuonClip, QK-clip **[verify]**); the stable-rank/nuclear-rank
comparison is one-step and worst case. Repairs: Polar Express and Gram–Newton–Schulz **[verify]**,
Hyperball (Wen et al., arXiv:2606.16899, 2026 **[verify]**), SOAP (Chapter 7), spectral-norm constraints via
weight decay (Chapter 13).

**Exercises (ideas).** Maximizer set of $\ip{G,D}$ over the spectral ball for rank-deficient $G$; the cubic
Newton–Schulz basin is exactly $(0,\sqrt3)$; design a quintic with prescribed slope at $0$; the polar factor is
the nearest orthogonal matrix in Frobenius norm; orthogonalized SGD on a rank-one gradient; simulation:
measure the nuclear rank of gradients of a small MLP; research: a theory of why gradients of trained
networks have high nuclear rank relative to activation stable rank.

### Chapter 4 — Norms for Networks (key `modular`)

**In the wild.** Under the standard parametrization the best learning rate drifts with width; under $\mu$P it
stays put, so hyperparameters tuned on a small model transfer (Yang and Hu, ICML 2021; Yang et al. 2022,
"Tensor Programs V: tuning large neural networks via zero-shot hyperparameter transfer"); Adam's
hidden-layer learning rate is scaled by $1/\mathrm{fan\_in}$ in $\mu$P; Muon scales its update by
$\sqrt{d_{\mathrm{out}}/d_{\mathrm{in}}}$ or matches an RMS; the Modula library builds the norm of a network
from its modules (Large et al., NeurIPS 2024). Everett et al. (ICML 2024, "Scaling exponents across
parameterizations and optimizers") measured per-layer prescriptions across optimizers.

**Idealization.** A network as a composition of layers, each with a norm on its weights (RMS→RMS operator norm
for linear layers); inputs and features measured in RMS; one gradient step from a single example (rank-one
layer gradients); the desideratum that each layer's features move by $\Theta(1)$ in RMS per step
("feature learning") and that no layer's update dominates.

**Theory.**
- *Theorem (the norm of a network and its dual; complete proof).* For
  $\norm{\theta}=\max_ls_l\norm{W_l}_{(l)}$ with scales $s_l>0$: $\norm{g}_*=\sum_l\norm{G_l}_{(l),*}/s_l$ and
  $\lmo(g)=(\lmo_{(l)}(G_l)/s_l)_l$. Normalized steepest descent in a max-norm *is* per-layer normalized
  descent with per-layer learning rates $\eta/s_l$; unnormalized descent couples the layers through the single
  factor $\norm g_*$ (Bernstein and Newhouse, ICML 2025, "Modular duality in deep learning"; Large et al. 2024).
- *Theorem (the learning rate each optimizer needs for width-independent feature updates; complete proof).*
  For a hidden linear layer with $d_{\mathrm{in}}\times d_{\mathrm{out}}$ weights, input $x$ with
  $\norm x_{\rms}=1$, and a single-example gradient $G=\delta x^\top$: (i) SGD with step $\eta$ changes the
  output by $\norm{\Delta Wx}_{\rms}=\eta\norm\delta_2\norm x_2^2/\sqrt{d_{\mathrm{out}}}$; (ii) sign descent
  (Adam without memory): $\Delta W=-\eta\,\sign(\delta)\sign(x)^\top$, so $\norm{\Delta W}_{\rms\to\rms}=\eta
  d_{\mathrm{in}}$ and $\norm{\Delta Wx}_{\rms}=\eta\norm x_1\cdot\norm{\sign\delta}_{\rms}/1$ — fix the exact
  form — of order $\eta d_{\mathrm{in}}$, whence $\eta\propto1/d_{\mathrm{in}}$ ($\mu$P's Adam rule); (iii) Muon
  scaled by $\sqrt{d_{\mathrm{out}}/d_{\mathrm{in}}}$: $\norm{\Delta W}_{\rms\to\rms}=\eta$ and
  $\norm{\Delta Wx}_{\rms}=\eta$ exactly for rank-one gradients. State the dependence on the width of $\delta$
  honestly (it is set by the layers above); give the corresponding SGD rule.
- *Theorem (the spectral condition; complete proof of the upper bound and of the rank-one lower bound).* For a
  deep linear network with $\norm{W_l}_{\rms\to\rms}=\Theta(1)$ and updates with
  $\norm{\Delta W_l}_{\rms\to\rms}=\Theta(1)$: features $h_l$ satisfy $\norm{h_l}_{\rms}=O(1)$ and
  $\norm{\Delta h_l}_{\rms}=O(1)$ by a telescoping product bound; with rank-one aligned updates the change is
  $\Theta(1)$. This is Yang, Simon and Bernstein, "A spectral condition for feature learning" (arXiv:2310.17813,
  2023 **[verify]**), whose statement in operator norms is
  $\norm{W_l}_{\op}=\Theta(\sqrt{d_l/d_{l-1}})$, $\norm{\Delta W_l}_{\op}=\Theta(\sqrt{d_l/d_{l-1}})$.
- *Statements.* Well-normed modules and width-independent Lipschitz and smoothness constants in the modular
  norm (Large et al. 2024) — sketch; depth: residual branches scaled by $1/\sqrt{N}$ or $1/N$ and depth-$\mu$P
  (Bordelon et al., ICLR 2024; Yang, Yu, Zhu, Hayou, ICLR 2024; CompleteP, Dey et al. 2025 **[verify]**) —
  statement only.

**Reading.** Figure (theory vs simulation): for a small MLP (numpy or torch), the optimal learning rate as a
function of width for SGD and Adam under the standard parametrization and for the scaled rules, plus Muon;
predicted slopes ($-1$ in $\log$–$\log$ for Adam-SP hidden layers, $0$ for the scaled rules) against the
measured optima. A second panel: $\norm{\Delta h}_{\rms}$ after one step versus width, predicted vs measured.

**Misses.** Idealized away: rank-one single-example gradients (minibatch gradients are low stable rank, not
rank one), $\Theta(1)$ heuristics in place of constants, linear layers, alignment at every step. Where it
breaks: embeddings and output layers need different norms (the max over layers is a modelling choice);
transfer degrades over long training and with weight decay (Wang and Aitchison 2025 **[verify]**); depth
transfer needs extra scaling; the norm says nothing about which $\eta$ is optimal, only how it scales.
Repairs: u-$\mu$P (Blake et al., ICLR 2025 **[verify]**), depth-$\mu$P, Hyperball's norm fixing (2026),
"Effective learning rate" analyses (Chapter 13).

**Exercises (ideas).** Dual of a weighted max norm; per-layer rates for an embedding layer (one-hot inputs make
$\ell_1\to\rms$ the natural norm); SGD's $\mu$P rule from the rank-one calculation; the output-layer scaling;
a two-layer network where the max norm and the sum norm give different steps; simulation: learning-rate
transfer; research: a principled norm for attention.

### Chapter 5 — Moving Averages (key `momentum`)

**In the wild.** Momentum $0.9$ almost everywhere; Nesterov momentum in Muon; Lion's two coefficients;
exponential moving averages of the weights (model EMA) and schedule-free methods (Defazio et al., NeurIPS 2024,
"The road less scheduled"; winner of AlgoPerf's self-tuning track **[verify]**). Observation: normalized and
sign methods need momentum at small batch sizes.

**Idealization.** Three settings: (a) deterministic quadratics (acceleration); (b) quadratics with additive
gradient noise (averaging); (c) normalized steepest descent in a general norm with stochastic gradients.

**Theory.**
- *Theorem (heavy ball on quadratics; complete proof).* For $\theta_{t+1}=\theta_t-\eta\nabla\Loss(\theta_t)+
  \beta(\theta_t-\theta_{t-1})$ on a quadratic with spectrum in $[\mu,L]$: with
  $\eta=4/(\sqrt L+\sqrt\mu)^2$, $\beta=((\sqrt\kappa-1)/(\sqrt\kappa+1))^2$ the spectral radius of every
  $2\times2$ mode matrix is $\sqrt\beta=(\sqrt\kappa-1)/(\sqrt\kappa+1)$ (Polyak 1964).
- *Proposition (stability regions; complete; editor checked numerically).* Heavy ball is stable on a mode with
  curvature $\lambda_i$ iff $0<\eta\lambda_i<2+2\beta$ (and $\abs\beta<1$); the EMA form
  $m_t=\beta m_{t-1}+(1-\beta)g_t$, $\theta_{t+1}=\theta_t-\eta m_t$ is heavy ball with step $\eta(1-\beta)$, so
  stable iff $\eta\lambda_i<(2+2\beta)/(1-\beta)$, which is $38$ for $\beta=0.9$ (`\cref{lem:toolkit:recurrence}`).
  Nesterov's region (derive). Reused in Chapter 8 for Adam's $38/\eta$.
- *Theorem (in the noise-dominated regime momentum is a rescaled step size; complete proof; editor checked).*
  On $\frac12\lambda\theta^2$ with additive noise of variance $\sigma^2$, heavy ball's stationary variance is
  $\frac{\eta\sigma^2(1+\beta)}{\lambda(1-\beta)(2(1+\beta)-\eta\lambda)}$ (`\cref{lem:toolkit:lyapunov}`), which
  agrees to first order in $\eta$ with SGD at step $\eta/(1-\beta)$; the mean contracts at the same rate to
  first order (Kidambi et al., ICLR 2018; Zhang et al., NeurIPS 2019, noisy quadratic model).
- *Theorem (normalized steepest descent with momentum in any norm; complete proof).* Let $\Loss$ be $L$-smooth
  w.r.t. $\norm\cdot$, $\E\norm{\zeta_t}_2^2\le\sigma^2$ with $\norm\cdot_*\le\kappa\norm\cdot_2$, and
  $\theta_{t+1}=\theta_t+\eta\lmo(m_t)$, $m_t=\beta m_{t-1}+(1-\beta)g_t$. Then
  $\frac1T\sum_t\E\norm{\nabla\Loss(\theta_t)}_*\le\frac{\Delta_0}{\eta T}+\frac{L\eta}2+\frac{2L\eta\beta}{1-\beta}
  +2\kappa\sigma\sqrt{\tfrac{1-\beta}{1+\beta}}+\frac{2\norm{m_{-1}-\nabla\Loss(\theta_0)}_*}{T(1-\beta)}$ (fix
  constants), giving $O(T^{-1/4})$ with $1-\beta\asymp T^{-1/2}$, $\eta\asymp T^{-3/4}$. Key step:
  $\ip{\nabla\Loss,\lmo(m)}\le-\norm{\nabla\Loss}_*+2\norm{m-\nabla\Loss}_*$. With $\beta=0$ the noise term
  $2\kappa\sigma$ does not shrink: normalized SGD without momentum stalls at a noise floor unless $B$ grows
  (Cutkosky and Mehta, ICML 2020, for $\ell_2$; general norms in Pethick et al. 2025, Kovalev 2025
  **[verify]**). This is the reason Muon, Lion and signSGD use momentum.
- *Statements.* Nesterov acceleration (one paragraph); schedule-free SGD's guarantee (Defazio et al. 2024) and
  primal averaging; model EMA ≈ learning-rate decay (heuristic ◇, with Chapter 12).

**Field guide entries.** SGD with momentum (heavy ball and EMA forms), Nesterov, Lion (full entry),
schedule-free SGD/AdamW.

**Reading.** Figure (theory vs simulation): the noise floor $\E\norm{\nabla\Loss}$ of normalized SGD with
momentum on a noisy quadratic as a function of $1-\beta$, predicted $\propto\sqrt{1-\beta}$ (plus the
$\eta/(1-\beta)$ lag term) vs simulation; and/or the stationary variance of heavy ball vs $\beta$ (exact formula
vs simulation).

**Misses.** Idealized away: quadratics and bounded-variance noise; constant $\beta$. Where it breaks: acceleration
is irrelevant once noise dominates; the worst-case $T^{-1/4}$ rate does not explain why $\beta=0.9$ works
universally; heavy-tailed noise; momentum interacts with the stability threshold (Chapter 8). Repairs:
variance reduction (STORM, Cutkosky and Orabona 2019), AdEMAMix (two EMAs; Pagliardini et al., ICLR 2025
**[verify]**), schedule-free and iterate averaging, Nesterov in Muon.

**Exercises (ideas).** The heavy-ball $2\times2$ matrix and its eigenvalues; Nesterov's stability region; EMA vs
heavy-ball equivalence and bias correction; normalized SGD without momentum on a 1-D noisy problem has a floor
$\approx\sigma$; Lion as a two-$\beta$ EMA; simulation: noise floor vs $\beta$; research: why $\beta=0.9$.

### Chapter 6 — Adaptive Preconditioning (key `adaptive`)

**In the wild.** Adam and AdamW with $\beta_2=0.999$ in vision and $0.95$ in language models (GPT-3, LLaMA
**[verify]**), $\epsilon=10^{-8}$ (smaller in some large runs); AdaGrad in sparse settings; Adafactor's
factored second moments (Shazeer and Stern, ICML 2018); bias correction. Observations: large $\beta_2$ is
associated with instability in large language models; Adam's nonconvergence counterexample (Reddi, Kale,
Kumar, ICLR 2018) contrasts with Adam converging in practice when $\beta_2$ is large (Zhang et al., NeurIPS
2022, "Adam can converge without any modification on update rules").

**Idealization.** Online convex optimization for AdaGrad; stochastic nonconvex optimization for Adam and
RMSProp; preconditioning as steepest descent in a data-adapted quadratic norm $\norm\cdot_{P_t}$.

**Theory.**
- *Theorem (diagonal AdaGrad; complete proof).* On convex losses over a box of $\ell_\infty$-radius $D_\infty$,
  regret $\le\sqrt2\,D_\infty\sum_i\sqrt{\sum_tg_{t,i}^2}$ (Duchi, Hazan, Singer, JMLR 2011; McMahan and
  Streeter, COLT 2010) via `\cref{lem:toolkit:adagrad}`; compare online gradient descent's
  $D_2\sqrt{\sum_t\norm{g_t}_2^2}$ (`\cref{thm:toolkit:ogd}`). AdaGrad is competitive with the best diagonal
  preconditioner in hindsight: $\big(\sum_i\norm{g_{1:T,i}}_2\big)^2=\min\{\sum_t\norm{g_t}^2_{\diag(s)^{-1}}:
  s\ge0,\sum_is_i=1\}$ (complete; Cauchy–Schwarz).
- *Proposition (Adam's nonconvergence; complete).* Reddi et al.'s online counterexample — re-derive with the
  chapter's conventions and check numerically.
- *Theorem (Adam/RMSProp converge for large $\beta_2$; statement + proof sketch, or a complete proof of a
  clean special case).* Défossez, Bottou, Bach, Usunier (TMLR 2022) "A simple convergence proof of Adam and
  Adagrad": $O(\log T/\sqrt T)$ with explicit dependence on $\beta_1,\beta_2$; Zhang et al. 2022; Li, Rakhlin,
  Jadbabaie (NeurIPS 2023). Prefer a complete proof of AdaGrad-Norm's adaptivity to unknown smoothness (Ward,
  Wu, Bottou, ICML 2019 / JMLR 2020; Levy, Yurtsever, Cevher 2018) if a full Adam proof does not fit.
- *Proposition (what $\beta_2$ controls; complete).* Chapter 2's bound on $\abs{m_t/\sqrt{v_t}}$ and the
  "quiet-then-loud" worst case: after a long stretch of tiny gradients, a coordinate whose gradient jumps gets
  a step of about $\eta(1-\beta_1)/\sqrt{1-\beta_2}$ ($3.16\eta$ at $\beta_2=0.999$, $0.45\eta$ at $0.95$; editor
  checked): the memory of $v$ is the timescale $1/(1-\beta_2)$ over which Adam's preconditioner lags.
- *Proposition (Adafactor's factorization; complete).* The rank-one nonnegative matrix minimizing the
  generalized KL (I-)divergence to $V$ is $(V\mathbf1)(\mathbf1^\top V)/(\mathbf1^\top V\mathbf1)$ (row sums times
  column sums).
- *Remark ($\epsilon$).* For $\abs{g}\ll\epsilon$ Adam is SGD with step $\eta/\epsilon$; $\epsilon$ interpolates
  between sign descent and SGD; why it is set so small, and when it matters (Chapter 10).
- *Adam's two limits.* $\beta_2\to0$: sign descent ($\ell_\infty$); $\beta_2\to1$ with stationary gradients:
  diagonal preconditioning by $\diag(\E g^2)^{1/2}$ — not the Fisher (Kunstner, Hennig, Balles, NeurIPS 2019).

**Field guide entries.** AdaGrad, RMSProp, Adam (full), AdamW (pointing to Chapter 13), Adafactor.

**Reading.** Figure (theory vs simulation): online logistic regression with sparse, heavy-tailed feature
frequencies — regret of OGD and AdaGrad vs their bounds. Figure 2 (optional): the largest per-coordinate Adam
step after a quiet period vs $\beta_2$, bound vs simulation.

**Misses.** Idealized away: convex losses for the regret theory; bounded gradients; $\epsilon\to0$. Where it
breaks: regret bounds say nothing about the nonconvex speed of Adam on transformers (Chapter 14); Adam's
convergence theory needs problem-dependent $\beta_2$; the diagonal preconditioner is not curvature.
Repairs: AMSGrad, Adam with model EMA (Ahn and Cutkosky, NeurIPS 2024 **[verify]**), Kronecker preconditioners
(Chapter 7), Adam-mini and block-wise learning rates (Zhang et al. 2024 **[verify]**).

**Exercises (ideas).** The AdaGrad lemma by induction; full-matrix AdaGrad regret; Reddi's example with
$\beta_1>0$; bias correction derivation; Adafactor's I-divergence minimizer; simulation: AdaGrad vs OGD on
sparse data; research: the right $\beta_2$ as a function of batch size (Marek et al., NeurIPS 2025, keep the
half-life in tokens fixed).

### Chapter 7 — Kronecker Preconditioners (key `kronecker`)

**In the wild.** Shampoo (Gupta, Koren, Singer, ICML 2018; distributed by Anil et al. 2020 **[verify]**), with
grafting and infrequent root computations; K-FAC (Martens and Grosse, ICML 2015); SOAP, Adam in Shampoo's
eigenbasis (Vyas et al., ICLR 2025); "Purifying Shampoo" (Eschenhagen et al. 2025 **[verify]**). AlgoPerf
(Dahl et al. 2023; Kasimbeg et al., ICLR 2025).

**Idealization.** Full-matrix AdaGrad on $\vect(W)$ and its Kronecker approximations; a layer loss whose
Hessian is a Kronecker product $B\otimes A$.

**Theory.**
- *Theorem (the Kronecker bound; complete proof).* For $G_1,\dots,G_T\in\R^{m\times n}$ of rank $\le r$ and
  $\eps>0$: $\eps I+\sum_t\vect(G_t)\vect(G_t)^\top\preceq r\,(\eps I+\sum_tG_tG_t^\top)^{1/2}\otimes
  (\eps I+\sum_tG_t^\top G_t)^{1/2}$ in the right vec convention (Gupta et al. 2018, Lemma 8). Proof: single-$G$
  case via the SVD and Cauchy–Schwarz; summation via joint concavity of the matrix geometric mean
  (`\cref{thm:toolkit:geomean}`) applied to the commuting pair $A_t\otimes I$, $I\otimes B_t$.
- *Theorem (Shampoo's regret; statement + proof sketch from the Kronecker bound and the full-matrix AdaGrad
  analysis).* $\sqrt{2r}\,D\,\tr(L_T^{1/4})\tr(R_T^{1/4})$ (check exact form in Gupta et al.).
- *Proposition (Kronecker-factored curvature; complete).* For $\Loss(W)=\frac12\norm{A^{1/2}(W-W_\star)B^{1/2}}_F^2$
  the Hessian on $\vect(W)$ is $B\otimes A$; gradient descent converges at rate $1-2/(\kappa(A)\kappa(B)+1)$ at the
  best step; the K-FAC step $A^{-1}GB^{-1}$ converges in one step; Shampoo's step $(GG^\top)^{-1/4}G(G^\top G)^{-1/4}$
  with accumulated factors — compute its effective conditioning (square roots of the condition numbers in the
  stationary case — derive and check numerically).
- *Proposition (K-FAC's approximation; complete).* For $G=\delta a^\top$, the Fisher block is
  $\E[(aa^\top)\otimes(\delta\delta^\top)]$ in a vec convention; K-FAC replaces it by
  $\E[aa^\top]\otimes\E[\delta\delta^\top]$, exact when $a$ and $\delta$ are independent.
- *Statements.* Shampoo as one step of power iteration toward the optimal Kronecker approximation of the
  AdaGrad matrix (Morwani, Shapira, Vyas, Malach, Kakade, Janson, ICLR 2025, "A new perspective on Shampoo's
  preconditioner" **[verify]**); SOAP = Adam in a slowly rotating eigenbasis; without accumulation Shampoo is
  Chapter 3's orthogonalization — the unification line closes.

**Field guide entries.** Shampoo (full), K-FAC, SOAP.

**Reading.** Figure (theory vs simulation): on matrix least squares with ill-conditioned $A$ and $B$, losses of
GD, Adam, Muon, one-sided and two-sided Kronecker preconditioning: predicted rates from the condition numbers
(curves) vs runs (points). Figure 2 (optional): the Kronecker inequality's tightness ratio vs $r$.

**Misses.** Idealized away: the Kronecker structure of curvature (true for one linear layer with independent
inputs and output gradients, approximate otherwise); online convexity in the regret theory; exact matrix roots
(computed rarely, in low precision). Where it breaks: regret bounds don't explain Shampoo's empirical wins;
memory and compute overheads; speedups shrink with scale (Wen et al. 2026). Repairs: eigenvalue-corrected
Shampoo/SOAP, Kronecker-factored Adam variants, PSGD (Li 2018 **[verify]**).

**Exercises (ideas).** The vec identity and Kronecker eigenvalues; single-$G$ Kronecker bound; the geometric
mean of commuting matrices; K-FAC for a two-layer linear network; Shampoo with $-1/2$ powers; simulation;
research: when is the gradient covariance close to a Kronecker product?

### Chapter 8 — The Edge of Stability (key `eos`)

**In the wild.** Practitioners pick the largest learning rate that does not diverge, and training then runs
with the sharpness at the stability threshold: full-batch gradient descent sharpens progressively to $2/\eta$
and stays there while the loss keeps falling (Cohen, Kaur, Li, Kolter, Talwalkar, ICLR 2021); Adam's
preconditioned sharpness hovers near $38/\eta$ for $\beta_1=0.9$ (Cohen et al. 2022, "Adaptive gradient methods
at the edge of stability", arXiv:2207.14484); with minibatches a "batch sharpness" stabilizes (Andreyev and
Beneventano 2025, "Edge of stochastic stability" **[verify]**). Volume 2, Chapter 15 solves the product model and
the catapult; cite it, do not repeat it.

**Idealization.** Quadratic stability thresholds; then the cubic model of Damian, Nichani and Lee (ICLR 2023,
"Self-stabilization: the implicit bias of gradient descent at the edge of stability"): the coordinate $x$ along
the top Hessian eigenvector and the sharpness offset $y=\eta S-2$ (or $S-2/\eta$), with progressive sharpening
rate $\alpha=-\ip{\nabla\Loss,\nabla S}>0$ and coupling $\beta=\frac12\norm{\nabla S}^2$. A concrete loss realizing
it exactly: $\Loss(x,y)=\frac12(2/\eta+y)x^2-\alpha y$, whose gradient descent map is
$x\mapsto-(1+\eta y)x$, $y\mapsto y+\eta(\alpha-\frac12x^2)$.

**Theory.**
- *Proposition (thresholds; complete).* GD $\eta S<2$; heavy ball $\eta S<2+2\beta$; EMA momentum
  $\eta S<(2+2\beta)/(1-\beta)$; Adam with a frozen preconditioner $P=\diag(\sqrt{\hat v})+\epsilon$:
  $\eta\lmax(P^{-1}H)<(2+2\beta_1)/(1-\beta_1)$ ($=38$ at $\beta_1=0.9$). Reuse Chapter 5's proposition.
- *Theorem (self-stabilization in the continuous limit; complete proof).* With $\tau=\eta t$ and
  $A=\log x^2$, the two-step-averaged dynamics is $\dot A=2y$, $\dot y=\alpha-\frac12e^A$ (derive the scaling
  carefully, including which quantities are $O(1)$). It conserves $\mathcal H=y^2+\frac12e^A-\alpha A$; every
  orbit is closed; over a period the time averages are exactly $\langle y\rangle=0$ (the sharpness averages
  $2/\eta$) and $\langle x^2\rangle=2\alpha$ (the oscillation energy that cancels progressive sharpening).
  Proof: conserved quantity; level sets of a strictly convex function of $(A,y)$ are closed curves; averages of
  derivatives over a period vanish.
- *Central flow (statement + derivation ◇).* In general, $\frac{d\theta}{d\tau}=-\nabla\Loss-\frac12\sigma^2\nabla S$
  with $\sigma^2=2\max(0,-\ip{\nabla\Loss,\nabla S})/\norm{\nabla S}^2$ on the edge — gradient flow projected onto
  $\{S\le2/\eta\}$ (Cohen, Damian, Talwalkar, Kolter, Lee, ICLR 2025, "Understanding optimization in deep
  learning with central flows"); the model above has $\sigma^2=2\alpha$, matching the theorem. Adaptive
  optimizers: "acceleration via regularization" (same paper) and the Adam/RMSProp central flow.
- *Discrete gradient descent (honest numerics; editor checked).* The exact GD map follows the conserved
  quantity for many oscillation cycles but drifts outward slowly (in the editor's runs with $\eta=0.01$,
  $\alpha=1$ started near the period-two orbit, $\mathcal H$ grew from $10$ to overflow within $10^5$ steps; with
  $\eta=0.002$ the drift was much slower). State what is proved (continuous limit) and what is observed
  (discrete drift), and explain where damping comes from in real networks (higher-order terms; Damian et al.'s
  assumptions and time horizon — check their theorem's exact statement and quote its hypotheses).

**Reading.** Figure 1 (theory vs simulation): GD iterates of the cubic model in the $(A,y)$ plane on top of the
level curves of $\mathcal H$; running averages of $\eta S$ and $x^2$ converging to $2$ and $2\alpha$. Figure 2:
Adam on a small network (numpy or torch), preconditioned sharpness vs $38/\eta$ (and GD's sharpness vs $2/\eta$),
sharpness by power iteration.

**Misses.** Idealized away: the cause of progressive sharpening ($\alpha$ is an input); constant $\alpha,\beta$; a
single unstable eigenvalue; full batch. Where it breaks: why sharpness rises is unexplained; with cross-entropy
the sharpness falls late in training; several eigenvalues at the edge; SGD's noise changes the threshold;
discrete-time drift. Repairs: central flows, multi-eigenvalue self-stabilization, stochastic EoS, volume 2's
catapult, Arora–Li–Panigrahi (ICML 2022), "minimalist" EoS models (Zhu et al., ICLR 2023; Ahn et al., NeurIPS
2023) **[verify]**, Adam rod flows (arXiv:2605.06821, 2026 **[verify]**).

**Exercises (ideas).** Nesterov's threshold; the frozen-preconditioner Adam threshold; derive the cubic model
from a Taylor expansion; the period of small oscillations of $\mathcal H$ ($2\pi/\sqrt{2\alpha\cdot\ldots}$ —
derive); a two-eigenvalue model; simulation: Adam's $38/\eta$; research: a model where progressive sharpening is
an output.

### Chapter 9 — Warmup (key `warmup`)

**In the wild.** Linear warmup over a fixed number of steps (2000 in many LLM recipes **[verify]**) or a few
percent of training; Post-LN transformers need it (Xiong et al., ICML 2020); for Adam the rule of thumb
$2/(1-\beta_2)$ steps (Ma and Yarats, AAAI 2021); large-batch SGD on ImageNet (Goyal et al. 2017, five epochs).
Observations: the main benefit of warmup is to make larger target learning rates trainable, and during warmup
the sharpness falls as $2/\eta_t$ rises (Kalra and Barkeshli, NeurIPS 2024, "Why warmup the learning rate?");
warmup reduces curvature and prevents early instabilities (Gilmer et al., ICLR 2022); RAdam attributes Adam's
need for warmup to the variance of its early second-moment estimate (Liu et al., ICLR 2020); Kosson, Messmer,
Jaggi (NeurIPS 2024) attribute it to large early angular updates.

**Idealization.** (a) Loss-dependent curvature: $\norm{\nabla^2\Loss(\theta)}_{\op}\le K_0+K_1(\Loss(\theta)-\Loss^\star)$
(Alimisis, Islamov, Lucchi, "Why do we need warm-up? A theoretical perspective", ICML 2026, arXiv:2510.03164;
Riabinin et al., "Where does warm-up come from?", NeurIPS 2026, arXiv:2602.05813 **[verify both]**); (b) the
edge of stability with a growing step size (Chapter 8); (c) Adam's first steps.

**Theory.**
- *Lemma (a local descent lemma under loss-dependent curvature; complete proof).* If $\eta\le1/(K_0+K_1\Delta)$
  at a point with suboptimality $\Delta$ and gradient $g$, then $\Loss(\theta-\eta g)\le\Loss(\theta)-\frac\eta2\norm g_2^2$.
  Proof by a continuity argument along the segment: as long as the loss on the segment stays below its
  starting value the curvature bound $K_0+K_1\Delta$ applies, which keeps the derivative negative.
- *Theorem (warmup is optimal under loss-dependent curvature; complete proof).* Under the PL inequality
  $\norm{\nabla\Loss}^2\ge2\mu\Delta$ (`\cref{lem:toolkit:pl}`): the adaptive step $\eta_t=1/(K_0+K_1\Delta_t)$
  gives $\Delta_{t+1}\le\Delta_t-\mu\Delta_t/(K_0+K_1\Delta_t)$: a linear decrease by about $\mu/K_1$ per step while
  $K_1\Delta\gg K_0$, then a linear rate $\mu/(2K_0)$; any constant step that is safe from the start must be
  $\le1/(K_0+K_1\Delta_0)$ and pays $(K_0+K_1\Delta_0)/\mu\cdot\log(\Delta_0/\eps)$ steps. The safe step size
  *increases* as the loss falls: a warmup, of hyperbolic shape $1/(K_0+K_1(\Delta_0-\mu t/K_1))$ (derive the
  exact profile; compare with Alimisis et al.'s upper and lower bounds and quote them).
- *Example (complete).* Exponential-type losses satisfy loss-dependent curvature: $\frac{d^2}{dx^2}e^{-x}=e^{-x}$
  ($K_0=0,K_1=1$); $2\cosh x-2$ has curvature exactly $2+\Loss$; linear softmax regression with $\norm x\le R$ has
  $\norm{\nabla^2\ell}\le R^2(1-p_y)\le R^2\ell$ (check). So cross-entropy has curvature bounded by the loss.
- *Proposition (Adam's first steps; complete).* With bias correction and $\epsilon=0$ the first update is
  $-\eta\sign(g_0)$ whatever the gradient's size; for $t\ll1/(1-\beta_2)$, $\hat v_t$ averages $t$ squared
  gradients (state precisely); heuristic ◇ for the $2/(1-\beta_2)$ rule.
- *Heuristic ◇ (warmup at the edge).* Combine Chapter 8: with $\eta_t$ increasing slowly, the central flow keeps
  $S(\theta_t)\approx2/\eta_t$, so warmup *drives the sharpness down*; if $\eta_t$ grows faster than the
  self-stabilization time scale, a catapult or divergence results (volume 2, Chapter 15).

**Reading.** Figure 1 (theory vs simulation): GD on a loss-dependent-curvature example (e.g. logistic
regression from a bad initialization, or $2\cosh x-2$ in many coordinates): adaptive step vs constant step vs
linear warmup — loss curves, theory bounds vs runs. Figure 2: a small network (numpy or torch) with linear
warmup to a large $\eta$: sharpness tracking $2/\eta_t$ during warmup; without warmup the same $\eta$ diverges.

**Misses.** Idealized away: a global loss-curvature inequality; PL; full batch; fixed preconditioner. Where it
breaks: the inequality is measured, not derived, for deep networks; warmup also interacts with Adam's
statistics and with normalization layers; the length of warmup in practice is set by rules of thumb that the
theory reproduces only up to constants. Repairs: adaptive schedules from the theory (Riabinin et al. 2026),
GI-Adam (Kalra and Barkeshli 2024), normalized updates that remove the need (Kosson et al. 2024).

**Exercises (ideas).** The curvature–loss bound for softmax regression; the schedule profile under PL;
warmup for $e^{-x}$ with $K_0=0$ (step can grow without bound — the large-step regime of logistic regression,
Wu, Bartlett, Telgarsky, Yu, COLT 2024 **[verify]**); RAdam's rectification term; simulation; research:
derive the curvature–loss inequality for a two-layer network.

### Chapter 10 — Clipping and Spikes (key `spikes`)

**In the wild.** Global gradient-norm clipping at $1.0$ in most LLM recipes; loss spikes in large runs, handled
by restarting from a checkpoint and skipping batches (PaLM, Chowdhery et al. 2022 **[verify]**); z-loss (PaLM)
and QK-layernorm (Dehghani et al., ICML 2023; Wortsman et al., ICLR 2024, "Small-scale proxies for
large-scale transformer training instabilities"); MuonClip/QK-clip for Muon (Kimi K2, 2025 **[verify]**);
lowering $\beta_2$ and $\epsilon$. Observations: attention-logit growth and output-logit divergence cause
instabilities at high learning rates in small proxies (Wortsman et al.); spikes follow Adam updates that are
large because $v$ is small (Molybog et al. 2023, "A theory on Adam instability in large-scale machine learning"
**[verify]**).

**Idealization.** $(L_0,L_1)$-smoothness $\norm{\nabla^2\Loss(\theta)}\le L_0+L_1\norm{\nabla\Loss(\theta)}$ (Zhang,
He, Sra, Jadbabaie, ICLR 2020, "Why gradient clipping accelerates training"); heavy-tailed noise (Zhang et al.,
NeurIPS 2020, "Why are adaptive methods good for attention models?"); the mechanisms of three instabilities.

**Theory.**
- *Lemma (local smoothness under $(L_0,L_1)$; complete proof).* For $\norm u\le c/L_1$:
  $\norm{\nabla\Loss(\theta+u)-\nabla\Loss(\theta)}\le(A L_0+BL_1\norm{\nabla\Loss(\theta)})\norm u$ with explicit
  $A,B$ depending on $c$ (Grönwall, `\cref{thm:toolkit:gronwall}`).
- *Theorem (clipped GD; complete proof).* Clipped GD with $\eta_t=\min\{\eta_c,\gamma/\norm{g_t}\}$ reaches
  $\norm{\nabla\Loss}\le\eps$ in $O\!\big(L_0\Delta_0/\eps^2+L_1^2\Delta_0/L_0\big)$ steps, while GD with a fixed
  step needs $\Omega\big((L_0+L_1M)\Delta_0/\eps^2\big)$ on some function of the class, $M$ the largest gradient
  norm on the sublevel set (Zhang et al. 2020; state the lower bound and prove the upper bound; check
  constants).
- *Proposition (normalized steepest descent is clipped by construction; complete).* For Chapter 1's normalized
  updates the step length is $\eta$ in the chosen norm, so the $(L_0,L_1)$ analysis applies to sign descent,
  Muon and (with Chapter 2's bound) Adam.
- *Proposition (anatomy of an Adam spike; complete).* Chapter 6's quiet-then-loud bound as a mechanism, with
  the effect of $\epsilon$; and output-logit drift: cross-entropy is invariant under adding a constant to all
  logits, so $\log Z$ is a flat direction that z-loss $(\log Z)^2$ pins (complete); attention logits
  $q^\top k$ are quadratic in the weight scale and QK-normalization bounds them by the gains (complete).
- *Statement.* Clipped SGD under heavy-tailed noise with only a $p$-th moment, $1<p\le2$: rate
  $T^{-(p-1)/(3p-2)}$ (Zhang et al. 2020) — sketch.

**Reading.** Figure 1 (theory vs simulation): GD vs clipped GD on an $(L_0,L_1)$-smooth function (e.g.
$\Loss(x)=x^4$ or $\cosh$ in several coordinates) from a range of initial gradient norms: iterations to target,
predicted scaling vs runs. Figure 2: Adam's largest step after a quiet period as a function of $\beta_2$ and
$\epsilon$, bound vs simulation; or a small attention model's logit growth with and without QK-norm.

**Misses.** Idealized away: deterministic gradients in the main theorem; global constants. Where it breaks:
real spikes have several interacting causes (data, precision, optimizer state); the $(L_0,L_1)$ condition is
measured, not derived; clipping thresholds in practice are rarely active late in training. Repairs: clipping
analyses for Adam (Zhang et al. 2020; Li et al. 2023), spike-aware optimizers, normalization layers.

**Exercises (ideas).** $(L_0,L_1)$ constants of $x^4$, $e^x$ and logistic loss; normalized GD on $e^x$;
z-loss's gradient; QK-norm bound; Adam's update with $\epsilon$; simulation; research: a solvable model of a
loss spike.

### Chapter 11 — Noise and Batch Size (key `noise`)

**In the wild.** Batch sizes in the millions of tokens; the linear scaling rule for SGD (Goyal et al. 2017;
Krizhevsky 2014) and the square-root rule for Adam (Hoffer, Hubara, Soudry 2017; Malladi, Lyu, Panigrahi,
Arora, NeurIPS 2022); critical batch size and the gradient noise scale (McCandlish, Kaplan, Amodei 2018,
"An empirical model of large-batch training"); perfect scaling then diminishing returns (Shallue et al., JMLR
2019); critical batch size grows with data size (Zhang et al., ICLR 2025, "How does critical batch size scale
in pre-training?" **[verify]**); batch-size warmup.

**Idealization.** The noisy quadratic model (NQM): $\Loss(\theta)=\frac12\sum_ih_i\theta_i^2$, stochastic gradient
$H\theta+\zeta$ with $\Cov(\zeta)=C/B$, typically $C\propto H$ (Zhang, Li, Nado, Martens, Sachdeva, Dahl,
Shallue, Grosse, NeurIPS 2019, "Which algorithmic choices matter at which batch sizes?").

**Theory.**
- *Theorem (exact dynamics of SGD in the NQM; complete proof).* $\E\theta_{t+1,i}^2=(1-\eta h_i)^2\E\theta_{t,i}^2+
  \eta^2c_i/B$; closed forms for the expected loss at every $t$ and the stationary loss
  $\sum_i\frac{\eta c_i}{2B(2-\eta h_i)}$; stability $\eta<2/h_{\max}$.
- *Theorem (steps to target and the critical batch size; complete proof in the NQM).* For a target loss, the
  number of steps $S(B)$ at the best constant step size (or the best schedule, if it fits) satisfies
  $S(B)\approx S_{\min}(1+B_{\mathrm{crit}}/B)$ with an explicit $B_{\mathrm{crit}}$ in terms of $h_i,c_i$ and
  the target — derive exactly what is true (the curvature-limited regime $\eta=\Theta(1/h_{\max})$ versus the
  noise-limited regime $\eta\propto B$), and relate to McCandlish et al.'s $B_{\mathrm{noise}}=\tr(H\Sigma)/
  (G^\top HG)$ and $(S/S_{\min}-1)(E/E_{\min}-1)=1$.
- *Theorem (preconditioning raises the critical batch size; complete in the NQM).* With preconditioning
  $\theta\leftarrow\theta-\eta H^{-p}g$, $0\le p\le1$, the stable step is $\eta<2/h_{\max}^{1-p}$ and the
  curvature-limited time falls by about $\kappa^{p}$; in the noise-limited regime (small $B$) the steps scale
  like $C_p/B$ with a constant $C_p$ that also depends on $p$ — derive it. *Editor's numerics* (exact
  recursions, best constant $\eta$ on a grid; $d=1000$, $h_i=1/i$, $c_i=h_i$, $\theta_0=\mathbf1$, target 5% of
  the initial loss): SGD needs $10071,2578,724,310,220,200,195$ steps at $B=1,4,16,64,256,1024,4096$;
  $p=\frac12$ needs $3394,850,214,55,16,11,10$. So the advantage grows from about $3\times$ at small batch to about
  $20\times$ at large batch, and the knee (critical batch size) moves right by a factor of about four — but the
  advantage does *not* vanish at small batch in this model. State precisely what is true, including how the
  small-batch ratio depends on the spectrum, the noise model $c_i$ and the initialization (find a natural
  case where it tends to $1$, if there is one, e.g. noise not proportional to curvature or a target dominated by
  stiff directions). Zhang et al. (2019) report that preconditioning and momentum extend perfect scaling to
  larger batches; check their exact claims. Momentum helps only in the curvature-limited regime (Chapter 5).
  Cross-reference Chapter 14 (Marek et al., NeurIPS 2025: SGD close to Adam at batch size one).
- *Heuristic ◇ (SDE scaling rules).* SGD's SDE depends on $\eta/B$ (linear rule); Adam's on $\eta/\sqrt B$ with
  $1-\beta_i\propto B$ (square-root rule) (Li, Tai, E 2017/2019; Malladi et al. 2022). State the conditions under
  which each rule fails (curvature-limited regime).

**Reading.** Figure (theory vs simulation): steps to target vs batch size in the NQM for SGD, momentum SGD and
preconditioned SGD (exact recursions for the theory curves, Monte Carlo for points): perfect scaling, a knee,
and saturation, with the knee moving right under preconditioning.

**Misses.** Idealized away: a fixed quadratic, Gaussian noise proportional to curvature, no feature learning.
Where it breaks: critical batch size changes during training and with data size; very small batches behave
better than the SDE suggests (Marek et al. 2025); learning-rate schedules interact with batch size. Repairs:
batch-size schedules, noise-scale estimators, the dynamics of $B_{\mathrm{crit}}$ (Merrill et al. 2025
**[verify]**).

**Exercises (ideas).** NQM stationary loss with momentum; derive $B_{\mathrm{noise}}$; the linear rule in the
NQM; square-root rule for sign descent; simulation; research: critical batch size from data statistics.

### Chapter 12 — Schedules (key `schedules`)

**In the wild.** Cosine decay to 10% of the peak (GPT-3, Chinchilla **[verify]**), linear decay to zero
(Bergsma et al., ICLR 2025, "Straight to zero" **[verify]**), warmup–stable–decay (WSD; MiniCPM, Hu et al. 2024;
Hägele et al., NeurIPS 2024, "Scaling laws and compute-optimal training beyond fixed training durations"),
inverse square root (Vaswani et al. 2017). Observations: the loss drops sharply during the cooldown; WSD with a
cooldown of 10–20% matches cosine; empirical laws predict loss curves across schedules (Tissue et al. 2024,
"Scaling law with learning rate annealing"; Luo et al., ICLR 2025, "A multi-power law for loss curve prediction
across learning rate schedules"); the last-iterate bound of convex SGD predicts these shapes (Schaipp, Hägele,
Taylor, Simsekli, Bach, ICML 2025, "The surprising agreement between convex optimization theory and
learning-rate scheduling for large model training" **[verify]**).

**Idealization.** (a) Convex Lipschitz stochastic optimization with an arbitrary schedule; (b) the river
valley: a slow "river" coordinate along which the loss decreases slowly and stiff "hill" coordinates with
gradient noise (Wen, Li, Wang, Hall, Liang, Ma, "Understanding warmup-stable-decay learning rates: a river
valley loss landscape perspective", arXiv:2410.05192, ICLR 2025 **[verify venue]**).

**Theory.**
- *Theorem (last iterate of SGD under any schedule; complete proof).* For convex $G$-Lipschitz $\Loss$ and SGD
  with steps $\eta_1,\dots,\eta_T$: an explicit bound on $\E\Loss(\theta_T)-\Loss^\star$ in terms of $D^2$, $G^2$ and
  the schedule (Defazio, Cutkosky, Mehta, Mishchenko 2023, "Optimal linear decay learning rate schedules and
  further refinements", arXiv:2310.07831 **[verify]**; Orabona 2020's last-iterate argument; Shamir and Zhang,
  ICML 2013). Corollaries: linear decay to zero attains the optimal $O(DG/\sqrt T)$ for the last iterate;
  constant step pays $\log T$; WSD with a linear cooldown of fraction $c$ — compute the bound and its
  optimal $c$; reproduce Schaipp et al.'s observation that the bound's shape matches LLM loss curves.
- *Theorem (the river valley; complete proof).* $\Loss(z,y)=r(z)+\frac12\sum_ih_iy_i^2$ with noise of variance
  $\sigma_i^2$ on $y_i$ and a slowly varying river $r$ (linear, or with small curvature). SGD under a schedule:
  $\E y_{T,i}^2=\sum_s\eta_s^2\sigma_i^2\prod_{u>s}(1-\eta_uh_i)^2$ (plus the initial term), and the river moves by
  $\approx-\sum_s\eta_sr'$. So the expected loss is river(area $\sum\eta$) + hill(exponentially weighted recent
  $\eta$): at constant $\eta$ the hill term is $\approx\eta\sigma^2/4$ (per coordinate, small $\eta h$); a
  cooldown removes it in time $\approx1/(\eta h)$ while costing little river progress. Derive the optimal
  cooldown length in this model and the form "loss = $f(\sum\eta)-g(\text{annealing})$" of the empirical
  scaling laws (Tissue et al.; Luo et al.); relate to the functional scaling law of Li, Chen, Huang, Wang, Wu
  (NeurIPS 2025, arXiv:2509.19189 **[verify]**).
- *Remark.* WSD's practical advantages (one run, many budgets; continual pretraining); schedule-free methods
  as implicit decay (Chapter 5; "Through the river", arXiv:2507.09846 **[verify]**).

**Reading.** Figure 1 (theory vs simulation): loss curves of constant, cosine, linear-decay and WSD schedules in
the river-valley model (exact recursion = curves, SGD runs = points). Figure 2: the convex last-iterate bound
for the same schedules vs SGD on a convex Lipschitz problem (Schaipp-style).

**Misses.** Idealized away: convexity or a separable valley; stationary noise; a fixed landscape (no feature
learning, no progressive sharpening). Where it breaks: the optimal cooldown fraction in practice depends on
scale and data; schedules interact with weight decay (Chapter 13) and with batch size (Chapter 11); the
river's existence in real networks is supported by visualizations (arXiv:2508.01483, arXiv:2601.09000
**[verify]**), not derived. Repairs: multi-power and functional scaling laws; schedule-free methods.

**Exercises (ideas).** The hill variance recursion; the noise floor of normalized steps (sign descent) under a
schedule; optimal cooldown in the model; the last-iterate bound for step decay; simulation; research: derive the
river from a network.

### Chapter 13 — Weight Decay (key `decay`)

**In the wild.** AdamW with decoupled weight decay $\lambda=0.1$ in LLM pretraining (Loshchilov and Hutter, ICLR
2019, "Decoupled weight decay regularization"); weight decay on matrices but not gains and biases; Muon needs
weight decay at scale (Liu et al. 2025); scaling rules that keep the timescale $1/(\eta\lambda)$ fixed as a
fraction of training (Wang and Aitchison, ICML 2025, "How to set AdamW's weight decay as you scale model and
dataset size" **[verify]**); Hyperball fixes the norms directly (Wen, Dang, Lyu, Ma, Liang 2026,
arXiv:2606.16899 **[verify]**). Observations: with normalization layers, weight decay controls an effective
learning rate (van Laarhoven 2017; Zhang, Wang, Xu, Grosse, ICLR 2019, "Three mechanisms of weight decay
regularization"); the angular update per step equilibrates to $\sqrt{2\eta\lambda}$ (Kosson, Messmer, Jaggi,
ICML 2024, "Rotational equilibrium"); in LLMs weight decay improves optimization rather than generalization
(D'Angelo, Andriushchenko, Varre, Flammarion, NeurIPS 2024 **[verify]**).

**Idealization.** (a) Normalized steepest descent plus decoupled weight decay on a convex objective; (b) a
scale-invariant loss $\Loss(c\theta)=\Loss(\theta)$ (a weight matrix followed by normalization).

**Theory.**
- *Theorem (decoupled weight decay turns steepest descent into Frank–Wolfe; complete proof).*
  $\theta_{t+1}=(1-\eta\lambda)\theta_t+\eta\lmo(g_t)=(1-\gamma)\theta_t+\gamma v_t$ with $\gamma=\eta\lambda$ and
  $v_t=\lmo(g_t)/\lambda\in\argmin_{\norm v\le1/\lambda}\ip{g_t,v}$. Consequences, all proved: (i) the ball
  $\norm\theta\le1/\lambda$ is invariant and attracts at rate $(1-\gamma)^t$; (ii) for convex $L$-smooth $\Loss$
  (w.r.t. $\norm\cdot$), with $\gamma_t=2/(t+2)$ the constrained gap is $\le2L(2/\lambda)^2/(t+2)$ (Frank–Wolfe;
  diameter $2/\lambda$), and with constant $\gamma$ it converges to $O(\gamma L/\lambda^2)$ — the learning-rate
  schedule *is* the Frank–Wolfe step-size schedule. Instances: AdamW without memory → $\ell_\infty$ ball (Xie and
  Li, ICML 2024, "Implicit bias of AdamW: $\ell_\infty$ norm constrained optimization"); Lion with weight decay
  (Chen, Liu, Liang, Liu, ICLR 2024, "Lion secretly solves constrained optimization"); Muon with weight decay →
  spectral-norm ball (Chen, Li, Liu 2025 **[verify]**; Pethick et al. 2025).
- *Theorem (rotational equilibrium; complete proof; editor checked).* For scale-invariant $\Loss$:
  $\ip{\nabla\Loss(\theta),\theta}=0$ and $\nabla\Loss(c\theta)=\nabla\Loss(\theta)/c$. SGD with weight decay and
  $\norm{g_t}=G/\norm{\theta_t}$ (idealization: constant gradient norm on the unit sphere):
  $u_{t+1}=(1-\eta\lambda)^2u_t+\eta^2G^2/u_t$ for $u_t=\norm{\theta_t}^2$, which converges for every $u_0>0$ to
  $u^\star=\eta G/\sqrt{1-(1-\eta\lambda)^2}\approx G\sqrt{\eta/(2\lambda)}$; the angular update converges to
  $\sqrt{1-(1-\eta\lambda)^2}\approx\sqrt{2\eta\lambda}$ and the effective step on the sphere to
  $\eta/u^\star\approx\sqrt{2\eta\lambda}/G$; relaxation time $\asymp1/(\eta\lambda)$. With $G_t\in[G_1,G_2]$, bounds
  by comparison. AdamW version ($\norm{u_t}_2=\sqrt d$, $u_t\perp\theta_t$): $\norm\theta\to\eta\sqrt d/\sqrt{1-(1-\eta\lambda)^2}
  \approx\sqrt{\eta d/(2\lambda)}$, same angular update $\sqrt{2\eta\lambda}$ (editor's simulation: angular update
  $0.031624$ vs $\sqrt{2\eta\lambda}=0.031623$ at $\eta=0.05$, $\lambda=0.01$).
- *Proposition (AdamW is an exponential moving average of its updates; complete).*
  $\theta_t=(1-\eta\lambda)^t\theta_0-\eta\sum_{s<t}(1-\eta\lambda)^{t-1-s}u_s$: a timescale $\tau=1/(\eta\lambda)$
  steps; consequences for scaling $\lambda$ with dataset size and with $\mu$P's $\eta\propto1/d_{\mathrm{in}}$
  (Wang and Aitchison 2025).
- *Statements.* Effective learning rate governs loss dynamics; Hyperball (2026) as the limit that fixes norms;
  weight decay and learning-rate decay together (the equilibrium shifts during decay, with lag $1/(\eta_t\lambda)$).

**Field guide entries.** AdamW (full), Muon with weight decay (MuonW), Hyperball (brief).

**Reading.** Figure 1 (theory vs simulation): for a scale-invariant model (a normalized linear layer or a small
normalized network), $\norm{\theta_t}$ and the angular update under SGD+WD and AdamW: recursion (curves) vs
training (points). Figure 2: the Frank–Wolfe picture — AdamW's iterates entering the $\ell_\infty$ ball of
radius $1/\lambda$ at rate $(1-\eta\lambda)^t$, and the constrained gap vs the Frank–Wolfe bound.

**Misses.** Idealized away: exact scale invariance, a constant gradient norm on the sphere, orthogonal updates,
convexity in the Frank–Wolfe theorem. Where it breaks: layers without normalization; the gradient norm on the
sphere changes over training; weight decay's generalization effects (in vision) are not captured; the
equilibrium is reached only if training is long compared with $1/(\eta\lambda)$. Repairs: Hyperball, Kosson et
al.'s rotational optimizers, effective-learning-rate analyses (arXiv:2608.24814 **[verify]**), Chapter 15's
implicit bias.

**Exercises (ideas).** Convergence of $u\mapsto au+b/u$; L2 regularization vs decoupled decay for Adam (they
differ); the equilibrium with momentum; FW with constant step; $\lambda$ scaling under $\mu$P; simulation;
research: weight decay and the river valley.

### Chapter 14 — Why Adam Beats SGD on Transformers (key `transformers`)

**In the wild.** Transformers are trained with Adam; SGD is much slower on language models but competitive on
convolutional networks. Evidence: heavy-tailed gradient noise and clipping (Zhang et al., NeurIPS 2020); the gap
persists in full batch and sign descent with momentum closes it (Kunstner et al., ICLR 2023); heavy-tailed
class imbalance — GD is slow on rare tokens, already for linear models (Kunstner, Milligan, Yadav, Schmidt,
Bietti, NeurIPS 2024, "Heavy-tailed class imbalance and why Adam outperforms gradient descent on language
models"); block heterogeneity of the Hessian across parameter blocks (Zhang, Chen, Ding, Li, Sun, Luo, NeurIPS
2024, "Why transformers need Adam: a Hessian perspective"); adaptivity on the last layer and LayerNorm
parameters suffices (Zhao, Morwani, Brandfonbrener, Vyas, Kakade, ICLR 2025, "Deconstructing what makes a
good optimizer for language models" **[verify]**); rotating parameters slows Adam (Xie et al., ICLR 2025); and
at batch size one SGD matches Adam, the gap growing with batch size (Marek, Lotfi, Somasundaram, Wilson,
Goldblum, NeurIPS 2025, "Small batch size training for language models: when vanilla SGD works, and why
gradient accumulation is wasteful").

**Idealization.** Three solvable models: (A) heavy-tailed class frequencies; (B) block-heterogeneous curvature;
(C) the batch-size regime (Chapter 11's NQM with heterogeneous curvature).

**Theory.**
- *Theorem A (Zipf frequencies; complete proof; editor checked).* Rows $w_j$ of an output layer (or a bigram
  table) with input-token frequencies $q_j$ and the square-loss surrogate $\Loss=\sum_jq_j\frac12\norm{w_j-y_j}^2$
  (then the softmax version as a proposition or sketch): GD with the largest stable step $\eta=1/q_1$ has
  $\Loss_t=\sum_jq_j\frac12(1-q_j/q_1)^{2t}\norm{y_j}^2$, which for Zipf $q_j\propto1/j$ over $K$ classes falls by
  only about $\ln10/(2H_K)$ per decade of $t$ ($H_K$ the harmonic number) — logarithmically; sign descent and
  Adam-like per-coordinate normalization make the rate independent of $q_j$. Exact formulas, then asymptotics.
  The softmax cross-entropy version: the gradient on row $j$ carries the factor $q_j$ (complete), and the
  frequency-bucket picture of Kunstner et al. 2024.
- *Theorem B (block heterogeneity; complete proof).* For a block-diagonal quadratic, GD's rate is governed by
  $\lmax(H)/\lmin(H)$ while blockwise step sizes (Adam's per-coordinate scale acting per block, or an
  "Adalayer") give $\max_b\lmax(H_b)/\lmin(H_b)$; transformer Hessians have $\kappa_{\mathrm{global}}\gg
  \max_b\kappa_b$ (Zhang et al. 2024), convolutional networks do not.
- *Theorem C (how the gap depends on the batch size; complete in the NQM).* With heterogeneous curvature, the
  ratio of steps-to-target of SGD and of preconditioned SGD grows with $B$, from a noise-limited constant at small
  batches to the condition-number factor at large batches — reuse Chapter 11 and its numerics (about $3\times$
  growing to $20\times$ in the editor's example). Say honestly whether and when the small-batch ratio tends to
  one; this is how the model speaks to Marek et al. (2025), and it may only partly reconcile them with the
  Hessian and class-imbalance explanations.
- *Proposition (rotation; reuse Chapter 2).* Adam's advantage relies on the coordinate system: a random rotation
  multiplies the $\ell_\infty$ smoothness constant by about $d\lmax/\tr H$.

**Reading.** Figure 1 (theory vs simulation): the Zipf model — loss vs steps for GD, sign descent and Adam, exact
formulas vs runs; inset: per-frequency-bucket losses. Figure 2: steps to target vs batch size for SGD and Adam
in a heterogeneous NQM (exact vs simulation), showing the gap open with $B$.

**Misses.** Idealized away: linear models and quadratics; one mechanism at a time. Where it breaks: no single
mechanism accounts for the whole gap; the mechanisms interact (heavy-tailed classes produce heterogeneous
curvature); results at scale are sensitive to tuning (Wen et al. 2026); the batch-size explanation is recent
and partly unrefereed (state status). Repairs: Adam-mini/blockwise rates, SGD with per-layer normalization,
SOAP/Muon.

**Exercises (ideas).** Asymptotics of $\sum_jq_j(1-q_j)^{2t}$ for Zipf exponents $\ne1$; sign descent on the
Zipf model; the softmax gradient factor; blockwise preconditioning of a $2$-block quadratic; simulation;
research: a unified model with all three mechanisms.

### Chapter 15 — Where Steepest Descent Ends (key `bias`)

**In the wild.** Adaptive methods sometimes generalize worse than SGD (Wilson, Roelofs, Stern, Srebro, Recht,
NeurIPS 2017, "The marginal value of adaptive gradient methods in machine learning"), with a constructed linear
example; AdamW's solutions are $\ell_\infty$-bounded (Chapter 13); different optimizers reach different
solutions at the same loss.

**Idealization.** Linearly separable data, exponential loss, steepest descent in a norm (normalized, with
decaying or small steps); homogeneous networks.

**Theory.**
- *Theorem (the margin of a norm; complete proof).* For separable data with $\norm{x_i}_*\le1$, exponential loss,
  and normalized steepest descent $\theta_{t+1}=\theta_t+\eta_t\lmo(\nabla\Loss(\theta_t))$ with suitable $\eta_t$
  ($\eta_t\to0$, $\sum\eta_t=\infty$; check what constant steps give): the normalized margin
  $\min_iy_i\ip{\theta_t,x_i}/\norm{\theta_t}$ converges to $\gamma_{\norm\cdot}=\max_{\norm\theta\le1}\min_iy_i
  \ip{\theta,x_i}$ (Gunasekar, Lee, Soudry, Srebro, ICML 2018, "Characterizing implicit bias in terms of
  optimization geometry"). Key inequality: $\norm{\nabla\Loss(\theta)}_*\ge\gamma_{\norm\cdot}\Loss(\theta)$; then the
  argument of volume 2, Chapter 4, with dual norms. Instances: GD → $\ell_2$ SVM; sign descent and Adam without
  memory → $\ell_\infty$ max margin (dense); $\ell_1$ descent → $\ell_1$ max margin (sparse, boosting); spectral
  descent on matrix classifiers → spectral-norm max margin.
- *Statements.* Adam (with $\beta_1\le\beta_2$, decaying steps, $\epsilon\to0$) → $\ell_\infty$ max margin (Zhang,
  Zou, Cao, NeurIPS 2024, "The implicit bias of Adam on separable data" **[verify]**); AdamW → KKT points of the
  $\ell_\infty$-constrained problem (Xie and Li 2024); Muon and spectral descent (Fan, Schmidt, Thrampoulidis
  2025; Tsilivis, Vardi, Kempe, ICLR 2025, "Flavors of margin" **[verify]**); homogeneous networks: KKT points of
  norm-max-margin problems (Lyu and Li, ICLR 2020, in $\ell_2$; general norms in Tsilivis et al.) — sketch.
- *Example (complete).* An explicit dataset where the $\ell_\infty$-max-margin classifier generalizes worse than
  the $\ell_2$ one (in the spirit of Wilson et al. 2017 — re-derive their construction or build a cleaner one,
  and compute both test errors).

**Reading.** Figure (theory vs simulation): normalized margins of GD, sign descent and $\ell_1$ descent on a
small separable dataset converging to the three max margins (computed by linear/quadratic programming:
horizontal lines) vs iterations ($\log$ scale).

**Misses.** Idealized away: separability, linear models, infinite time (rates are logarithmic). Where it breaks:
in deep networks the implicit bias interacts with feature learning and the edge of stability; generalization
differences between Adam and SGD are small or reversed on transformers; the KKT limits need not be global
max-margin solutions. Repairs: rates of margin convergence for steepest descent (Nacson et al. 2019; Wang et al.
2023 **[verify]**), implicit bias with weight decay (Chapter 13), mini-batch steepest descent bias
(arXiv:2602.11557 **[verify]**).

**Exercises (ideas).** Max margins of a 3-point dataset in $\ell_1,\ell_2,\ell_\infty$; the dual-norm inequality;
the constant-step variant; the $\ell_1$ case as boosting; simulation; research: implicit bias of Muon in deep
networks.

### Epilogue — What Theory Owes Practice (key `epilogue`) — written by the editor

A ledger: every practice of the book with the verdict of its chapter (explained / partly explained / unexplained)
and the hypotheses on which the explanation rests; what the idealizations miss together (fixed landscapes,
quadratic or convex surrogates, one mechanism at a time, no data); open problems (why progressive sharpening;
the right norm for embeddings and attention; why $\beta_1=0.9$; schedules at scale; why matrix optimizers' gains
shrink with scale; hyperparameter scaling laws; optimizers for mixtures of experts, reinforcement learning and
low precision); a recipe for turning a practice into a theorem.

### Appendix A — A Toolkit (key `toolkit`)

Sections and fixed labels as in the table above: norms and duality; matrix analysis (SVD, polar factor, von
Neumann, Löwner–Heinz, Kronecker products, geometric mean); linear recurrences and their stochastic versions;
smoothness and the PL inequality; online convex optimization; concentration and martingales. Complete proofs
where short (dual norms, subdifferential, conjugate, von Neumann via Birkhoff or via a direct argument, polar
factor, Löwner–Heinz for $r=\frac12$, Kronecker facts, geometric mean characterization and concavity, Jury test,
Lyapunov recursion, descent lemma, PL, AdaGrad lemma, OGD regret, Grönwall, Gauss's inequality); statements with
references for Azuma–Hoeffding and Doob. 15–22 pages; same voice; no field notes or "misses" sections; a short
opening paragraph says which chapters use each section.
