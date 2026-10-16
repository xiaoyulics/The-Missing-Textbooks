# The Hydrogen Atoms of Deep Learning — Book Plan and Chapter Specifications

## The thesis

Every physics student derives the harmonic oscillator, the hydrogen atom and the Ising model, and knows
where each one fails: hydrogen's Schrödinger spectrum misses the fine structure, the Lamb shift and the
hyperfine splitting; mean-field Ising gets the critical exponents wrong in low dimension. The failures
are as instructive as the solutions, because each failure names the physics the model left out.

Deep learning theory has accumulated a comparable collection of solvable models — ridge regression in
proportional asymptotics, deep and diagonal linear networks, single-index models, sparse parities, modular
addition, the toy model of superposition, linear attention on regression tasks, induction heads on Markov
chains, and more — but they are scattered across dozens of papers in incompatible notation, and the
places where they fail are rarely written down next to the solutions. This book does for deep learning
what the standard physics curriculum does for quantum mechanics: **each chapter takes one phenomenon,
gives one minimal solvable model, proves one main theorem completely, and then says precisely what the
model gets wrong.** All chapters use one notation (below), and each chapter ends with a "Translating the
literature" paragraph that maps the source papers' notation onto it.

A model earns a chapter if it passes four tests (stated in the prologue):
1. **It reproduces a phenomenon** that is observed in real networks.
2. **It is solvable**: its main claim has a complete proof, short enough to teach.
3. **It predicts a number**: an exponent, a threshold, a time scale, a phase boundary, a constant.
4. **It fails instructively**: the ways it is wrong are known, specific, and point to the next model.

Positioning (for the preface; verified 2026-10-17): Roberts, Yaida and Hanin, *The Principles of Deep
Learning Theory* (Cambridge University Press, 2022) develops one method, an effective field theory of
wide networks in a 1/width expansion. Haiping Huang, *Statistical Mechanics of Neural Networks* (Springer,
2021/2022) covers the classical statistical-physics models (spin glasses, Hopfield, perceptron, RBMs).
Zhilin Zhao, *From Approximation to Emergence: A Theory of Deep Learning* (arXiv:2607.01311, July 2026) is
a broad, proof-oriented monograph that gives each topic about one chapter; its preface says it is not
organized as a set of instructional units. Simon, Kunin, Atanasov et al., *There Will Be a Scientific Theory
of Deep Learning* (arXiv:2604.21691, April 2026) names "solvable idealized settings" as one of five strands
of an emerging "learning mechanics". Older ancestors: Engel and Van den Broeck, *Statistical Mechanics of
Learning* (Cambridge, 2001).

## Structure, keys, and files

| # | key | file | Title — Subtitle | Model tag (3rd arg of `\casechapter`) |
|---|-----|------|------------------|------|
| – | prologue | 00-prologue.tex | The Hydrogen Atom — What a solvable model is for | — (special chapter) |
| | | | **Part I. Linear Models That Already Surprise** | |
| 1 | doubledescent | 01-doubledescent.tex | Double Descent — Least squares in the proportional limit | Ridgeless least squares, Gaussian features |
| 2 | deeplinear | 02-deeplinear.tex | Learning in Stages — Exact dynamics of deep linear networks | Deep linear networks, whitened inputs |
| 3 | diagonal | 03-diagonal.tex | The Bias of Small Initialization — Diagonal linear networks, from kernels to sparsity | Diagonal linear networks |
| 4 | margin | 04-margin.tex | Training Past Zero Error — Logistic regression on separable data | Linear classifiers, exponential-tailed losses |
| | | | **Part II. Infinite Width** | |
| 5 | signalprop | 05-signalprop.tex | Order and Chaos at Initialization — Signal propagation in wide random networks | Infinitely wide random networks |
| 6 | lazy | 06-lazy.tex | Lazy Training — The neural tangent kernel and what it cannot learn | Linearization around initialization |
| | | | **Part III. Feature Learning in High Dimensions** | |
| 7 | singleindex | 07-singleindex.tex | The Search Phase — Single-index models and the information exponent | Single-index model, online SGD |
| 8 | committee | 08-committee.tex | Plateaus and Specialization — The committee machine and the ODE method | Soft committee machine, teacher and student |
| 9 | parity | 09-parity.tex | Hidden Progress — Sparse parities and the limits of gradient methods | Sparse parity on the hypercube |
| | | | **Part IV. What Networks Represent** | |
| 10 | grokking | 10-grokking.tex | Grokking — Modular addition and its Fourier circuit | Modular addition, quadratic two-layer network |
| 11 | superposition | 11-superposition.tex | Superposition — More features than dimensions | ReLU toy autoencoder |
| 12 | collapse | 12-collapse.tex | Neural Collapse — The geometry of the last layer | Unconstrained features |
| | | | **Part V. Transformers** | |
| 13 | icl | 13-icl.tex | Learning in Context — Linear attention as one step of gradient descent | Linear self-attention on regression tasks |
| 14 | induction | 14-induction.tex | Induction Heads — In-context Markov chains and a two-layer circuit | Attention-only transformer on random Markov chains |
| | | | **Part VI. Training at Scale** | |
| 15 | stability | 15-stability.tex | The Edge of Stability — Large step sizes and the catapult | The two-parameter product model |
| 16 | scaling | 16-scaling.tex | Scaling Laws — Power laws from spectra and from Zipf's law | Power-law linear regression and quanta |
| – | epilogue | 17-epilogue.tex | Fine Structure — What the solvable models get wrong together | — (special chapter) |
| A | toolkit | 18-toolkit.tex | A Toolkit — The mathematics the models share | — (appendix chapter, empty tag) |

Chapter labels are `ch:<key>`; `\cref{ch:<key>}` prints "Chapter N" (the appendix prints "Appendix A").
Each chapter starts with `\casechapter{Title}{Subtitle}{Model tag}{ch:key}` using exactly the strings in
the table. The appendix uses `\casechapter{A Toolkit}{The mathematics the models share}{}{ch:toolkit}`.

### Fixed appendix labels (chapters may cite these; the appendix must define exactly these)

| label | content |
|---|---|
| `thm:toolkit:stein` | Gaussian integration by parts (Stein's lemma), scalar and vector forms |
| `lem:toolkit:hermite` | Hermite polynomials: orthonormality of $h_k$, $h_k'=\sqrt k\,h_{k-1}$, generating function, $z h_k=\sqrt{k+1}h_{k+1}+\sqrt k h_{k-1}$ |
| `thm:toolkit:hermite-corr` | for unit $u,v$: $\E[h_j(\ip{u,x})h_k(\ip{v,x})]=\delta_{jk}\ip{u,v}^k$, $x\sim\cN(0,I_d)$ |
| `cor:toolkit:dual` | dual activation: $\E[\phi(u)\phi(v)]=\sum_k\hat\phi_k^2\rho^k$ for standard Gaussians with correlation $\rho$ |
| `lem:toolkit:price` | Price's theorem: $\partial_\rho\E[f(u)g(v)]=\E[f'(u)g'(v)]$ |
| `thm:toolkit:addition` | spherical harmonics: $\dim\cH_k=N_{d,k}$, addition theorem, Funk–Hecke formula |
| `thm:toolkit:inv-wishart` | $\E[(X^\top X)^{-1}]=I_p/(n-p-1)$ for $X\in\R^{n\times p}$ with i.i.d. $\cN(0,1)$ entries, $n\ge p+2$ |
| `thm:toolkit:mp` | Marchenko–Pastur law via the Stieltjes transform (complete proof for Gaussian entries) |
| `lem:toolkit:doob` | Doob's $L^2$ maximal inequality |
| `thm:toolkit:freedman` | Freedman's inequality (and Azuma–Hoeffding as a corollary) |
| `thm:toolkit:gronwall` | Grönwall's inequality, continuous and discrete |
| `thm:toolkit:ode-method` | the ODE method: a stochastic recursion with $O(1/d)$ increments tracks its mean ODE |
| `thm:toolkit:fourier-zp` | Fourier analysis on $\Z_p$: Plancherel, convolution theorem |
| `thm:toolkit:fourier-cube` | Fourier–Walsh expansion on $\{\pm1\}^d$, Parseval |
| `thm:toolkit:bonami` | Bonami's lemma / hypercontractivity for low-degree polynomials on the cube and in Gaussian space |
| `thm:toolkit:hoeffding` | Hoeffding's inequality |
| `thm:toolkit:bernstein` | Bernstein's inequality |
| `thm:toolkit:gauss-conc` | Gaussian concentration for Lipschitz functions |

### Cross-chapter threads (point them out with `\cref` where natural)

- **Escape from a saddle at small initialization** (logistic growth, time ∝ log(1/init)/strength, or a
  power of 1/init for higher-order saddles): deep linear networks (2) ↔ diagonal networks (3) ↔
  single-index search phase (7) ↔ committee plateaus (8) ↔ hidden progress in parities (9) ↔ induction
  heads (14). The prologue's quadratic is the degenerate case with no saddle.
- **Rotational invariance of Gaussian data**: double descent (1) ↔ kernel lower bound (6) ↔ single-index (7)
  ↔ committee machine (8) ↔ linear attention (13).
- **Implicit bias and margins**: diagonal networks (3) ↔ max margin (4) ↔ max-margin Fourier features (10)
  ↔ neural collapse (12).
- **Lazy versus rich**: initialization scale in diagonal networks (3) ↔ NTK (6) ↔ grokking as a lazy-to-rich
  transition (10) ↔ feature learning (7).
- **Fourier analysis on groups**: Boolean cube (9) ↔ $\Z_p$ (10) ↔ circular features (11) ↔ Hermite
  expansions as the Gaussian analogue (5, 7).
- **Spectra**: Marchenko–Pastur (1) ↔ kernel spectra and spectral bias (6) ↔ Jacobian spectra (5) ↔
  power-law spectra (16).
- **The ODE method / order parameters**: committee machine (8) ↔ single-index (7) ↔ appendix.
- **Stability of gradient descent**: prologue quadratic ↔ edge of stability (15) ↔ the $\eta^2$ term in the
  committee machine (8).
- **Plateaus and emergence**: (8) ↔ (9) ↔ (10) ↔ (14) ↔ (16, emergence versus smooth loss).
- **Max-margin solutions as attractors**: (4) ↔ (10) ↔ (12).

## Notation (global; summarized for readers in `frontmatter/notation.tex`)

- $n$ samples, $d$ input dimension, $m$ hidden width, $p$ features/parameters of a linear model, $K$
  classes, $N$ context length (Ch. 13–14; in Ch. 16 $N$ and $D$ follow the scaling-law literature:
  parameters and data), $L$ depth, $\ell$ layer index, $k$ a degree or a sparsity.
- $x$ input, $y$ label, $f(x;\theta)$ predictor, $\theta$ parameters, $w$ weight vector, $W$ weight matrix,
  $\phi$ activation. Subscript star: target/teacher ($w_\star$, $f_\star$). Superscript star: optimum
  ($\lambda^\star$, $\Gamma^\star$).
- $\Loss$ population loss, $\Lhat$ training loss; $\eta$ step size; $t$ time; $\lambda$ ridge/weight decay;
  $\alpha$ initialization scale; $\sigma$ noise level; $\sigma_w^2,\sigma_b^2$ initialization variances.
- $h_k=\He_k/\sqrt{k!}$ normalized Hermite polynomials; $\hat g_k=\E[g(z)h_k(z)]$.
- $\cN$, $\Unif$, $\Sph^{d-1}$; $\ip{\cdot,\cdot}$, $\norm{\cdot}$, $\norm{\cdot}_{\op}$, $\norm{\cdot}_F$;
  $\lesssim,\asymp,\tilde O$.

---

## Chapter specifications

Facts marked **[verify]** must be checked against sources before they go into the text. Theorem statements
below are the editor's plan; the author must re-derive each one, fix constants, check them numerically,
and correct the plan where it is wrong (and say so in the report). "Complete proof" is mandatory where
marked; elsewhere use judgment.

### Prologue — The Hydrogen Atom (key `prologue`) — written by the editor

The physics tradition (harmonic oscillator, hydrogen, Ising; what each gets wrong); the four tests; the
first hydrogen atom of optimization, gradient descent on a quadratic (complete solution: normal modes,
stability threshold $\eta<2/\lmax$, rates, condition number; what it gets wrong: the Hessian is not
constant (Ch. 15), losses are not convex (Ch. 2, 3, 7), minimizers are not unique (Ch. 3, 4)); the anatomy
of a chapter; a full-width map of the book (phenomenon / model / main theorem / gets wrong); proofs,
sketches and heuristic derivations (■, sketch, ◇); how to read the book.

### Chapter 1 — Double Descent (key `doubledescent`)

**Phenomenon.** Test error as a function of model size first follows the classical U, peaks at the
interpolation threshold, then descends again (Belkin, Hsu, Ma, Mandal, PNAS 2019; Nakkiran, Kaplun, Bansal,
Yang, Barak, Sutskever, ICLR 2020 / J. Stat. Mech. 2021: model-wise, epoch-wise and sample-wise double
descent; label noise accentuates the peak). Prehistory: Krogh and Hertz (1992), Opper and coauthors in the
1990s, Advani and Saxe (arXiv 2017; Neural Networks 2020), Spigler, Geiger et al. (2019, "jamming"), and the
PNAS letter by Loog, Viering, Mey, Krijthe and Tax (2020), "A brief prehistory of double descent" **[verify
all]**. Sample-wise version: more data can hurt (Nakkiran 2019, arXiv).

**Model.** Linear regression with isotropic Gaussian features: $y=\ip{w_\star,x}+\eps$. Two versions:
(a) well specified, $x\sim\cN(0,I_p)$, learner uses all $p$ features; (b) the "weak features" model of
Belkin, Hsu and Xu (SIAM J. Math. Data Sci. 2020, "Two models of double descent for weak features"): the
truth uses $D$ independent standard Gaussian features, the learner uses a subset of $p$; the omitted features
act as extra independent Gaussian noise of variance $\norm{w_{\star,\text{omitted}}}^2$. Estimators:
minimum-norm least squares $\hat w=X^+y$, and ridge $\hat w_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y$.
Dictionary: $p/n$ ↔ parameters per example; $\sigma^2$ ↔ label noise plus signal the model cannot
represent; ridge $\lambda$ ↔ weight decay / early stopping; smallest singular value of $X$ ↔ conditioning
of the interpolating solution.

**Solution.**
- *Theorem (exact finite-sample risk; complete proof).* $X\in\R^{n\times p}$ with i.i.d. $\cN(0,1)$
  entries, $y=Xw_\star+\xi$ with $\xi$ independent of $X$, mean zero, covariance $\sigma^2I_n$. Then
  $\E\norm{\hat w-w_\star}^2=\sigma^2p/(n-p-1)$ for $p\le n-2$; $=\norm{w_\star}^2(1-n/p)+\sigma^2n/(p-n-1)$
  for $p\ge n+2$; $=+\infty$ for $|p-n|\le1$ when $\sigma>0$. Test error $=\sigma^2+\E\norm{\hat w-w_\star}^2$.
  Proof: (i) $\E[(X^\top X)^{-1}]=I/(n-p-1)$ via rotational invariance plus the Schur-complement identity
  $[(X^\top X)^{-1}]_{11}=1/\norm{(I-P_{-1})X_{:1}}^2$ with $\norm{(I-P_{-1})X_{:1}}^2\sim\chi^2_{n-p+1}$ and
  $\E[1/\chi^2_k]=1/(k-2)$ ($=\infty$ for $k\le2$); (ii) for $p>n$, $\hat w=Pw_\star+X^\top(XX^\top)^{-1}\xi$
  with $P$ the projection on the row space, $\E P=(n/p)I$ by invariance under right multiplication by
  orthogonal matrices; variance via (i) with roles of $n$ and $p$ swapped. (Checked by Monte Carlo by the
  editor at $n=20$.)
- *Corollary (proportional limit).* $p/n\to\gamma$: risk $\to\sigma^2\gamma/(1-\gamma)$ for $\gamma<1$ and
  $\norm{w_\star}^2(1-1/\gamma)+\sigma^2/(\gamma-1)$ for $\gamma>1$; the peak at $\gamma=1$ is where the
  smallest eigenvalue $(1-\sqrt\gamma)^2$ of the Marchenko–Pastur law vanishes. Weak-features version:
  second descent below the first minimum when the signal is spread over many features; compute the
  location of the overparameterized minimum.
- *Theorem (ridge in proportional asymptotics; proof given the Marchenko–Pastur theorem,
  `\cref{thm:toolkit:mp}`).* Isotropic design, $w_\star$ uniform on the sphere of radius $r$ (or fixed and
  averaged over rotations), $p/n\to\gamma$: bias $\to r^2\lambda^2 m'(-\lambda)$, variance
  $\to\sigma^2\gamma\,(m(-\lambda)-\lambda m'(-\lambda))$, with $m$ the Stieltjes transform of the MP law
  of ratio $\gamma$ (state the quadratic equation for $m$ in the normalization $\hat\Sigma=X^\top X/n$)
  **[verify formulas; Dobriban and Wager, Ann. Statist. 2018; Hastie, Montanari, Rosset, Tibshirani, Ann.
  Statist. 2022]**. Optimal $\lambda^\star=\sigma^2\gamma/r^2$ (in this normalization — verify) removes the
  peak and gives risk monotone in $\gamma$ (Nakkiran, Venkat, Kakade, Ma, ICLR 2021, "Optimal regularization
  can mitigate double descent"). Consistency check: $\lambda\to0$ recovers the corollary.

**Reading.** Figure 1: exact finite-sample risk versus $p$ (fixed $n$), Monte Carlo points, both versions
(a) and (b). Figure 2 (optional): ridge risk curves for several $\lambda$ with the optimal one monotone.
Explain: the variance blows up because the smallest singular value of $X$ vanishes at $p=n$; past it the
min-norm solution spreads energy over more directions.

**Gets wrong.** Thrown away: fixed isotropic features (no feature learning), linear target, min-norm
interpolation as the training algorithm. Where it breaks: in deep networks the peak sits at an effective
complexity where training error first reaches zero (Nakkiran et al.'s "effective model complexity"), not at
a parameter count; epoch-wise double descent needs dynamics (Heckel and Yilmaz, ICLR 2021; Pezeshki et al.
2022 **[verify]**); with tuned regularization the peak usually disappears in practice; isotropic features
cannot give benign overfitting — the overparameterized risk tends to the null risk $\norm{w_\star}^2$ as
$\gamma\to\infty$; benign overfitting needs a covariance with a fast-decaying head and a heavy flat tail
(Bartlett, Long, Lugosi, Tsigler, PNAS 2020: effective ranks $r_k,R_k$ — state their theorem and explain
the two conditions). Curth, Jeffares and van der Schaar (NeurIPS 2023, "A U-turn on double descent") argue
the second descent reflects two different complexity axes. Fine structure: random features regression
(Mei and Montanari, CPAM 2022) — nonlinear features act as linear features plus effective noise; kernel
ridgeless regression (Liang and Rakhlin, Ann. Statist. 2020); benign/tempered/catastrophic overfitting
(Mallinar et al., NeurIPS 2022); anisotropic deterministic equivalents (cross-ref Ch. 16).

**Exercises (ideas).** Inverse-Wishart mean via Schur complements; sample-wise double descent from the
exact formula (fix $p$, vary $n$: more data can hurt); minimizer of the overparameterized branch; optimal
ridge and monotonicity; a spiked covariance (two eigenvalue levels) showing benign overfitting; simulation:
random ReLU features regression; research: a solvable model of epoch-wise double descent.

**Cross-links.** Ch. 16 (power-law spectra), Ch. 6 (kernels), Ch. 11 (Henighan et al. 2023, superposition
and double descent **[verify]**), appendix MP law.

### Chapter 2 — Learning in Stages (key `deeplinear`)

**Phenomenon.** Learning curves with plateaus and sudden drops; networks learn coarse structure before fine
(progressive differentiation, Saxe, McClelland, Ganguli, PNAS 2019, "A mathematical theory of semantic
development in deep neural networks"); networks learn functions of increasing complexity (Kalimeris et al.,
NeurIPS 2019 **[verify]**); low-rank, incremental learning.

**Model.** $f(x)=W_L\cdots W_1x$; square loss; whitened inputs $\E[xx^\top]=I$; then
$\Loss=\tfrac12\norm{\Sigma_{yx}-W}_F^2+\text{const}$ with $W=W_L\cdots W_1$ and $\Sigma_{yx}=\E[yx^\top]=USV^\top$
(singular values $s_1>s_2>\dots$). Gradient flow.

**Solution.**
- *Theorem (conservation law; complete proof).* Under gradient flow on $\ell(W_L\cdots W_1)$ the matrices
  $W_{j+1}^\top W_{j+1}-W_jW_j^\top$ are constant (Arora, Cohen, Hazan, ICML 2018; Du, Hu, Lee, NeurIPS 2018).
- *Theorem (Saxe–McClelland–Ganguli exact solution; complete proof).* Two layers, balanced and aligned
  initialization ($W_1(0)=O\,\diag(\sqrt{a_\alpha(0)})V^\top$, $W_2(0)=U\diag(\sqrt{a_\alpha(0)})O^\top$ with
  $O$ orthogonal — fix the precise form): the product stays $U\diag(a_\alpha(t))V^\top$ with
  $\dot a=2a(s-a)$ (time unit as in the chapter), so $a(t)=s/(1+(s/a_0-1)e^{-2st})$; mode $\alpha$ is learned
  at $t_\alpha\approx\log(s_\alpha/a_0)/(2s_\alpha)$ with transition width $\asymp1/s_\alpha$. Proof: the
  ansatz is consistent (uniqueness of ODE solutions), reduce to scalar ODEs, solve the logistic equation.
  Depth $L$: $\dot a=La^{2-2/L}(s-a)$; plateau length $\asymp a_0^{-(1-2/L)}/(Ls)$ for $L\ge3$ (verify), i.e.
  depth lengthens the plateau and sharpens the transition.
- *Theorem (end-to-end dynamics; complete proof).* Balanced initialization implies
  $\dot W=-\sum_{j=1}^L(WW^\top)^{(L-j)/L}\nabla\ell(W)(W^\top W)^{(j-1)/L}$ (Arora, Cohen, Hazan 2018).
- *Optional.* Baldi–Hornik (1989): for two-layer linear networks every local minimum is global; saddles
  correspond to subsets of singular modes (proof sketch or complete if space allows). Saddle-to-saddle from
  small random initialization (Jacot, Ged, Şimşek, Hongler, Gabriel 2021, arXiv **[verify]**; Li, Luo, Lyu,
  ICLR 2021, greedy low-rank learning) — statement and heuristic.

**Reading.** Loss staircase $\tfrac12\sum_{\beta\text{ not yet learned}}s_\beta^2$; figure: exact mode curves
versus gradient descent from small *random* (non-aligned) initialization; a hierarchical dataset where modes
are levels of a tree (broad distinctions first).

**Gets wrong.** Thrown away: nonlinearity (only linear maps; no new features), whitening, aligned
initialization (random small initialization aligns first: "silent alignment", Atanasov, Bordelon, Pehlevan,
ICLR 2022), population loss (Advani and Saxe 2020 analyze finite samples). Where it breaks: when linear
statistics vanish ($\Sigma_{yx}=0$, e.g. XOR or parity) linear-network theory predicts nothing is learnable,
yet nonlinear networks learn (cross-ref Ch. 7, 9); stage order in nonlinear networks is not set by the
singular values of $\Sigma_{yx}$; implicit low-rank bias of deep matrix factorization is not a norm (Arora,
Cohen, Hu, Luo, NeurIPS 2019; Razin and Cohen, NeurIPS 2020). Fine structure: unbalanced initialization with
exact solutions (Kunin et al., NeurIPS 2024, "Get rich quick" **[verify]**), gated deep linear networks (Saxe,
Sodhani, Lewallen, ICML 2022), prior knowledge (Braun et al., NeurIPS 2022), saddle-to-saddle.

**Exercises (ideas).** Per-neuron conservation for ReLU networks ($\norm{w_{\rm in}}^2-a_{\rm out}^2$
constant); depth-$L$ mode ODE and plateau scaling; unbalanced two-layer scalar model with conserved
$b^2-c^2$ solved exactly; non-whitened inputs with shared singular vectors; simulation: hierarchical data;
research: stage order in a nonlinear network.

### Chapter 3 — The Bias of Small Initialization (key `diagonal`)

**Phenomenon.** Among the many interpolators of an overparameterized problem, gradient descent picks one,
and which one depends on the initialization scale: kernel ("lazy") versus rich regimes (Chizat, Oyallon,
Bach, NeurIPS 2019; Woodworth et al., COLT 2020, "Kernel and rich regimes in overparametrized models"); small
initialization gives sparse or low-rank solutions (Gunasekar et al., NeurIPS 2017); incremental learning
(Gissin, Shalev-Shwartz, Daniely, ICLR 2020).

**Model.** $w=u\odot u-v\odot v\in\R^d$, $u(0)=v(0)=\alpha\mathbf 1$, $\Loss(w)=\frac1{2n}\norm{Xw-y}^2$,
$X\in\R^{n\times d}$ with $n<d$, gradient flow on $(u,v)$.

**Solution.**
- *Theorem (implicit bias; complete proof incl. convergence).* With $g=\nabla_w\Loss$:
  $u=\alpha e^{-2\int g}$, $v=\alpha e^{2\int g}$, $w=-2\alpha^2\sinh(4\int g)$, so
  $\tfrac{d}{dt}\nabla\Phi_\alpha(w)=-\nabla\Loss(w)$ with $\nabla\Phi_\alpha(w)_i=\tfrac14\arcsinh(w_i/2\alpha^2)$,
  $\Phi_\alpha(w)=\tfrac{\alpha^2}{4}\sum_i q(w_i/\alpha^2)$, $q(z)=2-\sqrt{4+z^2}+z\arcsinh(z/2)$ (re-derive
  constants). Convergence: for every interpolator $w^\circ$,
  $\tfrac{d}{dt}D_{\Phi}(w^\circ,w(t))=\ip{\nabla\Loss(w),w^\circ-w}\le-\Loss(w)$, $D_\Phi$ coercive ⇒ bounded
  trajectory, $\Loss(w(t))\le D_\Phi(w^\circ,0)/t$, limit point unique via monotonicity of $D_\Phi$ at the
  limit point. Limit = $\argmin\{\Phi_\alpha(w):Xw=y\}$ by KKT ($\nabla\Phi_\alpha(w_\infty)\in\operatorname{range}X^\top$).
- *Theorem (the two limits; complete proof).* $\alpha\to\infty$: minimum-$\ell_2$ interpolator (kernel
  regime); $\alpha\to0$: minimum-$\ell_1$ interpolator when unique (rich regime), since
  $\Phi_\alpha(w)/(\tfrac14\log(1/\alpha^2))\to\norm{w}_1$ uniformly on compacts (check the constant).
- *Theorem (incremental learning under orthogonal design; complete proof).* $X^\top X/n=I_d$: coordinates
  decouple, $u_iv_i=\alpha^2$ is conserved, $\dot w_i=-4\sqrt{w_i^2+4\alpha^4}\,(w_i-w_{\star,i})$; closed
  form via $w_i=2\alpha^2\sinh\theta_i$; coordinate $i$ is learned at $t_i\approx\log(1/\alpha)/(2|w_{\star,i}|)$
  (verify) — a staircase in accelerated time $\tau=t/\log(1/\alpha)$; same logistic mechanism as Ch. 2.
- *Saddle-to-saddle for general $X$ (Pesme and Flammarion, NeurIPS 2023): statement and sketch* — in
  accelerated time the trajectory converges, as $\alpha\to0$, to a piecewise constant path through saddles
  computed by a LARS/homotopy-like algorithm.

**Reading.** Figure: sparse recovery ($n<d$, sparse $w_\star$) — test error of the limit versus $\alpha$
(from $\ell_2$ to $\ell_1$ behaviour), theory = solve the $\Phi_\alpha$ problem numerically, simulation = run
the flow. Figure: incremental learning staircase (orthogonal design), exact solution versus simulation.

**Gets wrong.** Thrown away: the predictor is linear in $x$ (a reparameterized linear model), infinitesimal
steps, no noise, square loss. Where it breaks: step size and SGD noise change the bias (Pesme,
Pillaud-Vivien, Flammarion, NeurIPS 2021; Even, Pesme, Gunasekar, Flammarion, NeurIPS 2023; Nacson et al.,
ICML 2022 **[verify]**); classification gives max-$\ell_1$-margin with an initialization/accuracy trade-off
(Moroshko et al., NeurIPS 2020); the mirror-flow description exists only for commuting parametrizations (Li,
Wang, Lee, Arora, NeurIPS 2022, "Implicit bias of gradient descent on reparametrized models: on equivalence
to mirror descent") — general networks have no potential; for matrix factorization the nuclear-norm
analogue fails (Razin and Cohen 2020; Li, Luo, Lyu 2021). Fine structure: depth-$L$ diagonal networks
(Woodworth et al.), initialization shape (Azulay et al., ICML 2021), label-noise SGD and the Lasso
(Pillaud-Vivien et al. 2022 **[verify]**).

**Exercises (ideas).** Taylor expansions of $q$; the $\ell_1$ limit with a non-unique minimizer; depth-$L$
potential; closed-form integration of the orthogonal-design ODE; simulation: phase transition in $\alpha$
for sparse recovery; variation: unbalanced initialization $u(0)\ne v(0)$; research: SGD noise.

### Chapter 4 — Training Past Zero Error (key `margin`)

**Phenomenon.** After training error reaches zero, continuing to train keeps improving test loss and
sometimes test error, while weights grow without bound (Soudry et al., JMLR 2018; Hoffer, Hubara, Soudry,
NeurIPS 2017 "Train longer, generalize better"; Zhang et al., ICLR 2017 for memorization). Direction matters,
norm does not.

**Model.** Linear classifier $x\mapsto\ip{w,x}$, labels $y_i\in\{\pm1\}$, linearly separable data with
$\norm{x_i}\le1$, exponential loss $\Loss(w)=\frac1n\sum_ie^{-y_i\ip{w,x_i}}$ (logistic as a variant),
gradient flow from $w(0)=0$; max margin $\gamma^\star=\max_{\norm u=1}\min_iy_i\ip{u,x_i}>0$ attained at the
unique $u^\star$ (the hard-margin SVM direction).

**Solution.**
- *Theorem (margin convergence; complete proof).* $\Loss(w(t))\le1/(1+\gamma^{\star2}t)$ (from
  $\dot\Loss=-\norm{\nabla\Loss}^2\le-\gamma^{\star2}\Loss^2$, since $\norm{\nabla\Loss}\ge\ip{-\nabla\Loss,u^\star}\ge\gamma^\star\Loss$);
  with $N(t)=-\log(n\Loss(w(t)))\le\min_iy_i\ip{w(t),x_i}$ and $\dot N=\norm{\nabla\Loss}^2/\Loss\ge\gamma^\star\norm{\dot w}$,
  integrate: $\min_iy_i\ip{w(t),x_i}\ge\gamma^\star\norm{w(t)}-\log n$; and
  $\norm{w(t)}\ge\log((1+\gamma^{\star2}t)/n)$. Hence normalized margin gap
  $\le\log n/\log((1+\gamma^{\star2}t)/n)$.
- *Theorem (directional convergence; complete proof).* $w(t)/\norm{w(t)}\to u^\star$ (uniqueness of the
  max-margin direction + compactness). Mention the $O(1/\log t)$ direction rate of Soudry et al. and the
  $\log\log t/\log t$ refinement with a citation (state precisely what is known).
- *Discrete gradient descent* with $\eta\le1$ (exponential loss): same conclusion with $t\to\eta t$ — complete
  proof if it fits (local smoothness: $\norm{\nabla^2\Loss(w)}_{\op}\le\Loss(w)$), else an exercise with hints.
- *Homogeneous networks (Lyu and Li, ICLR 2020; Ji and Telgarsky, NeurIPS 2020): statement and proof sketch*
  — the smoothed margin increases once $\Loss<1/n$; limit directions are KKT points of
  $\min\norm\theta^2$ s.t. $y_if(x_i;\theta)\ge1$.

**Reading.** Figure: normalized margin versus $\log t$, loss $\sim1/t$, norm $\sim\log t$; theory bounds
versus simulation on a small separable dataset (e.g. 2-D Gaussian clusters).

**Gets wrong.** Thrown away: fixed features, separability, infinite time, no regularization. Where it breaks:
the rates are logarithmic — the limit is never reached in practice; for ReLU networks the KKT points need
not be even local margin maximizers (Vardi, Shamir, Srebro, NeurIPS 2022); margin-based bounds do not track
generalization in deep nets (Nagarajan and Kolter, NeurIPS 2019); large step sizes make the loss
non-monotone yet faster (Wu, Bartlett, Telgarsky, Yu, COLT 2024 **[verify]**); adaptive methods bias toward
other norms (Adam and $\ell_\infty$ margin, Zhang, Zou, Cao 2024 **[verify]**); noisy labels: benign
overfitting in classification (Chatterji and Long, JMLR 2021; Frei, Chatterji, Bartlett, COLT 2022). Fine
structure: linear convolutional networks and Fourier-domain $\ell_{2/L}$ margins (Gunasekar et al., NeurIPS
2018), diagonal networks and $\ell_1$ margins (cross-ref Ch. 3), non-separable data (Ji and Telgarsky, COLT
2019), normalized GD with faster rates (Nacson et al., AISTATS 2019).

**Exercises (ideas).** Logistic loss version; GD version; tightness of the $\log n$ term; compute the
max-margin direction for three points; $\ell_1$-margin for diagonal networks; simulation; research: rates for
homogeneous networks.

### Chapter 5 — Order and Chaos at Initialization (key `signalprop`)

**Phenomenon.** Deep networks are trainable only for carefully scaled initializations (Glorot and Bengio,
AISTATS 2010; He, Zhang, Ren, Sun, ICCV 2015); vanishing and exploding gradients (Hochreiter 1991; Bengio,
Simard, Frasconi 1994); the trainable depth is set by a correlation length that diverges at an order–chaos
boundary (Poole et al., NeurIPS 2016; Schoenholz, Gilmer, Ganguli, Sohl-Dickstein, ICLR 2017); 10,000-layer
vanilla CNNs trained with critical orthogonal initialization (Xiao et al., ICML 2018).

**Model.** $h^1=W^1x+b^1$, $h^{\ell+1}=W^{\ell+1}\phi(h^\ell)+b^{\ell+1}$, $W^\ell_{ij}\sim\cN(0,\sigma_w^2/n_{\ell-1})$,
$b^\ell_i\sim\cN(0,\sigma_b^2)$; widths to infinity. Order parameters: length $q^\ell$ and correlation $c^\ell$
between two inputs.

**Solution.**
- *Theorem (Gaussian-process limit, sequential widths; complete proof).* Conditionally on layer $\ell$, the
  coordinates of $h^{\ell+1}$ are exactly i.i.d. Gaussian with covariance $\sigma_b^2+\sigma_w^2\,
  \tfrac1{n_\ell}\sum_j\phi(h^\ell_j(x))\phi(h^\ell_j(x'))$; the law of large numbers makes this empirical kernel
  deterministic; induction. State the simultaneous-limit theorem with citations (Matthews et al., ICLR 2018;
  Hanin, Ann. Appl. Probab. 2023 **[verify]**); Neal (1996) for one hidden layer; Lee et al., ICLR 2018.
- *Theorem (phase diagram via dual activations; complete proof).* With the fixed point $q^*$ and
  $\phi(\sqrt{q^*}z)=\sum_k a_kh_k(z)$, the correlation map is $C(c)=(\sigma_b^2+\sigma_w^2\sum_ka_k^2c^k)/q^*$ —
  a power series with nonnegative coefficients (`\cref{cor:toolkit:dual}`; Daniely, Frostig, Singer,
  NeurIPS 2016), increasing and convex on $[0,1]$ with $C(1)=1$ and
  $C'(1)=\chi_1=\sigma_w^2\E[\phi'(\sqrt{q^*}z)^2]$. If $\chi_1<1$: $1-c^{\ell+1}\le\chi_1(1-c^\ell)$ — ordered
  phase, depth scale $\xi=-1/\log\chi_1$. If $\chi_1>1$: a fixed point $c^*<1$ attracts — chaotic phase.
  Critical line $\chi_1=1$. (Handle the length map: existence and stability of $q^*$ for tanh.)
- *Theorem (ReLU at He initialization; complete proof).* $\sigma_w^2=2,\sigma_b=0$ preserves lengths; the
  arc-cosine correlation map $C(c)=(\sqrt{1-c^2}+(\pi-\arccos c)c)/\pi$ (Cho and Saul, NeurIPS 2009) has
  $C(1-\eps)=1-\eps+\tfrac{2\sqrt2}{3\pi}\eps^{3/2}+O(\eps^{5/2})$, hence $1-c^\ell\sim\tfrac{9\pi^2}{2\ell^2}$
  (critical, polynomial decorrelation; confirmed numerically by the editor: $(1-c^\ell)\ell^2\to44.4$).
- *Gradients (heuristic ◇ with a stated assumption):* $\E\norm{\partial\Loss/\partial h^\ell}^2\approx
  \chi_1^{L-\ell}\E\norm{\partial\Loss/\partial h^L}^2$ under the gradient-independence assumption, rigorous in
  the infinite-width limit (Yang, NeurIPS 2019 / Tensor Programs **[verify]**); exact for linear $\phi$.
  Dynamical isometry (Saxe, McClelland, Ganguli, ICLR 2014; Pennington, Schoenholz, Ganguli, NeurIPS 2017):
  statement.

**Reading.** Figure 1: phase diagram in $(\sigma_w^2,\sigma_b^2)$ for tanh with the critical line and the
depth scale (computed by Gauss–Hermite quadrature). Figure 2: $c^\ell$ versus $\ell$ in the three phases,
theory versus finite-width simulation ($n=1000$). The physics: order parameter, critical slowing down
($\xi\to\infty$), mean-field exactness at infinite width.

**Gets wrong.** Thrown away: finite width, training, architecture (residual connections, normalization,
attention), data. Where it breaks: finite-width fluctuations grow with depth-to-width ratio $L/n$ (Hanin and
Nica, Comm. Math. Phys. 2020; Hanin, NeurIPS 2018; Roberts, Yaida, Hanin 2022) — at $L\sim n$ the GP picture
fails; criticality is necessary but not sufficient for trainability; NNGP/NTK kernels underperform trained
networks (Lee et al., NeurIPS 2020); batch normalization makes gradients explode at initialization (Yang et
al., ICLR 2019); rank collapse in attention (Dong, Cordonnier, Loukas, ICML 2021; Noci et al., NeurIPS 2022).
Fine structure: $1/n$ effective theory (Roberts–Yaida–Hanin), shaped activations and the neural covariance
SDE (Li, Nica, Roy, NeurIPS 2022), residual networks at the edge of chaos (Yang and Schoenholz, NeurIPS
2017), depth-$\mu$P (Yang et al. 2023; Bordelon et al. 2024 **[verify]**).

**Exercises (ideas).** Dual activation of erf (arcsine kernel) and its critical line; ReLU length
preservation; the $9\pi^2/2$ constant; leaky ReLU; simulation: fluctuations of $c^\ell$ versus $L/n$;
research: the $L/n$ regime.

### Chapter 6 — Lazy Training (key `lazy`)

**Phenomenon.** Very wide networks reach zero training loss while each weight barely moves, and their
predictions match kernel regression with the neural tangent kernel (Jacot, Gabriel, Hongler, NeurIPS 2018;
Lee et al., NeurIPS 2019; Du, Zhai, Póczos, Singh, ICLR 2019; Arora et al., NeurIPS 2019 — CNTK on CIFAR-10
**[verify accuracy numbers]**); spectral bias (Rahaman et al., ICML 2019; Basri et al., NeurIPS 2019). But
kernel models underperform trained networks (Lee et al., NeurIPS 2020; Geiger et al., J. Stat. Mech. 2020;
Fort et al., NeurIPS 2020).

**Model.** A parametric map $F:\R^P\to\R^n$ (outputs on the training set), Jacobian $J$, gradient flow on
$\tfrac12\norm{F(\theta)-y}^2$; specialization: $f(x;W)=m^{-1/2}\sum_{r=1}^m a_r\phi(\ip{w_r,x})$, fixed
$a_r\in\{\pm1\}$, $w_r(0)\sim\cN(0,I_d)$, $\norm{x_i}=1$, smooth $\phi$ with $|\phi'|,|\phi''|\le\beta$.

**Solution.**
- *Theorem (lazy training; complete proof).* If $\sigma_{\min}(J(\theta_0))\ge2\mu$ and $J$ is $L_J$-Lipschitz
  on $B(\theta_0,R)$ with $R=\norm{r_0}/\mu$ and $L_JR\le\mu$, then $\norm{r_t}\le e^{-\mu^2t}\norm{r_0}$ and
  $\norm{\theta_t-\theta_0}\le\norm{r_0}/\mu$ for all $t$ ($r=F-y$). Proof: on the exit time interval
  $\sigma_{\min}(J)\ge\mu$; $\tfrac{d}{dt}\norm{r}\le-\mu\norm{J^\top r}$ and $\norm{\dot\theta}=\norm{J^\top r}$.
  Corollary: closeness to the linearized model (Grönwall).
- *Theorem (two-layer networks; complete proof).* With $K^\infty_{ij}=\ip{x_i,x_j}\E_w[\phi'(\ip{w,x_i})\phi'(\ip{w,x_j})]$,
  $\lambda_0=\lmin(K^\infty)>0$ (cite positivity for non-parallel inputs), width
  $m\gtrsim n^2\log(n/\delta)/\lambda_0^2$ (verify) gives the hypotheses: $J(W_0)J(W_0)^\top\to K^\infty$ by
  concentration; $L_J\le\beta\sqrt{n/m}$; $\norm{r_0}\lesssim\sqrt{n\log(n/\delta)}$. Each neuron moves
  $O(\sqrt{n/(\lambda_0m)})$: lazy.
- *Spectral bias (complete for the linearized flow):* $r_t=e^{-tK}r_0$; eigenfunctions of rotation-invariant
  kernels on the sphere are spherical harmonics (Funk–Hecke, `\cref{thm:toolkit:addition}`); NTK eigenvalues
  decay with degree (Bietti and Mairal, NeurIPS 2019) ⇒ low degrees first.
- *Theorem (what a kernel cannot learn; complete proof).* $x\sim\Unif(\Sph^{d-1})$; $Z_w$ the zonal harmonic of
  degree $k$, $f_w=Z_w/\sqrt{N_{d,k}}$ ($\norm{f_w}=1$), $w$ uniform on the sphere. For every subspace $V$ of
  dimension $\le n$ chosen independently of $w$ (e.g. the span of $K(\cdot,x_i)$ for kernel ridge regression
  with any fixed kernel and any inputs), $\E_w\norm{f_w-\Pi_Vf_w}^2\ge1-n/N_{d,k}$, $N_{d,k}\asymp d^k/k!$.
  Proof: $\E_w[f_w\otimes f_w]=N_{d,k}^{-1}\Pi_{\cH_k}$ by the addition theorem; $\E\norm{\Pi_Vf_w}^2=
  \tr(\Pi_V\Pi_{\cH_k})/N_{d,k}$. Hence kernel methods need $n\gtrsim d^k$ for a degree-$k$ single-index target,
  versus far fewer for feature learning (Ch. 7). Cite Ghorbani, Mei, Misiakiewicz, Montanari (Ann. Statist.
  2021) and Yehudai and Shamir (NeurIPS 2019) for sharper versions.
- *The other infinite width (heuristic ◇ + statements):* mean-field scaling $1/m$ and Wasserstein gradient
  flow (Mei, Montanari, Nguyen, PNAS 2018; Chizat and Bach, NeurIPS 2018; Rotskoff and Vanden-Eijnden;
  Sirignano and Spiliopoulos); lazy training as an effect of scale (Chizat, Oyallon, Bach 2019); one-step
  feature movement $O(1/\sqrt m)$ versus $O(1)$; $\mu$P (Yang and Hu, ICML 2021).

**Reading.** Figure: relative weight change and kernel change versus width (simulation) with the predicted
$m^{-1/2}$ slope. Figure: kernel ridge regression versus a trained (non-lazy) network on a single-index target
of degree 2 or 3 in moderate $d$, showing the sample-complexity gap (keep it fast).

**Gets wrong.** Thrown away: features frozen; width $\gg$ poly($n$); small learning rates. Where it breaks: the
empirical NTK moves a lot early in training (Fort et al. 2020; silent alignment, Atanasov et al. 2022);
performance gap; sample-complexity separations (above; Allen-Zhu and Li 2019; Bai and Lee, ICLR 2020; Damian,
Lee, Soltanolkotabi, COLT 2022); no feature reuse. Fine structure: finite-width corrections (Hanin and Nica,
ICLR 2020; Huang and Yau, ICML 2020; Dyer and Gur-Ari, ICLR 2020), mean-field and $\mu$P, lazy-to-rich phase
diagrams (Geiger et al. 2020).

**Exercises (ideas).** NTK of a two-layer ReLU network (arc-cosine kernels); linearization error bound;
Gaussian-input version of the kernel lower bound with Hermite tensors; train both layers; simulation:
relative weight change versus $m$; research: quantify the lazy–rich crossover.

### Chapter 7 — The Search Phase (key `singleindex`)

**Phenomenon.** In feature learning the loss first stays flat while the network searches for the relevant
direction, then drops quickly; the length of the search depends on the target in a precise way (Ben Arous,
Gheissari, Jagannath, JMLR 2021, "Online stochastic gradient descent on non-convex losses from
high-dimensional inference"; Dudeja and Hsu, COLT 2018; Arnaboldi et al. 2023).

**Model.** $x\sim\cN(0,I_d)$, $y=g(\ip{w_\star,x})$, student $\phi(\ip{w,x})$ with $\norm w=1$, online
(fresh-sample) spherical SGD on the correlation loss $-y\phi(\ip{w,x})$. Information exponent
$k^\star=\min\{k\ge1:\hat g_k\hat\phi_k\ne0\}$. Overlap $r=\ip{w,w_\star}$.

**Solution.**
- *Lemma (Hermite correlation; `\cref{thm:toolkit:hermite-corr}`).* $\E[y\phi(\ip{w,x})]=\sum_k\hat g_k\hat\phi_kr^k=:\Psi(r)$.
- *Theorem (population flow; complete proof).* $\dot r=(1-r^2)\Psi'(r)$; from $r_0\asymp d^{-1/2}$ the time to
  reach $r=1/2$ is $\asymp d^{(k^\star-2)/2}$ for $k^\star\ge3$, $\asymp\log d$ for $k^\star=2$, $\asymp1$ for $k^\star=1$.
- *Theorem (online SGD for $k^\star=k\ge3$; complete proof — the hardest proof in the book).* Matched pure
  link $g=\phi=h_k$ (or more generally $\Psi'(r)\ge ckr^{k-1}$ on $[0,\tfrac12]$ and bounded moments).
  Update $w_{t+1}=(w_t-\eta g_t)/\norm{w_t-\eta g_t}$ with $g_t=P^\perp_{w_t}\nabla\ell_t(w_t)$ ⇒
  $r_{t+1}=(r_t+\eta k r_t^{k-1}(1-r_t^2)+\eta\xi_t)(1+\eta^2\norm{g_t}^2)^{-1/2}$ with martingale noise
  $\E[\xi_t^2|\cF_t]\le C_k$ and $\E[\norm{g_t}^2|\cF_t]\le C_kd$. If $r_0\ge d^{-1/2}$, $\eta=c_k\eps d^{-k/2}$,
  $T=C_kd^{k-1}/\eps$, then $\Pr(r_T\ge\tfrac12)\ge1-\eps$. Proof plan: doubling epochs $[2^jr_0,2^{j+1}r_0]$
  of length $T_j\asymp(2^jr_0)^{2-k}/\eta$; per epoch, drift gain $\gtrsim2^jr_0$; the martingale part is
  $\le2^jr_0/4$ except with probability $\le16C\eta^2T_j/(2^jr_0)^2\asymp\eta(2^jr_0)^{-k}$ (Doob,
  `\cref{lem:toolkit:doob}`), summable over $j$ to $\lesssim\eta d^{k/2}$; the normalization loss
  $\sum\eta^2\norm{g_t}^2r_t\lesssim\eta d\,(2^jr_0)^{3-k}$ per epoch, small since $\eta\ll d^{-k/2}$ (Markov).
  Then a second phase to $r\ge1-\eps'$. A random start has $r_0\ge d^{-1/2}$ with probability $\approx0.16$; for
  even $k$ the sign is irrelevant. Author: verify every constant, and make the proof self-contained.
- *Statements.* Ben Arous et al.: $n\asymp d^{k^\star-1}$ ($k^\star\ge3$), $d\log d$ ($k^\star=2$), $d$
  ($k^\star=1$), with a matching lower bound for online SGD; CSQ lower bound $d^{k^\star/2}$ (Damian et al.;
  Abbe et al.) **[verify the exact form, including odd $k^\star$]**.

**Reading.** Figure: escape time versus $d$ for $k=1,2,3$ (log–log), simulation versus predicted slopes.
Figure: stochastic trajectories $r_t$ versus the ODE. The dictionary between information exponent and
"how much the target leaks at low order".

**Gets wrong.** Thrown away: isotropic Gaussian inputs, one relevant direction, known link, single pass,
spherical constraint, single neuron. Where it breaks: reusing samples beats the information exponent
(generative exponent: Damian, Pillaud-Vivien, Lee, Bruna, COLT 2024; Dandi et al., ICML 2024; Arnaboldi et al.
2024 "Repetita iuvant"; Lee, Oko, Suzuki, Wu, NeurIPS 2024 **[verify]**); smoothing the loss reaches $d^{k^\star/2}$
(Damian, Nichani, Ge, Lee, NeurIPS 2023); the exponent depends on the loss; structured covariance aligned with
$w_\star$ helps (Ba et al., NeurIPS 2023; Mousavi-Hosseini et al. **[verify]**). Fine structure: multi-index
and staircase/leap complexity (Abbe, Boix-Adserà, Misiakiewicz, COLT 2022 and COLT 2023), one-step feature
learning (Ba et al., NeurIPS 2022), unknown link (Bietti, Bruna, Sanford, Song, NeurIPS 2022).

**Exercises (ideas).** Hermite lemma via generating functions; information exponents of ReLU, $|z|$,
$\He_3$, $\tanh$; square loss versus correlation loss; label noise; simulation: escape times; research:
generative exponent.

### Chapter 8 — Plateaus and Specialization (key `committee`)

**Phenomenon.** Training of networks with several hidden units shows long plateaus during which units are
redundant, followed by specialization (Saad and Solla, Phys. Rev. Lett. 1995 and Phys. Rev. E 1995; Biehl and
Schwarze, J. Phys. A 1995; Biehl, Riegler, Wöhler, J. Phys. A 1996) **[verify]**. Modern rigorous treatment:
Goldt, Advani, Saxe, Krzakala, Zdeborová (NeurIPS 2019); phase diagrams (Veiga et al., NeurIPS 2022);
Arnaboldi et al. (COLT 2023).

**Model.** Teacher $y=\sum_{k=1}^{m_\star}g(\ip{w_{\star k},x})$, student $\hat y=\sum_{j=1}^mg(\ip{w_j,x})$,
$g(z)=\erf(z/\sqrt2)$, $x\sim\cN(0,I_d)$, online SGD on square loss with step $\eta/d$ (fix the normalization
so that preactivations are $O(1)$; Saad–Solla conventions translated). Order parameters
$Q_{jl}=\ip{w_j,w_l}$, $R_{jk}=\ip{w_j,w_{\star k}}$, $T_{kl}=\ip{w_{\star k},w_{\star l}}$ (scaled
consistently).

**Solution.**
- *Lemma (erf integrals; complete proof).* For jointly Gaussian $(u,v)$ with $\E u^2=a$, $\E v^2=b$,
  $\E uv=c$: $\E[g(u)g(v)]=\tfrac2\pi\arcsin\big(c/\sqrt{(1+a)(1+b)}\big)$ (via Price's theorem,
  `\cref{lem:toolkit:price}`); the three- and four-point integrals needed by the ODEs, derived from it.
- *Theorem (ODE limit; complete proof via `\cref{thm:toolkit:ode-method}`).* As $d\to\infty$ with $t=$
  steps$/d$, $(Q,R)$ converges uniformly on $[0,T]$ in probability to the solution of explicit ODEs (the
  $\eta^2$ term in $\dot Q$ comes from $\norm x^2/d^2\approx1/d$). Verify every term.
- *Theorem (the symmetric plateau; complete proof of the symmetry claims, numerics for constants).* For
  $m=m_\star=2$, orthonormal teacher: the symmetric subspace (all $R_{jk}$ equal, $Q_{11}=Q_{22}$) is
  invariant; in the small-$\eta$ limit it contains a fixed point (characterized by a one-dimensional
  equation: both students equal to $a(w_{\star1}+w_{\star2})$) that is attracting within the subspace and
  unstable along the specialization direction $w_1-w_2\propto w_{\star1}-w_{\star2}$ with rate $\lambda_u>0$;
  plateau length $\approx\tfrac{1}{2\lambda_u}\log d$ from random initialization (asymmetry $\asymp d^{-1/2}$).
- *Single unit ($m=m_\star=1$; complete):* the $\eta^2$ term limits convergence: exponential convergence for
  $\eta<\eta_{\max}$ — a stability threshold, cross-ref Ch. 15.

**Reading.** Figure: generalization error versus $t$ from the ODEs with the plateau, overlaid with SGD at
two values of $d$; plateau length versus $\log d$.

**Gets wrong.** Thrown away: Gaussian inputs, $d\to\infty$ with tiny steps, fixed second layer, $m,m_\star=O(1)$,
single pass. Where it breaks: overparameterized students ($m>m_\star$) change plateaus and generalization
(Goldt et al. 2019); large steps and noise-dominated fixed points (Veiga et al. 2022); structured inputs
(hidden manifold model, Goldt et al., Phys. Rev. X 2020). Fine structure: leap complexity and saddle-to-saddle
dynamics for multi-index targets (Abbe, Boix-Adserà, Misiakiewicz, COLT 2023): sample complexity
$\tilde\Theta(d^{\max(\mathrm{Leap}-1,1)})$ **[verify]**; Gaussian equivalence; dynamical mean-field theory.

**Exercises (ideas).** Derive $\E[g(u)g(v)]$; ODEs for $m=m_\star=1$ and $\eta_{\max}$; invariance of the
symmetric subspace; locate the plateau numerically; simulation; research: plateau length with $m>m_\star$.

### Chapter 9 — Hidden Progress (key `parity`)

**Phenomenon.** On $k$-sparse parities, SGD training loss is flat for a long time, then drops abruptly, yet
measurable progress accumulates all along (Barak, Edelman, Goel, Kakade, Malach, Zhang, NeurIPS 2022, "Hidden
progress in deep learning: SGD learns parities near the computational limit"). Context: emergent abilities
(Wei et al., TMLR 2022) versus metric artifacts (Schaeffer, Miranda, Koyejo, NeurIPS 2023).

**Model.** $x\sim\Unif\{\pm1\}^d$, $y=\chi_S(x)=\prod_{i\in S}x_i$, $|S|=k$ unknown; a single neuron and a
two-layer network trained by (S)GD on the correlation loss.

**Solution.**
- *Theorem (statistical-query lower bound; complete proof).* Every SQ learner with tolerance $\tau$ needs at
  least $(\binom dk-1)\tau^2$ queries to identify $S$ (Kearns, JACM 1998; Blum et al., STOC 1994): orthonormal
  parities, $\sum_S\ip{\tilde q,\chi_S}^2\le1$, adversary answers the label-independent part.
- *Theorem (exact one-step signal of a monomial neuron; complete proof).* For $\phi(z)=z^k$:
  $\E[\chi_S(x)\ip{w,x}^k]=k!\prod_{i\in S}w_i$; at $w_0=d^{-1/2}\mathbf 1$ the population gradient is
  $k!\,d^{-(k-1)/2}$ on $S$ and exactly $0$ off $S$. Remark: for ReLU/threshold neurons at symmetric
  initialization the gradient equals $\hat\psi(S\setminus\{i\})$ on $S$ and $\hat\psi(S\cup\{i\})$ off $S$ for the
  symmetric function $\psi=\phi'(\sum_jx_j+b)$, a "Fourier gap" between levels $k-1$ and $k+1$ of order $d$
  (Barak et al.); state with the majority Fourier coefficients (O'Donnell, *Analysis of Boolean Functions*,
  2014).
- *Theorem (one gradient step recovers $S$; complete proof).* With $n=C_kd^{k-1}\log(d/\delta)$ samples, the
  $k$ largest coordinates of the empirical gradient at $w_0$ are exactly $S$ with probability $\ge1-\delta$
  (tail bound for degree-$(k-1)$ polynomials via `\cref{thm:toolkit:bonami}` or a truncation argument). A
  second layer of $k+1$ ReLUs computes the parity of the selected coordinates (explicit construction).
  Compare with the SQ bound: $d$ queries at tolerance $n^{-1/2}$ ⇒ $dn\gtrsim\binom dk$ ⇒ $n\gtrsim d^{k-1}/k!$:
  "near the computational limit".
- *Theorem (hidden progress; complete proof).* Population correlation ascent from small initialization: the
  support weights obey $\dot u=k!u^{k-1}$ (or the spherical version, a Boolean analogue of Ch. 7 with
  information exponent $k$): $u(t)=(u_0^{-(k-2)}-(k-2)k!t)^{-1/(k-2)}$; the correlation $k!u^k$ stays below $\eps$
  until $t\ge t^*(1-O(\eps^{(k-2)/k}))$ while $\log(u(t)/u_0)$ grows steadily from the start. Make precise.

**Reading.** Figure: SGD on a small MLP for $(d,k)=(30\text{–}50,3)$: loss (flat, then drop) and a progress
measure (weight mass on $S$, or Fourier gap) growing steadily; overlay the ODE.

**Gets wrong.** Thrown away: uniform inputs; exact parity with no low-degree correlations; special
initialization; one step / population dynamics. Where it breaks: natural tasks have staircase structure that
gives gradient signal at low degree (Abbe et al., NeurIPS 2021 "The staircase property"; COLT 2022);
non-uniform product distributions create low-degree correlations; width–luck–compute trade-offs with sparse
initialization (Edelman, Goel, Kakade, Malach, Zhang, NeurIPS 2023 "Pareto frontiers"); emergence may be a
metric artifact (Schaeffer et al.); transformers and parity: sensitivity (Hahn and Rofin, ACL 2024), chain of
thought helps (Kim and Suzuki 2025 **[verify]**). Fine structure: leap complexity; noisy parity / LPN hardness
beyond SQ.

**Exercises (ideas).** SQ bound details; Fourier coefficients of majority for small $d$; $k+1$ ReLUs for
parity; hypercontractive moment bound; biased inputs ($p$-biased cube) create degree-1 signal; simulation;
research: SGD lower bounds beyond SQ.

### Chapter 10 — Grokking (key `grokking`)

**Phenomenon.** On small algorithmic datasets (e.g. $a+b\bmod p$ with a fraction of the table as training
data) networks memorize quickly and generalize much later (Power, Burda, Edwards, Babuschkin, Misra 2022,
arXiv:2201.02177); weight decay matters; the trained network computes with a few key Fourier frequencies via
trigonometric identities, and progress measures show the circuit forming before the jump (Nanda, Chan,
Lieberum, Smith, Steinhardt, ICLR 2023). Effective theories: Liu et al. (NeurIPS 2022); Omnigrok (Liu,
Michaud, Tegmark, ICLR 2023).

**Model.** $\Z_p$, $p$ an odd prime (state the chapter's convention: the modulus is $p$ here, which
locally overrides "$p$ features"), one-hot inputs $(a,b)$, two-layer network with quadratic activation
$f_c(a,b)=\sum_rv_{rc}\,(u_r(a)+u'_r(b))^2$.

**Solution.**
- *Theorem (a Fourier network; complete proof).* With frequencies $\omega_j=2\pi j/p$, $j=1,\dots,(p-1)/2$,
  and phases $\varphi,\psi\in\{0,2\pi/3,4\pi/3\}$: neurons $(\cos(\omega_ja+\varphi)+\cos(\omega_jb+\psi))^2$
  with readout $\tfrac29\cos(\omega_jc+\varphi+\psi)$ give logits exactly $(p\,\ind\{c=a+b\}-1)/2$ (editor
  checked numerically to $10^{-13}$ at $p=23$). Width $9(p-1)/2$. Explain via product-to-sum identities and
  cancellation of the $a-b$, $2a$, $2b$ and constant terms. Remark: random phases (Gromov 2023, arXiv
  **[verify]**) cancel only at large width.
- *Theorem (max-margin neurons are single-frequency; complete proof).* The class-weighted margin of a single
  neuron reduces (the $u(a)^2$, $u'(b)^2$ terms cancel) to $\tfrac{2p}{p-1}\E_{a,b}[u(a)u'(b)(v(a+b)-\bar v)]$,
  a sum over frequencies of $\hat u\hat u'\overline{\hat v}$; under a norm constraint it is maximized by
  concentrating on one frequency pair $\pm\omega$ (convexity of $s\mapsto s^{3/2}$), giving
  $u=\cos(\omega a+\varphi)$, $u'=\cos(\omega b+\psi)$, $v=\cos(\omega c+\varphi+\psi)$.
- *Theorem (Morwani, Edelman, Oncescu, Zhao, Kakade, ICLR 2024; certificate lemma complete if possible,
  else sketch).* Max-margin networks use every frequency; the network of the first theorem attains equal
  margins on all points and is max-margin.
- *A solvable model of the delay (statement + heuristic ◇ + figure).* Large initialization plus small weight
  decay: an early kernel (lazy) phase that memorizes and a late rich phase that generalizes, with transition
  at $t\approx\lambda^{-1}\log(\alpha/\alpha_c)$ — Lyu, Jin, Li, Du, Lee, Hu, ICLR 2024 ("Dichotomy of early
  and late phase implicit biases can provably induce grokking"); Kumar, Bordelon, Gershman, Pehlevan, ICLR
  2024 ("Grokking as the transition from lazy to rich training dynamics"). Illustrate with the diagonal
  network of Ch. 3 plus weight decay on a sparse regression task (train error drops early, test error late).

**Reading.** Figure: logits/accuracy of the explicit network; DFT of a trained network's first layer
(simulation with torch on $p\approx 59$–$97$, fast); grokking curves in the diagonal-network caricature with
the predicted time.

**Gets wrong.** Thrown away: a specific algebraic task, one-hot inputs, quadratic activation, infinite time.
Where it breaks: different runs find different algorithms ("clock" and "pizza", Zhong, Liu, Tegmark,
Andreas, NeurIPS 2023); grokking is controlled by hyperparameters (initialization norm, weight decay) and can
be removed (Omnigrok); grokking without circuits: linear estimators (Levi, Beck, Bar-Sinai, ICLR 2024
"Grokking in linear estimators — a solvable model that groks without understanding"), slingshot dynamics
(Thilak et al. 2022), softmax collapse (Prieto et al., ICLR 2025 **[verify]**), circuit efficiency (Varma et
al. 2023); non-abelian groups (Chughtai, Chan, Nanda, ICML 2023; Stander et al., ICML 2024). Fine structure:
group representations; Fourier features in pretrained LLMs for addition (Zhou et al., NeurIPS 2024
**[verify]**).

**Exercises (ideas).** Minimal number of phases per frequency; multiplication via the discrete log;
$S_3$ and irreducible representations; the certificate; simulation: DFT of trained embeddings; research:
dynamics.

### Chapter 11 — Superposition (key `superposition`)

**Phenomenon.** Polysemantic neurons; more features than dimensions; sparse autoencoders recover
interpretable features (Elhage et al. 2022, "Toy models of superposition", Transformer Circuits Thread;
Olah et al. 2020; Arora, Li, Liang, Ma, Risteski, TACL 2018 on word senses; Bricken et al. 2023; Cunningham
et al., ICLR 2024; Templeton et al. 2024).

**Model.** Features $x\in[0,1]^n$, independent, $x_i=0$ with probability $S$ and $\Unif[0,1]$ otherwise;
importances $I_i$; $h=Wx\in\R^m$, $m<n$; $x'=\relu(W^\top Wx+b)$; loss $\sum_iI_i\E(x_i-x'_i)^2$. (Sparsity
$S$ is local notation.)

**Solution.**
- *Theorem (no superposition without a nonlinearity; complete proof).* For $x'=W^\top Wx+b$ the optimum keeps
  the $m$ features with largest $I_i\Var(x_i)$ in orthogonal directions and drops the rest (Eckart–Young for a
  diagonal weighted covariance).
- *Theorem (two features, one dimension; complete proof of the exact losses).* Exact expected losses of the
  canonical configurations (one feature dedicated, the other dropped; antipodal superposition $W=(w,-w)$) each
  with its optimal bias, as explicit functions of $S$ and the importance ratio, and the boundary between them.
  **Caution (editor's numerics):** with optimal biases the global optimum is *not* always one of these
  configurations; for dense features ($S\lesssim0.25$, equal importance) asymmetric configurations do slightly
  better, and with $b=0$ forced the comparison is degenerate. Find the cleanest true statement: e.g. a
  sparse-limit theorem — every configuration that does not superpose has loss $\ge c(1-S)$ while the antipodal
  one has loss $O((1-S)^2)$, so superposition is forced as $S\to1$ — plus the exact comparison of the canonical
  configurations, plus a numerically computed global phase diagram, honestly labeled.
- *Theorem (packing; complete proof).* For $\eps\in(0,1)$ there are $N=\lfloor e^{\eps^2m/4}\rfloor$ (verify)
  unit vectors in $\R^m$ with pairwise $|\ip{u_i,u_j}|\le\eps$ (random signs + Hoeffding + union bound).
- *Theorem (reading out sparse features through a ReLU; complete proof).* Unit columns with coherence $\mu$,
  at most $s$ active features in $[0,1]$: readout $\relu(\ip{w_i,Wx}-b)$ with $b=s\mu$ outputs $0$ on inactive
  features and is within $2s\mu$ (verify) of the truth on active ones — compressed sensing in a network.

**Reading.** Figure: phase diagram for $n=2,m=1$ (exact canonical boundaries + numerical global optimum).
Figure: trained $n=5$, $m=2$ model showing the pentagon at high sparsity (simulation).

**Gets wrong.** Thrown away: given, independent, uniform sparse features; reconstruction instead of
computation; one layer. Where it breaks: correlated features change the geometry; computation in
superposition differs (Hänni, Mendel, Vaintrob, Chan 2024 **[verify]**); SAE non-uniqueness and "dark matter"
(Engels et al. 2024 **[verify]**); multidimensional circular features (Engels, Michaud, Liao, Gurnee, Tegmark,
ICLR 2025 "Not all language model features are linear") — cross-ref Ch. 10; the linear representation
hypothesis (Park, Choe, Veitch, ICML 2024). Fine structure: polytope geometry, memorization and double
descent (Henighan et al. 2023), scaling of SAEs.

**Exercises (ideas).** Linear PCA proof; antipodal loss with bias; packing bound; simulation of the pentagon;
correlated features; research: computation in superposition.

### Chapter 12 — Neural Collapse (key `collapse`)

**Phenomenon.** In the terminal phase of training, last-layer features collapse to their class means, the
means form a simplex equiangular tight frame, classifiers align with the means, and prediction reduces to
nearest class center (Papyan, Han, Donoho, PNAS 2020; MSE version Han, Papyan, Donoho, ICLR 2022; layerwise
law, He and Su, PNAS 2023).

**Model.** Unconstrained (layer-peeled) features: free $h_{k,i}\in\R^d$, $d\ge K-1$, classifier $W$, cross-entropy
with norm constraints (Fang, He, Long, Su, PNAS 2021; Lu and Steinerberger, ACHA 2022; Graf et al., ICML 2021) or
with weight decay (Zhu et al., NeurIPS 2021); Mixon, Parshall, Pi (2022) **[verify venue]**.

**Solution.**
- *Theorem (global minimizers are neural collapse; complete proof).* Norm-constrained cross-entropy: every
  minimizer has $h_{k,i}=h_k$, $\{h_k\}$ a simplex ETF of the maximal norm, $w_k\propto h_k$. Proof: Jensen on
  the exponentials, then Cauchy–Schwarz and the constraint, with equality analysis.
- *MSE version (closed form, complete or proposition)* and the *benign landscape* (Zhu et al. 2021: every
  local minimum is global; others strict saddles) — statement and sketch.
- *Minority collapse (Fang et al. 2021)* — statement and heuristic for $K=2$ groups.
- Link to Ch. 4: without regularization, gradient descent on the UFM converges in direction to a max-margin
  solution, which is neural collapse (Ji, Lu, Zhang, Deng, Su, ICLR 2022).

**Reading.** Figure: NC metrics (within-class variability, ETF deviation, self-duality) versus epoch for a
small network trained past zero error on synthetic or small real data (torch, fast), compared with the UFM
optimum (zero).

**Gets wrong.** Thrown away: free features (no data or backbone constraints), training data only, balanced
classes, last layer only. Where it breaks: test features do not collapse and collapse can hurt transfer (Hui,
Belkin, Nakkiran 2022); for deep UFMs low-rank solutions beat deep neural collapse (Súkeník, Mondelli,
Lampert, NeurIPS 2024 **[verify]**); $K>d+1$ (vocabularies) makes a simplex impossible (generalized neural
collapse, Jiang et al., ICML 2024 **[verify]**); next-token prediction (Zhao, Thrampoulidis et al. 2024
**[verify]**; Wu and Papyan 2024). Fine structure: deep UFM and deep neural collapse (Súkeník et al., NeurIPS
2023), extended UFM (Tirer and Bruna, ICML 2022).

**Exercises (ideas).** ETF properties; MSE closed form; minority collapse for two groups; NC4 from NC1–3;
simulation; research: dynamics of collapse.

### Chapter 13 — Learning in Context (key `icl`)

**Phenomenon.** Transformers learn from examples in the prompt without weight updates (Brown et al.,
NeurIPS 2020); transformers trained on linear-regression prompts match least squares (Garg, Tsipras, Liang,
Valiant, NeurIPS 2022); "transformers learn in-context by gradient descent" (von Oswald et al., ICML 2023;
Akyürek et al., ICLR 2023); a task-diversity threshold (Raventós, Paul, Chen, Ganguli, NeurIPS 2023).

**Model.** Prompts $(x_1,y_1,\dots,x_N,y_N,x_q)$, $x_i\sim\cN(0,I_d)$ (then general $\Sigma$),
$w\sim\cN(0,I_d)$, $y_i=\ip{w,x_i}+\eps_i$; single-layer linear self-attention (no softmax) with embedding
$Z\in\R^{(d+1)\times(N+1)}$ and the standard mask; prediction read from the query column.

**Solution.**
- *Proposition (expressivity; complete).* Suitable $W^{KQ},W^{PV}$ give $\hat y_q=x_q^\top\Gamma\,\tfrac1N\sum_iy_ix_i$:
  one step of preconditioned gradient descent from $0$.
- *Theorem (the optimal one-step learner; complete proof).* For $\Sigma=I$: $\Gamma^\star=\tfrac{N}{N+d+1+\sigma^2}I$,
  minimal risk $\tfrac{d(d+1+\sigma^2)}{N+d+1+\sigma^2}$ (+ $\sigma^2$ with query noise) — editor checked by Monte
  Carlo. General $\Sigma$ via $\E[(xx^\top)A(xx^\top)]=2\Sigma A\Sigma+\tr(A\Sigma)\Sigma$ (verify; Ahn, Cheng,
  Daneshmand, Sra, NeurIPS 2023). Compare with the Bayes (ridge) risk.
- *Theorem (global optimum over all single-layer LSA parameters; complete proof or careful sketch with the
  key symmetry lemma proved).* The minimizer computes exactly the predictor above (Ahn et al. 2023; Mahankali,
  Hashimoto, Ma, ICLR 2024; Zhang, Frei, Bartlett, JMLR 2024 **[verify]**).
- *Gradient flow (Zhang, Frei, Bartlett): statement + sketch*; robustness to task shift but not covariate
  shift ($\Gamma^\star$ depends on $\Sigma$).
- *Task diversity (Raventós et al.; Lu, Letey, Zavatone-Veth, Maiti, Pehlevan, "Asymptotic theory of
  in-context learning by linear attention" **[verify venue/year]**): statement.*

**Reading.** Figure: ICL risk versus $N$: theory curve for one-step GD, trained LSA (simulation: GD on the
LSA parameters), least squares and ridge.

**Gets wrong.** Thrown away: softmax, depth, MLPs, tokenization, Gaussian linear tasks. Where it breaks:
softmax attention implements different estimators (Collins et al., NeurIPS 2024 **[verify]**); "ICL is GD" does
not describe trained LLMs (Shen, Mishra, Khashabi, ICML 2024 position paper **[verify]**); multilayer LSA
implements preconditioned multi-step or higher-order methods only under special conditions (Ahn et al. 2023;
Fu et al. 2024 **[verify]**); ICL in LLMs relies on induction heads and task recognition (Ch. 14; Pan et al.
2023); transience (Singh et al., NeurIPS 2023). Fine structure: Bayesian view (Xie et al., ICLR 2022), task
diversity, looped transformers, softmax dynamics (Huang, Cheng, Liang, ICML 2024 **[verify]**).

**Exercises (ideas).** Gaussian fourth-moment identity; $\Gamma^\star$ for general $\Sigma$; comparison with
ridge; two layers implement two GD steps; simulation; research: softmax.

### Chapter 14 — Induction Heads (key `induction`)

**Phenomenon.** Induction heads ($[A][B]\dots[A]\to[B]$) form in an abrupt phase change early in training,
coinciding with a jump in in-context learning (Elhage et al. 2021, "A mathematical framework for transformer
circuits"; Olsson et al. 2022, "In-context learning and induction heads"); data properties matter (Chan et
al., NeurIPS 2022); mechanistic studies of formation (Singh et al., ICML 2024 **[verify]**; Reddy, ICLR 2024).

**Model.** In-context Markov chains (Edelman, Edelman, Goel, Malach, Tsilivis, NeurIPS 2024, "The evolution of
statistical induction heads: in-context learning Markov chains" **[verify]**): each sequence draws a
$k$-state transition matrix with Dirichlet($\alpha\mathbf 1$) rows, then a sequence; next-token
cross-entropy; a two-layer attention-only transformer.

**Solution.**
- *Theorem (Bayes-optimal predictor; complete proof).* $\Pr(x_{T+1}=j\mid x_{1:T})=(N_{x_Tj}+\alpha)/(N_{x_T\cdot}+k\alpha)$
  (Dirichlet–multinomial conjugacy, initial state independent of the matrix). Unigram and uniform baselines and
  their losses.
- *Theorem (a two-layer circuit; complete construction).* Layer 1: previous-token head writes $x_{t-1}$ into a
  separate subspace at position $t$; layer 2: from the last position, attend to positions $t$ with
  $x_{t-1}=x_T$, read $x_t$; with attention logits scaled by $\beta\to\infty$ the output is the empirical bigram
  distribution $N_{x_Tj}/N_{x_T\cdot}$, with error exponentially small in $\beta$ (quantify). Smoothing by an
  attention sink as an exercise.
- *Theorem (why the circuit forms late: a solvable caricature; complete proof).* Composition creates a saddle:
  with layer strengths $a$ (previous-token) and $b$ (induction) and a unigram strength $c$, the gradient on
  $c$ is $O(1)$ at the origin while $\dot a\approx bG$, $\dot b\approx aG$: the unigram solution is learned
  first; the induction circuit escapes the saddle after time $\approx G^{-1}\log(1/\eps)$ — the same
  mechanism as Ch. 2. Check Edelman et al.'s own simplified model and theorem and prefer it if it is
  provable in a few pages; Nichani, Damian, Lee (ICML 2024) and Bietti et al. (NeurIPS 2023) for rigorous
  results — statements.

**Reading.** Figure: training a tiny attention-only transformer (torch) on in-context Markov chains ($k=3$):
loss versus step with plateaus at the uniform, unigram and bigram Bayes losses (horizontal lines computed
exactly or by Monte Carlo).

**Gets wrong.** Thrown away: semantics, real token statistics, MLPs, many heads; data whose only signal is
in-context statistics. Where it breaks: in LLMs the phase change coincides with other changes; function and
task vectors (Todd et al., ICLR 2024; Hendel et al. 2023 **[verify]**); burstiness and Zipf control in-context
versus in-weights learning (Chan et al. 2022); transience (Singh et al. 2023); single-layer transformers on
Markov data get stuck at unigram local minima (Makkuva et al. 2024 **[verify]**). Fine structure: causal
structure (Nichani et al.), associative memories (Bietti et al.), higher-order chains (Edelman et al.;
Rajaraman et al., NeurIPS 2024 **[verify]**).

**Exercises (ideas).** Posterior mean derivation; expected cross-entropy of unigram versus bigram estimators
for $k=2$; trigram circuit; attention-sink smoothing; simulation; research: rigorous dynamics.

### Chapter 15 — The Edge of Stability (key `stability`)

**Phenomenon.** With full-batch gradient descent the sharpness $\lmax(\nabla^2\Loss)$ rises during training
("progressive sharpening") until it reaches $2/\eta$ and then hovers there while the loss keeps decreasing
non-monotonically (Cohen, Kaur, Li, Kolter, Talwalkar, ICLR 2021); large initial steps produce a loss spike and
a drop in curvature, the catapult (Lewkowycz, Bahri, Dyer, Sohl-Dickstein, Gur-Ari 2020, arXiv:2003.02218);
classical theory says gradient descent diverges when $\eta>2/\lmax$ (the prologue).

**Model.** The two-parameter product model $\Loss(u,v)=\tfrac12(uv-y)^2$, $y>0$ (a width-one two-layer linear
network), and its wide version (Lewkowycz et al.). With $r=uv-y$ and $c=u^2-v^2$, gradient descent is the
map $c'=(1-\eta^2r^2)c$, $r'=r\,[1-\eta\lambda(r,c)+\eta^2r(r+y)]$, $\lambda(r,c)=\sqrt{4(r+y)^2+c^2}$ (verify); the
sharpness at the minimizer with imbalance $c$ is $\lambda_0(c)=\sqrt{4y^2+c^2}$.

**Solution.**
- *Theorem (dynamical stability; complete proof).* Quadratics: convergence iff $\eta<2/\lmax$. For the product
  model: if gradient descent converges to a minimizer with $\eta\lambda_0(c_\infty)>2$ then it hits the minimizer
  set exactly in finite time, which happens only for a Lebesgue-null set of initializations (the map is
  polynomial with a not-identically-zero Jacobian determinant). Cite the general statement (Lee et al.;
  Wu, Ma, E, NeurIPS 2018) for smooth losses.
- *Theorem (conservation law and the sharpness of the limit; complete proof).* Gradient flow conserves $c$, so
  its limit has sharpness $\lambda_0(c_0)$; gradient descent multiplies $c$ by $1-\eta^2r_t^2$ each step, so if
  it converges the limit sharpness is $\lambda_0(c_\infty)$ with $|c_\infty|=|c_0|\prod_t|1-\eta^2r_t^2|$ and
  $\eta\lambda_0(c_\infty)\le2$ (generically). Gradient descent can only reach a minimizer flatter than
  $2/\eta$ by oscillating: the catapult is forced.
- *The catapult phase diagram (heuristic ◇ + exact numerics).* Three regimes in $\eta\lambda_0(c_0)$: below 2
  (lazy: converges near the flow's limit); between 2 and about 4 (catapult: loss spike, landing below $2/\eta$);
  above about 4 (divergence). Near the edge, a flip-bifurcation normal form with slow sharpness drift gives
  $\dot A=2\eps A-\beta A^2$, $\dot\eps=-\kappa A$ ($A=r^2$, $\eps=\eta\lambda-2$; $\beta=4\eta^4y^2$,
  $\kappa=2\eta^2(1-\eta^2y^2)$ — re-derive), whose solution yields the **reflection law**
  $\eta\lambda_\infty\approx4-\eta\lambda_0$ to first order (second-order correction from $x e^{-x}$
  symmetry, Lambert $W$). Editor's numerics ($y=1,\eta=0.4$): $\eta\lambda_0=2.08\mapsto1.914$, $2.2\mapsto1.783$,
  $2.4\mapsto1.537$, $2.8\mapsto0.98$, $3.2\mapsto0.80$ (the flattest minimizer $2\eta y$), $4.2$ diverges. Present
  the derivation honestly as heuristic and check it against the exact map.
- *Self-stabilization (Damian, Nichani, Lee, ICLR 2023) and central flows (Cohen et al., ICLR 2025
  **[verify]**): statement and a two-dimensional heuristic.*

**Reading.** Figure 1: $\eta\lambda_\infty$ versus $\eta\lambda_0$ from the exact map, with the reflection-law
line and the three regimes. Figure 2: a catapult trajectory (loss and sharpness versus step). Optional:
progressive sharpening and EoS in a small MLP trained by full-batch GD (torch; sharpness by power iteration).

**Gets wrong.** Thrown away: the product model has no sustained progressive sharpening; full-batch only; square
loss; two parameters. Where it breaks: why sharpness rises in real networks is not explained by the model;
with cross-entropy the sharpness eventually decreases (Cohen et al. 2021); SGD changes the threshold
(Jastrzębski et al., ICLR 2020 "The break-even point"; stochastic edge of stability **[verify]**); Adam has an
adaptive edge where the preconditioned sharpness hovers near $38/\eta$ for $\beta_1=0.9$ (Cohen et al. 2022
**[verify]**); linear stability says nothing about the long EoS transient. Fine structure: self-stabilization,
central flows, minimalist examples with proofs (Zhu et al., ICLR 2023; Ahn et al., NeurIPS 2023 "Learning
threshold neurons via the edge of stability"; Arora, Li, Panigrahi, ICML 2022; Kalra, He, Barkeshli 2023
**[verify]**), large step sizes in logistic regression (Wu et al. 2024; cross-ref Ch. 4).

**Exercises (ideas).** GD on a quadratic; derive the $(r,c)$ map; compute $c_{\rm edge}$; the $y=0$ model;
three-factor product $uvw$; simulation: the catapult curve; research: progressive sharpening.

### Chapter 16 — Scaling Laws (key `scaling`)

**Phenomenon.** Test loss falls as a power law in parameters, data and compute over many orders of magnitude
(Hestness et al. 2017, arXiv; Kaplan et al. 2020, arXiv:2001.08361; Hoffmann et al., NeurIPS 2022 —
$L(N,D)=E+A/N^{\alpha}+B/D^{\beta}$ with fitted $\alpha\approx0.34$, $\beta\approx0.28$ **[verify]**; replication
issues: Besiroglu et al. 2024 **[verify]**); data-constrained scaling (Muennighoff et al., NeurIPS 2023); broken
scaling laws (Caballero et al., ICLR 2023); emergence versus smooth loss.

**Models.** (A) Power-law linear regression: features with covariance eigenvalues $\lambda_j\asymp j^{-a}$,
target energy $\lambda_jw_{\star,j}^2\asymp j^{-(1+b)}$; a model of size $N$ uses the first $N$ features; $D$
samples; Gaussian sequence model idealization: per-feature estimation noise $\sigma^2/D$ in prediction
units. (B) The quanta model (Michaud, Liu, Girit, Tegmark, NeurIPS 2023) and Hutter's learning-curve model
(Hutter 2021, arXiv:2102.04074): skills with Zipf frequencies $p_k\propto k^{-(1+\alpha)}$.

**Solution.**
- *Theorem (power laws from spectral tails; complete proof).* Approximation-limited loss
  $\sum_{j>N}\lambda_jw_{\star,j}^2\asymp N^{-b}$; with $D$ samples the projection estimator has loss
  $\asymp N^{-b}+N\sigma^2/D$, optimal $N^\star\asymp D^{1/(1+b)}$, loss $\asymp D^{-b/(1+b)}$ (re-derive in the
  chapter's parametrization; state exactly what is idealized).
- *Theorem (compute-optimal allocation; complete proof).* Minimizing $E+AN^{-\alpha}+BD^{-\beta}$ subject to
  $6ND=C$: $N^\star\propto C^{\beta/(\alpha+\beta)}$, $D^\star\propto C^{\alpha/(\alpha+\beta)}$, with explicit
  constants; evaluate with Chinchilla's fitted values and with the replication's.
- *Theorem (Zipf learning curves; complete proof with exact constant).* A learner that has memorized every item
  seen in $n$ samples errs with probability $\sum_kp_k(1-p_k)^n\sim\frac{\Gamma(\alpha/(1+\alpha))}{(1+\alpha)\,
  \zeta(1+\alpha)^{1/(1+\alpha)}}\,n^{-\alpha/(1+\alpha)}$ (editor checked numerically for
  $\alpha=0.5,1,2$); a model that can hold $N$ items has error $\sum_{k>N}p_k\sim N^{-\alpha}/(\alpha\zeta(1+\alpha))$.
  Data exponent $\alpha/(1+\alpha)$ versus parameter exponent $\alpha$.
- *Random-feature models with power-law spectra (statement + sketch via deterministic equivalents; cross-ref
  Ch. 1 and `\cref{thm:toolkit:mp}`)*: Bordelon, Canatar, Pehlevan (ICML 2020); Maloney, Roberts, Sully (2022,
  arXiv:2210.16859); Bahri, Dyer, Kaplan, Lee, Sharma (PNAS 2024); Paquette, Paquette, Xiao, Pennington
  (NeurIPS 2024, "4+3 phases"); Lin, Wu, Kakade, Bartlett, Lee (NeurIPS 2024) **[verify all]**.

**Reading.** Figure: learning curves of ridge regression on synthetic power-law data versus the predicted
exponents (log–log). Figure: Zipf memorization curve, exact sum versus asymptotic constant.

**Gets wrong.** Thrown away: feature learning (in (A) the exponent is the data spectrum's, an input not an
output), optimization, architecture, irreducible loss, data repetition. Where it breaks: real exponents are not
derived from first principles; downstream metrics jump while loss is smooth; data repetition and quality;
fitting choices change conclusions (Besiroglu et al.); feature learning can change exponents (Bordelon,
Atanasov, Pehlevan 2024/2025 **[verify]**). Fine structure: 4+3 phases; dynamical models; associative memories
(Cabannes, Dohmatob, Bietti, ICLR 2024 **[verify]**); data-manifold dimension (Sharma and Kaplan, JMLR 2022).

**Exercises (ideas).** Tail sums; Chinchilla allocation; Hutter's constant; combined quanta law with finite $N$
and $D$; simulation: power-law ridge; research: predicting exponents from data.

### Epilogue — Fine Structure (key `epilogue`) — written by the editor

Recurring mechanisms (saddle escape; rotational invariance; Fourier analysis on groups; margins as
attractors; infinite-width limits) in a table; what the models get wrong together (Gaussian or uniform data,
one mechanism at a time, asymptotic regimes, no architecture); the missing hydrogen atoms (adaptive optimizers,
normalization layers, data curricula and mixtures, reinforcement learning from feedback, chain-of-thought,
long-context retrieval); a recipe for building a new solvable model.

### Appendix A — A Toolkit (key `toolkit`)

Sections and fixed labels as in the table above. Complete proofs where short (Stein, Hermite facts and
correlation lemma, dual activations, Price, inverse-Wishart mean, Doob, Grönwall, the ODE method, Fourier on
$\Z_p$ and the cube, Hoeffding); a complete proof of the Marchenko–Pastur law for Gaussian matrices via the
Stieltjes transform (Schur complement + concentration of quadratic forms + variance bound); statements with
references for Bernstein, Freedman, Gaussian concentration, Bonami/hypercontractivity, and the addition theorem
and Funk–Hecke formula (with dimension count $N_{d,k}=\binom{d+k-1}{k}-\binom{d+k-3}{k-2}$). 15–22 pages. Same
voice; no model card or "gets wrong" sections; a short opening paragraph says how chapters use each section.
