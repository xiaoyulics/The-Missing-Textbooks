# The Replica Rosetta Stone — Plan

*The Replica Rosetta Stone: Physical Derivations and Rigorous Proofs, Side by Side.* Volume 7 of
*The Missing Textbooks*. Written by Claude; conceived and edited by Xiaoyu Li. English, about 350
pages. Plan written 2026-10-24 by the coordinating author from the four research digests in
`build/research/` (01 replicas and interpolation; 02 cavity, AMP and landscapes; 03 comparison and
regression; 04 dynamics, positioning and history). Where this plan and a digest disagree, the digest
section named in your chapter's specification is the source; tell the coordinator.

---

## Thesis

Many exact statements of high-dimensional statistics and machine-learning theory were first derived
by statistical physicists: the capacity of the perceptron, the threshold at which PCA sees a planted
spike, the error of the Bayes-optimal estimator in a generalized linear model, the learning curve of
ridge regression, the energy that gradient flow reaches in a rugged landscape. The derivations use
replicas, cavities, TAP equations, Kac–Rice counts and dynamical mean-field theory. Many of the same
statements were later proved by mathematicians, with interpolation, Gaussian conditioning,
comparison inequalities and large deviations, and the proofs rarely look like the derivations. The
two literatures describe the same objects in different languages and rarely read each other. This
book is a parallel text. It takes one prediction at a time, derives it as a physicist would, sets
the derivation side by side with the rigorous route step by step (the *stone*), proves the central
result completely in general or in a special case that keeps the idea, measures it in a short
script, and records on a *status board* what is proved, under which hypotheses, what is conjectured
and on what evidence, and what is known to be false. All examples come from inference and learning.

## Positioning (verified 2026-10-24; digest 04 §4)

Two literatures grew up around the mean-field theory of disordered systems. Physicists derived
exact predictions with replicas, cavities, TAP equations, Kac–Rice counting and dynamical mean-field
theory; mathematicians proved many of them with interpolation, approximate message passing,
Gaussian comparison and the probabilistic analysis of random dynamics. Each literature has excellent
books: Mézard–Parisi–Virasoro, Nishimori, Engel–Van den Broeck and Mézard–Montanari on one side;
Talagrand, Bovier and Panchenko on the other. A few texts already cross the border in one direction
or for a few models: Bandeira, Perry and Wein's notes on computational gaps (2018); Montanari and
Sen's tutorial (*A Friendly Tutorial on Mean-Field Spin Glass Techniques for Non-Physicists*, Found.
Trends Mach. Learn. 2024), which teaches replica and cavity computations alongside interpolation
bounds on two models; and Krzakala and Zdeborová's lecture notes (draft of June 2024), which follow
several replica calculations with their rigorous counterparts. What is missing is a systematic
parallel text for inference and learning that includes Gaussian comparison, landscapes and dynamics,
with a status board for every prediction.

**The book must not claim** (digest 04 §4.5): that no one has explained the physicists' methods to
mathematicians or the reverse; that it is the first text to put replica derivations next to proofs
(claim a *systematic* parallel treatment with status boards, "to our knowledge"); that DMFT for
gradient methods "is rigorous" without the fixed-horizon and smoothness qualifications; that
Langevin dynamics provably reaches the threshold energy in general; that aging or the dynamical
transition is a theorem; that n ≍ d^{k−1} is a lower bound for gradient methods; that Goldt et al.
2019 proved the Saad–Solla ODEs without noting the gaps fixed by Veiga et al. 2022; that a Les
Houches 2022 volume exists in print; that Montanari–Sen is titled "A short tutorial".

## Design (`rosetta.sty`)

Physics writes in **carnelian** (`physics`), proof in **lapis** (`proof`), and the structure of the
book is cut in **granite** (`stone`). Under each chapter title run three registers (physics,
numerics, proof). Each chapter opening carries at its foot a **status stone** (`\stonefoot{...}`)
for the chapter's main prediction: `proved`, `partial` (proved under extra hypotheses or in part of
the range), `conjectured`, `false`, `open`. The **stone** environment sets the derivation and the
proof side by side across the full width. The **status board** lists every prediction with a badge:
`\proved`, `\provedunder`, `\provedpreprint`, `\conjectured`, `\heuristicstatus`, `\knownfalse`,
`\openq`. The title page carries Gardner's capacity formula in three registers.

### The status words

- **proved**: a theorem in a refereed publication (or a classical result), with its source.
- **proved under**: proved with hypotheses beyond the physical prediction (a prior, a range of
  parameters, Gaussian data, smoothness, a fixed number of iterations); the board names them.
- **preprint** (`\provedpreprint`): proved only in an unrefereed preprint. Many 2025–2026
  preprints settle long-open questions (the de Almeida–Thouless line with a field, full replica
  symmetry breaking at zero field, the overlap law of SK, the Ising perceptron capacity by a
  computer-assisted proof). Report them as their authors state them, with the arXiv number and
  version, and say when the authors state that the proof was written with AI assistance (digest 01
  §0.2). Never upgrade them to "proved".
- **conjectured**: believed, with evidence (physics, numerics, partial results) named.
- **heuristic**: an argument without a claim of exactness.
- **known false**: a counterexample or a regime where the prediction fails, named.
- **open**: nobody knows, and nobody has a strong conjecture; or the question is posed in a cited
  source.

---

## Global notation and conventions

The two literatures disagree on almost every normalization. These are the book's; every chapter
uses them and translates its sources in "Translating the literature". The digests' convention
sections (01 "Conventions", 02 §0, 03 §0.5, 04 §1.2) have the full translation tables.

**Sizes.** $N$ is the number of spins or variables in a spin-glass or matrix model (Parts I–II, and
the spin glasses of Part IV); $d$ is the dimension of the unknown in a learning problem, $n$ the
number of samples or patterns, and $\alpha=n/d$ the load (Gardner's $\alpha$; Donoho's $\delta$;
$\gamma=d/n=1/\alpha$ only when quoting Hastie et al. or Dobriban–Wager). In Chapter 12, $N$ is the
number of random features and $\psi=N/d$.

**Gaussian notation.** $Z,z\sim\Normal(0,1)$; `\Dz` is the standard Gaussian measure in physics
notation, $\int f(z)\Dz=\E f(Z)$; $\phi$ and $\Phi$ the standard normal density and distribution
function; $H(x)=\Prob(Z>x)=1-\Phi(x)$ (the Gardner function).

**Free entropy.** The book works with the **free entropy** $\phi_N=\frac1N\E\log Z_N$ and its limit
$\phi$ (mathematicians' "pressure"; Zdeborová–Krzakala's "free entropy"). The physicist's free energy
is $f=-\phi/\beta$; when a chapter quotes it, it says so. Guerra's bound is an *upper* bound on
$\phi$ (a lower bound on $f$). In inference, $Z_N$ is the evidence (marginal likelihood) and the free
entropy is, up to a known term, minus the mutual information per variable.

**Gibbs measures and overlaps.** $\gibbs{\cdot}$ is the Gibbs (posterior) average, $\E$ the average
over the disorder (the data). Replicas $\sigma^1,\sigma^2,\dots$ are independent samples from the same
Gibbs measure. The overlap of two replicas is the random variable $R_{12}=\frac1N\ip{\sigma^1}{\sigma^2}$;
$q$ is its limit (or its typical value) in a replica-symmetric phase; $m=\frac1N\ip{\sigma}{x^*}$ is
the overlap with the truth.

**SK.** $H_N(\sigma)=\frac{\beta}{\sqrt N}\sum_{i<j}g_{ij}\sigma_i\sigma_j+h\sum_i\sigma_i$ with
$g_{ij}$ i.i.d. $\Normal(0,1)$, Gibbs weight $e^{H_N(\sigma)}$ on $\{\pm1\}^N$, covariance
$\E H_N(\sigma)H_N(\tau)=N\xi(R_{\sigma\tau})+O(1)$ with $\xi(q)=\beta^2q^2/2$; $\beta_c=1$. The
replica-symmetric solution: $q=\E\tanh^2(\beta\sqrt q\,Z+h)$ and
$\phi_{\mathrm{RS}}=\log2+\E\log\cosh(\beta\sqrt q\,Z+h)+\frac{\beta^2}4(1-q)^2$; the
de Almeida–Thouless condition $\beta^2\E\,\mathrm{sech}^4(\beta\sqrt q\,Z+h)\le1$. (Panchenko and
Auffinger–Chen use $\xi=\beta^2q^2$; translate.)

**REM.** $Z_N=\sum_{\sigma\in\{\pm1\}^N}e^{\beta\sqrt N X_\sigma}$ with $X_\sigma$ i.i.d. $\Normal(0,1)$;
$\beta_c=\sqrt{2\log2}$; $\phi(\beta)=\log2+\beta^2/2$ for $\beta\le\beta_c$ and $\beta\sqrt{2\log2}$
for $\beta\ge\beta_c$. (Derrida's energies have variance $NJ^2/2$.)

**Spiked Wigner.** $A=\frac\lambda N xx^\top+W$ with $W\sim\mathrm{GOE}(N)$ (off-diagonal variance
$1/N$, diagonal $2/N$, semicircle on $[-2,2]$), $x_i$ i.i.d. from a prior $P_0$ with $\E x_i^2=1$.
BBP at $\lambda=1$: for $\lambda>1$ the top eigenvalue tends to $\lambda+1/\lambda$ and the squared
overlap of the top eigenvector with $x/\norm x$ to $1-1/\lambda^2$. The scalar channel is
$Y=\sqrt\gamma X+Z$ with $\gamma=\lambda^2q$; the replica-symmetric fixed point is
$q=1-\mathrm{mmse}(\lambda^2q)$. Lelarge–Miolane and Deshpande–Abbe–Montanari write
$Y=\sqrt{\mathrm{snr}/N}\,XX^\top+Z$ with $\mathrm{snr}=\lambda^2$ (so their overlap is $1-1/\mathrm{snr}$).

**Perceptrons.** Patterns $g_\mu\sim\Normal(0,I_d)$, $\mu\le n=\alpha d$; weights $w\in\sqrt d\,S^{d-1}$
(spherical) or $w\in\{\pm1\}^d$ (Ising); the constraints $\ip{g_\mu}{w}/\sqrt d\ge\kappa$ (the margin
$\kappa$). Gardner's replica-symmetric capacity $\alpha_c(\kappa)=1/\E[(Z+\kappa)_+^2]
=1/[(1+\kappa^2)\Phi(\kappa)+\kappa\phi(\kappa)]$, $\alpha_c(0)=2$.

**GLMs and teacher–student.** $y_\mu=\varphi_{\mathrm{out}}(\ip{x_\mu}{w^*}/\sqrt d,\,A_\mu)$ with
$x_\mu\sim\Normal(0,I_d)$, $w^*\sim P_0^{\otimes d}$, $n=\alpha d$. Ising teacher (sign activation,
Rademacher prior): information-theoretic threshold $\alpha_{\mathrm{IT}}=1.2447$ (digest 01 §0.4:
recomputed from Barbier et al.'s formula, which the paper prints as 1.249), AMP threshold
$\alpha_{\mathrm{AMP}}=1.4926$. Chapter 7 must recompute both and state the discrepancy.

**Regression and classification (Part III; digest 03 §0.5).** Inputs $x_i\sim\Normal(0,\Sigma)$
($\Sigma=I_d$ unless stated), $y=\ip{x}{\beta^*}+\sigma\varepsilon$, $\E\varepsilon^2=1$;
$r^2=\ip{\beta^*}{\Sigma\beta^*}$. Ridge, per-sample normalized:
$\hat\beta=\argmin\frac1{2n}\norm{y-X\beta}^2+\frac\lambda2\norm\beta^2$. Excess risk
$R=\E_x(\ip{x}{\hat\beta}-\ip{x}{\beta^*})^2$; test error $R+\sigma^2$. Physics scaling
$w=\sqrt d\,\beta$. The LASSO and M-estimation chapter may use the scaling of Bayati–Montanari
($A=X/\sqrt n$, $x_0=\sqrt n\beta^*$, objective $\frac12\norm{y-Ax}^2+\lambda_{\mathrm{BM}}\norm x_1$)
with the translation $R=\mathrm{MSE}/\alpha$ stated once. Classification: $y\in\{\pm1\}$; $\kappa$ is
only the margin; test error $\Prob(y\ne\sgn\ip{x}{\hat\beta})$. Kernels: eigenvalues $\eta_k$,
effective ridge $\lambda_{\mathrm{eff}}$ (never $\kappa$).

**Spherical $p$-spin and landscapes (Part IV).** Auffinger–Ben Arous–Černý normalization:
$H_N$ on $S^{N-1}(\sqrt N)$ with $\E H_N(\sigma)H_N(\sigma')=N\,R(\sigma,\sigma')^p$; energies per
spin; ground state $-E_0(p)$ ($E_0(3)=1.65700$), threshold $-E_\infty(p)$,
$E_\infty=2\sqrt{(p-1)/p}$ ($1.63299$ for $p=3$). Physics (Crisanti–Sommers, Cugliandolo–Kurchan)
energies are those divided by $\sqrt2$ (CK threshold $-1.154701$, ground state $-1.171675$ for
$p=3$). The dynamics chapter states its dictionary once.

**Dynamics.** In Chapter 14, $C(t,s)$ and $R(t,s)$ are the correlation and response functions (the
overlap $R_{12}$ does not appear there). In Chapter 15, $\eta$ is the learning rate and time is the
number of samples divided by $d$.

**Volume macros.** `\Prob \iid \Normal \Unif \diag \Dz \gibbs{} \Ed{}`; class: `\E \R \Var \tr \sgn`
and the rest (STYLE.md §7).

---

## Structure

| # | File | Title — Subtitle | Dictionary (`\rstchapter` arg. 3) | Stone |
|---|---|---|---|---|
| P | `00-prologue` | Prologue — The number two | — | `proved` |
| | | **Part I. Two Languages** | | `proved` |
| 1 | `01-freeentropy` | Free Entropy — Posteriors, saddle points and self-averaging | saddle points\dict large deviations | `proved` |
| 2 | `02-rem` | The Random Energy Model — Replicas and the second moment | replicas\dict the second moment | `proved` |
| 3 | `03-sk` | The Sherrington–Kirkpatrick Model — Replica symmetry and its breaking | broken replicas\dict interpolation | `proved` |
| | | **Part II. Inference on the Nishimori Line** | | `partial` |
| 4 | `04-nishimori` | Bayes-Optimal Inference — Why replica symmetry holds | the Nishimori line\dict overlap concentration | `proved` |
| 5 | `05-spiked` | Spiked Matrices — PCA, the BBP transition and the information threshold | zero-temperature replicas\dict resolvents | `proved` |
| 6 | `06-amp` | Message Passing — The cavity, TAP and AMP | the cavity\dict Gaussian conditioning | `proved` |
| 7 | `07-glm` | Generalized Linear Models — The teacher–student perceptron and the hard phase | the replica formula\dict adaptive interpolation | `partial` |
| | | **Part III. Comparison and Convexity** | | `partial` |
| 8 | `08-gordon` | Gaussian Comparison — Gordon's theorem and the capacity of the perceptron | Gardner's volume\dict Gordon's min-max | `partial` |
| 9 | `09-ridge` | Ridge Regression — Three derivations of double descent | replicas\dict random matrices and the CGMT | `proved` |
| 10 | `10-lasso` | The LASSO and Robust Regression — Four routes to one fixed point | the replica fixed point\dict AMP and the CGMT | `proved` |
| 11 | `11-classification` | Logistic Regression and Max-Margin Classifiers — The capacity problem with a teacher | Gardner with a teacher\dict the CGMT | `proved` |
| 12 | `12-kernels` | Kernels and Random Features — Learning curves and Gaussian equivalence | the Gaussian approximation\dict universality | `partial` |
| | | **Part IV. Landscapes and Dynamics** | | `partial` |
| 13 | `13-kacrice` | Counting Critical Points — Kac–Rice and the spiked tensor model | TAP complexity\dict Kac–Rice and the second moment | `partial` |
| 14 | `14-dmft` | Dynamical Mean-Field Theory — Gradient flow in high dimension | the generating functional\dict Gaussian conditioning | `partial` |
| 15 | `15-sgd` | Online SGD — Teacher–student networks, plateaus and the information exponent | order-parameter equations\dict concentration of summary statistics | `proved` |
| E | `16-epilogue` | Epilogue — The stone, read across | — | `open` |
| A | `17-toolkit` | A Toolkit — Gaussian calculus, concentration and random matrices | — | — |

Chapter keys (for `\cref{ch:key}`): `freeentropy`, `rem`, `sk`, `nishimori`, `spiked`, `amp`,
`glm`, `gordon`, `ridge`, `lasso`, `classification`, `kernels`, `kacrice`, `dmft`, `sgd`, `toolkit`.

---

## Fixed labels (cite these across chapters; define them in your chapter if they are yours)

| Label | Content |
|---|---|
| `thm:toolkit:gaussianibp` | Gaussian integration by parts (Stein's lemma), multivariate |
| `lem:toolkit:interpolation` | Gaussian interpolation: $\frac{d}{dt}\E F(\sqrt t X+\sqrt{1-t}Y)$ in terms of the covariance difference |
| `thm:toolkit:poincare` | Gaussian Poincaré inequality |
| `thm:toolkit:lipschitz` | concentration of Lipschitz functions of Gaussian vectors (Tsirelson–Ibragimov–Sudakov) |
| `thm:toolkit:laplace` | Laplace's method and Varadhan's lemma |
| `def:toolkit:stieltjes` | Stieltjes transform and resolvent |
| `thm:toolkit:semicircle` | semicircle law for GOE (statement) and the norm $\norm W\to2$ |
| `thm:toolkit:mp` | Marchenko–Pastur law (statement) |
| `lem:toolkit:conditioning` | conditional distribution of a Gaussian matrix given linear constraints (the AMP conditioning lemma) |
| `thm:toolkit:slepian` | Slepian and Sudakov–Fernique comparison (statements; Gordon is Chapter 8) |
| `thm:toolkit:kacrice` | the Kac–Rice formula (statement, Adler–Taylor conditions) |
| `sec:toolkit:numerics` | Gaussian quadrature, damped fixed-point iteration, intervals (as the kit implements them) |
| `lem:freeentropy:nishimori` | the Nishimori identity (Bayes form) |
| `thm:freeentropy:immse` | the I-MMSE relation for the scalar Gaussian channel |
| `thm:freeentropy:concentration` | concentration of the free entropy of Gaussian disordered systems |
| `thm:rem:freeentropy` | the free entropy of the REM, with the freezing transition |
| `thm:sk:guerratoninelli` | existence of the limit of the SK free entropy |
| `thm:sk:rsbound` | Guerra's replica-symmetric bound (proved in Ch. 3) |
| `thm:sk:parisi` | the Parisi formula (Guerra; Talagrand), statement |
| `thm:nishimori:rsformula` | the replica-symmetric formula for spiked Wigner (Lelarge–Miolane), statement and the proof of Ch. 4 |
| `thm:spiked:bbp` | the BBP transition for spiked Wigner: eigenvalue and eigenvector overlap |
| `thm:amp:se` | AMP state evolution for a fixed number of iterations |
| `thm:glm:rsformula` | the replica formula for GLMs (Barbier et al. 2019), statement |
| `thm:gordon:gordon` | Gordon's comparison inequality |
| `thm:gordon:cgmt` | the convex Gaussian min-max theorem |
| `thm:gordon:capacity` | the capacity of the spherical perceptron for $\kappa\ge0$ |
| `thm:ridge:risk` | the proportional-limit ridge risk |
| `thm:lasso:risk` | the LASSO risk (fixed point of Bayati–Montanari) |
| `thm:classification:mle` | the existence threshold of the logistic MLE (Candès–Sur) |
| `thm:kacrice:annealed` | the annealed complexity of the pure spherical $p$-spin (Auffinger–Ben Arous–Černý) |
| `thm:dmft:fixedhorizon` | DMFT on fixed time horizons (statement) |
| `thm:sgd:ode` | the ODE limit of online SGD on fixed horizons |

Other labels carry your chapter key (STYLE.md §7).

---

## Shared citation keys

Use these keys; the BibTeX for each is in the digest named (all four digests now use them). If a
work you need is not in a digest, verify it (STYLE.md §7) and choose a key of the same form.

| Key | Work | Digest |
|---|---|---|
| `Cover1965` | Cover, geometrical and statistical properties of systems of linear inequalities | 01, 03 |
| `Gardner1988`, `GardnerDerrida1988` | the space of interactions; optimal storage properties | 01, 03, 04 |
| `GardnerDerrida1989a`, `GardnerDerrida1989b` | REM partition function distribution; three unfinished works | 01, 04 |
| `KrauthMezard1989` | storage capacity of the Ising perceptron | 01 |
| `Derrida1980`, `Derrida1981` | the random energy model | 01, 04 |
| `EdwardsAnderson1975`, `SherringtonKirkpatrick1975`, `ThoulessAndersonPalmer1977`, `deAlmeidaThouless1978` | the founding papers | 01, 02, 04 |
| `Parisi1979a`, `Parisi1979b`, `Parisi1980a` | Parisi's papers (Phys. Lett. A; PRL; J. Phys. A) | 01, 04 |
| `MezardEtAl1984a` | Mézard, Parisi, Sourlas, Toulouse, Virasoro, ultrametricity (PRL) | 01, 04 |
| `MezardParisiVirasoro1986`, `MezardParisiVirasoro1987` | the cavity paper (EPL); *Spin Glass Theory and Beyond* | 02, 04 |
| `GuerraToninelli2002`, `Guerra2003`, `AizenmanSimsStarr2003`, `Talagrand2006`, `Panchenko2013a` (Annals, ultrametricity), `Panchenko2013b` (book), `Panchenko2014` | the rigorous SK theory | 01, 04 |
| `Talagrand2003`, `Talagrand2011a`, `Talagrand2011b` | Talagrand's books (Vols. I and II) | 01, 03, 04 |
| `AizenmanLebowitzRuelle1987`, `Toninelli2002` | high temperature; RSB below AT | 01 |
| `Lopatto2026a`, `Lopatto2026b`, `Lopatto2026c`, `KusuokaNakajima2026a`, `KusuokaNakajima2026b`, `AronowLopatto2026a`, `AronowLopatto2026b`, `Zhou2026`, `Shmalo2026` | 2026 results (check each entry's arXiv number and status) | 01 |
| `Nishimori1981`, `Nishimori2001` | the Nishimori line; the book | 01, 04 |
| `GuoShamaiVerdu2005` | I-MMSE | 01, 02 |
| `LelargeMiolane2019`, `BarbierMacris2019`, `DeshpandeAbbeMontanari2016`, `LesieurKrzakalaZdeborova2017`, `BarbierEtAl2019` (GLMs, PNAS) | Bayes-optimal inference | 01, 02 |
| `BaikBenArousPeche2005`, `FeralPeche2007`, `BenaychGeorgesNadakuditi2011` | BBP | 01, 02 |
| `DonohoMalekiMontanari2009`, `Bolthausen2014`, `BayatiMontanari2011`, `JavanmardMontanari2013`, `MontanariVenkataramanan2021` | AMP and state evolution | 02 |
| `ChenPanchenko2018`, `FanMeiMontanari2021`, `CelentanoFanMei2023` | the TAP free energy | 02 |
| `Slepian1962`, `Gordon1985`, `Gordon1988`, `Stojnic2013a` (another look at the Gardner problem), `Stojnic2013b` (negative spherical perceptron), `ThrampoulidisOymakHassibi2015`, `ThrampoulidisAbbasiHassibi2018`, `AmelunxenEtAl2014` | comparison and the CGMT | 03 |
| `MontanariZhongZhou2024`, `DingSun2019`, `Huang2024` | perceptron capacity, recent | 01, 03 |
| `HastieEtAl2022`, `DobribanWager2018`, `KroghHertz1992`, `BayatiMontanari2012`, `SurCandes2019`, `CandesSur2020`, `MontanariEtAl2025` (max-margin), `MeiMontanari2022`, `BordelonEtAl2020`, `CanatarEtAl2021`, `HuLu2023`, `MontanariSaeed2022` | Part III | 03 |
| `AuffingerBenArousCerny2013`, `Subag2017a`, `BenArousMeiMontanariNica2019`, `RosEtAl2019` | Kac–Rice | 02, 04 |
| `CugliandoloKurchan1993`, `CrisantiHornerSommers1993`, `BenArousDemboGuionnet2001`, `BenArousDemboGuionnet2006`, `Sellke2024`, `CelentanoChengMontanari2021`, `GerbelotEtAl2024` | DMFT | 04 |
| `SaadSolla1995a`, `SaadSolla1995b`, `GoldtEtAl2020a`, `VeigaEtAl2022`, `BenArousGheissariJagannath2021`, `BenArousGheissariJagannath2024` | online SGD | 04 |
| `MezardMontanari2009`, `EngelVandenBroeck2001`, `ZdeborovaKrzakala2016`, `MontanariSen2024`, `KrzakalaZdeborova2024`, `BandeiraPerryWein2018` | books and bridges | 01, 02, 04 |

---

## Chapter specifications

Every chapter follows STYLE.md §2. "Read" lists the digest sections to read before writing; the
digests also hold verified BibTeX, numbers and scripts (in `build/research/scratch/`, which you may
read and adapt; copy what you need into your own script under `figures/scripts/`).

### Prologue — The number two (key: prologue; special chapter; written by the coordinator)

Four derivations of one number: a random set of $n$ points in general position in $\R^d$ can be
split by a hyperplane through the origin in every way, with high probability, if $n<2d$, and in
almost no way if $n>2d$. Cover's counting (1965) gives it exactly at every finite size; Gardner's
replica computation (1988) gives $\alpha_c(\kappa)$ for every margin; Gordon's comparison gives it
rigorously for $\kappa\ge0$ (Stojnic 2013; Shcherbina–Tirozzi 2003; Talagrand Vol. II, Ch. 8); the
kinematic formula gives it as the statistical dimension $n/2$ of the orthant (Amelunxen–Lotz–McCoy–
Tropp 2014; digest 03 §0.3 item 7). Then the book's plan, the stone and the status board, and how
to read. Figures: the probability of separability at $d=25$–$200$ against $\alpha$ (Cover's formula
and LP simulations); $\alpha_c(\kappa)$ with the region $\kappa<0$ marked (upper bound only;
digest 03 §1.5). Read: digest 01 §5, digest 03 §§1.4–1.5.

### Part I. Two Languages

*Blurb:* A physicist computes a free energy by replicating the system and finding a saddle point; a
probabilist bounds it by interpolation and concentration. On the simplest models the two
computations can be set side by side line by line, and they agree, except where the physicist's
first guess, replica symmetry, fails.

#### Chapter 1. Free Entropy — Posteriors, saddle points and self-averaging (key: freeentropy)

Read: digest 01 §1 (free energies, self-averaging, the replica trick), §4.1–4.2 (the Nishimori
identity), digest 02 §2.3 (the scalar channel numbers).

*The question.* A posterior over $N$ variables is a Gibbs measure, its normalizer is a partition
function, and the log-evidence per variable is a free entropy. When does it concentrate, what is its
limit, and how do the two languages compute it?
*The physics.* The Curie–Weiss model by Hubbard–Stratonovich and a saddle point (the self-consistent
equation $m=\tanh(\beta m+h)$); quenched versus annealed averages; the replica trick in its simplest
form ($\E Z^n$ at integer $n$ and the continuation $n\to0$), with its pitfalls (Carleman's condition
fails for Gaussian disorder: $\E Z^n\ge e^{cNn^2}$; Tanaka 2007's counterexamples).
*The stone.* Hubbard–Stratonovich ↔ Laplace's method / Varadhan; saddle point ↔ large deviations;
"self-averaging" ↔ Gaussian concentration (Poincaré, Tsirelson–Ibragimov–Sudakov); quenched
average ↔ Jensen's annealed bound; replica trick ↔ (no counterpart: the rigorous route never
continues in $n$).
*The proof.* Complete proofs of: the Curie–Weiss free entropy (Laplace); concentration of the free
entropy of a Gaussian disordered system (`thm:freeentropy:concentration`, with the explicit
constant $2e^{-Nu^2/\beta^2}$-type tail for SK, or a variance bound via Poincaré); the scalar Gaussian
channel: the free entropy, MMSE and the I-MMSE relation $\frac{d}{d\gamma}I=\frac12\mathrm{mmse}$
(`thm:freeentropy:immse`, Guo–Shamai–Verdú, by Gaussian integration by parts); the Nishimori
identity (`lem:freeentropy:nishimori`) and its first consequence $\E\gibbs{x}x^*=\E\gibbs{x}^2$.
*Measuring it.* Exact enumeration for small $N$ (SK or REM, $N\le16$): the spread of
$\frac1N\log Z_N$ against the concentration bound; quenched versus annealed; the scalar channel's
MMSE for Rademacher and Gaussian priors against Monte Carlo (digest 01 `r1_concentration_replica.py`,
`r1_nishimori_check.py`).
*Status board.* Self-averaging [proved]; existence of the limit for SK [proved, Guerra–Toninelli,
Ch. 3]; the replica trick [heuristic; not justifiable unconditionally]; integer-moment asymptotics
[proved]; Nishimori identity [proved].
*Where the translation fails.* The replica trick is not a theorem even at finite $N$; the
exchange of $n\to0$ and $N\to\infty$; annealed computations that are wrong at low temperature.

#### Chapter 2. The Random Energy Model — Replicas and the second moment (key: rem)

Read: digest 01 §2 (and §1.3).

*The question.* Derrida's REM as the random code ensemble: $2^N$ independent energies; decoding a
random code is finding the lowest. Its free entropy and the freezing transition at $\beta_c$.
*The physics.* The annealed computation; the replica computation with the 1RSB ansatz (Parisi
parameter $m=\beta_c/\beta$, $P(q)=m\delta_0+(1-m)\delta_1$); why replica symmetry gives a negative
entropy below $T_c$.
*The stone.* $\E Z^n$ ↔ first and second moments; the 1RSB saddle ↔ the extreme-value computation;
$m=\beta_c/\beta$ ↔ the Poisson–Dirichlet weights $PD(m,0)$; the exchange of limits ↔ the exact
$\varphi(n)=\lim N^{-1}\log\E Z_N^n$, which is non-analytic at $n=m$ (digest 01 §2.3: the book's own
exact computation; check and present it).
*The proof.* Complete: the free entropy of the REM at every $\beta$ (`thm:rem:freeentropy`),
by the first and second moments plus concentration; the exact moment function $\varphi(n)$ and the
failure of the naive continuation. Stated: the Poisson–Dirichlet limit of the Gibbs weights
(Ruelle 1987; Bovier 2006), fluctuations (Bovier–Kurkova–Löwe 2002), the GREM.
*Measuring it.* Simulate the REM at $N\le20$: the free entropy against the formula (with the
$\log N/N$ corrections), the participation ratio against $1-m$, the PD weights via GEM sampling.
*Status board.* Free entropy and freezing [proved]; 1RSB description of the Gibbs weights
[proved]; fluctuations [proved]; RS solution below $T_c$ [known false].

#### Chapter 3. The Sherrington–Kirkpatrick Model — Replica symmetry and its breaking (key: sk)

Read: digest 01 §3 (all), §0.2 (the 2026 preprints); digest 04 §5 (history).

*The question.* SK as the posterior of $\Z_2$ synchronization when there is no signal (Chapter 4
makes this precise). Its free entropy; replica symmetry, the de Almeida–Thouless line, the Parisi
formula.
*The physics.* The replica computation: replica symmetry, its instability (AT), Parisi's
hierarchical ansatz and the Parisi formula; ultrametricity; the zero-temperature ground-state
energy $-0.7633$ (with this book's normalization; check).
*The stone.* Replicas and the RS saddle ↔ Guerra's interpolating Hamiltonian; the sign of
$\E\gibbs{(R_{12}-q)^2}$ ↔ the positivity of the remainder; Parisi's hierarchy ↔ Ruelle cascades in
Guerra's broken-RSB bound; the matching lower bound ↔ Talagrand's two-replica bound and
Aizenman–Sims–Starr; ultrametricity ↔ Ghirlanda–Guerra identities and Panchenko's theorem.
*The proof.* Complete: Guerra–Toninelli superadditivity and existence of the limit
(`thm:sk:guerratoninelli`); Guerra's replica-symmetric bound (`thm:sk:rsbound`); the RS formula at
high temperature with $h=0$, $\beta<1$ (by the second moment or by the bound plus concentration,
as you prefer). Stated: the Parisi formula (`thm:sk:parisi`), Guerra's broken-RSB bound (sketch),
Talagrand 2006, Panchenko 2013, Auffinger–Chen; Toninelli 2002 (RSB below AT); the 2026 preprints
on the AT line with a field (Lopatto; Kusuoka–Nakajima), full RSB at zero field (Lopatto), the
overlap law (Aronow–Lopatto) — all `\provedpreprint`, with arXiv versions.
*Measuring it.* The RS fixed point and the AT line; the RS free entropy against exact enumeration
at small $N$ and against the Parisi value; the negative zero-temperature entropy of RS
($-1/(2\pi)$).
*Status board.* Digest 01 §7 rows on SK.

### Part II. Inference on the Nishimori Line

*Blurb:* When the posterior is the true Bayesian posterior, the replicas behave like the truth, the
overlap concentrates, and replica symmetry is not an assumption but a theorem. The physicist's
formulas for Bayes-optimal errors are then exact, and the remaining question is whether an efficient
algorithm can reach them.

#### Chapter 4. Bayes-Optimal Inference — Why replica symmetry holds (key: nishimori)

Read: digest 01 §4; digest 02 §2.2–2.3 (RS formula statements, numbers).

*The question.* $\Z_2$ synchronization / spiked Wigner with a Rademacher prior: the mutual
information and the MMSE in the limit; the Bayes posterior is a Gibbs measure at the Nishimori
temperature.
*The physics.* The replica computation on the Nishimori line: RS ansatz justified by the Nishimori
symmetry; the scalar fixed point $q=\E\tanh(\lambda^2q+\lambda\sqrt q\,Z)$; the continuous
transition at $\lambda=1$ for the Rademacher prior; first-order transitions for sparse priors.
*The stone.* Replicas ↔ Guerra interpolation between the matrix channel and a scalar channel; RS
on the Nishimori line ↔ overlap concentration from the Nishimori identity plus a small perturbation;
the saddle point ↔ the matching upper and lower bounds (Aizenman–Sims–Starr scheme or Barbier–
Macris adaptive interpolation).
*The proof.* Complete: the Nishimori identity in the matrix model; the upper bound of the RS formula
by Guerra's interpolation for the Rademacher prior (or a bounded prior); the overlap concentration
lemma (with the perturbation), in a form you can prove completely (the derivative of the free
entropy in a small side-channel SNR). Stated: the full theorem (`thm:nishimori:rsformula`,
Lelarge–Miolane 2019, any prior with finite second moment; Barbier–Macris 2019), with what the lower
bound adds. Note that "RS is exact on the Nishimori line" is model-specific as a theorem (digest 01
§0.4 item 6).
*Measuring it.* Exact posterior enumeration at $N=12$ and AMP at $N=2000$ against the MMSE curve;
the free entropy against the formula.
*Status board.* Digest 01 §7 and digest 02 §6 rows on the Nishimori line and spiked Wigner.

#### Chapter 5. Spiked Matrices — PCA, the BBP transition and the information threshold (key: spiked)

Read: digest 02 §2 (all), §5; digest 01 §4.3–4.4.

*The question.* When does the top eigenvector of $A=\frac\lambda Nxx^\top+W$ see $x$, and when does
the Bayes estimator do better? Rademacher and sparse Rademacher priors.
*The physics.* The top eigenvalue as the ground state of a spherical model: the replica (or
saddle-point) computation at zero temperature giving $\lambda+1/\lambda$ and the overlap
$1-1/\lambda^2$; the sparse-prior phase diagram from the RS free entropy (Lesieur–Krzakala–
Zdeborová): continuous versus first-order transitions, $\rho_{\mathrm{tri}}=0.0964$, the hard phase.
*The stone.* Zero-temperature spherical replicas ↔ the resolvent and a rank-one perturbation
(Sherman–Morrison, the secular equation); the RS potential of the sparse prior ↔ Lelarge–Miolane
plus Montanari–Venkataramanan; "AMP is optimal" ↔ restricted-class optimality (first-order methods,
low degree) and the conjecture beyond.
*The proof.* Complete: the BBP eigenvalue limit and the eigenvector overlap
(`thm:spiked:bbp`) given the semicircle law and $\norm W\to2$ (`thm:toolkit:semicircle`), by the
resolvent. Stated: Baik–Ben Arous–Péché 2005 (Wishart), Féral–Péché, Benaych-Georges–Nadakuditi;
the IT thresholds for the sparse prior (from Lelarge–Miolane; the numbers are theorems up to
numerical evaluation); Montanari–Venkataramanan's Bayes-AMP result.
*Measuring it.* The top eigenvalue and overlap at $N=2000$ against $\lambda+1/\lambda$ and
$1-1/\lambda^2$; the sparse-Rademacher phase diagram; the hard phase seen by AMP from weak and strong
initializations (digest 02 `02_bbp_and_pspin_gd.py`, `02_sparse_rademacher.py`,
`02_hard_phase_demo.py`).
*Status board.* Digest 02 §6 rows on BBP, sparse priors, the hard phase, SDP.
*Note.* The often-quoted "$\rho\approx0.09$" mixes two critical densities (0.0964 and 0.0856;
digest 02 corrections item 4); state both.

#### Chapter 6. Message Passing — The cavity, TAP and AMP (key: amp)

Read: digest 02 §1 (all), §3 (TAP free energy).

*The question.* An algorithm that computes the posterior means approximately, and a theory that
predicts its error exactly.
*The physics.* The cavity method; belief propagation on the dense graph; the TAP equations and the
Onsager reaction term; AMP as the iterative TAP; state evolution as the cavity prediction for the
iterates.
*The stone.* Removing a spin (cavity) ↔ conditioning the Gaussian matrix on the past iterates
(Bolthausen); the Onsager term ↔ the correction that makes the conditional field Gaussian; the
cavity field's Gaussianity ↔ the conditioning lemma (`lem:toolkit:conditioning`); TAP free energy
↔ Chen–Panchenko / Fan–Mei–Montanari / Celentano–Fan–Mei.
*The proof.* Complete: state evolution (`thm:amp:se`) for symmetric AMP with a GOE matrix and
Lipschitz denoisers, for a fixed number of iterations, by Bolthausen's conditioning. If the full
induction does not fit, prove it completely for the first two iterations and for polynomial
denoisers, and give the general induction as a sketch with every step named (Bayati–Montanari 2011;
Javanmard–Montanari 2013; Berthier–Montanari–Nguyen 2020). Stated: universality (Bayati–Lelarge–
Montanari; Chen–Lam), growing iterations (Rush–Venkataramanan), rotationally invariant matrices
(Fan 2022), the TAP-free-energy theorems.
*Measuring it.* Bayes-AMP for $\Z_2$ synchronization at $N=3000$, $\lambda=1.5$, against state
evolution, with and without the Onsager term; spectral initialization with the corrected convention
$m^{-1}=h^0/\lambda^2$ (digest 02: a misprint in Celentano–Fan–Mei arXiv v3).
*Status board.* Digest 02 §6 rows on AMP, TAP.

#### Chapter 7. Generalized Linear Models — The teacher–student perceptron and the hard phase (key: glm)

Read: digest 01 §6; digest 02 §5; digest 03 §4.6.

*The question.* A teacher perceptron labels random patterns; how well can a student learn, and can
it do so efficiently? The Bayes-optimal error and the first-order transition for the Ising teacher.
*The physics.* Gardner–Derrida (1989), Györgyi (1990), Sompolinsky–Tishby–Seung (1990): the replica
formula for the teacher–student GLM; the Ising teacher's first-order transition to perfect learning
at $\alpha_{\mathrm{IT}}$ and the spinodal $\alpha_{\mathrm{AMP}}$; the spherical teacher's Bayes
error $\varepsilon\sim0.442/\alpha$ and Gibbs learning $\sim0.625/\alpha$.
*The stone.* Replica formula ↔ adaptive interpolation (Barbier et al. 2019); the Nishimori
symmetry ↔ overlap concentration; the spinodal ↔ GAMP state evolution; "the hard phase is hard" ↔
evidence only (OGP, low degree, restricted-class optimality).
*The proof.* Complete: one direction of the replica formula (the interpolation bound) for a
Gaussian or Rademacher prior and a channel you choose, or a complete proof of a weaker but exact
statement (for example the Nishimori consequences for the Gibbs-learning error). Stated: Barbier
et al. 2019 (hypotheses (h1)–(h5)), with what each direction needs.
*Measuring it.* Solve the scalar equations: the Ising teacher's potential and its zero crossing
(recompute $\alpha_{\mathrm{IT}}=1.2447$; the paper prints 1.249: report this as a discrepancy
between the paper's formula and its printed value, checked at high precision), the GAMP spinodal
1.4926; run GAMP at $d=2000$ against state evolution; the spherical teacher's learning curves.
*Status board.* Digest 01 §7 rows on GLMs and the teacher–student perceptron; digest 02 §5.3 on
"AMP is optimal among polynomial algorithms" [conjectured].

### Part III. Comparison and Convexity

*Blurb:* Many learning problems are convex, and for convex problems with Gaussian data a single
comparison inequality, Gordon's, replaces the whole replica calculation. The replica-symmetric
saddle point and Gordon's auxiliary problem turn out to be the same equations, which is why the
physicists were right; where the problem stops being convex, Gordon gives only a bound, and the
replicas break.

#### Chapter 8. Gaussian Comparison — Gordon's theorem and the capacity of the perceptron (key: gordon)

Read: digest 03 §1 (all); digest 01 §5.

*The question.* The capacity of the spherical perceptron with margin $\kappa$; Gardner's formula.
*The physics.* Gardner's replica computation of the volume of solutions (RS); the AT instability
for $\kappa<0$; the Ising perceptron: why the RS capacity $4/\pi$ is wrong and the RS entropy zero
gives Krauth–Mézard's $0.83308$.
*The stone.* Gardner's volume ↔ Gordon's min-max; the RS order parameter $q\to1$ at capacity ↔ the
auxiliary optimization over a norm; the replica-symmetric saddle point ↔ the scalar auxiliary
problem (digest 03 §1.6: they are the same equations); convexity ↔ the reverse direction of the
CGMT; RSB at $\kappa<0$ ↔ Gordon gives only an upper bound.
*The proof.* Complete: Gordon's comparison inequality (`thm:gordon:gordon`) by Gaussian
interpolation (from `lem:toolkit:interpolation`); the capacity for $\kappa\ge0$
(`thm:gordon:capacity`): the upper bound by Gordon and the lower bound by the CGMT (or by convex
duality), so that $\alpha_c(0)=2$. Stated: the CGMT (`thm:gordon:cgmt`, Thrampoulidis–Oymak–Hassibi
2015 Thm 3; Thrampoulidis–Abbasi–Hassibi 2018 Thm 6.1; credit Stojnic for the duality idea); the
negative margin (Stojnic's upper bound, beaten for $\kappa\lesssim-0.63$; full RSB predicted;
Montanari–Zhong–Zhou to leading order); the Ising perceptron (Ding–Sun and Huang under numerical
conditions; Shmalo 2026 `\provedpreprint` computer-assisted; Altschuler–Tikhomirov's
$\alpha_c<0.847$, and the closed form $(1+\log2)/2$ the same method gives, digest 01 §0.4 item 2:
check, and present as the book's observation); the symmetric binary perceptron (Aubin–Perkins–
Zdeborová; Perkins–Xu; Abbe–Li–Sly).
*Measuring it.* Capacity by linear programming at $d=50$–$400$ against $\alpha_c(\kappa)$ for
$\kappa\ge0$ and against the upper bound for $\kappa<0$; Cover's finite-$d$ formula.
*Status board.* Digest 03 §6 rows 1–5; digest 01 §7 perceptron rows.

#### Chapter 9. Ridge Regression — Three derivations of double descent (key: ridge)

Read: digest 03 §2 (all).

*The question.* The test error of ridge regression when $n$ and $d$ are comparable; the
interpolation peak at $\alpha=1$; the optimal ridge.
*The physics.* The replica-symmetric zero-temperature computation (digest 03 §2.3); the
random-matrix precursors (Krogh–Hertz 1991–92; Hertz–Krogh–Thorbergsson 1989).
*The stone.* The replica saddle point ↔ the CGMT scalar problem (identical equations) ↔ the
Stieltjes transform of the Marchenko–Pastur law; the effective ridge ↔ the fixed-point equation of
the Stieltjes transform.
*The proof.* Complete: the limiting ridge risk in the isotropic Gaussian case
(`thm:ridge:risk`) by one rigorous route (random matrices with the Marchenko–Pastur law stated in
`thm:toolkit:mp`, or the CGMT from Chapter 8), including the ridgeless limits
$\sigma^2/(\alpha-1)$ and $r^2(1-\alpha)+\sigma^2\alpha/(1-\alpha)$ and the optimal ridge
$\lambda^*=\sigma^2/(\alpha r^2)$ (check against digest 03: $\lambda^*=\sigma^2\gamma/r^2$ with
$\gamma=1/\alpha$). Stated: Hastie–Montanari–Rosset–Tibshirani 2022 (general covariance), Dobriban–
Wager 2018, Dicker 2016, Bartlett–Long–Lugosi–Tsigler 2020 (benign overfitting, non-asymptotic).
*Measuring it.* Simulation at $n=1000$ against the three derivations (one-minute settings, digest
03 §2.8).
*Status board.* Digest 03 §6 rows 6–9.

#### Chapter 10. The LASSO and Robust Regression — Four routes to one fixed point (key: lasso)

Read: digest 03 §3 (all).

*The question.* The risk of the LASSO for a sparse signal; the Donoho–Tanner phase transition for
noiseless $\ell_1$ recovery; robust M-estimation with $p/n$ fixed.
*The physics.* Kabashima–Wadayama–Tanaka 2009 (replica, noiseless), Rangan–Fletcher–Goyal 2012
(RS decoupling with noise); Advani–Ganguli 2016 (robust regression).
*The stone.* The replica fixed point ↔ AMP's state evolution (Bayati–Montanari 2012) ↔ the CGMT
(Thrampoulidis–Abbasi–Hassibi 2018; Miolane–Montanari 2021); the noiseless limit ↔ the statistical
dimension of the $\ell_1$ descent cone (Amelunxen et al. 2014) and polytope angles (Donoho–Tanner).
*The proof.* Complete: the Donoho–Tanner threshold as a statistical-dimension computation (the
upper and lower bounds from the kinematic formula, stated, with the $\ell_1$ computation done in
full), or the LASSO risk by the CGMT given `thm:gordon:cgmt` (`thm:lasso:risk`). Stated: Bayati–
Montanari 2012, Miolane–Montanari 2021, the El Karoui–Bean–Bickel–Lim–Yu system and its three proofs
(El Karoui 2013; Donoho–Montanari 2016; Thrampoulidis–Abbasi–Hassibi 2018).
*Measuring it.* The LASSO risk for a Bernoulli–Gaussian signal against simulation; the
Donoho–Tanner curve computed three ways (agreeing to eight digits; $\rho(\delta=0.5)=0.38568967$);
Huber regression at $n=1000$, $p=300$.
*Status board.* Digest 03 §6 rows 10–14.

#### Chapter 11. Logistic Regression and Max-Margin Classifiers — The capacity problem with a teacher (key: classification)

Read: digest 03 §4 (all).

*The question.* When does the logistic MLE exist, how biased is it when it does, and how well does
the max-margin classifier generalize when the data are separable?
*The physics.* Gardner's computation at fixed teacher overlap (digest 03 §4.2.3, heuristic) gives
the separability threshold; Opper–Kinzel and Aubin et al. for convex ERM with a perceptron teacher;
the order of discovery (physics first only for Gardner's capacity, the optimal-stability error, the
Bayes-optimal perceptron and mixtures; Sur–Candès were derived and proved directly by
statisticians, digest 03 §4.8).
*The stone.* Gardner with a teacher ↔ the CGMT for the separability problem; the cavity /
AMP fixed point of the MLE ↔ Sur–Candès' system; the max-margin RS saddle ↔ Montanari–Ruan–Sohn–Yan.
*The proof.* Complete: the separability threshold at zero signal ($\alpha=2$, Cover again) and,
if it fits, the threshold curve $\alpha_{\mathrm{MLE}}(r)$ (`thm:classification:mle`, Candès–Sur
2020) by the kinematic formula or Gordon. Stated: Sur–Candès 2019 (bias, variance inflation,
rescaled LRT), Montanari–Ruan–Sohn–Yan (max-margin), Aubin et al. 2020, Deng–Kammoun–Thrampoulidis
(note the reversed inequality in their v2 Prop. 3.3), Mignacco et al. 2020.
*Measuring it.* The MLE-existence curve against LP separability at $n=400$–$1000$; the logistic
MLE's bias and variance inflation against simulation.
*Status board.* Digest 03 §6 rows 15–24.

#### Chapter 12. Kernels and Random Features — Learning curves and Gaussian equivalence (key: kernels)

Read: digest 03 §5 (all).

*The question.* Learning curves of kernel ridge regression and of random-features regression in
high dimension; the "Gaussian equivalence principle" that physics derivations assume.
*The physics.* Sollich; Bordelon–Canatar–Pehlevan 2020 and Canatar–Bordelon–Pehlevan 2021 (the
replica/cavity learning curve with the effective ridge); Gerace et al. 2020 and Goldt et al. 2020b
(random features and Gaussian equivalence); Loureiro et al. 2021 (Gaussian covariate model).
*The stone.* The learning-curve formula ↔ deterministic equivalents (Cheng–Montanari 2024;
Misiakiewicz–Saeed 2024); Gaussian equivalence ↔ universality theorems (Mei–Montanari 2022 Thm 7;
Hu–Lu 2023; Montanari–Saeed 2022) with their hypotheses; the linearization of kernel matrices
(El Karoui 2010) ↔ its failure in the polynomial regime $n\asymp d^\ell$.
*The proof.* Complete: one exact statement within reach, for example the deterministic equivalent
of kernel ridge regression for Gaussian features with a finite number of distinct eigenvalues, or
the reduction of the flat-spectrum case to Chapter 9's ridge formula, or El Karoui's linearization
in its simplest case. Stated: Mei–Montanari 2022 (random features, $\lambda>0$), Hu–Lu 2023,
Montanari–Saeed 2022, Ghorbani et al. 2021 (polynomial regime), Misiakiewicz 2022, Xiao et al.
Where Gaussian equivalence fails: digest 03 §6 rows 34–40, and the book's own counterexample at zero
ridge in fixed low dimension (digest 03 §5.3.3: check it independently before presenting it as the
book's own).
*Measuring it.* Kernel ridge learning curves on the sphere against the formula; random-features
regression against Mei–Montanari's curve; a Gaussian-equivalence test that fails (polynomial regime
or one large gradient step).
*Status board.* Digest 03 §6 rows 25–40.

### Part IV. Landscapes and Dynamics

*Blurb:* Inference by optimization asks a different question: not what the posterior knows, but
where an algorithm goes. Physicists answer by counting critical points and by writing closed
equations for the dynamics; mathematicians have proved both, on fixed time horizons and for
specific models, and the gap between the energy dynamics reaches and the energy that exists is one
of the cleanest facts the two languages share.

#### Chapter 13. Counting Critical Points — Kac–Rice and the spiked tensor model (key: kacrice)

Read: digest 02 §4 (all), §5.2 (ALG); digest 04 §1.5 (numbers).

*The question.* How many local minima does the landscape of the pure spherical $p$-spin and of the
spiked tensor model have at each energy, and where do gradient methods get stuck?
*The physics.* The complexity by the TAP approach and replicas (Crisanti–Sommers;
Cavagna–Giardina–Parisi; Bray–Dean; Fyodorov); the threshold energy; Ros–Ben Arous–Biroli–
Cammarota 2019 for the spiked tensor (replicated Kac–Rice, RS: heuristic).
*The stone.* TAP complexity ↔ the Kac–Rice formula (`thm:toolkit:kacrice`) with the Hessian a
shifted GOE; the annealed computation ↔ the GOE large-deviation computation; "quenched = annealed"
↔ Subag's second moment; the threshold ↔ the edge of the semicircle.
*The proof.* Complete: the annealed expected number of critical points of the pure spherical
$p$-spin (total and by index or energy, as far as the GOE determinant computation allows; ABC 2013
Thms 2.5, 2.8 give the target; `thm:kacrice:annealed`), given the Kac–Rice formula and the GOE
facts of the toolkit. Stated: Subag 2017a (concentration), Subag–Zeitouni, Ben Arous–Mei–Montanari–
Nica 2019 (spiked tensor), Huang–Sellke (ALG), Fan–Mei–Montanari's TAP complexity bound for $\Z_2$.
*Measuring it.* $E_\infty$, $E_0$ and the complexity curves for $p=3$; gradient descent on the
3-spin at $N=200$–$800$ ending near $-E_\infty$ (digest 02 `02_pspin_complexity.py`,
`02_bbp_and_pspin_gd.py`); the spiked tensor's annealed threshold $\lambda_c(k)$.
*Status board.* Digest 02 §6 Kac–Rice rows; note ABC Remark 2.7's misprint ($\sqrt2E$, not
$E/\sqrt2$).

#### Chapter 14. Dynamical Mean-Field Theory — Gradient flow in high dimension (key: dmft)

Read: digest 04 §1, §2 (all).

*The question.* Closed equations for the correlation and response of gradient flow and Langevin
dynamics in high dimension: for the spherical $p$-spin, and for gradient descent on a GLM.
*The physics.* The generating functional (Martin–Siggia–Rose / De Dominicis), the
Crisanti–Horner–Sommers and Cugliandolo–Kurchan equations; aging; the threshold energy; DMFT for
learning (Agoritsas–Biroli–Urbani–Zamponi 2018; Mignacco et al. 2020).
*The stone.* The generating functional and the saddle point ↔ large deviations (Ben Arous–Guionnet)
and Gaussian conditioning via AMP (Celentano–Cheng–Montanari; Gerbelot et al. 2024); time
translation invariance and FDT ↔ the high-temperature theorem; "dynamics reaches the threshold" ↔
Sellke 2024 (pure $p$-spin, $\beta\to\infty$); "DMFT at $t\to\infty$ = statics" ↔ convex cases only
(CCM Thms 3–4) and the counterexample $E_{\mathrm{th}}\ne E_{\mathrm{gs}}$.
*The proof.* Complete: an exactly solvable case, for example gradient flow on least squares
(the dynamics through the Marchenko–Pastur spectrum, the DMFT kernel computed explicitly) or Langevin
dynamics on the spherical $p=2$ model (diagonal in the eigenbasis; aging for $p=2$ is Ben Arous–
Dembo–Guionnet 2001). Stated: Ben Arous–Dembo–Guionnet 2006 (CK equations, soft constraint, fixed
horizon), Sellke 2024 (threshold), Dembo–Subag 2020, Celentano–Cheng–Montanari (unpublished, arXiv
v3 2026), Gerbelot et al. 2024 (SIAM J. Math. Data Sci.). Normalizations: state the CK ↔ ABC
dictionary once (factor $\sqrt2$).
*Measuring it.* Integrate the CK equations for $p=3$ (energy and two-time correlation at several
temperatures; $T_d=\sqrt{3/8}$); gradient flow on least squares against the Marchenko–Pastur curve
(digest 04 `d4_ck_integrate.py`, `d4_gf_leastsquares.py`).
*Status board.* Digest 04 §6 rows D1–D13, L1–L9.

#### Chapter 15. Online SGD — Teacher–student networks, plateaus and the information exponent (key: sgd)

Read: digest 04 §3 (all).

*The question.* Online SGD with one fresh sample per step on a teacher–student network: the
learning curve, the plateau, and how many samples a random start needs.
*The physics.* Saad–Solla and Biehl–Schwarze (1995): the order-parameter ODEs for soft committee
machines; the symmetric plateau and its escape.
*The stone.* Order-parameter ODEs ↔ concentration of summary statistics (martingale arguments,
fixed horizons: Goldt et al.; Veiga et al. 2022 fixing gaps); the plateau ↔ a fixed point of the
ODE; the escape time ↔ (heuristic: the $\ln N$ plateau length is not a theorem); the search phase
from a random start ↔ the information exponent (Ben Arous–Gheissari–Jagannath 2021, with log
factors; sharp for vanilla online SGD only; smoothing gives $d^{k/2}$; reuse beats it).
*The proof.* Complete: the ODE limit for online SGD on a single neuron (or a linear or erf
student-teacher pair) on a fixed time horizon (`thm:sgd:ode`), by a martingale/Grönwall argument.
Stated: Goldt et al. 2019/2020a with Veiga et al. 2022; Ben Arous–Gheissari–Jagannath 2021 (Thms
1.3–1.5, log factors) and 2024 (effective dynamics); Damian et al. 2023 (smoothing); Arnaboldi et
al. 2023.
*Measuring it.* The Saad–Solla ODE for $K=M=2$ against SGD at $d=500$–$2000$: plateau at
$\varepsilon^*=0.011611$ ($\eta\to0$; $0.012086$ at $\eta=0.1$); steps to weak recovery for
Hermite targets with information exponent 1, 2, 3 (digest 04 `d4_saad_solla.py`,
`d4_info_exponent.py`).
*Status board.* Digest 04 §6 rows O1–O12.

### Epilogue — The stone, read across (key: epilogue; written by the coordinator)

The fifteen status boards side by side; what is proved, under which hypotheses, what is conjectured,
what is false; the 2025–2026 preprints and the AI-assisted proofs among them, reported as their
authors report them; open problems; what to read next.

### Appendix A. A Toolkit — Gaussian calculus, concentration and random matrices (key: toolkit)

Written by a delegated author with the fixed `toolkit` labels above: proofs of Gaussian integration
by parts, the Gaussian interpolation lemma, the Gaussian Poincaré inequality, Lipschitz concentration
(with the Maurey–Pisier proof or a reference for the sharp constant), Laplace's method and Varadhan's
lemma; statements with references of the semicircle and Marchenko–Pastur laws, Stieltjes transforms
and resolvent identities, Slepian and Sudakov–Fernique, the Gaussian conditioning lemma (with proof:
it is short), the Kac–Rice formula (statement, Adler–Taylor); the numerics used by the scripts
(Gauss–Hermite quadrature, damped fixed-point iteration, intervals). Read: digest 01 §1.2,
digest 02 §1.2 and §4.1, digest 03 §1.1.
