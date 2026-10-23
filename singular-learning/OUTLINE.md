# Singular Learning Theory for the Rest of Us — Book Plan and Chapter Specifications

*The Missing Textbooks, volume 6.* Working title: **Singular Learning Theory for the Rest of Us:
From y = abx to Neural Networks** (Chinese working title 奇异学习理论：写给深度学习研究者).
Written by Claude; conceived and edited by Xiaoyu Li, from the editor's proposal "M-5": a readable
textbook of singular learning theory for deep-learning researchers, built on one running family of
toy models that starts from the two-parameter model y = abx, on which every concept is computed by
hand; Hironaka's resolution of singularities used as a black box; the last chapters connected to
the empirical results of deep learning.

## The thesis

A statistical model is *regular* when different parameters give different distributions and the
Fisher information is invertible at the true parameter. Then the posterior is asymptotically
normal, the log marginal likelihood is $-nS_n-\tfrac d2\log n+O_p(1)$ (Schwarz's BIC), and the
expected generalization error is $\tfrac{d}{2n}$ (Akaike's AIC). Neural networks are not regular.
Permuting hidden units, flipping signs, rescaling the two sides of a ReLU, or switching a unit off
gives the same function from different weights; the set of parameters that realize the truth is
not a point but an analytic set with singularities, and the Fisher information is degenerate on it.
Then the classical formulas are not just inexact; they are wrong in a systematic direction.

Sumio Watanabe's singular learning theory (from 1999) says what replaces them. One birational
invariant of the loss, the *real log canonical threshold* $\lambda$ — Watanabe's *learning
coefficient* — with its multiplicity $m$, replaces $\tfrac d2$: the free energy is
$nS_n+\lambda\log n-(m-1)\log\log n+O_p(1)$, the expected Bayes generalization error is
$\lambda/n$, and the information criteria WAIC and WBIC estimate these quantities without the
regularity assumptions. The learning coefficient is computed by resolution of singularities, the
theorem of Hironaka (1964) that Atiyah (1970), and Bernstein and S. I. Gel'fand (1969), used to
answer I. M. Gel'fand's 1954 question about the analytic continuation of $\int\abs{f}^z$.
Watanabe's theory treated layered neural networks from its first papers (2001). Since about 2020 it
has been brought to modern deep learning: the *local* learning coefficient of a trained network has
been estimated by stochastic-gradient Langevin dynamics, compared across networks, and followed
along training, where its jumps mark stages of development.

The theory is hard to learn. Watanabe's books begin with algebraic geometry and empirical processes
on manifolds; the deep-learning papers begin with estimators. Between them there is no textbook
(see "Positioning"). This book starts from a model with two parameters, $y=abx$, whose free energy,
posterior, learning coefficient, generalization error and information criteria can all be computed
by hand, and adds machinery only when a computation demands it. **Each chapter teaches one concept:
it states what the regular theory says about it, computes it by hand on the running toy models,
proves its central theorem completely (if necessary under a stated simplifying hypothesis that the
general theory removes), measures it in a script, records the result on a ledger card, and says
where the theory stops.** The last part asks what all this says about trained networks, and is as
specific about what singular learning theory does not explain as about what it does.

## The ledger

Every chapter ends its "The ledger" section with a card (`\ledgercard`) with six fields: the
**quantity**; what the **regular theory** predicts for it; what the **singular theory** predicts
(with the theorem's label); its value in the chapter's **running example**; what the script
**measured**; and what it **rests on**. The epilogue, *The ledger*, collects the fifteen cards
into two tables (regular versus singular, and assumptions versus deep-learning practice).

## The running models

The book's toy models, with their fixed values. Unless a chapter says otherwise: inputs
$x\sim\Normal(0,1)$ (or $\Normal(0,I_k)$ for $k$ inputs), Gaussian noise of variance $1$, the prior
uniform on $[-1,1]^d$, and the truth $w_0=0$ (so that $y$ is pure noise); then $K(w)=\tfrac12
\E_x[f(w,x)^2]$ for a regression $y=f(w,x)+\text{noise}$. The script kit (`sltkit.py`) implements
all of them. Chapter authors must use these names and values; a chapter that computes a value
differently (e.g. with another prior) says so.

| name | model | $d$ | $K(w)$ near the singular point | $W_0$ | $\lambda$ | $m$ | $\tfrac d2$ |
|---|---|---|---|---|---|---|---|
| the regular line | $y=ax$ | 1 | $a^2/2$ | a point | $1/2$ | 1 | $1/2$ |
| the regular plane | $y=ax_1+bx_2$ | 2 | $(a^2+b^2)/2$ | a point | $1$ | 1 | $1$ |
| **the product model** | $y=abx$ | 2 | $a^2b^2/2$ | the cross $ab=0$ | $1/2$ | 2 | $1$ |
| the product model, generic truth | $y=abx$, truth $a_0b_0=c\neq0$ | 2 | $(ab-c)^2/2$ | the hyperbola $ab=c$ | $1/2$ | 1 | $1$ |
| the cusp model | $y=ab^2x$ | 2 | $a^2b^4/2$ | the cross | $1/4$ | 1 | $1$ |
| the cubic cusp | $y=ab^3x$ | 2 | $a^2b^6/2$ | the cross | $1/6$ | 1 | $1$ |
| the quartic model | $y=ax_1+b^2x_2$ | 2 | $(a^2+b^4)/2$ | a point | $3/4$ | 1 | $1$ |
| the sextic model | $y=ax_1+b^3x_2$ | 2 | $(a^2+b^6)/2$ | a point | $2/3$ | 1 | $1$ |
| the quartic cup | $y=a^2x_1+b^2x_2$ | 2 | $(a^4+b^4)/2$ | a point | $1/2$ | 1 | $1$ |
| the mixed cup | $y=a^2x_1+b^3x_2$ | 2 | $(a^4+b^6)/2$ | a point | $5/12$ | 1 | $1$ |
| the chain | $y=a_1a_2\cdots a_Lx$ | $L$ | $(a_1\cdots a_L)^2/2$ | the union of the coordinate hyperplanes | $1/2$ | $L$ | $L/2$ |
| the tanh neuron | $y=a\tanh(bx)$, $x\sim\Unif[-1,1]$, prior on $[-2,2]^2$ | 2 | $\tfrac{a^2}2\E[\tanh^2(bx)]\asymp a^2b^2$ | the cross | $1/2$ | 2 | $1$ |
| the ReLU neuron | $y=a\relu(bx)$ | 2 | $a^2b^2/4$ exactly | the cross | $1/2$ | 2 | $1$ |
| the mixture | $(1-a)\Normal(0,1)+a\Normal(b,1)$, $a\in[0,1]$, $b\in[-2,2]$ (default; Ch. 10 uses $[-1,1]$, Ch. 1 $[-4,4]$ at its regular truth, each stated there), truth $\Normal(0,1)$ | 2 | $K\asymp a^2b^2$ (not analytic in $a$ at $a=0$) | $\{a=0\}\cup\{b=0\}$ | $1/2$ | 2 | $1$ |
| rank-one regression | $y=b\,a^\top x$, $a\in\R^N$, $b\in\R^M$, truth $0$ | $M+N$ | $\norm{a}^2\norm{b}^2/2$ | $\{a=0\}\cup\{b=0\}$ | $\min(M,N)/2$ | 2 if $M=N$, else 1 | $(M+N)/2$ |
| reduced-rank regression | $y=BAx$, $B\in\R^{M\times H}$, $A\in\R^{H\times N}$, truth of rank $r$ | $H(M+N)$ | $\norm{BA-B_0A_0}_F^2/2$ | — | Aoyagi–Watanabe formula; e.g. $M=N=2$, $r=1$: $H=1$: $3/2$; $H=2$: $2$ ($m=2$). $M=N=3$, $r=1$: $H=1,2,3$: $5/2$, $7/2$ ($m=2$), $4$ | | $H(M+N)/2$ |

**Corrections from digest 02 (`build/research/02-computing-rlct.md` §6).** (1) With Gaussian
inputs the tanh neuron's $K$ is $C^\infty$ but **not real-analytic** at $b=0$ (the Taylor series has
radius $0$: the complex poles of $\tanh$ meet the unbounded Gaussian support), so Watanabe's
analyticity condition fails; with $x\sim\Unif[-1,1]$ the radius is $\pi/2$ and the theorems apply
verbatim. The book therefore uses $x\sim\Unif[-1,1]$ for the tanh neuron (then $\E[x^2]=\tfrac13$).
(2) The mixture's $K$ is not analytic in $a$ at $a=0$ for $b\neq0$; its values $\lambda=\tfrac12$,
$m=2$ (attained at $(0,0)$, on the boundary $a=0$) follow by comparison with $a^2b^2$, but the
general theorems do not apply verbatim (Watanabe treats mixtures separately; Watanabe 2021,
*Behaviormetrika*). Chapters that use it say so. (3) The ReLU neuron's $K$ is a polynomial, but the
model is not analytic in $b$ at $b=0$ (a kink), so the same caution applies. (4) Generic truths of
the product model: if the hyperbola meets the prior box only at a corner, $\lambda$ jumps (to $1$ for
$ab=1$ on $[-1,1]^2$); use truths whose true set crosses the interior. (5) The cusp attains $\tfrac14$
along the whole segment $b=0$, not only at the origin ("one constraint, yet $\lambda=\tfrac14$"). (6)
In the chain, a point with exactly $j$ vanishing coordinates has local value $(\tfrac12,j)$: depth
raises $m$, not $\lambda$. (7) The tanh neuron at a nonzero truth $(a_0,b_0)$ is regular: the true
set is $\{\pm(a_0,b_0)\}$ and $\lambda=1$, though one Fisher eigenvalue is small (at $(1,1)$ with
Gaussian $x$: $0.0196$), so the posterior looks singular for small $n$.

The values for the product model, the chain, the cusps, and the sums of independent monomials are
proved in the book (Chapters 3–6); the tanh neuron, the ReLU neuron and the mixture are reduced to
the product model in Chapter 2 (verify the expansions); rank-one regression is Chapter 7's main
theorem. The first fact a reader should take away: **the product model has two parameters but
learns like a model with one**, and pays a little less than one, by $\log\log n$, because its set
of true parameters is a cross.

Generic truths: at a generic truth of the product model the set of true parameters is a smooth
curve, and $\lambda=\tfrac12$, $m=1$: the model still has one effective parameter, but no
$\log\log n$ term. Chapter 12 uses truths close to the cross (at distance $n^{-1/2}$) to show
phase transitions.

## Positioning (verified 2026-10-24; digest in `build/research/04-positioning-classics.md`, §1)

Watanabe's two books (2009, a research monograph; 2018, a graduate text with exercises) are the
rigorous sources and the place to go after this one; his Japanese books (2006, 2012) have no English
translations. Suzuki's exercise books (Springer 2023, R and Python editions; Japanese editions
2023–2024) work through Watanabe's theory with exercises and Stan code. The algebraic-statistics
texts of Drton, Sturmfels and Sullivant (2009, ch. 5) and Sullivant (2018, ch. 17), and Lin's thesis
(2011), compute Bayesian integrals of algebraic models for mathematicians. Amari and Watanabe's
Japanese book (August 2026) adds chapters on the singularities of deep networks to a book on
statistical neurodynamics. For deep-learning researchers the entry points are an invitation paper
(Wei et al. 2023), blog sequences (Carroll 2023; Agrawal 2026, unfinished), exercise posts and
recorded lectures; the field's own reading list sends readers from these straight to Watanabe's
books. In October 2026 we found no book, lecture notes or course that teaches the theory to
deep-learning researchers by computing every quantity on a family of small models.

**Do not claim** that this is the first textbook of singular learning theory (Watanabe's books are
textbooks or monographs). The book's claims to difference: (i) one running family of toy models on
which every quantity is computed by hand; (ii) Hironaka's theorem as a black box; (iii) complete
proofs of special cases (one-coefficient models, the linear models of Chapter 8) rather than either
full generality or no proofs; (iv) a deep-learning part that is honest about what the theory does
not explain; (v) scripts that reproduce every number. Cite Watanabe 2018 and Suzuki 2023 as the
sequel and the companion, Carroll 2023 as the informal introduction, Amari and Watanabe 2026 as the
related Japanese book, and the critiques of Skalse (2023) and Erdil (2023) where the theory stops.

Also from the digest (§4.4): Watanabe analysed layered neural networks from 2001 (Watanabe 2001a,
2001b, 2007), so no chapter should suggest that the theory ignored neural networks before 2020; the
1969 co-author of Bernstein is S. I. (Sergei) Gel'fand, not I. M. Gel'fand, who asked the question.

## Structure, keys, and files

| # | key | file | Title — Subtitle | the model | Newton foot (`vertices`, `t`, `λ`, `K`) |
|---|-----|------|------------------|-----------|------|
| – | prologue | 00-prologue.tex | Prologue — Two parameters | (special chapter) | `{2/2}{2}{\tfrac12}{a^2b^2}` |
| | | | **Part I. Where Regular Theory Ends** | | `{2/2}{2}{\tfrac12}{a^2b^2}` |
| 1 | regular | 01-regular.tex | Regular Models — What the Laplace approximation promises | `$y=ax_1+bx_2$` | `{0/2,2/0}{1}{1}{a^2+b^2}` |
| 2 | singular | 02-singular.tex | Singular Models — When parameters fold | `$y=a\tanh(bx)$` | `{2/2}{2}{\tfrac12}{a^2b^2}` |
| 3 | coefficient | 03-coefficient.tex | The Free Energy by Hand — The whole theory in one coefficient | `$y=abx$, $y=ab^2x$` | `{2/4}{4}{\tfrac14}{a^2b^4}` |
| | | | **Part II. The Learning Coefficient** | | `{0/4,2/0}{4/3}{\tfrac34}{a^2+b^4}` |
| 4 | volume | 04-volume.tex | Volume — The learning coefficient as a scaling exponent | `$y=a^2x_1+b^2x_2$` | `{0/4,4/0}{2}{\tfrac12}{a^4+b^4}` |
| 5 | blowups | 05-blowups.tex | Blow-Ups — Resolution of singularities by hand | `$y=ax_1+b^2x_2$` | `{0/4,2/0}{4/3}{\tfrac34}{a^2+b^4}` |
| 6 | newton | 06-newton.tex | Newton Polygons — The toric shortcut | `$y=a^2x_1+b^3x_2$` | `{0/6,4/0}{12/5}{\tfrac5{12}}{a^4+b^6}` |
| 7 | catalogue | 07-catalogue.tex | A Catalogue — Learning coefficients of models in use | `$y=BAx$` | `{2/2}{2}{\tfrac12}{a^2b^2}` |
| | | | **Part III. Watanabe's Theorems** | | `{0/6,2/2,6/0}{2}{\tfrac12}{a^6+a^2b^2+b^6}` |
| 8 | freeenergy | 08-freeenergy.tex | The Free Energy Formula — Model selection without the Laplace approximation | `$y=ax_1+b^3x_2$` | `{0/6,2/0}{3/2}{\tfrac23}{a^2+b^6}` |
| 9 | generalization | 09-generalization.tex | Generalization — The error λ/n and the equations of state | `$y=ab^3x$` | `{2/6}{6}{\tfrac16}{a^2b^6}` |
| 10 | waic | 10-waic.tex | WAIC — Estimating generalization without knowing λ | `$(1-a)\Normal(0,1)+a\Normal(b,1)$` | `{2/2}{2}{\tfrac12}{a^2b^2}` |
| 11 | temperature | 11-temperature.tex | Temperature — WBIC and the thermodynamics of the posterior | `$y=ax_1+b^2x_2$` | `{0/4,2/0}{4/3}{\tfrac34}{a^2+b^4}` |
| 12 | phases | 12-phases.tex | Phase Transitions — How the posterior chooses between singularities | `$y=abx$ near the cross` | `{0/6,2/2,6/0}{2}{\tfrac12}{a^6+a^2b^2+b^6}` |
| | | | **Part IV. Deep Learning** | | `{2/4,4/2}{3}{\tfrac13}{a^4b^2+a^2b^4}` |
| 13 | llc | 13-llc.tex | The Local Learning Coefficient — Measuring a trained network | `$y=a\relu(bx)$, deep linear networks` | `{2/2}{2}{\tfrac12}{a^2b^2}` |
| 14 | stages | 14-stages.tex | Stages — Learning coefficients along training | `the toy model of superposition` | `{2/4,4/2}{3}{\tfrac13}{a^4b^2+a^2b^4}` |
| 15 | sgd | 15-sgd.tex | SGD Is Not Bayes — What the theory does not explain | `$y=abx$ under gradient descent` | `{2/2}{2}{\tfrac12}{a^2b^2}` |
| – | epilogue | 16-epilogue.tex | Epilogue — The ledger | (special chapter) | `{0/2,2/0}{1}{1}{a^2+b^2}` |
| A | toolkit | 17-toolkit.tex | A Toolkit (appendix) | | none |

Files: `chapters/NN-key.tex` and `chapters/NN-key.bib`; scripts `figures/scripts/NN-key-name.py`;
data `figures/data/NN-key-name.dat`. Chapter labels `ch:key`. The opening of each chapter is

```
\sltchapter{Title}{Subtitle}{the model}{ch:key}
\newtonfoot{vertices}{t}{lambda}{K}
```

with the values in the table (the model in math mode as listed, or in words). The prologue and the
epilogue open with `\specialchapter{Title}{Subtitle}{{vertices}{t}{lambda}{K}}` and number their
floats with `\sltfloatprefix{P}` (epilogue: `E`) after the heading and `\sltfloatrestore` at the
end. The appendix opens with `\appendixchapter{A Toolkit}{Integrals, singularities, and samplers}{ch:toolkit}`.

## Global notation

All chapters use this notation (also in `frontmatter/notation.tex`). A chapter that needs a clashing
local symbol keeps the global meaning and chooses another symbol; a source's notation is translated
in "Translating the literature".

**Data and models.** Data $X^n=(X_1,\dots,X_n)$, independent with distribution $q$ (the truth);
for regression $X_i=(x_i,y_i)$ with $y_i=f(w,x_i)+\varepsilon_i$, $\varepsilon_i\sim\Normal(0,\sigma^2)$
($\sigma=1$ unless stated). A model $p(x\mid w)$ with parameter $w\in W\subset\R^d$ ($W$ compact unless
stated), prior density $\varphi(w)$. Coordinates of $w$ in the toy models: $a,b$ (two parameters),
$a_1,\dots,a_L$ (the chain); matrices $A,B$ (reduced rank).

**Losses and entropies.** $L_n(w)=-\frac1n\sum_{i=1}^n\log p(X_i\mid w)$ (`\Ln`); $L(w)=-\E_q[\log
p(X\mid w)]$; $S=-\E_q[\log q(X)]$ and $S_n=-\frac1n\sum_i\log q(X_i)$ (`\Sn`). A true (optimal)
parameter $w_0$ (`\wzero`) minimizes $L$; $K(w)=L(w)-L(w_0)\geq0$, which in the realizable case
$q=p(\cdot\mid w_0)$ is the Kullback–Leibler divergence $\KL(q\,\|\,p(\cdot\mid w))$;
$K_n(w)=L_n(w)-L_n(w_0)$ (`\Kn`); the set of true parameters $W_0=\{w\in W:K(w)=0\}$ (`\Wzero`).

**Posteriors.** The posterior at inverse temperature $\beta>0$ has density proportional to
$\varphi(w)e^{-n\beta L_n(w)}$; its expectation is $\Ewb{f}$ (`\Ewb{f}`), and $\Ew{f}$ (`\Ew{f}`) at
$\beta=1$. The predictive density is $p^*(x)=\Ew{p(x\mid w)}$. $\E$ and $\Prob$ without a subscript
are over the data $X^n$ (and fresh test points); say so when it matters.

**Free energy.** $F_n=-\log\int_W\varphi(w)e^{-nL_n(w)}\dd w$ (`\Fn`), the negative log marginal
likelihood; $F_n(\beta)$ at inverse temperature $\beta$; the *normalized* free energy is
$F_n-nS_n=-\log\int\varphi(w)e^{-nK_n(w)}\dd w$. The deterministic counterpart
$\bar F(n)=-\log\int\varphi(w)e^{-nK(w)}\dd w$ is the *population free energy*.

**Learning coefficient.** The zeta function $\zeta(z)=\int_WK(w)^z\varphi(w)\dd w$ ($\Re z>0$); its
largest pole $-\lambda$ and that pole's order $m$: the **learning coefficient** $\lambda$ (the real
log canonical threshold of $K$ with respect to $\varphi$) and its **multiplicity** $m$. The volume
function $V(\eps)=\int_{K(w)<\eps}\varphi(w)\dd w$ and the state density $v(t)=V'(t)$. Local versions
at a point or on a set carry it as an argument: $\lambda(w^*)$, $\lambda(U)$.

**Generalization.** Bayes generalization loss $G_n=-\E_X[\log p^*(X)]$ and error $G_n-S$; Bayes
training loss $T_n=-\frac1n\sum_i\log p^*(X_i)$; functional variance
$V_n=\sum_i\Var_w[\log p(X_i\mid w)]$; $\WAIC=T_n+V_n/n$ (per-sample scale: never multiplied by
$2n$ in this book); $\WBIC=\E^{\beta}_w[nL_n(w)]$ at $\beta=1/\log n$; the singular fluctuation
$\nu$; Gibbs losses with a subscript $\mathrm{g}$ when needed.

**Deep learning (Part IV).** A trained parameter (local minimum) $w^*$ (`\wstar`); the local
learning coefficient $\lambda(w^*)$ and its estimate $\hat\lambda(w^*)$; SGLD step size $\epsilon$,
localization strength $\gamma$, minibatch size $b$; training steps $t$.

**Asymptotics and probability.** $O_p,o_p$ in $n$ with the randomness of the data; "converges in
law", "in probability"; $\log$ is the natural logarithm; $a_n\sim b_n$ means $a_n/b_n\to1$;
$a\asymp b$ means bounded above and below by constant multiples.

**Regular and singular (the book's convention).** Following Watanabe from 2010 on (not his 2009
book, where "singular models" include the regular ones), *regular* and *singular* describe the truth
relative to the model: the pair is **regular** if $W_0$ is a single point $w_0$ in the interior of
$W$ and the Hessian $\nabla^2L(w_0)$ (the Fisher information, in the realizable case) is positive
definite; otherwise it is **singular**. (The interior condition is the book's addition: Watanabe
2010, Definition 2, has none — Chapter 2 says so.) "A singular model" is shorthand for a model with singular
truths of interest (e.g. $y=abx$ at $ab=0$). The product model at a generic truth is singular too:
$W_0$ is a curve, though $\lambda=\tfrac12$ is then the regular value for the one-dimensional
reparametrization $c=ab$.

**What the digests settle (read `build/research/01-core-theory.md` before writing Part III).**
(1) $K=\KL(q\,\|\,p_w)$ only in the realizable case; in general $K(w)=\KL(q\,\|\,p_w)-\KL(q\,\|\,p_{w_0})$.
(2) $\E[G_n]-L(w_0)=\lambda/n+o(1/n)$ holds at $\beta=1$; at inverse temperature $\beta$ the
coefficient is $(\lambda-\nu)/\beta+\nu$. (3) The free energy expansion is pathwise with $nS_n$ (or
$nL_n(w_0)$); its $O_p(1)$ remainder converges in law, not in probability. (4) $\lambda\le d/2$ needs a
prior positive at some true parameter (with Jeffreys' prior $\lambda\ge d/2$). (5) WBIC is not $F_n+O_p(1)$:
$F_n-\WBIC=-(m-1)\log\log n-U_n\sqrt{\lambda\log n/2}+O_p(1)$. (6) The WAIC theorems assume $s\ge8$
moments, the WBIC theorems $s\ge6$, and both assume relatively finite variance, not realizability.
(7) **Theorem numbers.** No theorem number inside Watanabe's 2018 book is verified, and those of
the 2009 book are known only as Watanabe cites them in his 2013 paper (W09 Theorems 2.4, 4.6,
5.9, 5.10, 6.1–6.3, Main Theorem 6.4). Cite the verified paper theorems (Watanabe 2010a Theorems
1–3 and Corollaries 1–3 for the equations of state; Watanabe 2010 Lemma 1, Theorem 1, Corollary 1,
Theorem 2 for WAIC and cross-validation; Watanabe 2013 Theorems 1–5 and Corollaries 1–3 for WBIC),
and cite the books by chapter and section (the digest has the 2009 section map and the 2018
chapter titles). Never invent a theorem number.

## The appendix and its fixed labels

The appendix (`ch:toolkit`, `17-toolkit.tex`) collects what several chapters use, with proofs where
they are short. Chapters cite these labels and may prove what they need inline when it is short.

| label | content |
|---|---|
| `thm:toolkit:laplace` | Laplace's method in $d$ dimensions with an interior nondegenerate minimum: the leading term and the $O(1/n)$ error (with proof) |
| `lem:toolkit:gaussian` | Gaussian integrals and moments; Gaussian tails (with proofs) |
| `thm:toolkit:stein` | Stein's identity $\E[Zf(Z)]=\E[f'(Z)]$ and the heat equation for Gaussian smoothing (with proofs) |
| `def:toolkit:gamma` | the Gamma function; $\int_0^\infty t^{s-1}e^{-t}\dd t$, $\int_0^1t^{s}(\log\frac1t)^{k}\dd t$ |
| `def:toolkit:mellin` | the Mellin transform; meromorphic continuation by integrating by parts; poles and the behaviour at $0$ |
| `thm:toolkit:karamata` | the Abelian direction of Karamata's theorem for Laplace transforms of regularly varying functions (with proof) |
| `def:toolkit:analytic` | real analytic functions; power series; the identity theorem; analytic sets |
| `thm:toolkit:hironaka` | Hironaka's resolution of singularities, in the form used in the book (statement, black box) |
| `thm:toolkit:splitting` | the splitting lemma (Morse lemma with parameters) (statement) |
| `thm:toolkit:empirical` | the normalized log-likelihood-ratio process on a resolved parameter space: uniform tightness and convergence in law (Watanabe; statement, black box) |
| `def:toolkit:kl` | Kullback–Leibler divergence, entropy, cross-entropy; Pinsker's inequality |
| `def:toolkit:convergence` | convergence in law and in probability; $O_p$ and $o_p$; Slutsky's lemma |
| `sec:toolkit:mcmc` | random-walk Metropolis, Hamiltonian Monte Carlo, tempering, thermodynamic integration (as used in the scripts) |
| `sec:toolkit:sgld` | stochastic-gradient Langevin dynamics; the localized tempered posterior |
| `sec:toolkit:intervals` | Student-$t$, Clopper–Pearson and bootstrap intervals (as used in the scripts) |

## Results that other chapters cite (fixed labels)

Authors of the cited chapters must use exactly these labels for these results; authors of citing
chapters may rely on them (and should also say in prose what they cite). The statements are given
in the chapter specifications.

| label | result |
|---|---|
| `thm:regular:laplace` | the free energy of a regular model: $F_n=nL_n(\hat w)+\tfrac d2\log\tfrac n{2\pi}+\tfrac12\log\det J(\hat w)-\log\varphi(\hat w)+o_p(1)$ |
| `thm:regular:aic` | the expected Bayes generalization error of a regular realizable model is $\tfrac d{2n}+o(1/n)$ |
| `prop:singular:fisher` | if $W_0$ contains a smooth curve through $w_0$, then the Fisher information at $w_0$ is singular |
| `thm:singular:scales` | the posterior of the product model at the singular truth spreads uniformly over scales: $\log\abs a/\log n\to\Unif(-\tfrac12,0)$ |
| `thm:coefficient:free` | the free energy of a one-coefficient model whose coefficient density behaves like $\abs{c}^{2\lambda-1}(\log\frac1{\abs c})^{m-1}$: $F_n-nS_n=\lambda\log n-(m-1)\log\log n+O_p(1)$ |
| `lem:volume:abelian` | if $V(\eps)\sim C\eps^\lambda(\log\frac1\eps)^{m-1}$, then $\int e^{-nK}\varphi\sim C\,\Gamma(\lambda+1)n^{-\lambda}(\log n)^{m-1}$ |
| `prop:volume:monomial` | learning coefficient and multiplicity of a normal-crossing function: $\lambda=\min_j\frac{h_j+1}{2k_j}$, $m=$ the number of minimizers |
| `thm:volume:zeta` | the volume, the state density and the zeta function: the largest pole of $\zeta$ and the asymptotics of $V$ determine each other |
| `thm:blowups:pullback` | the learning coefficient is computed on a resolution: pulling back $K$ and $\varphi\,\dd w$ by a proper analytic map that is an isomorphism off a null set |
| `prop:blowups:rules` | rules: invariance under analytic isomorphism, comparison, sums of functions of separate variables, $\lambda\le d/2$ |
| `prop:blowups:morsebott` | a critical set that is a smooth manifold of dimension $s$ with nondegenerate normal Hessian has $\lambda=(d-s)/2$, $m=1$ ("minimally singular"; used in Chapters 13–14) |
| `thm:newton:plane` | the Newton-polygon formula $\lambda=1/t$ in two variables, as proved in Chapter 6: for sums of monomials with positive coefficients and even exponents (the nondegenerate case in $d$ variables is the stated `thm:newton:general`) |
| `thm:catalogue:rankone` | the learning coefficient of rank-one regression: $\lambda=\min(M,N)/2$, $m=2$ if $M=N$ |
| `thm:catalogue:aoyagi` | the Aoyagi–Watanabe formula for reduced-rank regression (statement) |
| `thm:freeenergy:linear` | the free energy formula for models $y=G(w)x+$noise (matrix-valued $G$, so rank-one and reduced-rank regression are covered; only $\E\norm{x}^2<\infty$; no compactness — the geometry enters through $V(\eps)\sim C\eps^\lambda(\log\frac1\eps)^{m-1}$) (complete proof, as written by Chapter 8) |
| `thm:freeenergy:main` | Watanabe's free energy formula $F_n=nS_n+\lambda\log n-(m-1)\log\log n+O_p(1)$ (statement, proof sketch) |
| `thm:generalization:onecoef` | $n(\E[G_n]-S)\to\lambda$ for one-coefficient models (complete proof) |
| `thm:generalization:main` | Watanabe's generalization theorem: $\E[G_n]-S=\lambda/n+o(1/n)$ and the equations of state (statement) |
| `thm:waic:main` | WAIC is asymptotically unbiased for the Bayes generalization loss and asymptotically equivalent to leave-one-out cross-validation (statement) |
| `prop:temperature:mean` | the tempered mean energy: $\E^\beta_w[nK]=\lambda/\beta-(m-1)/(\beta\log n\beta)+\dots$ (population version, complete proof). As written by Chapter 11 it assumes a two-term *volume expansion* (`ass:temperature:volume`), true for analytic $K$ and for the running models: the $-(m-1)$ term does not follow from $V(\eps)\sim C\eps^\lambda(\log\frac1\eps)^{m-1}$ alone. Chapter 11 also defines `thm:temperature:wbiclinear` and `cor:temperature:estimators` |
| `thm:temperature:wbic` | WBIC $=F_n+O_p(\sqrt{\log n})$ (statement; complete proof for the linear models of Chapter 8) |
| `thm:phases:local` | local free energies and the posterior's choice between phases |
| `def:llc:llc` | the local learning coefficient $\lambda(w^*)$ |
| `thm:llc:hessian` | Hessian bounds: $\tfrac r2\le\lambda(w^*)\le\tfrac r2+\tfrac{d-r}4$ for a Hessian of rank $r$ (Chapter 13 proves it for any $C^4$ loss $K\ge0$; the upper bound is Lin 2017, Cor. 4.15) |
| `thm:sgd:balance` | gradient flow on the product model conserves $a^2-b^2$ and converges to the point of the hyperbola it selects |

## Shared citation keys

Chapters cite the same work with the same key (each chapter's `.bib` holds its own copy of the
entry; the entries must agree). Take entries from the digests in `build/research/` — the helper
`python build/scratch/bibpick.py chapters/NN-key.bib KEY1 KEY2 ...` copies verified entries by key
into your bib file (it drops the digests' verification notes; check what it writes). Where two
digests use different keys for one work, **the key in this table wins** (rename the entry's key).
New entries follow the convention AuthorYear, AuthorAuthorYear, AuthorEtAlYear (three or more
authors), with a letter suffix only for the same author and year.

| key | work |
|---|---|
| `Watanabe1999` | "Algebraic analysis for singular statistical estimation", ALT 1999, LNCS 1720, 39–50 |
| `Watanabe2001a` | "Algebraic analysis for nonidentifiable learning machines", *Neural Computation* 13(4), 899–933 |
| `Watanabe2001b` | "Algebraic geometrical methods for hierarchical learning machines", *Neural Networks* 14(8), 1049–1060 |
| `Watanabe2001c` | "Learning efficiency of redundant neural networks in Bayesian estimation", *IEEE Trans. Neural Networks* 12(6), 1475–1486 (digest 01; digest 04 uses this key for a Japanese book — do not use that one) |
| `Watanabe2007` | "Almost all learning machines are singular", IEEE FOCI 2007, 383–388 |
| `Watanabe2009` | *Algebraic Geometry and Statistical Learning Theory*, Cambridge University Press, 2009 |
| `Watanabe2010` | "Asymptotic equivalence of Bayes cross validation and widely applicable information criterion in singular learning theory", *JMLR* 11, 3571–3594 (digest 01 calls it `Watanabe2010e`) |
| `Watanabe2010a` | "Equations of states in singular statistical estimation", *Neural Networks* 23(1), 20–34 |
| `Watanabe2010b` | the unrealizable regular case, *IEICE Trans. Fundamentals* E93-A(3), 617–626 |
| `Watanabe2010d` | the renormalizable condition, *J. Phys. Conf. Ser.* 233, 012014 |
| `Watanabe2013` | "A widely applicable Bayesian information criterion", *JMLR* 14, 867–897 |
| `Watanabe2018` | *Mathematical Theory of Bayesian Statistics*, CRC Press, 2018 |
| `Watanabe2024` | "Recent advances in algebraic geometry and Bayesian statistics", *Information Geometry* 7 (2024) (arXiv 2211.10049) |
| `Gelfand1957` | I. M. Gel'fand, "Some aspects of functional analysis and algebra", Proc. ICM 1954, vol. 1 (1957), 253–276; the question is Problem I in §3, pp. 262–263 (digest 01 read the scan; digest 01 calls the entry `Gelfand1954`) |
| `Hironaka1964a`, `Hironaka1964b` | resolution of singularities in characteristic zero, I and II, *Annals of Math.* 79 |
| `BernsteinGelfand1969` | I. N. Bernstein and S. I. Gel'fand, "Meromorphic property of the functions $P^\lambda$" (digest 01: `BernshteinGelfand1969`) |
| `Atiyah1970` | "Resolution of singularities and division of distributions", *CPAM* 23 (not 13, a misprint in Watanabe's citations) |
| `Bernstein1972` | the b-function paper, *Funct. Anal. Appl.* 6(4) (digest 01: `Bernshtein1972`) |
| `Kashiwara1976`, `Varchenko1976` | rationality of the roots of b-functions; Newton polyhedra and oscillatory integrals |
| `YamazakiWatanabe2003`, `AoyagiWatanabe2005` | mixtures; reduced-rank regression |
| `DrtonPlummer2017` | the singular BIC, *JRSS B* 79 |
| `DrtonEtAl2009`, `Sullivant2018`, `Lin2011` | algebraic statistics (ch. 5; ch. 17); Lin's thesis |
| `Suzuki2023a`, `AmariWatanabe2026` | WAIC and WBIC exercise book (R); the Japanese book with chapters on singularities |
| `WeiEtAl2023` | "Deep learning is singular, and that's good", *IEEE TNNLS* 34 (2023) (arXiv 2010.11560) |
| `LauEtAl2025` | the local learning coefficient, AISTATS 2025, PMLR 258 (arXiv 2308.12108) |
| `FurmanLau2024`, `ChenEtAl2023` | the LLC at scale (merged into later versions of the LLC paper); dynamical versus Bayesian phase transitions in the toy model of superposition (arXiv 2310.06301; rejected at ICLR 2024, no venue — cite as a preprint; digest 03 calls it `ChenEtAl2023a`) |
| `HooglandEtAl2025` | "Loss landscape degeneracy and stagewise development in transformers", TMLR 2025 (arXiv 2402.02364; digest 04's `HooglandEtAl2024` is the same work — use this key) |
| `WangEtAl2025a`, `HitchcockHoogland2025`, `PepinLehalleurEtAl2025`, `UrdshalsEtAl2025`, `BakerEtAl2026` | refined LLCs (ICLR 2025); the local-sampling benchmark (preprint, under review); the programme's position paper; MDL and the LLC; susceptibilities (ICLR 2026) |
| `Aoyagi2024`, `Aoyagi2025`, `PepinLehalleurRimanyi2024`, `NagayasuWatanabe2023` | deep linear networks: the exact learning coefficient (*Neural Networks* 172, 106132); its restatement; fibres of the multiplication map; free energy of overparameterized deep ReLU networks (arXiv) |
| `ElhageEtAl2022`, `OlssonEtAl2022`, `GargEtAl2022` | toy models of superposition; induction heads; in-context linear regression |
| `Sussmann1992`, `ChenEtAl1993`, `AlbertiniSontag1993`, `FarrugiaRoberts2023a`, `FukumizuAmari2000`, `WeiEtAl2008`, `CousseauEtAl2008`, `WatanabeAmari2003` (truths at distance of order $1/n$ from the singular ones; added by Ch. 12's referee) | symmetries and fibres of networks; plateaus and dynamics near singularities (several are verified bibliographically only — digest 03 §8) |
| `MacKay1992a`, `MacKay2003`, `Balasubramanian1997` | the Bayesian Occam factor; MacKay's book (Ch. 28); Occam's razor and statistical mechanics |
| `DinhEtAl2017`, `MandtEtAl2017`, `IzmailovEtAl2021`, `SaxeEtAl2014`, `KuninEtAl2021` | sharp minima can generalize; SGD as approximate Bayesian inference; posteriors of real networks; exact dynamics of deep linear networks; symmetries and conservation laws |
| `Murfet2023a` | Murfet's reply to Skalse (LessWrong comment, 2023-11-20) |
| `Carroll2023`, `Skalse2023`, `Erdil2023`, `Bushnaq2024` | LessWrong: the DSLT sequence (digest 04's entry; the single post "DSLT 2" is `Carroll2023b`); two critiques; the Hessian-rank bound |
| `Schwarz1978`, `Akaike1974`, `Haughton1988` | BIC; AIC; BIC for curved families and its consistency |
| `vanderVaart1998`, `ClarkeBarron1990`, `KassRaftery1995`, `TierneyKadane1986` | asymptotic statistics; Bayes asymptotics; Bayes factors; Laplace approximations |
| `Hartigan1985`, `Fukumizu1996`, `Fukumizu2003`, `SelfLiang1987`, `Davies1977`, `LiuShao2003`, `DacunhaCastelleGassiat1999`, `Drton2009` | classical failures of regular theory |
| `AmariEtAl2006` | Amari, Park and Ozeki, "Singularities affect dynamics of learning in neuromanifolds" |
| `GelmanMeng1998`, `FrielPettitt2008`, `VehtariEtAl2017`, `GelmanEtAl2014` | path sampling; power posteriors; PSIS-LOO; predictive information criteria (digest 01: `VehtariGelmanGabry2017`, `GelmanHwangVehtari2014`) |
| `WellingTeh2011`, `MacKay1992` | SGLD; Bayesian Occam factor |

| `Lin2017`, `Greenblatt2010`, `Saito2007`, `Kollar1997`, `ArnoldEtAl1988` | ideal-theoretic RLCTs, Newton polyhedra and the Hessian bound (*J. Algebraic Statistics*); sublevel volumes and Newton polyhedra; real log canonical thresholds; singularities of pairs; Arnold, Gusein-Zade and Varchenko, *Singularities of Differentiable Maps* II |
| `AoyagiNagata2012`, `Aoyagi2010a`, `Aoyagi2019`, `RusakovGeiger2005`, `Hayashi2021`, `KariyaWatanabe2022`, `YamazakiWatanabe2005a` | Vandermonde-type singularities; restricted Boltzmann machines; further network results; naive Bayes networks; LDA; two-component mixtures; HMMs (check each entry's content in digest 02 §5 before quoting) |
| `Watanabe2021`, `Watanabe2021b` | "WAIC and WBIC for mixture models", *Behaviormetrika* 48 (2021); "Information criteria and cross validation for Bayesian inference in regular and singular cases", *JJSDS* 4 (2021) (the coordinator renamed digest 01's key for the latter) |

Digest 03 calls Watanabe's 2001 *Neural Computation* paper `Watanabe2001` and the 2024 review
`Watanabe2022`; digest 02 uses `BernshteinGelfand1969`, `NagayasuWatanabe2023a/b` and
`HooglandEtAl2024`: use the keys in this table (`Watanabe2001a`, `Watanabe2024`,
`BernsteinGelfand1969`, `NagayasuWatanabe2023`, `HooglandEtAl2025`).

## Chapter specifications

Each specification gives the opening, the question with its regular answer, the hand computations,
the theorems (main results with proof plans), the script for "Measuring it", the expected ledger
card, where the theory stops, exercise ideas, cross-links and key references. "Verify" marks a fact
that the chapter author must check against the source before printing. The author may improve
every proposal here; where the chapter and the plan differ, the chapter is correct, and the author
reports the difference.

Several proofs below are proposed by the coordinator and checked numerically, not found in a
source (marked **book's own**). Their authors must check them line by line and say in the notes
that the argument is the book's; if a published proof exists, cite it.

---

### Prologue — Two parameters (key: prologue; special chapter)

Written by the coordinating author. Opens with the model $y=abx$: two numbers multiplied together,
fitted to data that are pure noise. Draws of the posterior (Figure P.1; the title page's emblem is
a separate grid sample at $n=400$ from `00-emblem.py`) crowd onto the cross $ab=0$. Classical statistics would charge two parameters $\log n$; the
measured free energy grows like $\tfrac12\log n-\log\log n$ (Figure P.2, from
`figures/scripts/00-prologue-free.py`). The two numbers $\tfrac12$ and $m=2$ are the book's subject:
where they come from (geometry: the volume of nearly true parameters), what they control (the free
energy, the generalization error, the information criteria), how to compute them (blow-ups, Newton
polygons), and what they say and do not say about trained networks. The running models and the
ledger; the anatomy of a chapter; prerequisites; how to read (Part I is the whole theory in a
special case; Part II is geometry; Part III is statistics; Part IV is deep learning; Chapters 1–3,
4, 8–9, 13, 15 form a short course). Ten to twelve pages.

### Part I. Where Regular Theory Ends

Part blurb (in `main.tex`). The part establishes the classical baseline, shows how the cast of toy
models violates it, and computes the free energy of the one-coefficient models completely, which
is the whole theory in miniature.

### Chapter 1. Regular Models — What the Laplace approximation promises (key: regular)

**Opening.** `\sltchapter{Regular Models}{What the Laplace approximation promises}{$y=ax_1+bx_2$}{ch:regular}` and `\newtonfoot{0/2,2/0}{1}{1}{a^2+b^2}`.

**The question.** What does a parameter cost? A Bayesian compares models by their marginal
likelihood; a frequentist by an estimate of test loss. For regular models both have classical
answers with a $d$ in them. Regular answer box: $F_n=nS_n+\tfrac d2\log n+O_p(1)$ (Schwarz 1978)
and $\E[G_n]-S=\tfrac d{2n}+o(1/n)$ (Akaike 1974 in its Bayesian form).

**By hand.** (1) The regular plane $y=ax_1+bx_2$ with a Gaussian prior: the posterior, the
marginal likelihood and the predictive are Gaussian and exact; read off $\tfrac d2\log n$ and
$\tfrac d{2n}$ with all constants. (2) The same model with the uniform prior on $[-1,1]^2$: the
Laplace approximation and its error (the boundary is exponentially far). (3) Where the $\tfrac d2$
comes from: each direction in which the posterior has width $n^{-1/2}$ contributes $\tfrac12\log n$.

**The theorem.** `thm:regular:laplace` (complete proof): for a model with compact $W$, a unique
minimizer $w_0$ of $L$ in the interior, $\varphi$ continuous and positive at $w_0$, $\log p(x\mid w)$
three times continuously differentiable near $w_0$ with an integrable envelope for the derivatives,
the Hessian $J=\nabla^2L(w_0)$ positive definite, and uniform convergence $\sup_W\abs{L_n-L}\to0$ in
probability: $F_n=nL_n(\hat w)+\tfrac d2\log\tfrac n{2\pi}+\tfrac12\log\det J-\log\varphi(w_0)+o_p(1)$,
with $\hat w$ the maximum-likelihood estimate. Plan: localize to a ball of radius $n^{-1/2+\delta}$
around $\hat w$ by uniform convergence and identifiability; inside, Taylor expansion with the
third-derivative envelope; outside, an exponentially small contribution. The author may state the
smoothness hypotheses in the most convenient equivalent form. Then `thm:regular:aic`: for regular
realizable models, $\E[G_n]-S=\tfrac d{2n}+o(1/n)$ — complete proof for the Gaussian linear
model (exact computation); the general statement with a proof sketch (expand the predictive around
$\hat w$; cite). State Bernstein–von Mises (van der Vaart 1998, Theorem 10.1; verify number and
hypotheses) and BIC's consistency for nested regular models (Haughton 1988; verify).

**Measuring it.** `figures/scripts/01-regular-free.py`: $F_n-nS_n$ for the regular line and the
regular plane by quadrature (`sk.free_energy_grid`), averaged over 200 data sets for
$n=10^2,\dots,10^6$, against $\tfrac d2\log n+\text{const}$ with the constant of the theorem;
`01-regular-gen.py`: $n(\E[G_n]-S)$ against $\tfrac d2$. Both agree to within the intervals —
the regular theory is exact here, and the book's later figures carry its slate dashed line as the
reference.

**Ledger.** Quantity: the free energy and the generalization error. Regular: $\tfrac d2\log n$,
$\tfrac d{2n}$. Singular: identical here ($\lambda=\tfrac d2$, $m=1$; a regular model is the special
case). Running example: the regular plane, $\lambda=1$. Measured: slopes. Rests on: identifiability,
positive definite Fisher information, interior truth, smoothness.

**Where the theory stops.** Assumed: identifiability, invertible Fisher information, an interior
truth. Breaks: mixtures (Hartigan 1985: the likelihood ratio statistic diverges), the boundary of
the parameter space (Self and Liang 1987: mixtures of $\chi^2$; verify), neural networks (Fukumizu
2003). Open: none here — the questions are the book's.

**Exercises.** Laplace for an exponential family; BIC's derivation; the ridge prior; the Laplace
error term; posterior normality fails at the boundary; a variation with unknown noise variance; a
measurement of the Laplace error at small $n$.

**Cross-links.** Every later chapter compares with this one. **Key references.** Schwarz1978,
Akaike1974, vanderVaart1998, TierneyKadane1986, KassRaftery1995, Haughton1988, ClarkeBarron1990.

### Chapter 2. Singular Models — When parameters fold (key: singular)

**Opening.** `\sltchapter{Singular Models}{When parameters fold}{$y=a\tanh(bx)$}{ch:singular}` and `\newtonfoot{2/2}{2}{\tfrac12}{a^2b^2}`.

**The question.** Two networks with different weights can compute the same function: permute hidden
units, flip signs of tanh units, rescale the two sides of a ReLU, or switch a unit off by zeroing its
outgoing weight. What does the set of parameters that realize the truth look like, and what happens
to the posterior on it? Regular answer box: the posterior concentrates at one point and is
asymptotically normal with covariance $(nJ)^{-1}$.

**By hand.** (1) The true sets of the cast (table in OUTLINE.md): the cross of the product model,
the hyperbola of a generic truth, the cross of the tanh and ReLU neurons (derive $K$ for both;
the ReLU neuron's $K=a^2b^2/4$ exactly), the two lines of the mixture (derive the expansion
$K\approx\tfrac{a^2}2(e^{b^2}-1)$; verify), rank-one regression. (2) Fisher information of the
product model: $I(a,b)=\begin{pmatrix}b^2&ab\\ab&a^2\end{pmatrix}$, singular everywhere and zero at
the origin. (3) The posterior of the product model at the singular truth, exactly: the marginal of
$a$ is a ratio of Gaussian distribution functions; it is flat in $\log\abs a$ between $n^{-1/2}$
and $1$.

**The theorem.** `prop:singular:fisher` (complete proof): if $W_0$ contains a $C^1$ curve through
$w_0$ with nonzero tangent $v$, then $I(w_0)v=0$. `thm:singular:scales` (complete proof, book's own;
checked numerically): for the product model at the singular truth with the uniform prior on
$[-1,1]^2$, for every $u\in[-\tfrac12,0]$, the posterior probability of $\{\log\abs a\le u\log n\}$
converges in probability to $2u+1$. Plan: the marginal posterior density of $a$ is
$\propto\abs a^{-1}[\Phi((\abs a-\hat c)/s)-\Phi((-\abs a-\hat c)/s)]$ with $s=S_{xx}^{-1/2}$,
$\hat c=S_{xy}/S_{xx}$; bound it above and below by multiples of $\min(1,\,s/\abs a)$ on the
appropriate ranges, integrate in $\log\abs a$. Corollaries: the posterior is not asymptotically
normal in any smooth reparametrization near the origin; the posterior mean and the MAP behave
differently. The symmetries and fibres of networks (digest 03 §1.1): generic symmetries
(permutations, sign flips of odd activations, positive rescaling of ReLU units) and non-generic
degeneracies (dead outputs, constant units, duplicated units, inactive ReLU units); Sussmann's
uniqueness theorem for minimal one-hidden-layer tanh networks (1992) and the equioutput group of
Chen, Lu and Hecht-Nielsen (1993) — the digest verified these only bibliographically, so state them
as standard and cite carefully, or fetch the papers; Farrugia-Roberts (2023) on functional
equivalence classes of reducible networks. Plateaus near singularities (Fukumizu and Amari 2000;
Amari, Park and Ozeki 2006; Wei et al. 2008; Cousseau et al. 2008) as the dynamical side, which
Chapter 15 returns to. Fukumizu (1996)'s characterization of a nondegenerate Fisher information is
unverified in wording (digest 03 §8): verify or do not quote it.

**Measuring it.** `02-singular-scales.py`: posterior draws (`sk.sample_grid`) of the product model
at $n=10^2,10^4,10^6$ (a scatter figure with the `draws` style) and the distribution of
$\log\abs a/\log n$ against $\Unif(-\tfrac12,0)$ (note the slow convergence; show the exact finite-$n$
prediction from the formula); `02-singular-zoo.py`: the posteriors of the tanh neuron and the mixture
at $n=10^3$ on a grid, which look like the product model's.

**Ledger.** Quantity: the shape of the posterior and of $W_0$. Regular: a point, a Gaussian of width
$n^{-1/2}$, invertible Fisher information. Singular: $W_0$ a cross; Fisher information singular
everywhere; the posterior uniform over scales. Running example: the product model and the tanh
neuron. Measured: quartiles of $\log\abs a/\log n$. Rests on: realizability, the uniform prior,
$x\sim\Normal(0,1)$.

**Where the theory stops.** Assumed: realizability; analytic models (ReLU is not analytic, though
the ReLU neuron's $K$ happens to be a polynomial); a compact $W$ with a positive prior. Breaks: in
large networks $W_0$ has many components of different dimensions; the prior matters (a prior that
vanishes on the cross changes the picture). Open: the geometry of $W_0$ for deep networks (e.g.
work on functional dimension of ReLU networks; verify sources), the structure of its singularities.

**Exercises.** $K$ for the tanh neuron to order $b^4$; the Fisher information of a two-unit tanh
network; the posterior of the generic-truth product model (it concentrates on the hyperbola with
which density?); the mixture's $K$; a variation with a Gaussian prior; Sussmann-type symmetries for
a two-unit network; a measurement of the scales law with the chain $L=3$.

**Key references.** Watanabe2009, Watanabe2018, Sussmann1992, FukumizuAmari2000, AmariParkOzeki2006,
Fukumizu2003, Hartigan1985, WeiEtAl2022.

### Chapter 3. The Free Energy by Hand — The whole theory in one coefficient (key: coefficient)

**Opening.** `\sltchapter{The Free Energy by Hand}{The whole theory in one coefficient}{$y=abx$, $y=ab^2x$}{ch:coefficient}` and `\newtonfoot{2/4}{4}{\tfrac14}{a^2b^4}`.

**The question.** What is the marginal likelihood of $y=abx$? Regular answer box: BIC charges two
parameters $\log n$.

**By hand.** (1) The energy of a one-coefficient model $y=g(w)x+\text{noise}$ depends on $w$ only
through $c=g(w)$: $nL_n(w)=\tfrac12S_{xx}(c-\hat c)^2+\text{const}$, so $F_n=-\log\int
e^{-nL_n(c)}\rho(c)\dd c$ with $\rho$ the density of $c$ under the prior (computation box).
(2) The density of $ab$ for independent uniforms is $\tfrac12\log\frac1{\abs c}$; of
$a_1\cdots a_L$, $(\log\frac1{\abs c})^{L-1}/(2(L-1)!)$ (proof by induction or Mellin transforms);
of $ab^2$, $\tfrac12(\abs c^{-1/2}-1)$; of $ab^3$ (compute). (3) A Gaussian of width $n^{-1/2}$
against these densities: $\tfrac12\log n-\log\log n$ for $y=abx$, $\tfrac14\log n$ for $y=ab^2x$.
The exponent of $\abs c$ in $\rho$ gives $\lambda$; the power of the logarithm gives $m-1$.

**The theorem.** `thm:coefficient:free` (complete proof): let $y=g(w)x+\varepsilon$ with
$\E[x^2]=1$, finite fourth moments, Gaussian noise of variance 1, truth $c_0=g(w_0)$, and suppose
the density of $c=g(w)$ under $\varphi$ satisfies $\rho(c)=\abs{c-c_0}^{2\lambda-1}
(\log\frac1{\abs{c-c_0}})^{m-1}\ell(c)$ near $c_0$ with $\ell$ continuous and positive at $c_0$ (from
both sides, or one side with the obvious change). Then $F_n-nS_n=\lambda\log n-(m-1)\log\log
n+O_p(1)$, and $F_n-nS_n-\lambda\log n+(m-1)\log\log n$ converges in law to an explicit random
variable (a function of $Z\sim\Normal(0,1)$; give it). Plan: substitute $t=\sqrt n(c-c_0)$,
use $\sqrt n(\hat c-c_0)\to Z$ and dominated convergence; the logarithm becomes
$(\tfrac12\log n+\log\frac1{\abs t})^{m-1}$. Corollaries: the product model ($\lambda=\tfrac12$,
$m=2$), the chain ($\tfrac12$, $L$), the cusps ($\tfrac14$, $\tfrac16$), the regular line
($\tfrac12$, 1). Remark: the density $\rho$ is the state density of Chapter 4 in disguise.

**Measuring it.** `03-coefficient-free.py`: exact quadrature (`sk.free_energy_chain`,
`sk.coefficient_free_energy`) of $F_n-nS_n$ for the regular line, the product model, the chain
$L=3$, the cusp and the cubic cusp, $n=10^2,\dots,10^8$, 200 data sets each; plot against the
prediction with the theorem's limit constant and against BIC's $\tfrac d2\log n$. The coordinator's
check found the $m\ge2$ curves converge slowly (the gap shrinks like $1/\log n$); show the exact
finite-$n$ prediction (keep the slowly varying factor) as a second bronze curve.

**Ledger.** Quantity: the free energy. Regular: $\log n$ for two parameters. Singular:
$\lambda\log n-(m-1)\log\log n$. Running example: product $\tfrac12,2$; cusp $\tfrac14,1$.
Measured: (numbers). Rests on: one coefficient, Gaussian noise, a prior density with the stated
behaviour at $c_0$.

**Where the theory stops.** Assumed: the energy depends on one scalar; a fixed truth as $n$ grows.
Breaks: a truth at distance $n^{-1/2}$ from the singular point (Chapter 12); several coefficients
(Chapter 4 replaces $\rho$ by the volume function). Open: none specific; point to Chapter 8.

**Exercises.** The density of $ab^3$; the free energy with a Gaussian prior on $(a,b)$ (the
density of a product of Gaussians has a logarithmic singularity too: compute it); the limit law;
a variation with unknown noise variance; the chain with a generic truth; a measurement of how slowly
the $m=3$ curve converges.

**Key references.** Watanabe2009, Watanabe2018, AoyagiWatanabe2005 (for the chain), Drton et al.
for densities of products (verify).

### Part II. The Learning Coefficient

### Chapter 4. Volume — The learning coefficient as a scaling exponent (key: volume)

**Opening.** `\sltchapter{Volume}{The learning coefficient as a scaling exponent}{$y=a^2x_1+b^2x_2$}{ch:volume}` and `\newtonfoot{0/4,4/0}{2}{\tfrac12}{a^4+b^4}`.

**The question.** What replaces the coefficient density $\rho$ when the energy depends on several
coefficients? The volume of nearly true parameters, $V(\eps)=\varphi\{K<\eps\}$. Regular answer box:
near a nondegenerate minimum $\{K<\eps\}$ is an ellipsoid, $V(\eps)\sim C\eps^{d/2}$.

**By hand.** $V(\eps)$ exactly for $a^2b^2$ ($t(1+\log\frac1t)$ with $t=\sqrt{2\eps}$ for
$K=a^2b^2/2$; check), for the quartic cup $a^4+b^4$ (homogeneous: an exact power), for $a^2+b^4$;
the state density $v=V'$; the zeta function $\zeta(z)=\int K^z\varphi=\int_0^\infty t^z\,\dd V(t)$ of a
monomial on the unit box, $\prod_j(2k_jz+h_j+1)^{-1}$ up to constants, with its poles; the
population free energy $\bar F(n)=-\log\int e^{-nK}\varphi$ as the Laplace transform of $\dd V$.

**The theorem.** `lem:volume:abelian` (complete proof): if $V(\eps)\sim C\eps^\lambda
(\log\frac1\eps)^{m-1}$ as $\eps\to0$, then $\int e^{-nK}\varphi\sim C\,\Gamma(\lambda+1)\,n^{-\lambda}
(\log n)^{m-1}$; hence $\bar F(n)=\lambda\log n-(m-1)\log\log n+O(1)$. `prop:volume:monomial`
(complete proof): for $K=\prod_jw_j^{2k_j}$ and $\varphi=b(w)\prod_j\abs{w_j}^{h_j}$ ($b$ continuous,
positive) on a box around $0$, $\lambda=\min_j\frac{h_j+1}{2k_j}$ (over $j$ with $k_j>0$) and $m$ is
the number of minimizing $j$; with $V(\eps)\sim C\eps^\lambda(\log\frac1\eps)^{m-1}$ and an explicit
$C$. `thm:volume:zeta`: the largest pole of $\zeta$ and the asymptotics of $V$ determine each other —
complete proof of the direction from $V$ to $\zeta$ (a Mellin transform of a regularly varying
measure), the converse sketched (Mellin inversion and a residue at $-\lambda$; cite Watanabe 2009 and
Arnold–Gusein-Zade–Varchenko; verify), with the fact that for analytic $K$ the zeta function is
meromorphic with poles at negative rationals (Atiyah 1970, Bernstein–Gel'fand 1969; statement).
History: Gel'fand's question (verify where and when posed), Atiyah, Bernstein, Bernstein–Sato
polynomials (one paragraph).

**Measuring it.** `04-volume-mc.py`: Monte Carlo volume fractions (`sk.volume_fraction`) for the
quartic cup, the product, the quartic and the sextic models, $\eps$ from $10^{-1}$ to $10^{-8}$
(say how many draws are needed and why), on log–log axes against $C\eps^\lambda
(\log\frac1\eps)^{m-1}$; numerical $\zeta(z)$ for real $z$ near the pole.

**Ledger.** Quantity: the learning coefficient as a volume exponent. Regular: $d/2$. Singular:
$\lambda$ (and $m$ as the power of the logarithm). Running example: the quartic cup, $\lambda=\tfrac12$
with $d/2=1$. Measured: fitted exponents. Rests on: $K\ge0$ analytic (or the monomial form), a
prior positive near $W_0$.

**Where the theory stops.** Assumed: we know $K$. In practice $K$ is unknown and high-dimensional;
volumes cannot be computed by Monte Carlo when $\lambda$ is small relative to $d$ and $\eps$ tiny
(the fraction is $\eps^\lambda$). Breaks: priors that vanish on $W_0$ change $\lambda$ (show).
Open: efficient volume estimation in high dimension (point to Chapter 13).

**Exercises.** Volumes for $a^2+b^2+c^2$, for $(a^2+b^2)^2$, for the chain; the zeta function of
$a^2b^4$ with a prior $\abs a$; Karamata's Tauberian converse (statement and use); a variation with
a Gaussian prior; a measurement of the Monte Carlo cost.

**Key references.** Watanabe2009 (Chapters 2–4; verify), Atiyah1970, BernsteinGelfand1969,
Bernstein1972, ArnoldEtAl1988 (Singularities of Differentiable Maps II; verify), BinghamEtAl1987
(regular variation; verify).

### Chapter 5. Blow-Ups — Resolution of singularities by hand (key: blowups)

**Opening.** `\sltchapter{Blow-Ups}{Resolution of singularities by hand}{$y=ax_1+b^2x_2$}{ch:blowups}` and `\newtonfoot{0/4,2/0}{4/3}{\tfrac34}{a^2+b^4}`.

**The question.** How do we compute $\lambda$ when $K$ is not a monomial, e.g. $a^2+b^4$? By changing
coordinates until it is. Regular answer box: the Morse lemma — near a nondegenerate minimum,
$K=\sum u_i^2$ in suitable coordinates, so $\lambda=d/2$; the regular case is "already resolved".

**By hand.** The blow-up of the origin of the plane in two charts, $(a,b)=(u,uv)$ and $(uv,v)$, with
Jacobians $\abs u$ and $\abs v$, drawn (TikZ: the exceptional divisor as a line of directions). Then:
$(ab)^2$ (already normal crossing); $(a^2+b^2)^2$ (one blow-up; $\lambda=\tfrac12$); $a^2+b^4$ (two
blow-ups, or one weighted blow-up $(a,b)=(uv^2,v)$; $\lambda=\tfrac34$) — computation boxes; the
product model with a generic truth (local coordinates on the hyperbola; $\lambda=\tfrac12$, $m=1$);
the cusp $K=(a^2-b^3)^2$, which needs three point blow-ups and has $\lambda=\tfrac5{12}$, $m=1$ (digest
02 §2.7, with the charts written out); Watanabe's three-parameter example $K=(ab+c)^2+a^2b^4$,
$\lambda=\tfrac34$ (digest §2.8). The form of Hironaka's theorem that the book uses is Watanabe's
"standard representation" (Watanabe 2013, Theorem 1, whose proof is given in Theorem 6.1 of the 2009
book according to that paper; digest 02 §1); Atiyah's local form is what Lin's thesis says Watanabe
(2009) uses as Theorem 2.3 (secondary). Cite them that way.

**The theorem.** `thm:toolkit:hironaka` is stated in the appendix (black box); here
`thm:blowups:pullback` (complete proof): if $g\colon U\to W$ is proper, real analytic, an analytic
isomorphism outside sets of measure zero, and in local coordinates $K\circ g=u^{2k}a(u)$,
$\varphi(g(u))\abs{\det g'(u)}=\abs{u^h}b(u)$ with $a,b>0$, then $\zeta_K(z)=\int_U(K\circ g)^z
(\varphi\circ g)\abs{\det g'}\dd u$, and $\lambda$, $m$ are the minimum of $\frac{h_j+1}{2k_j}$ over all
charts and coordinates and the largest number of coordinates achieving it at one point. Plan: change
of variables, a finite partition of unity (compactness and properness), `prop:volume:monomial` on
each piece. `prop:blowups:rules` (complete proofs): invariance under analytic isomorphisms;
comparison ($c_1K_1\le K_2\le c_2K_1$ gives equal $\lambda,m$; $K_1\le CK_2$ gives
$\lambda(K_1)\le\lambda(K_2)$); sums of functions of separate variables ($\lambda$ adds, $m=m_1+m_2-1$);
products ($\min$, multiplicities add when equal); $\lambda\le d/2$ when $K$ is $C^2$ at an interior
$w_0$ with $\varphi(w_0)>0$; the global $\lambda$ is the minimum of the local ones.

**Measuring it.** `05-blowups-check.py`: for $a^2+b^4$ and two more functions, the numerical zeta
function and volume in the original coordinates and on the blown-up charts agree, and both give
the predicted $\lambda$; a figure of the posterior of the quartic model in the original and in the
blown-up coordinates (where it becomes a product of simple densities).

**Ledger.** Quantity: $\lambda$ by resolution. Regular: $d/2$ (Morse). Singular: $\lambda$ from the
normal crossing form. Running example: $a^2+b^4$, $\tfrac34$. Measured. Rests on: analyticity and
Hironaka's theorem.

**Where the theory stops.** Assumed: analytic $K$ (Hironaka needs it; ReLU networks are piecewise
analytic). Breaks: resolution is existential; algorithms exist (Villamayor; Bierstone–Milman;
verify) but are infeasible for high-dimensional losses. Open: computing $\lambda$ for networks of
moderate size.

**Exercises.** Blow up $a^3-b^2$, $a^2b^2+a^4$, $(a^2+b^2)^3$, the chain with $L=3$ (no blow-up
needed), $(ab-cd)^2$ near the origin of $\R^4$ (hard: give hints); the multiplicity of $a^2b^2+a^4b^4$;
show $\lambda\le d/2$ fails if $\varphi(w_0)=0$.

**Key references.** Hironaka1964a, Hironaka1964b, Watanabe2009, Lin2011, Sullivant2018 (chapter on Bayesian
integrals; verify), Kollar1997 (log canonical thresholds; verify), Saito2007.

### Chapter 6. Newton Polygons — The toric shortcut (key: newton)

**Opening.** `\sltchapter{Newton Polygons}{The toric shortcut}{$y=a^2x_1+b^3x_2$}{ch:newton}` and `\newtonfoot{0/6,4/0}{12/5}{\tfrac5{12}}{a^4+b^6}`.

**The question.** Is there a way to read $\lambda$ off the exponents of $K$ without blowing up?
Regular answer box: for a quadratic $K$ the exponents are all $2$ and $\lambda=d/2$.

**By hand.** Newton diagrams (`\newtondiagram`) of the cast; $a^4+b^6$: $t=12/5$, $\lambda=5/12$;
a vertex hit ($a^6+a^2b^2+b^6$: $m=2$); the degenerate counterexample $(a-b)^2$ (the diagram says
$\lambda=1$, the truth is $\tfrac12$) and why (the face polynomial has a singular zero in the torus);
$(a^2-b^3)^2$ (degenerate for $K$ but correct through $f=a^2-b^3$; verify).

**The theorem.** `thm:newton:plane` (complete proof, book's own route; verify it against the
literature): for $K=\sum_{v\in S}c_va^{v_1}b^{v_2}$ with $c_v>0$ and all exponents even, on a box
with a prior positive at $0$, $\lambda=1/t$ where $(t,t)$ is the meeting point of the diagonal with
the Newton boundary, and $m=2$ if that point is a vertex, $1$ otherwise. Plan: $K\asymp\max_va^{v_1}
b^{v_2}$; in logarithmic coordinates $a=e^{-s},b=e^{-u}$ the set $\{K<\eps\}$ becomes the polyhedron
$\{v\cdot(s,u)\ge\log\frac1\eps+O(1)\ \forall v\}$ and the volume an integral of $e^{-(s+u)}$ over
it; the exponent is the value of a linear program, $L/t$, by LP duality, and the power of $L$ is the
dimension of the optimal face. Then the general statements (digest 02 §3): in **every**
coordinate system $\lambda\le1/t$ (Greenblatt 2010, Theorem 1.2(a)); equality holds when $K$ is
nondegenerate, meaning that no compact-face polynomial vanishes on the real torus $(\R^*)^d$
(Greenblatt 2010, Theorem 1.6; Lin 2017, Theorem 1.3; Varchenko 1976 for the analytic origin, whose
condition $t>1$ matters only for oscillatory integrals with a sign-changing phase, not for $K\ge0$);
"$\lambda$ depends only on the ideal" (Lin 2017, Proposition 3.6), stated exactly. Counterexamples:
$(a-b)^2$ (the diagram says $1$, the truth is $\tfrac12$) and $(a-b^2)^2$ (the diagram says $\tfrac34$,
the truth is $\tfrac12$); and the cusp $(a^2-b^3)^2$, degenerate as written but with
$\lambda=\tfrac5{12}$, the same as the chapter's nondegenerate emblem $a^4+b^6$.

**Measuring it.** `06-newton-random.py`: for 30 random sums of even monomials in two variables, the
Newton prediction against Monte Carlo volume exponents; the degenerate examples off the line.

**Ledger.** Quantity: $\lambda$ from exponents. Regular: $d/2$. Singular: $1/t$. Running example:
$a^4+b^6$, $\tfrac5{12}$. Measured. Rests on: non-degeneracy in the chosen coordinates.

**Where the theory stops.** Assumed: non-degeneracy, which depends on coordinates; network losses
are often degenerate in their natural coordinates ($(ab-cd)^2$). Breaks: show a network-like loss
where Newton fails. Open: choosing coordinates (or a toric change) that makes a given loss
non-degenerate.

**Exercises.** Newton for $a^2b^2+a^6+b^6$, for the cusps, for a three-variable example;
non-degeneracy checks; the LP proof in three variables; when a weighted blow-up resolves.

**Key references.** Varchenko1976, Lin2011, ArnoldEtAl1988, Watanabe2009, Sullivant2018.

### Chapter 7. A Catalogue — Learning coefficients of models in use (key: catalogue)

**Opening.** `\sltchapter{A Catalogue}{Learning coefficients of models in use}{$y=BAx$}{ch:catalogue}` and `\newtonfoot{2/2}{2}{\tfrac12}{a^2b^2}`.

**The question.** What are the learning coefficients of the models people use: reduced-rank
regression, deep linear networks, mixtures, shallow networks, Boltzmann machines? Regular answer box:
half the number of parameters.

**By hand.** Rank-one regression $y=b\,a^\top x$ at the zero truth: $K=\norm a^2\norm b^2/2$; polar
coordinates in $a$ and in $b$; $\lambda=\min(M,N)/2$; the chain; the tanh neuron.

**The theorem.** `thm:catalogue:rankone` (complete proof). `thm:catalogue:aoyagi`: the Aoyagi–Watanabe
(2005) formula for reduced-rank regression with $H$ hidden units and a truth of rank $r$ (exact
statement with its case distinctions from digest 02 §5.1 — the digest worked from restatements; say
so), checked against the rank-one value and the digest's small cases ($M=N=2$, $r=1$: $3/2$, $2$;
$M=N=3$, $r=1$: $5/2$, $7/2$, $4$). Then the catalogue (digest 02 §5, with each entry's status):
deep linear networks (Aoyagi 2024; the multiplicity is $a(\ell-a)+1$ and can exceed $2$); three-layer
networks with analytic activations and Vandermonde-type singularities (Aoyagi and Nagata 2012 and
related; inputs on $[-1,1]$); normal mixtures (Yamazaki and Watanabe 2003 bounds; Kariya and Watanabe
2022 for the two-component case); restricted Boltzmann machines (Aoyagi 2010 — the digest found two
entries that look wrong by its own computation: check before quoting); naive Bayesian networks
(Rusakov and Geiger 2005); LDA (an exact formula, Hayashi 2021) and NMF (bounds); ReLU networks
(only free-energy upper bounds are published, Nagayasu and Watanabe; exact three-layer values are
claimed in an unrefereed 2025 preprint by Aoyagi). A table of $\lambda$ versus $d/2$ for small
instances. The singular BIC (Drton and Plummer 2017): how these values enter model selection
(statement and a worked example). A pre-asymptotic warning (digest 02 §8.3): for a tanh network with
two hidden units, plain Monte Carlo sees $\lambda\approx0.5$ down to $\eps=10^{-5}$ while the true value
is $2/3$.

**Measuring it.** `07-catalogue-rrr.py`: for reduced-rank regression with $M=N=3$ and $H=1,2$ and
truths of rank $0,1$, estimate $\lambda$ by the two-temperature estimator with Metropolis
(`sk.metropolis`; validate on the regular plane first) and compare with the Aoyagi–Watanabe value
and with $d/2$.

**Ledger.** Quantity: $\lambda$ for models in use. Regular: $d/2$. Singular: the formulas. Running
example: rank-one regression. Measured. Rests on: realizability, the truth's rank, the prior.

**Where the theory stops.** Exact values are known for few architectures; deep nonlinear networks
have only bounds. Open problems with citations.

**Exercises.** Rank-one with $M\ne N$; the formula for $H=2$, $r=1$, small $M,N$; mixtures of two
Bernoullis; an upper bound by a sub-model; sBIC computation.

**Key references.** AoyagiWatanabe2005, Aoyagi2024 (verify), YamazakiWatanabe2003, Watanabe2001,
Aoyagi2010, RusakovGeiger2005, DrtonPlummer2017.

### Part III. Watanabe's Theorems

### Chapter 8. The Free Energy Formula — Model selection without the Laplace approximation (key: freeenergy)

**Opening.** `\sltchapter{The Free Energy Formula}{Model selection without the Laplace approximation}{$y=ax_1+b^3x_2$}{ch:freeenergy}` and `\newtonfoot{0/6,2/0}{3/2}{\tfrac23}{a^2+b^6}`.

**The question.** The marginal likelihood of a singular model in general. Regular answer box: BIC.

**By hand.** For the product model, $nK_n(w)=t-\sqrt{2t}\,\xi_n$ with $t=nK(w)$-like and $\xi_n$ a
Gaussian variable: the fluctuation is at most of the order of the square root of the energy; the
sandwich $e^{-(1+\eta)t-\xi^2/(2\eta)}\le e^{-nK_n}\le e^{-(1-\eta)t+\xi^2/(2\eta)}$ (compute).

**The theorem.** `thm:freeenergy:linear` (complete proof; book's own presentation of a standard
argument): for $y=g(w)^\top x+\varepsilon$ with $x\sim\Normal(0,\Sigma)$ (or finite fourth moments and
$\Sigma\succ0$), Gaussian noise, $W$ compact, $\varphi$ bounded, and $K(w)=\tfrac12(g(w)-g(w_0))^\top
\Sigma(g(w)-g(w_0))$ with $V(\eps)\sim C\eps^\lambda(\log\frac1\eps)^{m-1}$: $F_n-nS_n=\lambda\log
n-(m-1)\log\log n+O_p(1)$. Plan: $nK_n=\tfrac n2\delta^\top\Sigma_n\delta-\sqrt n\,\delta^\top Z_n$ with
$\delta=g(w)-g(w_0)$, $Z_n\to\Normal(0,\Sigma)$; Cauchy–Schwarz and $2xy\le\eta x^2+y^2/\eta$ give the
sandwich; `lem:volume:abelian` applied to $(1\pm\eta)K$. `thm:freeenergy:main` (statement with
exact hypotheses, realizable and with relatively finite variance; proof sketch: resolution,
`thm:toolkit:empirical`, the same sandwich; the finer convergence in law; verify the theorem
numbers in Watanabe 2009/2018). Consequences: BIC overcharges singular models; the singular BIC
(Drton–Plummer) as a practical criterion; Bayesian model selection between a singular and a regular
model with the same $d$.

**Measuring it.** `08-freeenergy-models.py`: $F_n-nS_n$ by quadrature for the sextic ($\tfrac23$),
quartic ($\tfrac34$), quartic cup ($\tfrac12$), product ($\tfrac12$, $m=2$) and regular plane (1) over
$n=10^2,\dots,10^6$: fitted slopes against $\lambda$ (one figure, five lines); `08-freeenergy-sbic.py`:
rank selection in reduced-rank regression by BIC and by sBIC over 200 data sets (frequency of the
correct rank against $n$).

**Ledger.** Quantity: the free energy. Regular: $\tfrac d2\log n$. Singular:
$\lambda\log n-(m-1)\log\log n$. Running example: the sextic model, $\tfrac23$ against $1$. Measured:
slopes. Rests on: realizability (or relatively finite variance), analyticity, compactness.

**Where the theory stops.** Assumed: fixed model and truth as $n\to\infty$; compact $W$ (Watanabe:
infinity can itself be a singularity); relatively finite variance; essential uniqueness of the
optimal distribution (when several optimal distributions exist the asymptotics change: Nagayasu
and Watanabe; verify). Breaks: the $O_p(1)$ term dominates for practical $n$ when $\lambda$
differences are small (for the product model the constant is invisible even at $n=10^{100}$:
digest 01 §7.3); non-renormalizable unrealizable pairs (Watanabe 2010d: learning curves of order
$n^{-2/3}$). Open: free energy asymptotics when $d$ grows with $n$.

**Exercises.** The sandwich with unknown noise variance; the free energy of rank-one regression;
sBIC by hand; a variation with a Gaussian prior; the limit law for the product model.

**Key references.** Watanabe2009, Watanabe2018, DrtonPlummer2017, Schwarz1978.

### Chapter 9. Generalization — The error λ/n and the equations of state (key: generalization)

**Opening.** `\sltchapter{Generalization}{The error $\lambda/n$ and the equations of state}{$y=ab^3x$}{ch:generalization}` and `\newtonfoot{2/6}{6}{\tfrac16}{a^2b^6}`.

**The question.** The expected test loss of the Bayes predictive distribution. Regular answer box:
$\tfrac d{2n}$.

**By hand.** For a one-coefficient model with Gaussian noise, the Kullback–Leibler divergence from
the truth to the predictive is $\tfrac12(\Ew{c}-c_0)^2\E[x^2]+O(n^{-3/2})$: the second-order terms in
the posterior spread cancel (computation box; the coordinator checked it). So the generalization
error is the squared error of the posterior mean. For the cubic cusp the posterior of $t=\sqrt
n\,c$ is $\propto e^{-(t-Z)^2/2}\abs t^{-2/3}$ and its mean is pulled toward $0$.

**The theorem.** `thm:generalization:onecoef` (complete proof; book's own; the coordinator checked
it numerically to four digits for $\lambda=\tfrac14,\tfrac12,\tfrac34,1$): for a one-coefficient
model whose coefficient density is $\abs{c-c_0}^{2\lambda-1}\ell(c)$ with $\ell$ continuous and
positive at $c_0$ ($m=1$), $n(\E[G_n]-S)\to\lambda$. Plan: (i) the expansion of the divergence with a
uniform remainder; (ii) the rescaled posterior $\propto e^{-(t-Z_n)^2/2}\abs t^{\alpha}$,
$\alpha=2\lambda-1$, whose mean is $M(Z)=Z+h'(Z)/h(Z)$ with $h(Z)=\int e^{-(t-Z)^2/2}\abs t^\alpha\dd t$
(Tweedie's formula); (iii) homogeneity of $h$ and the heat equation give $h''=\alpha h-Zh'$, so
$r=h'/h$ satisfies $r'=\alpha-Zr-r^2$; (iv) Stein's identity $\E[Zr]=\E[r']$ for $Z\sim\Normal(0,1)$
gives $2\E[Zr]+\E[r^2]=\alpha$, hence $\E[M(Z)^2]=1+\alpha=2\lambda$; (v) uniform integrability to
pass to the limit. Then `thm:generalization:main`: Watanabe's theorem $\E[G_n]-S=\lambda/n+o(1/n)$ in
general (statement; proof sketch), the Gibbs generalization error and the singular fluctuation
$\nu$, and the equations of state relating Bayes and Gibbs generalization and training losses at
inverse temperature $\beta$: take the exact statements from digest 01, §4 (Watanabe 2010a,
Theorem 2 and Corollary 3; Watanabe 2010, Lemma 3): $E[B^*_g]=(\lambda-\nu)/\beta+\nu$,
$E[B^*_t]=(\lambda-\nu)/\beta-\nu$, $E[G^*_g]=\lambda/\beta+\nu$, $E[G^*_t]=\lambda/\beta-\nu$, and the
conservation law. Two examples from the digest: the product model has $\nu(\beta)=\tfrac12=\lambda$ for
every $\beta$ (its limit posterior in $c=ab$ is Gaussian, so at order $1/n$ it behaves like a regular
one-parameter model, and the singularity shows only through $m=2$ and $\lambda<d/2$); the variant
$y=a\,(b^\top x)$ with $x\sim\Normal(0,I_2)$, $a\in[-1,1]$ and $b$ uniform on the unit disk has
$\lambda=\tfrac12$, $m=1$ and $\nu(1)\approx0.6997\neq\lambda$ (digest §7.4; numerical — recompute it).
If the author can compute $\nu$ for the power-law one-coefficient class (the cusps) by the Tweedie
tools, do so and check numerically. Note also the slow convergence at $m=2$: $n\E[B_g]\approx0.37$ at
$n=10^3$ and $0.40$ at $10^4$ for the product model (digest §4.2(g)), a $-(m-1)/(n\log n)$ correction.

**Measuring it.** `09-generalization-onecoef.py`: $n(\E[G_n]-S)$ by exact quadrature of the
predictive for the regular line ($\tfrac12$), the product model ($\tfrac12$, slow), the cusp
($\tfrac14$) and the cubic cusp ($\tfrac16$), $n=10^2,\dots,10^6$, 500 data sets each, with 95%
intervals, against $\lambda$ (bronze) and against $d/2$ (slate; $1$ for two parameters).

**Ledger.** Quantity: the Bayes generalization error. Regular: $\tfrac d{2n}$. Singular:
$\tfrac\lambda n$. Running example: the cubic cusp, $\tfrac16$ against $1$. Measured. Rests on:
realizability; the Bayes predictive at $\beta=1$.

**Where the theory stops.** Assumed: the Bayes predictive (not a point estimate); fixed truth.
Breaks: for $m>1$ the $o(1/n)$ term decays only like $1/(n\log n)$; MAP and SGD solutions do not
inherit the result (Chapter 15). Open: rates for the remainder; finite-sample bounds.

**Exercises.** The divergence expansion; Tweedie's formula; the heat-equation identity for $h$; the
Gibbs error for the regular line; $\nu$ for the product model (research-level if not computable);
a measurement at $m=2$.

**Key references.** Watanabe2009, Watanabe2018, Watanabe2010, Efron2011 (Tweedie's formula; verify),
Stein1981 (verify).

### Chapter 10. WAIC — Estimating generalization without knowing λ (key: waic)

**Opening.** `\sltchapter{WAIC}{Estimating generalization without knowing $\lambda$}{$(1-a)\Normal(0,1)+a\Normal(b,1)$}{ch:waic}` and `\newtonfoot{2/2}{2}{\tfrac12}{a^2b^2}`.

**The question.** We rarely know $\lambda$. Can we estimate $G_n$ from the data and the posterior?
Regular answer box: AIC, training loss $+\,d/n$ (per-sample scale).

**By hand.** $V_n$ for the regular line ($\approx d$) and for the product model; leave-one-out by
importance sampling, $\mathrm{LOO}=\frac1n\sum_i\log\Ew{1/p(X_i\mid w)}$.

**The theorem.** The cumulant expansion (complete proof under a stated hypothesis that posterior
cumulants of $\log p(X_i\mid w)$ are $O_p(n^{-k/2})$ uniformly in $i$; verify that Watanabe proves
this hypothesis for singular models and cite): $\mathrm{LOO}-T_n-V_n/n=O_p(n^{-2})$ at $\beta=1$.
This is Watanabe 2010, Corollary 1 (digest 01, §5.2, which also gives the functional-cumulant
expansions of Theorem 1 at general $\beta$, where the difference is only $O_p(n^{-3/2})$).
`thm:waic:main` (Watanabe 2010, Lemma 1 and Theorem 2): $\E[\WAIC]=\E[G_n]+o(1/n)$ under
relatively finite variance with $s\ge8$ moments (realizability is not needed), the exact identity
$\E[\mathrm{CV}_n]=\E[G_{n-1}]$, and $B_g+C_v=2\lambda/n+o_p(1/n)$ at $\beta=1$, which makes the
error of WAIC negatively correlated with the generalization error (measure it). Practical WAIC
and PSIS-LOO (Vehtari, Gelman, Gabry 2017); the deviance scale of Gelman et al. is $2n$ times the
book's (digest §5.1). "Where it breaks": the non-renormalizable example of Watanabe 2010d
($q=\Normal(0,I_2)$, $p(x,y\mid a)=\Normal((a,\sqrt{a^4-a^2+1}),I_2)$): the functional variance grows
like $n^{1/3}$ and the learning curve like $n^{-2/3}$ (digest §1.4; a sketch in the source — check).

**Measuring it.** `10-waic-mixture.py`: for the mixture and the product model, over 300 data sets,
$\WAIC-G_n$, $\mathrm{AIC}-G_n$ (with $d=2$) and $\mathrm{LOO}-\WAIC$ against $n$; WAIC is unbiased,
AIC overestimates the error of singular models.

**Ledger.** Quantity: estimating $G_n$. Regular: AIC. Singular: WAIC. Running example: the mixture.
Measured: biases. Rests on: posterior samples of good quality; independent data.

**Where the theory stops.** Assumed: i.i.d. data, exact posterior samples. Breaks: high variance of
WAIC; dependent data; poor MCMC. Open: WAIC for SGLD samples of networks.

**Exercises.** Cumulant expansions; WAIC for a Gaussian model exactly; deviance scale; LOO identity.

**Key references.** Watanabe2010, Watanabe2018, VehtariGelmanGabry2017, GelmanHwangVehtari2014.

### Chapter 11. Temperature — WBIC and the thermodynamics of the posterior (key: temperature)

**Opening.** `\sltchapter{Temperature}{WBIC and the thermodynamics of the posterior}{$y=ax_1+b^2x_2$}{ch:temperature}` and `\newtonfoot{0/4,2/0}{4/3}{\tfrac34}{a^2+b^4}`.

**The question.** The free energy is an integral that sampling cannot see directly; its derivative in
the temperature is a posterior mean. Regular answer box: equipartition, $\Ewb{nK}\approx\tfrac d{2\beta}$.

**By hand.** $\Ewb{nK}$ for the product model and the quartic model by quadrature; thermodynamic
integration $F_n=\int_0^1\Ewb{nL_n}\dd\beta$; equipartition broken: a singular model stores
$\lambda/\beta$, not $\tfrac d{2\beta}$.

**The theorem.** `prop:temperature:mean` (complete proof, population version): under the
two-term volume expansion `ass:temperature:volume` (as written by Chapter 11; the asymptotics
$V(\eps)\sim C\eps^\lambda(\log\frac1\eps)^{m-1}$ alone do **not** suffice — a slowly oscillating
counterexample is an exercise there), $\Ewb{nK}=\frac\lambda\beta-\frac{m-1}{\beta\log(n\beta)}
+o(\frac1{\beta\log(n\beta)})$ as $n\beta\to\infty$.
`thm:temperature:wbic` (Watanabe 2013, Theorem 4, Corollaries 1–3; digest 01 §6:
$\WBIC=nL_n(w_0)+\lambda\log n+U_n\sqrt{\lambda\log n/2}+O_p(1)$ with $\E U_n=0$, $U_n=0$ under odd
parity; hence $F_n=\WBIC+O_p(\sqrt{\log n})$, and $F_n-\WBIC$ contains $-(m-1)\log\log n$ even when
$U_n=0$) and a complete proof for the linear models of `thm:freeenergy:linear` at least in the
population version; Watanabe 2013, Theorem 3 (the unique $\beta^*$ with $F_n=\E^{\beta^*}_w[nL_n]$, by
concavity of $F_n(\beta)$; prove it); the two-temperature estimator of $\lambda$ (Corollary 3) and its
bias $\approx-(m-1)/\log n$ (the digest's toy numbers: $0.33$, $0.37$, $0.42$ at $n=10^3,10^4,10^6$ for
$\lambda=\tfrac12$); the "specific heat" $\beta^2\Var^\beta_w(nL_n)$ as a second estimator (argued; no
source attributes it — mark it as the book's heuristic).

**Measuring it.** `11-temperature-lambda.py`: $\hat\lambda$ from two temperatures for the quartic,
product, sextic and regular plane by quadrature and by Metropolis; WBIC against $F_n$ (difference
against $\sqrt{\log n}$).

**Ledger.** Quantity: WBIC and $\hat\lambda$. Regular: $\tfrac d2\log n$, equipartition. Singular:
$\lambda/\beta$. Running example: the quartic model, $\tfrac34$. Measured. Rests on: sampling at
$\beta=1/\log n$.

**Where the theory stops.** Assumed: exact tempered posteriors. Breaks: bias from $m>1$; the
$O_p(\sqrt{\log n})$ error; in networks the tempered posterior is sampled by SGLD (Chapter 13).
Open: variance-reduced estimators of $\lambda$.

**Exercises.** Thermodynamic integration by hand for the regular line; the heat capacity
$\partial_\beta\Ewb{nK}$; WBIC for rank-one regression; optimal temperature pairs.

**Key references.** Watanabe2013, Watanabe2018, GelmanMeng1998, FrielPettitt2008.

### Chapter 12. Phase Transitions — How the posterior chooses between singularities (key: phases)

**Opening.** `\sltchapter{Phase Transitions}{How the posterior chooses between singularities}{$y=ax_1+b^2x_2$ near a singular truth}{ch:phases}` and `\newtonfoot{0/6,2/2,6/0}{2}{\tfrac12}{a^6+a^2b^2+b^6}`.

(Change the "model" argument to the model actually used.)

**The question.** When the posterior can sit near different singular regions with different losses
and learning coefficients, which one wins, and when does it switch? Regular answer box: never — the
posterior concentrates at the maximum-likelihood point.

**By hand.** Two phenomena, kept apart. (1) *A crossover.* The quartic model $y=ax_1+b^2x_2$ with
truth $b_0^2=\delta>0$ small: while $n\delta^2\ll1$ the data cannot see $\delta$, the posterior of $b$
lives at the scale $n^{-1/4}$ of the singular model, and the free energy grows like $\tfrac34\log n$;
once $n\delta^2\gg1$ it splits into two regular peaks at $\pm\sqrt\delta$ and the growth is
$1\cdot\log n$. The effective coefficient $\lambda_{\mathrm{eff}}(n)=\frac{\dd(F_n-nS_n)}{\dd\log n}$ moves
from $\tfrac34$ to $1$ around $n\approx\delta^{-2}$; compute it exactly by quadrature. (Note that
$b=0$ is a local maximum of $K$ in $b$ here: the regions are not separated.) (2) *A transition
between separated regions.* A model whose true distribution is fitted exactly only in a region of
large learning coefficient, while a region of small learning coefficient fits it almost: the
mixture $(1-a)\Normal(0,1)+a\Normal(b,1)$ with a truth $(a_0,b_0)$ off the axes but with small
$a_0$ is the proposed example (verify that the regions separate; the author may choose a cleaner
model). The cross has $\lambda=\tfrac12$ and misfit $n\Delta$; the true point is regular,
$\lambda=1$; local free energies $F_n(U_\alpha)=nL_n(w_\alpha)+\lambda_\alpha\log n
-(m_\alpha-1)\log\log n+O_p(1)$ predict the switch near $n\Delta\approx\tfrac12\log n$; the posterior
mass of each region by quadrature.

**The theorem.** `thm:phases:local` (complete proof): for a partition of $W$ into closed regions with
local free energies satisfying local formulas, the posterior mass of a region tends to $1$ in
probability when its $nL_n(w_\alpha)+\lambda_\alpha\log n-(m_\alpha-1)\log\log n$ is smaller than the
others' by a margin that tends to infinity; and the exact transition curve in the quartic example.
Discussion: internal model selection (Watanabe; Chen et al. 2023; verify), "accuracy versus
complexity", why a growing data set makes the posterior jump to more complex singularities.

**Measuring it.** `12-phases-crossover.py`: $\lambda_{\mathrm{eff}}(n)$ of the quartic model for
several $\delta$ against $\log(n\delta^2)$ (the curves collapse onto one); `12-phases-transition.py`:
the posterior mass of each region of the two-region example against $n$, with the predicted switch;
posterior draws before and after.

**Ledger.** Quantity: the posterior's choice of region. Regular: the MLE's neighbourhood. Singular:
minimize $nL+\lambda\log n$. Running example: the quartic model. Measured: transition points.
Rests on: local formulas, separated regions.

**Where the theory stops.** Assumed: $n\to\infty$ with fixed $\delta$; transitions in $n$, not in
training time. Breaks: when regions are not separated at the scale of the posterior. Open: the link
between Bayesian transitions in $n$ and dynamical transitions in training time (Chapter 14).

**Exercises.** The transition in the mixture with small separation; three regions; transitions in
$\beta$ at fixed $n$.

**Key references.** Watanabe2018 (verify chapter), ChenEtAl2023, Watanabe2009.

### Part IV. Deep Learning

### Chapter 13. The Local Learning Coefficient — Measuring a trained network (key: llc)

**Opening.** `\sltchapter{The Local Learning Coefficient}{Measuring a trained network}{$y=a\relu(bx)$, deep linear networks}{ch:llc}` and `\newtonfoot{2/2}{2}{\tfrac12}{a^2b^2}`.

**The question.** A trained network is one point $w^*$, not a posterior. How singular is it there,
and how does that compare with Hessian-based "effective dimension"? Regular answer box: the
Laplace approximation at $w^*$ counts the Hessian's rank.

**By hand.** Local learning coefficients in the cast: rank-one regression at $0$ and at points with
$a\ne0$, $b=0$ ($M/2$ versus $\min(M,N)/2$); the ReLU neuron at $(0,0)$ and on its true set;
Hessians at the same points.

**The theorem.** `def:llc:llc` (the local learning coefficient as the learning coefficient of $K$
restricted to a small neighbourhood; well defined for analytic $K$). `thm:llc:hessian` (as written
by Chapter 13: complete proof by volumes for any $C^4$ loss $K\ge0$, with no splitting lemma and no
analyticity; only `cor:llc:minimal` uses `thm:toolkit:splitting`. The plan below was the
specification): if $K$ is analytic, $w^*\in W_0$ is
interior, $\varphi(w^*)>0$ and $\nabla^2K(w^*)$ has rank $r$, then
$\tfrac r2\le\lambda(w^*)\le\tfrac r2+\tfrac{d-r}4$. Proof plan: the analytic splitting lemma writes
$K=\sum_{i\le r}u_i^2+g(z)$ in new coordinates; $\lambda$ adds over separate variables
(`prop:blowups:rules`); $g\ge0$ has no quadratic part, so its lowest-order part has even degree at
least $4$ and $g\le C\abs z^4$, whence $0\le\lambda(g)\le\frac{d-r}4$ by comparison. Sources: the
upper bound is published, in Jacobian-rank form, as Lin (2017), Corollary 4.15 (= Lin 2011,
Corollary 4.21; digest 02 §0.2); it was rediscovered in Bushnaq's LessWrong post of 2024-08-08
(`Bushnaq2024`), whose upper bound was improved to $\frac{d-r}4$ by Dmitry Vaintrob in an edit of
2024-08-16. The book proves both bounds itself and credits Lin first. Both bounds are attained: $a^2$ in $\R^2$ (lower), $a^2+b^4$ and
$a^2b^2$ (upper). The local learning coefficient as Lau et al. define it (2025, Definition 1: the exponent of the
local volume $V(\eps)$ on a ball where $w^*$ minimizes $L$; well-definedness in their App. B needs
$w^*$ at least as degenerate as nearby minimizers and relatively finite variance), its invariance
under local loss-preserving diffeomorphisms (their App. C), hence under ReLU rescaling, the symmetry
Dinh et al. (2017) used against Hessian measures. The estimator exactly as they state it (digest 03
§2.2): the localized tempered posterior $\propto\exp\{-n\beta L_n(w)-\tfrac\gamma2\norm{w-w^*}^2\}$,
$\hat\lambda(w^*)=n\beta^*[\E_{w\mid w^*,\beta^*,\gamma}L_n(w)-L_n(w^*)]$ with $\beta^*=1/\log n$, the
SGLD update and the recycled minibatch losses. Its status, said plainly: the global version is
WBIC (Watanabe 2013, Theorem 4; proved); that the localized version recovers the LLC is
**conjectured** by the authors (their App. D); the Gaussian localizer lies outside the theorem's
hypotheses; the data are used twice. The population consistency of the unlocalized mean energy
follows from `prop:temperature:mean`. Known biases: step size (the most important hyperparameter),
localization, chain length, $w^*$ not a minimum (negative estimates), and the finite-$n$ multiplicity
bias $\approx-(m-1)/\log(n\beta)$ (digest 01 §6.2 measures it on the product model; digest 03 §3.5
derives it heuristically — say so). Conventions differ across papers and code ($n\beta=n/\log n$ in
Lau et al., $m/\log m$ with the batch size $m$ in later papers and `devinterp`, a fixed $30$ in Wang
et al.; a factor $2$ in what "$\gamma$" means): state yours. The deep-linear-network validation
(up to $10^8$ parameters) is empirical and compared estimates at random optimal parameters with
Aoyagi's *global* coefficient, which is only a lower bound there (Hitchcock and Hoogland 2025, a
preprint, who build a benchmark with exactly known local values). State Aoyagi's formula for deep
linear networks (Aoyagi 2024; restated in Lau et al. Theorem 1 and Aoyagi 2025 Definition 4 —
check the definition of the auxiliary integer $a$ there) and use it for the validation in
"Measuring it".

**Measuring it.** `13-llc-sgld.py`: SGLD estimates (numpy or torch, tiny models) for the ReLU neuron,
rank-one regression ($M=N=3$) and deep linear networks of depth 2–4 with known $\lambda$ (Aoyagi;
verify), against the exact value, the Hessian bounds, and $d/2$; sensitivity to $\epsilon$ and $\gamma$.

**Ledger.** Quantity: $\lambda(w^*)$. Regular: Hessian rank$/2$. Singular: $\lambda(w^*)$ between
$\tfrac r2$ and $\tfrac r2+\tfrac{d-r}4$. Running example: rank-one regression. Measured: SGLD
estimates. Rests on: $w^*$ a local minimum; a good sampler; analyticity.

**Where the theory stops.** Assumed: $w^*$ a local minimum of an analytic $K$; the localized posterior
is meaningful. Breaks: at scale the estimator's accuracy is unverifiable (no ground truth); saddle
points; non-analytic activations. Open: error bars for LLC estimates in large models.

**Exercises.** Hessian bound examples (tight on both sides); the LLC of the chain at a point with one
nonzero coordinate; SGLD stationary distribution; a measurement of bias versus step size.

**Key references.** LauEtAl (verify year and venue), FurmanLau2024, WeiEtAl2022, DinhEtAl2017,
WellingTeh2011, Aoyagi2024.

### Chapter 14. Stages — Learning coefficients along training (key: stages)

**Opening.** `\sltchapter{Stages}{Learning coefficients along training}{the toy model of superposition}{ch:stages}` and `\newtonfoot{2/4,4/2}{3}{\tfrac13}{a^4b^2+a^2b^4}`.

**The question.** Training curves have plateaus and sudden drops. Does the local learning coefficient
change at those moments, and do its changes mark "stages"? Regular answer box: the regular theory
has nothing to say about training dynamics.

**By hand.** The toy model of superposition of Elhage et al. (2022) in the setting of Chen et al.
(2023, a preprint): $f(x,w)=\relu(W^\top Wx+b)$, $W\in\R^{2\times6}$, $b\in\R^6$, $d=18$, data in the
high-sparsity limit (one coordinate, uniform length in $[0,1]$); the closed-form population loss
(their Lemma 3.1); the $k$-gon critical points; and their learning coefficients from the rule that a
critical set which is a smooth manifold of dimension $s$ with a nondegenerate normal Hessian has
$\lambda=(d-s)/2$, $m=1$ ("minimally singular"; Chapter 5 proves it as `prop:blowups:morsebott`):
the $6$-gon is nondegenerate modulo the rotation symmetry, so $\lambda=(18-1)/2=8.5$.
Check Chen et al.'s Table 1 ($5$-gon $\lambda=7$, $L=0.06874$; $5^+$-gon $\lambda=8.5$, $L=0.06180$;
$6$-gon $\lambda=8.5$, $L=0.04819$) against the rule and numerically, and say which values are proved
in their paper and which are not (digest 03 §2.3).

**The theorem.** The learning coefficient of a minimally singular critical set (complete proof) and
its application to the TMS $k$-gons; the Bayesian $5\to6$ transition predicted by local free energies
at $n_{\rm cr}\approx600$ and seen with NUTS between $600$ and $700$ (Chen et al.; the authors flag
the tension of using an asymptotic formula at finite $n$); the "opposing staircases" of SGD (loss
down, $\hat\lambda$ up) and the Bayesian Antecedent Hypothesis, stated as they state it and marked
**conjectured**; the Bayesian $5\to6$ transition was not observed in SGD. Observations, each as an
`observation` with its setting and size (digest 03 §2.4–2.7): stagewise development in a two-layer
attention-only language model and in an in-context linear-regression transformer (Hoogland et al.,
TMLR 2025: stages with *decreasing* LLC exist; Hessian statistics did not show the stage boundaries
in the regression setting; the LLC was applied to non-minima throughout training); refined LLCs and
head specialization (Wang et al., ICLR 2025); later work from the digest, marked by status.

**Measuring it.** `14-stages-tms.py`: train a tiny TMS (CPU, under two minutes) and estimate the LLC
along training with SGLD; plot loss and $\hat\lambda$ against steps; mark the plateaus and the $k$-gon
values.

**Ledger.** Quantity: $\hat\lambda(w_t)$ along training. Regular: no prediction. Singular: jumps at
transitions between critical points with different $\lambda$. Running example: TMS. Measured.
Rests on: SGLD estimates at non-minima; small models.

**Where the theory stops.** Assumed: the trajectory passes near critical points; LLC estimates are
meaningful away from minima. Breaks: (from the digest; critiques). Open: a theory of stages for SGD.

**Exercises.** Critical points of the smallest TMS; LLC of a $k$-gon; designing a stagewise toy.

**Key references.** ChenEtAl2023, Elhage2022 (verify key), HooglandEtAl (verify), WangEtAl (verify),
LauEtAl.

### Chapter 15. SGD Is Not Bayes — What the theory does not explain (key: sgd)

**Opening.** `\sltchapter{SGD Is Not Bayes}{What the theory does not explain}{$y=abx$ under gradient descent}{ch:sgd}` and `\newtonfoot{2/2}{2}{\tfrac12}{a^2b^2}`.

**The question.** Singular learning theory is about Bayesian posteriors as $n\to\infty$ with the model
fixed; deep learning trains one network by stochastic gradient descent, often with more parameters
than data. What transfers? Regular answer box: in a regular model with flat priors, the posterior
mean and the minimizer agree to first order.

**By hand.** Gradient flow on the product model: $\frac{\dd}{\dd t}(a^2-b^2)=0$; it converges to the
point of the hyperbola $ab=\hat c$ with the same value of $a^2-b^2$; from a small balanced
initialization at the singular truth it ends at scale $n^{-1/4}$ near the origin, while the posterior
spreads uniformly over scales (Chapter 2). The two "implicit biases" compared.

**The theorem.** `thm:sgd:balance` (complete proof): conservation, convergence (except from a
measure-zero set of initializations), the limit point. The conservation law is classical for deep
linear networks (Saxe et al. 2014; Kunin et al. 2021 derive such laws from symmetries; cite what you
verify). Then (digest 03 §3): Langevin dynamics and SGLD sample the tempered posterior in the limit
of small steps (statement; cite); constant-step SGD is not a posterior sampler (Mandt et al. 2017;
statement as they give it); what is known about posteriors of real networks (Izmailov et al. 2021;
observation); the programme's own bridge, that SGD at step $t$ behaves like the posterior at some
sample size $n(t)$, is a stated hypothesis (Pepin Lehalleur et al. 2025). The critiques (digest 03
§4), stated fairly: Skalse (2023): low $\lambda$ means "simple" only given assumptions on the prior and
the parameter-function map; the $n\to\infty$ regime is not where the generalization puzzle lives;
singular points dominate only with analytic models and infinite data. Murfet's reply (2023)
separates (i) *that* overparameterized singular models *can* generalize, which the theory nearly
settles, from (ii) *why neural networks do*, which it reduces to why natural data give low
$L_0,\lambda,\nu$; Lau et al. (App. F) call explaining simplicity bias by a low-LLC bias "tautological".
No refereed paper whose main purpose is to critique the theory's relevance was found (as of
October 2026). Flatness, Hessian measures, minimum description length and $\lambda$ (digest §4.4–4.5).

**Measuring it.** `15-sgd-product.py`: final $(a,b)$ of gradient descent from random initializations
against posterior draws, for the product model; the distribution of $\log\abs a/\log n$ for both;
optionally SGD with finite step size on a tiny network against SGLD samples.

**Ledger.** Quantity: what SGD finds versus what the posterior averages. Regular: they agree. Singular:
they differ; SGD selects by a conserved quantity, the posterior by volume. Running example: the
product model. Measured. Rests on: gradient flow (no noise), the product model.

**Where the theory stops.** This chapter is that section for the book: a list of what singular learning
theory explains (parameter counts are the wrong complexity; Bayesian model selection in singular
models; the LLC as a measured complexity) and what it does not (why SGD generalizes; the regime
$d\gg n$; non-analytic architectures; the role of training time), each with sources. The three
paragraphs still close it.

**Exercises.** Conservation laws for deep linear networks (balancedness; Arora et al. 2018; verify);
gradient flow for the chain; Langevin stationary density; a research exercise on SGD and $\lambda$.

**Key references.** Skalse2023 (online; verify), MandtHoffmanBlei2017, WellingTeh2011,
IzmailovEtAl2021, AroraCohenHazan2018 (verify), DuHuLee2018 (balancedness; verify), DinhEtAl2017.

### Epilogue — The ledger (key: epilogue; special chapter)

Written by the coordinating author from the chapters' final cards: Table E.1, the quantities with
their regular and singular values; Table E.2, the assumptions of the theorems against deep-learning
practice; a list of open problems from the chapters; what to read next.

### Appendix A. A Toolkit (key: toolkit)

Written by the coordinating author (or a delegated author) with the fixed labels above: proofs of
Laplace's method, Gaussian facts, Stein's identity and the heat equation, the Abelian lemma; precise
statements of Hironaka's theorem, the splitting lemma, and the empirical-process theorem; the
samplers and intervals used by the scripts.
