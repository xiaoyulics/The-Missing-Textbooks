# Optimizers in the Wild — Writing Guide for Chapter Authors

*Optimizers in the Wild: Steepest Descent and the Dynamics of Training* is a graduate textbook in English,
volume 3 of *The Missing Textbooks*. Optimization textbooks stop at convex analysis and at "SGD converges to a
stationary point"; the decisions that decide a training run — Adam or Muon, warmup, weight decay, the
learning-rate schedule, the batch size, training at the edge of stability — are not in them. This book gives
their theory along two lines: the **unification line** (almost every optimizer is steepest descent under a
norm, plus preconditioning and moving averages) and the **dynamics line** (stability, warmup, noise,
schedules, equilibria). Each chapter starts from a practice in the wild, builds the smallest idealization that
can explain it, proves one main theorem completely, checks a prediction against a simulation, and says what the
idealization misses and what the theory does and does not tell a practitioner to do.

Read this guide and `OUTLINE.md` (the book plan, the global notation, the fixed appendix labels, and your
chapter's specification) before writing. For voice and LaTeX quality read one finished chapter of volume 2,
`../hydrogen-atoms/chapters/15-stability.tex` (the closest in topic) or
`../hydrogen-atoms/chapters/07-singleindex.tex`; this volume keeps that voice and those conventions, with its
own chapter anatomy.

---

## 1. Audience and level

Graduate students and researchers in machine learning, optimization, statistics and applied mathematics, and
practitioners who train large models and want to know which of their rules are understood, in what sense, and
how far the understanding reaches. Assume undergraduate analysis, linear algebra and probability, and gradient
descent on smooth functions. Do **not** assume the background of your chapter: a strong student who has read no
paper on your topic must be able to read the chapter on its own. Shared tools (dual norms, the polar factor,
von Neumann's trace inequality, Kronecker products, the matrix geometric mean, stability of linear recurrences,
Lyapunov recursions, the descent lemma, the PL inequality, online gradient descent, Grönwall, concentration)
are in the appendix (`\cref{ch:toolkit}`, fixed labels in OUTLINE.md); cite them there, or prove what you need
inline when it is short. Earlier chapters' results may be cited by `\cref` with the labels you expect them to
use (`thm:steepest:...`); the editor fixes cross-chapter labels at integration, so also say in prose what you
are citing.

## 2. The chapter anatomy (every chapter follows it)

```
\casechapter{Title}{Subtitle}{In-the-wild tag}{ch:key}
\begin{refsection}[chapters/NN-key.bib]

\begin{chaptersummary} 3–5 sentences: the practice, the idealization, what was proved, what it misses. \end{chaptersummary}

\section{In the wild: <specific title>}          % ~12–15%: a practice box + observations with evidence
\section{The idealization: <specific title>}     % ~10–12%: the model; a dictionary box; field-guide entries
\section{The theory: <specific title>}           % ~35–40%; may be split into two "The theory" sections
\section{Reading the theory}                     % ~10%: predictions, figure(s) theory vs simulation
\section{What the theory misses}                 % ~12–15%: three mandatory paragraphs
\section{Back in the wild}                       % ~3–5%: \fieldnotes card + 1–2 paragraphs
\begin{exercises} ... \end{exercises}
\notesandrefs ... \printchapterbib
\end{refsection}
```

**In the wild.** What practitioners do and what has been observed, by whom, when, with what evidence. State the
practice in a `practice` box — the rule as it is actually used, with typical values and where they come from:
```
\begin{practice}[GPT-3, \textcite{BrownEtAl2020}; LLaMA, \textcite{TouvronEtAl2023}]
Warm the learning rate up linearly from $0$ over the first $2000$ steps, then decay it ...
\end{practice}
```
State each empirical regularity precisely in an `observation` environment
(`\begin{observation}[\textcite{CohenEtAl2021}]...\end{observation}`), and be honest about its status: how
robust it is, in which settings it has been seen, which settings contradict it. End the section with the
question that the theory will answer.

**The idealization.** Define the setting completely (assumptions in `assumption` environments). Say what was
thrown away and why each simplification is the right one to make first. Include one `dictionary` box
translating quantities of the theory into quantities of a training run (left column: the theory; right column:
a training run). In Parts I–II, give a **field-guide entry** for each optimizer the chapter derives (see §7);
say there which norm, which preconditioner and which moving averages make it up. Say what one had to notice to
see the practice this way.

**The theory.** The heart of the chapter. At least **one main theorem with a complete proof** — a proof that a
careful graduate student can check line by line, with every hypothesis used and every constant right. Then
further results: complete proofs where they fit, `proofsketch` (saying what is omitted and where to find it)
where they do not, and `heuristic` derivations (ending with ◇) where the honest status is "derived, not
proved". Never let a heuristic look like a proof, and never call a sketch a proof.

**Reading the theory.** What the theory predicts, quantitatively: thresholds, rates, scaling rules, time
scales, equilibria, crossovers. At least **one figure comparing the theory's exact prediction (curve) with a
simulation (points)**, produced by a script (see §6). Explain the shape of every curve and say where the
agreement must fail.

**What the theory misses.** Three mandatory paragraphs, each opened with `\paragraph{...}`:
- `\paragraph{What was idealized away.}` Each simplification and what it costs. Be concrete.
- `\paragraph{Where it breaks.}` Specific, cited results — theorems or experiments — where the theory's
  predictions are wrong, or where the practice works for a different reason than the theory says. This is the
  most valuable paragraph of the chapter; make it specific, sourced and fair.
- `\paragraph{Repairs and open problems.}` The refinements that repair the failures, with citations, and what
  is still open.

**Back in the wild.** The `\fieldnotes` card (see §7), then one or two paragraphs: what the theory tells a
practitioner (and what it does not — do not overclaim; mark folklore as folklore), and how the chapter connects
to others with `\cref{ch:key}`.

**Exercises.** 10–14 items: about 5 `\consolidation` (re-derive or check results of the chapter), 3–4
`\variation` (change one assumption and re-solve — a different norm, momentum on, a different noise model),
2–3 `\simulation` (a numerical experiment with explicit parameters, stating what the reader should observe and
why), 1–2 `\research` (open-ended but well posed; say what is open). Mark difficulty with `\hard` or `\harder`
after the tag; give `\hint{...}` for starred items. Every exercise must be correct and solvable as stated.
Label exercises `\label{ex:key:name}` when you refer to them.

**Notes and references.** Three to six paragraphs of history and pointers with author–year citations, ending
with `\paragraph{Translating the literature.}`, which maps the notation of the two or three main source papers
onto this book's (e.g. "Bernstein and Newhouse write $g^\dagger$ for our $g^\sharp$ and call $\lmo$ the dualized
gradient up to sign; Pethick et al. write $\mathrm{lmo}(g)=\argmin_{\norm x\le\rho}\ip{g,x}$, which is $\rho$
times ours"). This is a service the book promises. Then `\printchapterbib`.

## 3. Length

Target **15–20 compiled pages** per chapter (about 7,000–9,500 words including mathematics). The theory sections
must contain at least one complete proof of a nontrivial theorem.

## 4. Global notation (all chapters use it — see OUTLINE.md §Notation for the full list)

- $\theta\in\R^d$ the parameters ($d$ = number of parameters); $W\in\R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$ a
  weight matrix; $G\in\R^{m\times n}$ a generic matrix or matrix gradient; layers $W_1,\dots,W_N$, index $l$;
  $n$ samples, $B$ batch size, $K$ classes, $T$ steps, $t$ the step.
- $\Loss(\theta)$ the objective, $\ell(\theta;\xi)$ a per-example or per-batch loss, $\Loss^\star$, $\theta^\star$,
  $\Delta_t=\Loss(\theta_t)-\Loss^\star$. Superscript star: optimal; subscript star: target or teacher.
- $g_t$ a stochastic gradient, $\zeta_t=g_t-\nabla\Loss(\theta_t)$, $\Sigma$ the per-example gradient covariance,
  $H=\nabla^2\Loss$, sharpness $S=\lmax(H)$, preconditioned sharpness $S_P=\lmax(P^{-1}H)$.
- $\eta$, $\eta_t$, $\eta_{\max}$, $T_{\mathrm w}$; $\beta$ momentum; Adam's $\beta_1,\beta_2,\epsilon$, $m_t,v_t$,
  $\hat m_t,\hat v_t$; decoupled weight decay $\lambda$; eigenvalues always with a subscript ($\lambda_i$,
  `\lmax`, `\lmin`), never a bare $\lambda$.
- Norms: $\norm{\cdot}$ generic (never silently Euclidean), $\norm{\cdot}_*$ dual, $\norm\cdot_2,\norm\cdot_1,
  \norm\cdot_\infty$; `\norm{W}_{\op}` spectral, $\norm W_F$, `\norm{W}_{\nuc}` nuclear, $\norm W_{p\to q}$ induced;
  `\norm{x}_{\rms}` and `\norm{W}_{\rms\to\rms}`; $\norm{x}_P$ for $P\succ0$.
- `\lmo(g)`$\in\argmin_{\norm u\le1}\ip{g,u}$; $g^\sharp=-\norm g_*\lmo(g)$; normalized steepest descent
  $\theta_{t+1}=\theta_t+\eta\lmo(g_t)$; `\sign` coordinatewise ($\sign0=0$); `\polar(G)`$=UV^\top$.
- $L$-smooth w.r.t. a norm: $\norm{\nabla\Loss(\theta)-\nabla\Loss(\theta')}_*\le L\norm{\theta-\theta'}$; $(L_0,L_1)$
  and $(K_0,K_1)$ as in OUTLINE.md.
- Asymptotics: $a\lesssim b$ means $a\le Cb$ for a constant $C$ that depends only on parameters declared fixed;
  $a\asymp b$; $O,\Omega,\Theta,o$; $\tilde O$ hides polylogarithmic factors. Say what is fixed.

If your chapter needs a symbol that clashes with this list, keep the global meaning and choose another symbol
for the local one; if a source paper's notation clashes, translate it (and record the translation in
"Translating the literature").

## 5. Voice and proof style

### 5.1 Voice
- Precise, warm and economical: a good graduate textbook with an essayist's sense of story (Nesterov's and
  Boyd–Vandenberghe's clarity, O'Donnell's *Analysis of Boolean Functions*, Tao's expository writing). "We" means
  the author and the reader.
- The "in the wild" frame is a frame, not a decoration: it names the movement of every chapter — from a practice
  observed in training runs, to an idealization studied on paper, back to the practice. Use naturalist language
  sparingly (one or two touches per chapter at most); never as ornament.
- Tell the history with real dates and people; never invent anecdotes, motives or quotations. Paraphrase; quote
  at most a short phrase (under 15 words), attributed.
- No hype ("revolutionary", "beautiful theory", "magic", "mysterious", "secretly", "dramatically"). Let the
  mathematics impress. Do not oversell a practice or a paper; do not dismiss one either.
- Be explicit about status: **proved** (theorem with proof), **sketched**, **derived** (heuristic, ◇),
  **observed** (observation, with the evidence), **conjectured**, **folklore** (a rule used without evidence
  you can cite). Many practices in this book are folklore; say so.
- Practice changes fast. Date every empirical claim about "current practice" ("as of 2025, most open LLM
  recipes ...") and give a source.

### 5.2 Proofs (Halmos rules)
1. State the result; then prove it. Open every proof longer than a paragraph with its plan in one sentence ("The
   plan is to …").
2. Write forward, from hypotheses to conclusion. Motivate a choice where it is made.
3. Before a chain of (in)equalities, say in words what it does. Split chains that do more than two things.
4. Name trivial and degenerate cases and dispose of them (zero gradients, ties in $\lmo$, rank-deficient
   matrices, $\beta=0$).
5. Never repeat an argument: extract a lemma instead.
6. Cite every hypothesis where it is used, including parameter ranges.
7. Number only displays that are cited later (`equation` + `\label`); use `\[ \]` otherwise. Punctuate displays
   as part of the sentence.
8. Proofs end with the tombstone (automatic); heuristic derivations end with ◇ (automatic). If either ends in a
   display, put `\qedhere` in the display.
9. `proofsketch` for sketches: say what is omitted and where to find it.

### 5.3 Words and symbols
- Never start a sentence with a symbol ("The step $\eta$ is ...", not "$\eta$ is ...").
- Pair every "if" with "then". Write "every", "each", or "some" — never "any".
- Banned: clearly, obviously, trivially, evidently, "it is easy to see", "it is straightforward". If something is
  easy, show it in one line.
- Define symbols before a display, not in a trailing "where".
- One word, one meaning: "step size" and "learning rate" are synonyms here — pick "learning rate" in the
  practice sections and "step size" in the theory, and say once that they are the same; "normalized" always
  means "the step has a fixed length in the chosen norm".
- No ∀, ∃, ⇒ in prose. American spelling. Serial comma.

## 6. Figures and numerics

- Every chapter has **at least one figure that compares the theory's exact prediction (a curve) with a
  simulation (points)**, and may have up to three more (TikZ schematics welcome).
- Numerical figures: write a script `figures/scripts/NN-key-name.py` (numpy, scipy and torch only;
  deterministic seeds; under 2 minutes on a laptop CPU — call `torch.set_num_threads(2)` if you use torch; several
  authors share one 4-core machine, so keep scripts light) that writes whitespace-separated data with a header
  row to `figures/data/NN-key-name.dat` (at most a few hundred rows per file). Plot it with pgfplots:
  ```
  \begin{figure}[t]
  \centering
  \begin{tikzpicture}
  \begin{axis}[wplot,xlabel={step $t$},ylabel={loss},ymode=log]
  \addplot[theory] table[x=t,y=exact] {figures/data/12-schedules-river.dat};
  \addplot[simulation] table[x=t,y=sgd] {figures/data/12-schedules-river.dat};
  \legend{exact recursion,SGD}
  \end{axis}
  \end{tikzpicture}
  \caption{... say what was simulated: sizes, number of runs, seeds.}\label{fig:key:name}
  \end{figure}
  ```
  Styles defined in `wild.sty`: `wplot` (axis), `theory`, `theory2`, `theory3` (curves), `simulation`,
  `simulation2` (marks), `guide` (dashed reference lines), `band` (filled regions); the default cycle list uses
  the series palette. Use `fullwidth` for wide figures. Do not use colours outside the palette (`sea`, `seadeep`,
  `seafoam`, `seamist`, `sand`, `sandline`, `ink`, `muted`, and mixes such as `sandline!55!black`).
- Run every script yourself, check the agreement, and state in the caption what was simulated (sizes, runs,
  seed). A figure whose simulation disagrees with the theory is a finding: report it and explain it.
- Use numerics also to check your theorems: constants, signs, thresholds and exercise answers. Keep scratch work in
  `build/` or in your scratch directory, not in `figures/`.

## 7. LaTeX conventions

- **Engine:** pdfLaTeX + biber. Build your chapter with `tools/build_chapter.sh NN` and fix every error it reports.
  Do not edit the class, `wild.sty`, `main.tex`, other chapters, or the tools. If you need a feature they lack,
  say so in your report.
- **Labels** carry your chapter key: `ch:key`, `sec:key:...`, `thm:key:...`, `lem:key:...`, `prop:key:...`,
  `cor:key:...`, `def:key:...`, `ass:key:...`, `obs:key:...`, `eq:key:...`, `fig:key:...`, `ex:key:...`. Refer with
  `\cref{...}` (sentence start `\Cref{...}`). Other chapters: `\cref{ch:key}` with the keys in OUTLINE.md;
  appendix results with the fixed `toolkit` labels.
- **Environments:** `theorem lemma proposition corollary conjecture claim` (boxed, italic);
  `definition problem assumption` (sand box); `example remark observation` (unboxed); starred versions of the
  classical ones; `proof`, `proofsketch`, `heuristic` (◇); `chaptersummary`; `aside[Title]`; `fullwidth`;
  `exercises` with tags `\consolidation`, `\variation`, `\simulation`, `\research`, difficulty `\hard`, `\harder`,
  `\hint{...}`. This volume adds:
  - `practice[source]` — the "in the wild" box: a rule as practitioners use it, with typical values.
  - `dictionary` — rows `theory side & training-run side \\` (header "in the theory | in a training run").
  - `fieldguide{Name}{first described}` with rows `\fgrow{label}{content}` — one entry per optimizer, e.g.
    ```
    \begin{fieldguide}{Adam}{\textcite{KingmaBa2015}}
    \fgrow{update}{$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$, \ $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$, \
      $\theta_{t+1}=\theta_t-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$}
    \fgrow{norm}{$\ell_\infty$ when $\beta_1=\beta_2=0$ (sign descent, \cref{ch:sign})}
    \fgrow{preconditioner}{diagonal, $\diag(\sqrt{\hat v_t})+\epsilon$}
    \fgrow{averages}{EMA of $g$ ($\beta_1$) and of $g^2$ ($\beta_2$)}
    \fgrow{state}{two numbers per parameter}
    \fgrow{in the wild}{$\beta_1=0.9$, $\beta_2\in\{0.95,0.999\}$, $\epsilon=10^{-8}$}
    \end{fieldguide}
    ```
    The standard rows are `update`, `norm`, `preconditioner`, `averages`, `state`, `in the wild`; add others
    (`variants`, `caution`) when useful. Keep each entry under a third of a page.
  - `\fieldnotes{practice}{idealization}{proved}{predicts}{misses}{takeaway}` — the end-of-chapter card, one or
    two lines per field. `takeaway` says what the theory suggests a practitioner do or expect, with its status
    (proved in the idealization / suggested / folklore).
- **Margin notes:** `\sidenote{...}` (numbered) for history, cross-references and side comments; `\marginaside{...}`
  for unnumbered ones. Use 6–15 per chapter. Never put essential mathematics in a side note. **Side notes only in
  running text** — not inside boxes, theorem-like environments, displays, captions, tables, the dictionary, field
  guide entries or the field notes (LaTeX will error). Do **not** use `\footnote`.
- **Terms:** `\term{...}` at the defining occurrence and `\index{...}` for 8–20 key terms per chapter (plain words;
  subentries with `!`; avoid `@ | !` inside terms otherwise).
- **Macros available:** from the class: `\R \N \Z \Q \C \F \E \eps \ind \dd \defeq \Var \Cov \argmin \argmax \rank
  \tr \sgn \supp \poly \conv \diam \dist \abs{} \norm{} \ip{} \ceil{} \floor{}` and `\cA … \cZ`; from `wild.sty`:
  `\Loss \Lhat \lmo \sign \polar \op \nuc \rms \vect \lmax \lmin \eqd \Unif \diag \relu \softmax \erf \Sph`
  (`\op`, `\nuc`, `\rms` are upright subscripts: write `\norm{W}_{\op}`, `\norm{W}_{\nuc}`, `\norm{x}_{\rms}`).
  Need more? Define it at the top of your chapter with a name that starts with your chapter key (e.g.
  `\newcommand{\eosH}{\mathcal H}`), never a generic name.
- **Citations:** keys `AuthorYear`, `AuthorAuthorYear` or `AuthorEtAlYear` (e.g. `KingmaBa2015`,
  `CohenEtAl2021`, `BernsteinNewhouse2024`). Your bib file is `chapters/NN-key.bib`. Every entry must be real and
  verified with a web search (authors, title, venue, year; volume and pages only when verified). For preprints give
  the arXiv number in `eprint` (with `eprinttype = {arxiv}`) or `note` only if verified. Mark unrefereed preprints
  as such in the text when a claim rests on them. Blog posts (Muon) are `@online` with `url` and `urldate`.
  12–40 entries per chapter.
- **No** `\usepackage`, no changes to page layout, no `\newpage` tricks, no colours outside the palette, no
  `\footnote`.

## 8. Quality checklist before you report back

- [ ] `tools/build_chapter.sh NN` runs with zero LaTeX errors, zero biber errors, no undefined references or
      citations belonging to your chapter, and no overfull box over 3pt.
- [ ] The lint (run by the build script) shows no warning, or each remaining one is deliberate.
- [ ] Every theorem is true as stated, with all hypotheses; every proof is complete or explicitly a sketch or a
      heuristic; constants are right (checked numerically where possible); edge cases are named.
- [ ] Every figure script runs, and the figure shows what the caption says.
- [ ] Every historical and bibliographic claim is verified; no invented quotations or anecdotes.
- [ ] The anatomy is followed, including the practice box, the dictionary, field-guide entries (Parts I–II), the
      three paragraphs of "What the theory misses", the field notes, the exercise mix, and "Translating the
      literature".
- [ ] The global notation is used; deviations are deliberate and explained.
- [ ] Length is 15–20 pages.

When you finish, report: the page count; the list of main results, saying which have complete proofs, which are
sketched and which are heuristic; the figures and what they show; where you corrected the editor's plan and why;
any facts you could not verify (and what you did about them); and anything the editor should check.
