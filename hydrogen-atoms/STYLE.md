# The Hydrogen Atoms of Deep Learning — Writing Guide for Chapter Authors

*The Hydrogen Atoms of Deep Learning: Solvable Models and What They Get Wrong* is a graduate textbook in
English, volume 2 of *The Missing Textbooks*. Physics students learn the harmonic oscillator, the
hydrogen atom and the Ising model: each is solved exactly, and each is known to be wrong in specific,
instructive ways. Deep learning theory has accumulated solvable models of the same kind, but they are
scattered over dozens of papers with incompatible notation. This book collects them, solves each one
completely in **one notation**, and says exactly where each one fails.

Read this guide and `OUTLINE.md` (the book plan, the global notation, and your chapter's specification)
before writing. For voice and LaTeX quality, read one chapter of volume 1, for example
`../dissolving-problems/chapters/08-fourier.tex` or `../dissolving-problems/chapters/11-vc.tex`: this volume
keeps that voice and those conventions, with a different chapter anatomy.

---

## 1. Audience and level

Readers are graduate students and researchers in machine learning, statistics, applied mathematics,
theoretical computer science and physics. Assume undergraduate analysis, linear algebra and probability,
basic optimization (gradient descent), and the maturity to read proofs. Do **not** assume the background
of your chapter: a strong student who has read no paper on your topic must be able to read the chapter
on its own. Tools that several chapters share (Gaussian integration by parts, Hermite polynomials,
Wishart matrices, the Marchenko–Pastur law, martingale inequalities, the ODE method, Fourier analysis on
finite groups, spherical harmonics) are in the appendix (`\cref{ch:toolkit}`, labels in OUTLINE.md);
cite them there, or prove what you need inline when it is short.

## 2. The chapter anatomy (every chapter follows it)

```
\casechapter{Title}{Subtitle}{Model tag}{ch:key}
\begin{refsection}[chapters/NN-key.bib]

\begin{chaptersummary} 3–5 sentences: the phenomenon, the model, what was proved, what it gets wrong. \end{chaptersummary}

\section{The phenomenon: <specific title>}       % ~12–15%
\section{The model: <specific title>}            % ~12–15%, contains a dictionary box
\section{The solution: <specific title>}         % ~35–40%; may be split into two "The solution" sections
\section{Reading the solution}                   % ~10%: predictions, figure(s), limits, phase diagram
\section{What the model gets wrong}              % ~12–15%, three mandatory paragraphs
\section{Model card}                             % ~3%: \modelcard + 1–2 paragraphs of cross-links
\begin{exercises} ... \end{exercises}
\notesandrefs ... \printchapterbib
\end{refsection}
```

**The phenomenon.** What is observed in real networks, by whom, when, with what evidence. State the
empirical regularity precisely in an `observation` environment (e.g.
`\begin{observation}[\textcite{Nakkiran2021}]...\end{observation}`), and be honest about its status: how
robust it is, in which settings it has been seen, and which settings contradict it. End the section with
the question that the model will answer.

**The model.** Define the minimal solvable model completely (assumptions in `assumption` environments
where helpful). Say what was thrown away and why each simplification is the right one to make first.
Include one `dictionary` box translating model quantities into network quantities (left column: the model;
right column: a trained network). Say what a reader would have had to notice to invent this model.

**The solution.** The heart of the chapter. At least **one main theorem with a complete proof** — a proof
that a careful graduate student can check line by line, with every hypothesis used and every constant
right. Then further results: complete proofs where they fit, `proofsketch` (saying what is omitted and
where to find it) where they do not, and `heuristic` derivations (physics-style arguments, ending with ◇)
where the honest status is "derived, not proved". Never let a heuristic look like a proof, and never call
a sketch a proof.

**Reading the solution.** What the solution predicts, quantitatively: scaling laws, thresholds, phase
diagrams, time scales, exponents. At least **one figure comparing the model's exact prediction (curve)
with a simulation (points)**, produced by a script (see §6). Explain the shape of every curve.

**What the model gets wrong.** Three mandatory paragraphs, each opened with `\paragraph{...}`:
- `\paragraph{What was thrown away.}` Each simplification and what it costs. Be concrete.
- `\paragraph{Where it breaks.}` Specific, cited results — theorems or experiments — where the model's
  predictions are wrong or where the phenomenon occurs for a different reason than the model says.
  This is the most valuable paragraph of the chapter; make it specific, sourced and fair.
- `\paragraph{Fine structure.}` The corrections and successor models that repair the failures (the
  analogue of the fine structure of hydrogen), with citations, and what is still open.

**Model card.** `\modelcard{phenomenon}{model}{solution}{predicts}{gets wrong}{successors}` with one or two
lines per field, then one or two paragraphs relating the model to other chapters with `\cref{ch:key}`.

**Exercises.** 10–14 items: about 5 `\consolidation` (re-derive or check results of the chapter),
3–4 `\variation` (change one assumption of the model and re-solve — this is the physicist's skill the
book teaches), 2–3 `\simulation` (a numerical experiment with explicit parameters, stating what the reader
should observe and why), 1–2 `\research` (open-ended but well posed; say what is open). Mark difficulty
with `\hard` or `\harder` after the tag; give `\hint{...}` for starred items. Every exercise must be
correct and solvable as stated (research items must say what is open). Label exercises
`\label{ex:key:name}` when you refer to them.

**Notes and references.** Three to six paragraphs of history and pointers with author–year citations,
ending with a paragraph `\paragraph{Translating the literature.}` that maps the notation of the two or
three main source papers onto this book's notation (e.g. "Saxe et al. write $s_\alpha$ for our
$s_\alpha$, $\tau$ for our $1$, and $W^{21}$ for our $W_2$"). This is a service the book promises.
Then `\printchapterbib`.

## 3. Length

Target **15–20 compiled pages** per chapter (about 7,000–9,500 words including mathematics). The
solution sections must contain at least one complete proof of a nontrivial theorem.

## 4. Global notation (all chapters use it — see also OUTLINE.md §Notation)

- Samples $n$; input dimension $d$; hidden width $m$; number of features/parameters of a linear model $p$;
  number of classes $K$; context length $N$ (Chapters 13–14); depth $L$; layer index $\ell$.
- Inputs $x\in\R^d$, labels $y$, predictor $f(x)$ or $f(x;\theta)$, parameters $\theta$, weight vectors $w$,
  weight matrices $W$. **A subscript star marks the target or teacher:** $w_\star$, $f_\star$, $W_\star$,
  $\theta_\star$. A superscript star marks an optimal value: $\gamma^\star$, $\lambda^\star$, $\Gamma^\star$.
- Activation $\phi$; noise standard deviation $\sigma$; initialization variances $\sigma_w^2,\sigma_b^2$;
  initialization scale $\alpha$.
- Population loss $\Loss(\theta)$ (`\Loss`), training loss $\Lhat(\theta)$ (`\Lhat`); test error in words.
- Step size $\eta$; time $t$ (continuous for flows, integer for iterations); weight decay and ridge
  parameter $\lambda$. Eigenvalues carry a subscript ($\lambda_k$, `\lmax`, `\lmin`); never use a bare
  $\lambda$ for an eigenvalue.
- Gaussians $\cN(\mu,\Sigma)$; uniform law `\Unif`; sphere $\Sph^{d-1}$ (`\Sph`).
- Hermite polynomials: probabilists' $\He_k$ (`\He`), normalized $h_k=\He_k/\sqrt{k!}$, orthonormal in
  $L^2(\cN(0,1))$; Hermite coefficients $\hat g_k=\E[g(z)h_k(z)]$.
- Overlaps (order parameters): $r=\ip{w,w_\star}$ for unit vectors; matrices $Q,R,T$ in Chapter 8.
- Inner product $\ip{u,v}$ (`\ip{u,v}`), Euclidean norm $\norm{u}$, operator norm $\norm{A}_{\op}$,
  Frobenius norm $\norm{A}_F$, transpose $A^\top$, identity $I_d$, all-ones vector $\mathbf 1_d$,
  indicator $\ind\{\cdot\}$.
- Asymptotics: $a\lesssim b$ means $a\le Cb$ for a constant $C$ that depends only on parameters declared
  fixed; $a\asymp b$; $O,\Omega,\Theta,o$; $\tilde O$ hides polylogarithmic factors. Say what is fixed.

If your chapter needs a symbol that clashes with this list, keep the global meaning and choose another
symbol for the local one; if a source paper's notation clashes, translate it (and record the translation
in "Translating the literature").

## 5. Voice and proof style

### 5.1 Voice
- Precise, warm and economical: a good graduate textbook with an essayist's sense of story (O'Donnell's
  *Analysis of Boolean Functions*, Tao's expository writing, a physics text such as Feynman's lectures or
  Zee). "We" means the author and the reader.
- The physics analogy is a frame, not a decoration: use it where it explains (normal modes, order
  parameters, phase transitions, mean-field exactness, perturbation theory), never as ornament.
- Tell the history with real dates and people; never invent anecdotes, motives or quotations. Paraphrase;
  quote at most a short phrase (under 15 words), attributed.
- No hype ("revolutionary", "beautiful theory", "magic", "mysterious"). Let the mathematics impress.
- Be explicit about status: **proved** (theorem with proof), **sketched**, **derived** (heuristic, ◇),
  **observed** (observation, with the evidence), **conjectured**, **folklore**.

### 5.2 Proofs (Halmos rules — run the linter; see §7)
1. State the result; then prove it. Open every proof longer than a paragraph with its plan in one
   sentence ("The plan is to …").
2. Write forward, from hypotheses to conclusion. Motivate a choice where it is made.
3. Before a chain of (in)equalities, say in words what it does. Split chains that do more than two things.
4. Name trivial and degenerate cases and dispose of them.
5. Never repeat an argument: extract a lemma instead.
6. Cite every hypothesis where it is used, including parameter ranges.
7. Number only displays that are cited later (`equation` + `\label`); use `\[ \]` otherwise. Punctuate
   displays as part of the sentence.
8. Proofs end with the tombstone (automatic); heuristic derivations end with ◇ (automatic). If either
   ends in a display, put `\qedhere` in the display.
9. `proofsketch` for sketches: say what is omitted and where to find it.

### 5.3 Words and symbols
- Never start a sentence with a symbol ("The matrix $X$ has rank $n$", not "$X$ has rank $n$").
- Pair every "if" with "then". Write "every", "each", or "some" — never "any".
- Banned: clearly, obviously, trivially, evidently, "it is easy to see", "it is straightforward".
  If something is easy, show it in one line.
- Define symbols before a display, not in a trailing "where".
- One word, one meaning. No ∀, ∃, ⇒ in prose.
- American spelling. Serial comma.

## 6. Figures and numerics

- Every chapter has **at least one figure that compares the model's exact prediction (a curve) with a
  simulation (points)**, and may have up to three more figures (schematics in TikZ are welcome).
- Numerical figures: write a script `figures/scripts/NN-key-name.py` (numpy, scipy, and torch only;
  deterministic seeds; under 2 minutes on a laptop CPU — call `torch.set_num_threads(2)` if you use torch)
  that writes whitespace-separated data with a header row to `figures/data/NN-key-name.dat` (at most a few
  hundred rows per file). Plot it with pgfplots in the chapter:
  ```
  \begin{figure}[t]
  \centering
  \begin{tikzpicture}
  \begin{axis}[hplot,xlabel={$\gamma=p/n$},ylabel={test error},ymin=0]
  \addplot[theory] table[x=gamma,y=exact] {figures/data/01-doubledescent-risk.dat};
  \addplot[simulation] table[x=gamma,y=mc] {figures/data/01-doubledescent-risk.dat};
  \legend{exact formula,simulation}
  \end{axis}
  \end{tikzpicture}
  \caption{... say what was simulated: sizes, number of trials, seeds.}\label{fig:key:name}
  \end{figure}
  ```
  Styles defined in `hydrogen.sty`: `hplot` (axis), `theory`, `theory2`, `theory3` (curves),
  `simulation`, `simulation2` (marks), `guide` (dashed reference lines), `band` (filled regions); the
  default cycle list uses the series palette. Use `fullwidth` for wide figures. Do not use colours outside
  the palette (`sea`, `seadeep`, `seafoam`, `seamist`, `sand`, `sandline`, `ink`, `muted`, and mixes of
  them such as `sandline!55!black`).
- Run every script yourself, check the agreement, and state in the caption what was simulated (sizes,
  trials, seed). A figure whose simulation disagrees with the theory is a finding: report it.
- Use numerics also to check your theorems: constants, signs, thresholds and exercise answers. Keep
  scratch work in `build/`, not in `figures/`.

## 7. LaTeX conventions

- **Engine:** pdfLaTeX + biber. Build your chapter with `tools/build_chapter.sh chapters/NN-key.tex` and fix
  every error it reports. Do not edit the class, `hydrogen.sty`, `main.tex`, other chapters, or the tools.
  If you need a feature they lack, say so in your report.
- **Labels** carry your chapter key: `ch:key`, `sec:key:...`, `thm:key:...`, `lem:key:...`, `prop:key:...`,
  `cor:key:...`, `def:key:...`, `ass:key:...`, `obs:key:...`, `eq:key:...`, `fig:key:...`, `ex:key:...`.
  Refer with `\cref{...}` (sentence start `\Cref{...}`). Other chapters: `\cref{ch:key}` with the keys in
  OUTLINE.md; appendix results with the fixed `toolkit` labels listed there.
- **Environments:** `theorem lemma proposition corollary conjecture claim` (boxed, italic);
  `definition problem assumption` (sand box); `example remark observation` (unboxed); starred versions
  of the classical ones; `proof`, `proofsketch`, `heuristic` (◇); `chaptersummary`; `dictionary`
  (rows `model side & network side \\`); `\modelcard{...}{...}{...}{...}{...}{...}`; `aside[Title]` (grey
  box for a digression or an old argument); `fullwidth`; `exercises` with tags `\consolidation`,
  `\variation`, `\simulation`, `\research`, difficulty `\hard`, `\harder`, and `\hint{...}`.
- **Margin notes:** `\sidenote{...}` (numbered) for history, cross-references, and side comments;
  `\marginaside{...}` for unnumbered ones. Use 6–15 per chapter. Never put essential mathematics in a side
  note. **Side notes only in running text** — not inside boxes, theorem-like environments, displays,
  captions, tables, or the dictionary (LaTeX will error). Do **not** use `\footnote`.
- **Terms:** `\term{...}` at the defining occurrence and `\index{...}` for 8–20 key terms per chapter (plain
  words; subentries with `!`; avoid `@ | !` inside terms otherwise).
- **Macros available:** from the class: `\R \N \Z \Q \C \F \E \eps \ind \dd \defeq \Var \Cov \argmin \argmax
  \rank \tr \sgn \supp \poly \conv \diam \dist \abs{} \norm{} \ip{} \ceil{} \floor{}` and `\cA … \cZ`;
  from `hydrogen.sty`: `\Loss \Lhat \Sph \lmax \lmin \eqd \op \He \Unif \diag \relu \softmax \erf \arcsinh
  \arccosh \polylog`. Need more? Define it at the top of your chapter with a name that starts with your
  chapter key (e.g. `\newcommand{\icloss}{...}`), never a generic name.
- **Citations:** keys `AuthorYear` or `AuthorAuthorYear` or `AuthorEtAlYear` (e.g. `Belkin2019`,
  `SaadSolla1995`, `NakkiranEtAl2021`). Your bib file is `chapters/NN-key.bib`. Every entry must be real and
  verified with a web search (authors, title, venue, year; volume and pages only when verified). For
  preprints give the arXiv number in the `eprint`/`note` field only if verified. Mark unrefereed preprints
  as such in the text when a claim rests on them. 12–35 entries per chapter.
- **No** `\usepackage`, no changes to page layout, no `\newpage` tricks, no colours outside the palette, no
  `\footnote`.

## 8. Quality checklist before you report back

- [ ] `tools/build_chapter.sh` runs with zero LaTeX errors, zero biber errors, no undefined references or
      citations belonging to your chapter, and no overfull box over 3pt.
- [ ] The Halmos linter (run by the build script) shows no WARN, or each remaining one is deliberate.
- [ ] Every theorem is true as stated, with all hypotheses; every proof is complete or explicitly a sketch
      or a heuristic; constants are right (checked numerically where possible); edge cases are named.
- [ ] Every figure script runs, and the figure shows what the caption says.
- [ ] Every historical and bibliographic claim is verified; no invented quotations or anecdotes.
- [ ] The anatomy is followed, including the three paragraphs of "What the model gets wrong", the model
      card, the exercise mix, and "Translating the literature".
- [ ] The global notation is used; deviations are deliberate and explained.
- [ ] Length is 15–20 pages.

When you finish, report: the page count; the list of main results, saying which have complete proofs,
which are sketched and which are heuristic; the figures and what they show; any facts you could not
verify (and what you did about them); and anything the editor should check.
