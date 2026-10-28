# The Replica Rosetta Stone — Writing Guide for Chapter Authors

*The Replica Rosetta Stone: Physical Derivations and Rigorous Proofs, Side by Side* is a graduate
textbook in English, volume 7 of *The Missing Textbooks*. Many exact predictions of high-dimensional
statistics and machine-learning theory were first derived by statistical physicists, with replicas,
cavities, TAP equations, Kac–Rice counts and dynamical mean-field theory; many were later proved by
mathematicians, with interpolation, Gaussian conditioning, comparison inequalities and large
deviations. The two literatures rarely read each other. **Each chapter takes one prediction about
inference or learning, derives it as a physicist would, sets the derivation side by side with the
rigorous route step by step, proves the central result completely (in general or in a special case
that keeps the idea), measures it in a script, and records on a status board which predictions are
proved, which are conjectures, and which are known to be wrong.**

Read this guide and `OUTLINE.md` (the plan, the global notation, the fixed labels, the shared
citation keys, and your chapter's specification) before writing, and the research digests in
`build/research/` that your specification names. For voice and LaTeX quality read one finished
chapter of volume 6, for example `../singular-learning/chapters/03-coefficient.tex`, or of volume 5,
`../verifiable-intelligence/chapters/05-sumcheck.tex`; this volume keeps their voice and conventions,
with its own chapter anatomy and design (`rosetta.sty`). Read the docstring of
`figures/scripts/rosettakit.py`.

---

## 1. Audience and level

Two readers at once. A mathematician or statistician (probability at the level of a first graduate
course, Gaussian processes at the level of Vershynin's *High-Dimensional Probability*) who has met
"the replica method predicts" in a paper and wants to know what was predicted, why physicists
believe it, and what has been proved. And a physicist or machine-learning theorist who can run a
replica computation and wants to know what the rigorous proof does and which steps of the
computation it justifies. Assume linear algebra, multivariable calculus, probability (laws of large
numbers, the central limit theorem, Gaussian integration by parts), and basic convex optimization.
Do **not** assume spin-glass theory, random-matrix theory beyond the semicircle and
Marchenko–Pastur laws (the appendix states them), or either side's textbooks: a strong student must
be able to read your chapter with the appendix (`\cref{ch:toolkit}`) and the chapters it cites.
Define every notion when you first use it in the chapter, in both languages when they differ
("the overlap $q$, which a probabilist would call the limit of the normalized inner product of two
independent samples from the posterior").

## 2. The chapter anatomy (every chapter follows it)

```
\rstchapter{Title}{Subtitle}{physics method\dict rigorous method}{ch:key}
\stonefoot{proved|partial|conjectured|false|open}
\begin{refsection}[chapters/NN-key.bib]

\begin{chaptersummary} 3–5 sentences: the prediction, how physics derives it, how it is proved,
what was measured, what is still open. \end{chaptersummary}

\section{The question: <specific title>}     % ~8%: the model, the quantity, the physical prediction
\section{The physics: <specific title>}      % ~20%: the derivation as a physicist does it
\section{The stone}                          % ~15%: the step-by-step translation
\section{The proof: <specific title>}        % ~25–30%; may be split into two "The proof" sections
\section{Measuring it}                       % ~10%: a script against the prediction and the bounds
\section{The status board}                   % ~5%: a paragraph and the \statusboard
\section{Where the translation fails}        % ~8%: three mandatory paragraphs
\begin{exercises} ... \end{exercises}
\notesandrefs ... \printchapterbib
\end{refsection}
```

**The opening.** `\rstchapter` prints the title, the subtitle, "the dictionary — *physics method ↔
rigorous method*" (write the arrow with `\dict`, e.g. `replicas\dict interpolation`), and the three
registers. `\stonefoot{...}` draws at the foot of the page the status stone of the chapter's main
prediction: `proved`, `partial` (proved under extra hypotheses or in part of the parameter range),
`conjectured`, `false`, or `open`. OUTLINE.md gives the values; change the status only if your
research shows it is wrong, and say so in your report.

**The question.** Start concretely: a learning or inference problem a reader meets (how many random
patterns can a perceptron store, when does PCA see a planted spike, what is the test error of ridge
regression when the number of features is comparable to the number of samples). Define the model and
the quantity precisely. State the physical prediction in a `physicsbox`:
```
\begin{physicsbox}[Gardner 1988]
The capacity of the spherical perceptron with margin $\kappa$ is $\alpha_c(\kappa)$, where ...
\end{physicsbox}
```
End the section with the two questions the chapter answers: how do physicists derive this, and what
is proved?

**The physics.** The derivation as a physicist would give it, honestly and completely: the
partition function or the Gibbs measure, the replicas (or the cavity, the TAP equations, the
generating functional), the order parameters, the ansatz, the saddle point, the limit. Write it in
`heuristic` environments (ending with ◇) or in running text, never as a proof. Name every
uncontrolled step where it is taken: the exchange of limits, the analytic continuation in the number
of replicas, the choice of the saddle point, the ansatz itself, the assumption that the order
parameter concentrates. Do not caricature: a physicist reading the section should recognize the
derivation and find it fair, and a mathematician should be able to follow every algebraic step.

**The stone.** The signature of the book: the physical derivation and the rigorous route side by
side, step by step, in the `stone` environment, 4–8 rows:
```
\begin{stone}{the replica computation}{Guerra's interpolation}
\stonestep{Replicate}{physics side of the step}{proof side of the step}
\stonestep{...}{...}{...}
\end{stone}
```
Each row pairs one step of the derivation with the step of the proof that does the same work: the
replicated partition function with the interpolating Hamiltonian, the saddle point with the
remainder that has a sign, the replica-symmetric ansatz with the concentration of the overlap. When a
physical step has no rigorous counterpart, say so in its row ("no counterpart: the proof never
continues in $n$"), and when the proof needs an idea the physics does not have, give it its own row.
A row is at most a third of a page; split longer steps. No side notes, footnotes or floats inside
the stone. After the stone, one or two paragraphs on what the translation shows: which physical
steps the proof justifies, which it replaces, and where the two routes part.

**The proof.** The heart of the chapter: at least **one main theorem with a complete proof** that a
careful graduate student can check line by line, with every hypothesis used and every constant
right. It is acceptable, and often right, to prove the theorem in a special case that keeps the idea
(high temperature, a Gaussian or Rademacher prior, a convex loss, a fixed number of iterations, a
Gaussian design), provided that the special hypotheses are in the statement, the general theorem is
stated separately with its source (theorem number, paper, year), and the chapter says exactly what
the general proof adds. Further results: complete proofs where they fit, `proofsketch` (saying what
is omitted and where to find it) where they do not.

**Measuring it.** Measure the chapter's quantity in a script (§6) and show at least **one figure
that compares measurements (points with 95% intervals) with the physical prediction (carnelian
curve) and, where they exist, rigorous bounds or proved curves (lapis dashed)**. Typical
measurements: a simulation at finite size (n = 200 to 5000) against the asymptotic curve, AMP
against its state evolution, the solution of a convex program against its CGMT prediction, an ODE
against SGD. Explain every curve, the finite-size effects, and where agreement is good or poor and
why. A measurement that contradicts a proved result is a bug in the script or in the theorem: find
it. Report sizes, repetitions, seeds, and what the intervals are.

**The status board.** One or two paragraphs that say what is settled, then the board:
```
\begin{statusboard}
\statusrow{the prediction, in a phrase}{\proved}{who proved it, where (Theorem 1.2 of ...)}
\statusrow{...}{\provedunder}{the extra hypotheses, and who}
\statusrow{...}{\provedpreprint}{the unrefereed preprint, with its arXiv number and version}
\statusrow{...}{\conjectured}{the evidence: numerics, partial results}
\statusrow{...}{\knownfalse}{the counterexample or the regime where it fails}
\statusrow{...}{\openq}{}
\statusrow{...}{\heuristicstatus}{an argument without a claim of exactness}
\end{statusboard}
```
List every prediction the chapter discusses, 4–10 rows. The status must be exactly right: a
prediction proved only for some priors or some parameter range is `\provedunder`, with the range; a
prediction that the physics itself says is wrong in some regime (replica symmetry beyond the de
Almeida–Thouless line) is `\knownfalse` there. The epilogue collects the fifteen boards.

**Where the translation fails.** Three mandatory paragraphs, each opened with `\paragraph{...}`:
- `\paragraph{What was assumed.}` The hypotheses of the rigorous results that the physical
  prediction does not need, and the idealizations of the model itself (Gaussian data, proportional
  limits, Bayes-optimality, known priors).
- `\paragraph{Where the physics is wrong, or not yet right.}` Specific, cited cases where the
  physical prediction fails, needs replica-symmetry breaking, holds only on average, or disagrees
  with simulations at accessible sizes; and cases where a rigorous result shows that a physical
  argument was incomplete.
- `\paragraph{Open problems.}` What is open, precisely stated, with citations where someone has
  posed it.

**Exercises.** 10–14 items: about 4–5 `\consolidation` (re-derive or check results of the chapter),
3–4 `\byhand` (carry out a replica, cavity or comparison computation on a variant), 2–3 `\translate`
(translate a physical step into a rigorous statement and prove it, or find the physical step that a
lemma of the proof replaces), 1–2 `\measurement` (run or modify a script; say what to observe and
why), 1–2 `\research` (open-ended but well posed). Mark difficulty with `\hard` or `\harder` after
the tag; give `\hint{...}` for starred items. Every exercise must be correct and solvable as stated;
compute the answer of every `\byhand` exercise yourself, numerically where possible. Label exercises
`\label{ex:key:name}` when you refer to them.

**Notes and references.** Three to six paragraphs of history and pointers with author–year
citations, ending with `\paragraph{Translating the literature.}`, which maps the notation and terms
of the main physics and mathematics sources onto this book's (e.g. "Mézard and Montanari write
$\Phi$ for our free entropy $\phi$ and $\mathrm{D}z$ for the standard Gaussian measure; Talagrand
writes $p_N(\beta)$ for $\frac1N\E\log Z_N$ and $R_{1,2}$ for the overlap $q$"). Then
`\printchapterbib`.

## 3. Length

Target **16–20 compiled pages** per chapter (about 7,500–9,500 words including mathematics,
figures and boxes). The proof sections must contain at least one complete proof of a nontrivial
theorem.

## 4. Notation

Use the global notation of OUTLINE.md; it fixes the conventions that the two literatures disagree on
(sign of the free energy, normalizations of the couplings and of the aspect ratio, the overlap, the
Gaussian measure). If your chapter needs a symbol that clashes with that list, keep the global
meaning and choose another symbol for the local one; translate source notation in "Translating the
literature". Macros of the volume: `\Prob \iid \Normal \Unif \diag \Dz \gibbs{} \Ed{}` (with the
class's `\E \R \Var \tr \sgn` and the others in §7).

## 5. Voice, fairness and proofs

### 5.1 Voice
- Precise, warm and economical: a good graduate textbook with an essayist's sense of story
  (O'Donnell's *Analysis of Boolean Functions*, Vershynin's *High-Dimensional Probability*,
  Mézard and Montanari's *Information, Physics, and Computation* at its clearest). "We" means the
  author and the reader.
- Respect both languages. The physical derivation is not a sloppy proof, and the proof is not a
  pedantic derivation; each sees things the other does not. Say what each sees.
- Tell the history with real dates and people; never invent anecdotes, motives or quotations.
  Paraphrase; quote at most a short phrase (under 15 words), attributed.
- No hype ("revolutionary", "magic", "mysterious", "beautiful", "secretly", "dramatically",
  "unreasonable effectiveness"). Let the mathematics impress.
- Be explicit about status, with the board's words: **proved** (theorem with proof), **proved
  under** (extra hypotheses, named), **conjectured** (with the evidence), **heuristic** (argued, ◇),
  **known false**, **open**; and **observed** for an experiment, with its setting and size.
- Date every claim about the state of research ("as of 2026, ...") and give a source.

### 5.2 Fairness
- State each result as its authors stated it, including its hypotheses and hedges, before using or
  examining it.
- Never attribute to a paper a claim it does not make. Check every theorem you attribute against the
  source (its statement, hypotheses and constants), and every number against its tables or text.
- Credit physicists for predictions and mathematicians for proofs, and credit the people who built
  the bridges (those who turned a cavity argument into an algorithm, or an algorithm into a proof).
  When a prediction was proved, say whether the proof followed the physical derivation or found
  another route.

### 5.3 Proofs (Halmos rules — the build script runs a linter)
1. State the result; then prove it. Open every proof longer than a paragraph with its plan in one
   sentence ("The plan is to …").
2. Write forward, from hypotheses to conclusion. Motivate a choice where it is made.
3. Before a chain of (in)equalities, say in words what it does. Split chains that do more than two
   things.
4. Name trivial and degenerate cases and dispose of them (zero temperature, a vanishing prior, an
   aspect ratio of 0 or ∞, a degenerate spike).
5. Never repeat an argument: extract a lemma instead.
6. Cite every hypothesis where it is used, including parameter ranges.
7. Number only displays that are cited later (`equation` + `\label`); use `\[ \]` otherwise.
   Punctuate displays as part of the sentence.
8. Proofs end with the tombstone (automatic); heuristic derivations end with ◇ (automatic). If
   either ends in a display, put `\qedhere` in the display.
9. `proofsketch` for sketches: say what is omitted and where to find it.
10. Limit statements say what is random, in what sense they hold (almost surely, in probability, in
    expectation over the disorder), and which limits are taken in which order (the proportional
    limit $n,d\to\infty$ with $n/d\to\alpha$; the number of iterations fixed as $n\to\infty$).

### 5.4 Words and symbols
- Never start a sentence with a symbol ("The overlap $q$ ...", not "$q$ ...").
- Pair every "if" with "then". Write "every", "each", or "some" — never "any".
- Banned: clearly, obviously, trivially, evidently, "it is easy to see", "it is straightforward".
  If something is easy, show it in one line.
- Define symbols before a display, not in a trailing "where".
- One word, one meaning: "free energy" and "free entropy" as OUTLINE.md fixes them; "replica
  symmetric" for the ansatz, never for a property that has been proved (say "the overlap
  concentrates"); "the prediction" for what physics derives, "the theorem" for what is proved;
  "exact" only for statements that hold in the stated limit.
- No ∀, ∃, ⇒ in prose. American spelling. Serial comma.

## 6. Measuring it: scripts and figures

- Scripts: `figures/scripts/NN-key-name.py`, using numpy and scipy, and the shared kit:
  ```python
  import os, sys
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  import rosettakit as rk
  ```
  Deterministic seeds; at most about two minutes per script on a laptop CPU; run from the volume
  root (`python figures/scripts/NN-key-name.py`). Each script writes whitespace-separated data with
  a header row to `figures/data/NN-key-name*.dat` with `rk.write_dat`, and prints every number the
  text quotes. Keep files small (at most a few thousand rows). Do not modify `rosettakit.py`; define
  what you need in your script. Several authors share one Windows machine: keep scripts light.
- Solve saddle-point and state-evolution equations with the kit's Gaussian quadrature and damped
  fixed-point iteration, check convergence, and say which branch (fixed point) you report when there
  are several. Simulate at sizes where the run fits the budget, and show the finite-size trend when
  it matters.
- Plot with pgfplots in the chapter, with the styles of `rosetta.sty`:
  ```
  \begin{figure}[t]
  \centering
  \begin{tikzpicture}
  \begin{axis}[rplot,xlabel={$\alpha=n/d$},ylabel={test error}]
  \addplot[physpred] table[x=alpha,y=pred] {figures/data/09-ridge-curve.dat};
  \addplot[measured,ebars] table[x=alpha,y=err,y error plus=hi,y error minus=lo] {figures/data/09-ridge-sim.dat};
  \legend{prediction, simulation ($d=500$)}
  \end{axis}
  \end{tikzpicture}
  \caption{... say what was run: sizes, repetitions, seeds, what the intervals are.}\label{fig:key:name}
  \end{figure}
  ```
  Styles: `rplot` (axis); `physpred`, `physpred2` (physical predictions, carnelian); `proofpred`
  (rigorous bounds or proved curves, lapis dashed); `measured` (filled ink marks), `measured2`
  (hollow carnelian circles), `measured3` (hollow lapis squares); `ebars` (95% intervals from error
  columns); `guide` (dashed reference lines); `band` (filled region). Use `fullwidth` for wide
  figures. Colours: only the palette — `stone`, `stonedeep`, `stonelight`, `physics`,
  `physicswash`, `proof`, `proofwash`, `papyrus`, `papyrusline`, `ink`, `muted`, and tints such as
  `physics!40`.
- Every caption says what was run (sizes, number of repetitions, seeds) and what the intervals are.
  Run every script yourself and check that the figure shows what the caption and the text say. Use
  numerics also to check theorems, constants, and exercise answers. Scratch work goes in `build/`,
  not in `figures/`.

## 7. LaTeX conventions

- **Engine:** pdfLaTeX + biber. Build your chapter with `tools/build_chapter.sh NN` (run it with
  `bash` from the volume root) and fix every error it reports. Do not edit the class,
  `rosetta.sty`, `main.tex`, other chapters, `rosettakit.py`, or the tools. If you need a feature
  they lack, say so in your report.
- **Labels** carry your chapter key: `ch:key`, `sec:key:...`, `thm:key:...`, `lem:key:...`,
  `prop:key:...`, `cor:key:...`, `def:key:...`, `ass:key:...`, `obs:key:...`, `eq:key:...`,
  `fig:key:...`, `tab:key:...`, `comp:key:...`, `ex:key:...` (exercises), `exa:key:...`
  (examples), `conj:key:...`, `rem:key:...`. Refer with `\cref{...}` (sentence start `\Cref{...}`).
  Other chapters: `\cref{ch:key}` with the keys in OUTLINE.md, and their results with the labels
  OUTLINE.md fixes; appendix results with the fixed `toolkit` labels.
- **Environments (class):** `theorem lemma proposition corollary conjecture claim` (boxed, italic);
  `definition problem` (grey box); `example remark` (unboxed); starred versions; `proof`,
  `proofsketch`; `chaptersummary`; `aside[Title]` (grey box for a digression); `fullwidth`;
  `exercises` with `\consolidation`, `\research`, `\hard`, `\harder`, `\hint{...}`.
- **Environments (volume):**
  - `physicsbox[short title]` — the physical prediction (carnelian rule, carnelian tint).
  - `stone{physics heading}{proof heading}` with `\stonestep{name}{physics}{proof}` — the
    side-by-side translation across the full width.
  - `statusboard` with `\statusrow{prediction}{status}{proof or evidence}` and the statuses
    `\proved \provedunder \provedpreprint \conjectured \heuristicstatus \knownfalse \openq`;
    `\statusstone{status}` draws a small stone inline. A result proved only in an unrefereed
    preprint (many from 2025–2026) is `\provedpreprint`, never `\proved`; name the preprint with its
    arXiv number and version, and if its authors state that the proof was written with AI
    assistance, say so in their words.
  - `computation{Title}` — a hand computation as numbered `\step`s; numbered "Computation N.k";
    label `comp:key:name`.
  - `assumption` (grey box), `observation[source]` (an empirical result as reported, with a left
    rule), `heuristic[title]` (a physical derivation, ending with ◇; the default title is
    "Physical derivation").
  - Exercise tags `\byhand`, `\translate`, `\measurement` (with `\consolidation`, `\research`).
- **Margin notes:** `\sidenote{...}` (numbered) for history, cross-references and side comments;
  `\marginaside{...}` for unnumbered ones. Use 6–15 per chapter. Never put essential mathematics in a
  side note. **Side notes only in running text** — not inside boxes, the stone, the status board,
  theorem-like environments, displays, captions, tables, computations, or proofs' displays (LaTeX
  will error). Do **not** use `\footnote`.
- **Terms:** `\term{...}` at the defining occurrence and `\index{...}` for 8–20 key terms per
  chapter (plain words; subentries with `!`, e.g. `\index{overlap!concentration}`).
- **Macros available:** class: `\R \N \Z \Q \C \F \E \eps \ind \dd \defeq \Var \Cov \argmin \argmax
  \rank \tr \sgn \supp \poly \conv \diam \dist \abs{} \norm{} \ip{} \ceil{} \floor{}` and
  `\cA … \cZ`; volume: `\Prob \iid \Normal \Unif \diag \Dz \gibbs{} \Ed{}`. Note that `\N` is the
  natural numbers; the normal distribution is `\Normal`. Need more? Define it at the top of your
  chapter with a name that starts with your chapter key (e.g. `\newcommand{\skParisi}{...}`), never a
  generic name.
- **Tables:** booktabs rules; numbers aligned (`siunitx` `S` columns are available).
- **Citations:** keys `AuthorYear`, `AuthorAuthorYear` or `AuthorEtAlYear` (three or more authors).
  Your bib file is `chapters/NN-key.bib`. Every entry must be real and verified (authors in order,
  title, venue, year; volume and pages only when verified). The digests in `build/research/` list
  verified entries; reuse them, with their keys. For arXiv preprints give `eprint = {...}`,
  `eprinttype = {arxiv}` (printed by this volume); give the refereed venue when there is one and it
  is verified. Mark unrefereed preprints as such in the text when a claim rests on them. 15–40
  entries per chapter. Use the same key for the same work as other chapters do (OUTLINE.md lists the
  shared keys).
- **Web search is a shared, limited budget.** Verify sources by fetching arXiv
  (`https://export.arxiv.org/api/query?id_list=...`), Crossref
  (`https://api.crossref.org/works?query.bibliographic=...`), zbMATH, OpenAlex, or publishers' pages
  directly; search the web only when those fail.
- **No** `\usepackage`, no changes to page layout, no `\newpage` tricks, no colours outside the
  palette, no `\footnote`. Write files with the file-editing tools, not shell heredocs (on this
  Windows machine the shell can collapse `\\`).

## 8. Quality checklist before you report back

- [ ] `tools/build_chapter.sh NN` runs with zero LaTeX errors, zero biber errors, no undefined
      references or citations belonging to your chapter, and no overfull box over 3pt.
- [ ] The style lint shows no warnings, or each remaining one is deliberate.
- [ ] Every theorem is true as stated, with all hypotheses; every proof is complete or explicitly a
      sketch; every physical derivation is marked as such; constants are right (checked numerically
      where possible); every hand computation is checked numerically.
- [ ] Every script runs within its budget from the volume root and writes its data; every figure
      shows what its caption and the text say; every number quoted from a script matches its output;
      no measurement contradicts a proved result.
- [ ] Every status on the board is exactly right and sourced; every claim attributed to a paper is
      checked; every historical and bibliographic claim is verified; no invented quotations or
      anecdotes.
- [ ] The anatomy is followed, including the physics box, the stone, at least one computation box
      or a complete hand derivation, the status board, the three paragraphs of "Where the
      translation fails", the exercise mix, and "Translating the literature".
- [ ] Length is 16–20 pages.

When you finish, report: the page count; the main results, saying which have complete proofs, which
are sketched and which are physical derivations; the scripts, their runtimes and what they showed;
the status board in one line per row; any facts you could not verify and what you did about them;
and anything the editor should check.
