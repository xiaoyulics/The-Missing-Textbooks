# Singular Learning Theory for the Rest of Us — Writing Guide for Chapter Authors

*Singular Learning Theory for the Rest of Us: From y = abx to Neural Networks* is a graduate
textbook in English, volume 6 of *The Missing Textbooks*. Neural networks are singular statistical
models: many parameters give the same function, the Fisher information is degenerate, and the
classical asymptotics behind the Laplace approximation, BIC, and AIC do not apply. Sumio Watanabe
built the theory that does apply, but his books ask the reader to know resolution of singularities
first. This book teaches the same theory to deep-learning researchers. **Each chapter teaches one
concept of singular learning theory, computes it by hand on the book's running family of toy
models (starting from the two-parameter model y = abx), proves its central theorem completely,
measures it in a script, and says plainly where the theory stops.** Hironaka's resolution of
singularities is used as a black box: stated precisely, applied freely, never proved.

Read this guide and `OUTLINE.md` (the plan, the global notation, the running toy models with their
fixed values, the fixed labels, and your chapter's specification) before writing. For voice and
LaTeX quality read one finished chapter of volume 5, for example
`../verifiable-intelligence/chapters/05-sumcheck.tex`, or of volume 2,
`../hydrogen-atoms/chapters/01-doubledescent.tex`; this volume keeps their voice and conventions,
with its own chapter anatomy and design (`singular.sty`). Read the docstring of
`figures/scripts/sltkit.py`.

---

## 1. Audience and level

Deep-learning researchers and graduate students in machine learning and statistics who have met
the claim that "neural networks are singular" and want the theory behind it. Assume multivariable
calculus, linear algebra, probability at the level of a first graduate course (laws of large
numbers, the central limit theorem, convergence in law), maximum likelihood and Bayes' rule, and
the basics of training neural networks. Do **not** assume algebraic geometry, complex analysis
beyond power series, empirical-process theory, or any of Watanabe's books: a strong student who has
never heard of a blow-up must be able to read your chapter on its own, with the appendix
(`\cref{ch:toolkit}`) for shared tools. Define every geometric notion when you first use it in the
chapter (one sentence and a picture or an example on the toy models, with a pointer to the appendix
or to the chapter that develops it). Earlier chapters' results may be cited by `\cref` with the
labels listed in OUTLINE.md; also say in prose what you are citing.

The promise of the title is a promise about the reader's effort, not about rigor: every statement
is exact, every theorem is stated with all its hypotheses, and everything the book does not prove
is named and sourced.

## 2. The chapter anatomy (every chapter follows it)

```
\sltchapter{Title}{Subtitle}{the model}{ch:key}
\newtonfoot{vertices}{t}{lambda}{K}
\begin{refsection}[chapters/NN-key.bib]

\begin{chaptersummary} 3–5 sentences: the question, what was computed by hand, the theorem, what
was measured, where the theory stops. \end{chaptersummary}

\section{The question: <specific title>}  % ~10%: the concept; the regular answer; why it fails
\section{By hand: <specific title>}        % ~20–25%: the concept computed on the running models
\section{The theorem: <specific title>}    % ~30–35%; may be split into two "The theorem" sections
\section{Measuring it}                     % ~10%: a figure from a script against the predictions
\section{The ledger}                       % ~5%: a paragraph and the \ledgercard
\section{Where the theory stops}           % ~10%: three mandatory paragraphs
\begin{exercises} ... \end{exercises}
\notesandrefs ... \printchapterbib
\end{refsection}
```

**The opening.** `\sltchapter` prints the title, the subtitle, "the model — *model*", and the
crossing rule; `\newtonfoot` draws at the foot of the page the Newton diagram of the chapter's
emblem function K(a,b): its boundary vertices from top left to bottom right as `i/j` pairs, the
point t where the diagonal meets the boundary, the learning coefficient λ as printed, and K as
printed. The values for your chapter are in OUTLINE.md; keep them unless your chapter changes its
model. Do not put `\cite` in the opening arguments.

**The question.** Start concretely, from something a deep-learning researcher meets: two networks
with different weights that compute the same function, a marginal likelihood that does not behave
like BIC, an estimate of "effective dimension" that is not the parameter count, a training curve
with plateaus. Then state the chapter's concept precisely, and what the **regular** theory says
about it in a `regularanswer` box:
```
\begin{regularanswer}[the free energy]
If the model is regular, $F_n = nS_n + \tfrac d2\log n + O_p(1)$ \parencite{Schwarz1978}.
\end{regularanswer}
```
Then say why the regular answer fails, or says nothing, for singular models, on the running toy
model. End the section with the question the chapter answers.

**By hand.** The signature of the book. Compute the chapter's concept on one or more of the running
models (OUTLINE.md, "The running models") completely and explicitly: every integral, every change
of variables, every limit, with numbers the reader can check. Put the central computation in a
`computation` box as numbered steps:
```
\begin{computation}{The free energy of $y=abx$}\label{comp:key:name}
\step The energy depends on $(a,b)$ only through $c=ab$, so ...
\step The density of $c$ under the prior is ...
\step ...
\end{computation}
```
Say what one had to notice to make the computation possible. The hand computation comes before
the general theorem, and the theorem's proof should be recognizable as the hand computation done
in general.

**The theorem.** The heart of the chapter. At least **one main theorem with a complete proof** — a
proof that a careful graduate student can check line by line, with every hypothesis used and every
constant right. It is acceptable, and often right, to prove the theorem under a stated simplifying
hypothesis that the general theory removes (the loss already in normal-crossing form, a model whose
energy depends on the parameter only through one coefficient, a compact parameter set with a
positive prior, realizability), provided that the hypothesis is in the statement, the general
version is stated separately with its source, and the chapter says exactly what the general proof
adds. Further results: complete proofs where they fit, `proofsketch` (saying what is omitted and
where to find it) where they do not, and `heuristic` derivations (ending with ◇) where the honest
status is "argued, not proved". Never let a heuristic look like a proof, and never call a sketch a
proof. When a result is Watanabe's, cite the theorem (book and theorem number, or paper and
theorem number) as given in OUTLINE.md or verified by you.

**Measuring it.** Measure the chapter's quantity in a script (§6) on models where the answer is
known, and show at least **one figure that compares measurements (points with 95% intervals) with
the singular prediction (bronze curve) and the regular prediction (slate dashed curve)**. Explain
every curve, where the agreement is good, where it is not and why (finite n, lower-order terms such
as log log n, sampler bias, grid error). A measurement that contradicts a proved result is a bug in
the script or in the theorem: find it. Report the sizes, the number of repetitions, the seeds, and
what the intervals are.

**The ledger.** One or two paragraphs that say what the chapter established, then the card:
```
\ledgercard{quantity}{regular theory}{singular theory}{running example}{measured}{rests on}
```
one or two lines per field: the quantity, what the regular theory predicts, what the singular
theory predicts (with the theorem's label), its value on the chapter's running model, what the
script measured, and the hypotheses everything rests on. The epilogue collects the fifteen cards.

**Where the theory stops.** Three mandatory paragraphs, each opened with `\paragraph{...}`:
- `\paragraph{What was assumed.}` The hypotheses that the chapter's results need and that deep
  learning may violate: realizability, a compact parameter set, analyticity (ReLU is not
  analytic), a fixed model as n grows, exact Bayesian posteriors at a given temperature, a prior
  positive at the singularity. Be concrete about what each assumption costs here.
- `\paragraph{Where it breaks.}` Specific, cited results — theorems, counterexamples, or
  experiments — where the chapter's result fails, does not apply, or has not been observed. This is
  the most valuable paragraph of the chapter; make it specific, sourced, and fair.
- `\paragraph{Open problems.}` What is open, precisely stated, with citations where someone has
  posed it.

**Exercises.** 10–14 items: about 4–5 `\consolidation` (re-derive or check results of the chapter),
3–4 `\byhand` (compute a learning coefficient, a free energy, a posterior, a blow-up, on a new toy
model), 1–2 `\variation` (change the model, the prior, the truth, or the temperature and redo the
analysis), 1–2 `\measurement` (run or modify a script; say what to observe and why), 1–2
`\research` (open-ended but well posed; say what is open). Mark difficulty with `\hard` or
`\harder` after the tag; give `\hint{...}` for starred items. Every exercise must be correct and
solvable as stated; compute the answer of every `\byhand` exercise yourself, numerically where
possible. Label exercises `\label{ex:key:name}` when you refer to them.

**Notes and references.** Three to six paragraphs of history and pointers with author–year
citations, ending with `\paragraph{Translating the literature.}`, which maps the notation and terms
of the two or three main sources onto this book's (e.g. "Watanabe (2009) writes $K(w)$ for our
$K(w)$ but $F$ for the free energy without the entropy term, our $F_n - nS_n$; Lau et al. write
$\hat\lambda(w^*)$ and $\beta^*$ for our ..."). Then `\printchapterbib`.

## 3. Length

Target **16–20 compiled pages** per chapter (about 7,500–9,500 words including mathematics,
figures and boxes). The theorem sections must contain at least one complete proof of a nontrivial
theorem.

## 4. Notation

Use the global notation of OUTLINE.md. In brief:
- Data $X^n=(X_1,\dots,X_n)$ i.i.d. from the true distribution $q$; for regression, pairs
  $(x_i,y_i)$ with $y_i=f(w,x_i)+$ Gaussian noise. Model $p(x\mid w)$, parameter $w\in W\subset\R^d$,
  prior $\varphi(w)$. Defaults (the kit's): $x\sim\Normal(0,1)$ or $\Normal(0,I_k)$, unit noise
  variance, $\varphi$ uniform on $[-1,1]^d$.
- Losses `\Ln` $L_n(w)=-\frac1n\sum_i\log p(X_i\mid w)$, $L(w)=\E_q[-\log p(X\mid w)]$; entropy $S$,
  empirical entropy `\Sn`; $K(w)=L(w)-L(w_0)$ (the Kullback–Leibler divergence in the realizable
  case), `\Kn` $K_n(w)=L_n(w)-L_n(w_0)$; true parameters `\wzero`, the set `\Wzero`
  $W_0=\{w:K(w)=0\}$.
- Free energy `\Fn` $F_n=-\log\int\varphi(w)e^{-nL_n(w)}\dd w$; at inverse temperature $\beta$,
  $F_n(\beta)$; posterior expectations `\Ew{f}`, `\Ewb{f}`.
- Zeta function $\zeta(z)=\int K(w)^z\varphi(w)\dd w$; **learning coefficient** $\lambda$ (the real
  log canonical threshold) and its **multiplicity** $m$; volume $V(\eps)=\int_{K(w)<\eps}\varphi(w)\dd w$.
- Bayes generalization loss $G_n$, Bayes training loss $T_n$, functional variance $V_n$,
  `\WAIC`, `\WBIC`, singular fluctuation $\nu$; the local learning coefficient $\lambda(w^*)$ at a
  local minimum `\wstar` and its estimate $\hat\lambda(w^*)$.
- `\Prob`, `\E`, `\Var`, `\Normal`, `\Unif`, `\KL`, `\vol`, `\relu`, `\iid`.
If your chapter needs a symbol that clashes with this list, keep the global meaning and choose
another symbol for the local one; translate source notation in "Translating the literature".
Write "learning coefficient" in running text; say "real log canonical threshold" when you speak of
it as an invariant of a function, and say once per chapter that the two are the same number.

## 5. Voice, fairness and proofs

### 5.1 Voice
- Precise, warm and economical: a good graduate textbook with an essayist's sense of story
  (O'Donnell's *Analysis of Boolean Functions*, Vershynin's *High-Dimensional Probability*,
  MacKay's sense of the concrete). "We" means the author and the reader.
- The running toy models are the point of the book, not a decoration: every chapter moves from a
  question, to a computation on a toy model the reader can hold in one hand, to a theorem, to a
  measurement, and back to neural networks. Do not oversell the bridge to deep learning: say where
  a toy computation transfers exactly, where it is a useful idealization, and where it fails.
- Tell the history with real dates and people; never invent anecdotes, motives or quotations.
  Paraphrase; quote at most a short phrase (under 15 words), attributed.
- No hype ("revolutionary", "magic", "mysterious", "beautiful theory", "secretly", "dramatically",
  "explains generalization"). Let the mathematics impress. Do not oversell singular learning
  theory, and do not dismiss its critics or its proponents.
- Be explicit about status: **proved** (theorem with proof), **sketched**, **argued** (heuristic,
  ◇), **observed** (an experiment, with its setting and size), **conjectured**, **open**. Much of
  the deep-learning side is observed in small experiments or argued; say so.
- Date every claim about current practice or the state of research ("as of 2026, ...") and give a
  source.

### 5.2 Fairness
- State each result as its authors stated it, including its hypotheses and hedges, before using
  or examining it.
- Never attribute to a paper a claim it does not make. Check every theorem you attribute against the
  source (its statement, hypotheses and constants), and every number against its tables or text.
- Credit the mathematical origin of an idea (Gel'fand, Atiyah, Bernstein, Hironaka, Varchenko) and
  the statistical one (Watanabe, his students, Drton and others) when a machine-learning paper uses
  it, and credit the machine-learning paper for what it adds.

### 5.3 Proofs (Halmos rules — the build script runs a linter)
1. State the result; then prove it. Open every proof longer than a paragraph with its plan in one
   sentence ("The plan is to …").
2. Write forward, from hypotheses to conclusion. Motivate a choice where it is made.
3. Before a chain of (in)equalities, say in words what it does. Split chains that do more than two
   things.
4. Name trivial and degenerate cases and dispose of them (a regular point, a prior that vanishes,
   a truth on the boundary of $W$, $n$ too small for an asymptotic statement to say anything).
5. Never repeat an argument: extract a lemma instead.
6. Cite every hypothesis where it is used, including parameter ranges.
7. Number only displays that are cited later (`equation` + `\label`); use `\[ \]` otherwise.
   Punctuate displays as part of the sentence.
8. Proofs end with the tombstone (automatic); heuristic derivations end with ◇ (automatic). If
   either ends in a display, put `\qedhere` in the display.
9. `proofsketch` for sketches: say what is omitted and where to find it.
10. Asymptotic statements say what is random and in what sense they hold ("$F_n-nS_n-\lambda\log n
    +(m-1)\log\log n$ converges in law", "in probability", "in expectation over $X^n$"), and
    which limits are taken in which order.

### 5.4 Words and symbols
- Never start a sentence with a symbol ("The loss $K$ ...", not "$K$ ...").
- Pair every "if" with "then". Write "every", "each", or "some" — never "any".
- Banned: clearly, obviously, trivially, evidently, "it is easy to see", "it is straightforward".
  If something is easy, show it in one line.
- Define symbols before a display, not in a trailing "where".
- One word, one meaning: "learning coefficient" is $\lambda$, never the slope of a training curve;
  "singular" means that the map from parameters to distributions is not one-to-one or the Fisher
  information is degenerate at a true parameter (say which); "regular" is its opposite;
  "posterior" is the exact Bayesian posterior at the stated temperature, never the output of
  a training run; "free energy" is $F_n$ (with the empirical entropy) unless you say otherwise.
- No ∀, ∃, ⇒ in prose. American spelling. Serial comma.

## 6. Measuring it: scripts and figures

- Scripts: `figures/scripts/NN-key-name.py`, using numpy and scipy (torch only if your chapter
  trains or samples a network; call `torch.set_num_threads(2)`), and the shared kit:
  ```python
  import os, sys
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  import sltkit as sk
  ```
  Deterministic seeds; at most about two minutes per script on a laptop CPU; run from the volume
  root (`python figures/scripts/NN-key-name.py`). Each script writes whitespace-separated data with
  a header row to `figures/data/NN-key-name*.dat` with `sk.write_dat`, and prints every number the
  text quotes. Keep files small (at most a few thousand rows). Do not modify `sltkit.py`; define
  what you need in your script. Several authors share one Windows machine: keep scripts light.
- Prefer exact or quadrature computations where the kit offers them (two-parameter models on a
  grid; one-coefficient models through the density of the coefficient) and say so; use MCMC or
  SGLD where you must, and then report the sampler, its settings, and a check of its accuracy on a
  case with a known answer.
- Plot with pgfplots in the chapter, with the styles of `singular.sty`:
  ```
  \begin{figure}[t]
  \centering
  \begin{tikzpicture}
  \begin{axis}[splot,xlabel={sample size $n$},ylabel={$F_n-nS_n$},xmode=log]
  \addplot[regularpred,domain=100:1e6] {ln(x)};
  \addplot[theory] table[x=n,y=pred] {figures/data/03-abx-free.dat};
  \addplot[measured,ebars] table[x=n,y=F,y error plus=hi,y error minus=lo] {figures/data/03-abx-free.dat};
  \legend{regular: $\log n$, singular: $\tfrac12\log n-\log\log n+c$, measured}
  \end{axis}
  \end{tikzpicture}
  \caption{... say what was run: sizes, repetitions, seeds, what the intervals are.}\label{fig:key:name}
  \end{figure}
  ```
  Styles: `splot` (axis); `theory`, `theory2` (singular predictions, bronze); `regularpred` (regular
  prediction, slate dashed); `measured` (filled ink marks), `measured2` (hollow bronze circles),
  `measured3` (hollow slate squares); `ebars` (95% intervals from error columns); `guide` (dashed
  reference lines); `band` (filled region); `draws` (posterior draws in a scatter plot).
  `xmode`/`ymode` are given directly as axis options. Use `fullwidth` for wide figures. Colours:
  only the palette — `singular`, `singulardeep`, `regular`, `regularwash`, `amberwash`, `vellum`,
  `vellumline`, `ink`, `muted`, and tints such as `singular!40`.
- Every caption says what was run (sizes, number of repetitions, seeds) and what the intervals are.
  Run every script yourself and check that the figure shows what the caption and the text say. Use
  numerics also to check theorems, constants, and exercise answers. Scratch work goes in `build/`,
  not in `figures/`.
- TikZ pictures of blow-ups (charts and the exceptional divisor), Newton diagrams
  (`\newtondiagram{vertices}{t}{lambda}{K}` draws one in the text), level sets, and posterior
  shapes are welcome; draw them in the palette.

## 7. LaTeX conventions

- **Engine:** pdfLaTeX + biber. Build your chapter with `tools/build_chapter.sh NN` (run it with
  `bash` from the volume root) and fix every error it reports. Do not edit the class,
  `singular.sty`, `main.tex`, other chapters, `sltkit.py`, or the tools. If you need a feature
  they lack, say so in your report.
- **Labels** carry your chapter key: `ch:key`, `sec:key:...`, `thm:key:...`, `lem:key:...`,
  `prop:key:...`, `cor:key:...`, `def:key:...`, `ass:key:...`, `obs:key:...`, `eq:key:...`,
  `fig:key:...`, `tab:key:...`, `comp:key:...`, `ex:key:...`. Refer with `\cref{...}` (sentence
  start `\Cref{...}`). Other chapters: `\cref{ch:key}` with the keys in OUTLINE.md, and their
  results with the labels OUTLINE.md fixes; appendix results with the fixed `toolkit` labels.
- **Environments (class):** `theorem lemma proposition corollary conjecture claim` (boxed, italic);
  `definition problem` (grey box); `example remark` (unboxed); starred versions; `proof`,
  `proofsketch`; `chaptersummary`; `aside[Title]` (grey box for a digression); `fullwidth`;
  `exercises` with `\consolidation`, `\research`, `\hard`, `\harder`, `\hint{...}`.
- **Environments (volume):**
  - `regularanswer[short title]` — what the regular theory says (slate rule, slate tint).
  - `computation{Title}` — a hand computation as numbered `\step`s; numbered "Computation N.k";
    label `comp:key:name`.
  - `\ledgercard{quantity}{regular theory}{singular theory}{running example}{measured}{rests on}`.
  - `assumption` (grey box), `observation[source]` (an empirical result as reported, with a left
    rule: `\begin{observation}[\textcite{LauEtAl2025}]`), `heuristic` (◇).
  - Exercise tags `\byhand`, `\variation`, `\measurement` (with `\consolidation`, `\research`).
- **Margin notes:** `\sidenote{...}` (numbered) for history, cross-references and side comments;
  `\marginaside{...}` for unnumbered ones. Use 6–15 per chapter. Never put essential mathematics in
  a side note. **Side notes only in running text** — not inside boxes, theorem-like environments,
  displays, captions, tables, computations, the card, or proofs' displays (LaTeX will error). Do
  **not** use `\footnote`.
- **Terms:** `\term{...}` at the defining occurrence and `\index{...}` for 8–20 key terms per
  chapter (plain words; subentries with `!`, e.g. `\index{learning coefficient!local}`).
- **Macros available:** class: `\R \N \Z \Q \C \F \E \eps \ind \dd \defeq \Var \Cov \argmin \argmax
  \rank \tr \sgn \supp \poly \conv \diam \dist \abs{} \norm{} \ip{} \ceil{} \floor{}` and
  `\cA … \cZ`; volume: `\Prob \iid \Normal \Unif \diag \vol \relu \KL \WAIC \WBIC \BIC \AIC \Res
  \Ln \Kn \Fn \Sn \Wzero \wzero \wstar \Ew{} \Ewb{}`. Note that `\N` is the natural numbers; the
  normal distribution is `\Normal`. Need more? Define it at the top of your chapter with a name
  that starts with your chapter key (e.g. `\newcommand{\blowupchart}{...}`), never a generic name.
- **Tables:** booktabs rules; numbers aligned (`siunitx` `S` columns are available).
- **Citations:** keys `AuthorYear`, `AuthorAuthorYear` or `AuthorEtAlYear` (three or more authors:
  `LauEtAl2025`). Your bib file is `chapters/NN-key.bib`. Every entry must be real and verified
  (authors in order, title, venue, year; volume and pages only when verified). The digests in
  `build/research/` list verified entries; reuse them, with their keys. For arXiv preprints give
  `eprint = {2308.12108}`, `eprinttype = {arxiv}` (printed by this volume); give the refereed venue
  when there is one and it is verified. Mark unrefereed preprints as such in the text when a claim
  rests on them. 15–40 entries per chapter. Use the same key for the same work as other chapters
  do (OUTLINE.md lists the shared keys).
- **Web search is a shared, limited budget.** Verify sources by fetching arXiv
  (`https://export.arxiv.org/api/query?id_list=...`), Crossref
  (`https://api.crossref.org/works?query.bibliographic=...`), DBLP, OpenAlex, or publishers' pages
  directly; search the web only when those fail.
- **No** `\usepackage`, no changes to page layout, no `\newpage` tricks, no colours outside the
  palette, no `\footnote`.

## 8. Quality checklist before you report back

- [ ] `tools/build_chapter.sh NN` runs with zero LaTeX errors, zero biber errors, no undefined
      references or citations belonging to your chapter, and no overfull box over 3pt.
- [ ] The style lint shows no warnings, or each remaining one is deliberate.
- [ ] Every theorem is true as stated, with all hypotheses; every proof is complete or explicitly a
      sketch or a heuristic; constants are right (checked numerically where possible); every hand
      computation is checked numerically.
- [ ] Every script runs within its budget from the volume root and writes its data; every figure
      shows what its caption and the text say; every number quoted from a script matches its output;
      no measurement contradicts a proved result.
- [ ] Every claim attributed to a paper is checked; every historical and bibliographic claim is
      verified; no invented quotations or anecdotes.
- [ ] The anatomy is followed, including the regular-answer box, at least one computation box, the
      ledger card, the three paragraphs of "Where the theory stops", the exercise mix, and
      "Translating the literature".
- [ ] Length is 16–20 pages.

When you finish, report: the page count; the main results, saying which have complete proofs,
which are sketched and which are argued; the hand computations and the numbers they give; the
scripts, their runtimes and what they showed; the ledger card in one line per field; any facts you
could not verify and what you did about them; and anything the editor should check.
