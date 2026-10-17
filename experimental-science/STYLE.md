# Deep Learning as an Experimental Science — Writing Guide for Chapter Authors

*Deep Learning as an Experimental Science: A Laboratory Manual* is a graduate textbook in English, volume 4
of *The Missing Textbooks*. It teaches experimental method for deep learning: how to measure a training
run, compare methods fairly, identify causes, fit and extrapolate scaling laws, and know what a metric
measures. **Each chapter opens with a famous conclusion that turned out to be wrong or much narrower than
claimed, and performs its post-mortem**, then teaches the method that would have caught it, reproduces
the failure and the fix in a laptop-sized laboratory experiment, and ends with a protocol.

Read this guide, `OUTLINE.md` (the plan, the notation, and your chapter's specification) and the
docstring of `figures/scripts/labkit.py` before writing. For voice and LaTeX quality, read one chapter of
volume 2, for example `../hydrogen-atoms/chapters/01-doubledescent.tex`: this volume keeps that voice and
those conventions, with a different chapter anatomy and a different design (`laboratory.sty`).

---

## 1. Audience and level

Graduate students and researchers who train networks: machine learning, statistics, natural language
processing, vision, reinforcement learning, and the sciences that now use deep learning. Assume
undergraduate probability, statistics at the level of a first course (means, variances, the central
limit theorem, a $t$-test), linear algebra and calculus, and practical familiarity with training a
network. Do **not** assume the background of your chapter: a strong student who has read none of its
papers must be able to follow it. Statistical tools that several chapters share are in the appendix
(`\cref{ch:toolkit}`, labels in OUTLINE.md); cite them there, or prove what you need inline when short.

## 2. The chapter anatomy (every chapter follows it)

```
\labchapter{Title}{Subtitle}{The claim}{Its source}{The verdict, one sentence}{ch:key}
\begin{refsection}[chapters/NN-key.bib]

\begin{chaptersummary} 3–5 sentences: the claim, what went wrong, the method, what the lab shows. \end{chaptersummary}

\section{The claim: <specific title>}            % ~12%
\section{The experiment: <specific title>}       % ~8%, contains an experiment record
\section{The post-mortem: <specific title>}      % ~15%, ends with a verdict box
\section{The method: <specific title>}           % ~30%; may be split into two "The method" sections
\section{In the laboratory: <specific title>}    % ~15%: lab sheets, figures from scripts, results
\section{Protocol}                               % ~5%: one protocol box and a paragraph on its limits
\begin{exercises} ... \end{exercises}
\notesandrefs ... \printchapterbib
\end{refsection}
```

**The opening.** `\labchapter` prints the title, the subtitle, *the claim* in quotation marks with its
source in the margin, *the verdict* in one sentence, and the ruler. The claim is the conclusion as the
field received it, in one sentence of at most about fifteen words; the source says whose claim it is and,
where the received version is broader than the paper, says so ("The received reading of …"). Do not put
`\cite` in the opening arguments (they are typeset before the bibliography section starts); write
"Wilson et al.\ (2017)" as plain text. Proposed values are in OUTLINE.md; verify and improve them.

**The claim.** What was claimed, by whom, when, on what evidence, and how the field received it: citations,
adoption, the practice it shaped. State the original claim exactly (paraphrase; quote at most a short
phrase, attributed) and separate it from the received version. Put the central reported result in a
`finding` environment, e.g. `\begin{finding}[\textcite{WilsonEtAl2017}]...\end{finding}`. Say what the
original authors got right, and why the claim was reasonable to believe at the time.

**The experiment.** Reconstruct the original experiment precisely in an `experiment` record (see §7):
question, treatment, levels, what was tuned, what was held fixed, response, replicates, analysis. Then say,
in prose, what this design could and could not show. This section prepares the reader to see the flaw
before the post-mortem names it.

**The post-mortem.** What later work found, how it found it, and the specific mechanism of the failure.
Be concrete and sourced. If the post-mortem literature itself is contested, say so and give both sides.
End the section with a `verdict` box: `\begin{verdict}{reversed}...\end{verdict}` with one of the verdict
words of OUTLINE.md (it is stamped in the margin) and two to four sentences. A post-mortem is **blameless**:
the subject is the method, not the people.

**The method.** The general lesson, taught properly. Definitions, the statistics or mathematics that make
it precise, worked examples, and at least **one theorem or proposition with a complete proof** that a
careful graduate student can check line by line, with every hypothesis used. Use `proofsketch` (saying
what is omitted and where to find it) and `heuristic` (an argument that is not a proof, ending with ◇)
where honest. The method must be general: it should apply to the next experiment of this kind, not only
to the famous one.

**In the laboratory.** At least one `lab` sheet and **at least one figure produced by your scripts from real
runs**, showing individual runs (hollow marks), means with 95% intervals, and the comparison the chapter
is about. Report what happened, with numbers and intervals. If the miniature did not reproduce the famous
effect, say so plainly and explain why; adjust the setting only in ways you report. Each lab sheet has
three parts: `\labpart{Setup}` (what is trained, how many runs, seeds, runtime), `\labpart{Predict first}`
(a question the reader should answer before running), `\labpart{Look for}` (what to check in the output).

**Protocol.** One `protocol` box of 5–9 imperative steps that a researcher can follow next time: the
chapter's method as a checklist. Then one paragraph on the protocol's limits: what it does not protect
against, and when it is too expensive and what to do then.

**Exercises.** 10–14 items: about 4 `\consolidation` (re-derive or check results of the chapter), 3 `\labwork`
(run or modify a script; say what to observe and why), 2–3 `\design` (design an experiment for a stated
question under a stated budget), 2 `\critique` (audit a described experiment or reporting practice; the
description is invented and must not be attributed to a real paper unless it accurately describes it),
1–2 `\research` (open-ended but well posed; say what is open). Mark difficulty with `\hard` or `\harder`
after the tag; give `\hint{...}` for starred items. Label exercises `\label{ex:key:name}` when referred to.

**Notes and references.** Three to six paragraphs of history and pointers with author–year citations,
ending with `\paragraph{Translating the literature.}`: map the notation and terminology of the two or three
main sources onto this book's (e.g. "Hoffmann et al.\ write $A,B,\alpha,\beta$ for our
$A_N,A_D,\alpha_N,\alpha_D$"; "what Dodge et al.\ call expected validation performance is our
$\E[M_k]$"). Then `\printchapterbib`.

## 3. Length

Target **15–20 compiled pages** per chapter (about 7,000–9,500 words including mathematics, figures and
boxes).

## 4. Notation

Use the global notation of OUTLINE.md (runs and statistics, training, scale, evaluation). In particular:
$n$ is the number of runs per arm; arms are 1 and 2; $Y$ is a run's outcome; $\SE$, $\SD$; $\Prob$;
$e_k$ for the expected maximum of $k$ standard normals; $k$ for a tuning budget in trials; $N,D,C$ for
parameters, data and compute; $\Loss(N,D)=E+A_NN^{-\alpha_N}+A_DD^{-\alpha_D}$; $m$ for the number of
test items; $B$ for batch size; Adam's $\epsilon$ as `\epsilon`. If a source paper's notation clashes,
translate it and record the translation in "Translating the literature".

## 5. Voice, fairness and proofs

### 5.1 Voice
- Precise, warm and economical: a good graduate textbook with an essayist's sense of story, and the
  practical authority of a laboratory manual. "We" means the author and the reader.
- Concrete before abstract: numbers, settings, sizes, the actual experiment.
- Tell the history with real dates and people; never invent anecdotes, motives, or quotations.
  Paraphrase; quote at most a short phrase (under 15 words), attributed.
- No hype and no sneering ("debunked", "myth", "embarrassing", "revolutionary", "magic"). Let the
  evidence speak.
- Be explicit about status: **proved** (theorem with proof), **sketched**, **derived** (heuristic, ◇),
  **observed** (finding, with the evidence and its size), **reported** (a number from a paper you could
  not check), **conjectured**, **folklore**.

### 5.2 Fairness (the rule of this book)
- State each claim as its authors stated it before stating how it was received. If the paper hedged,
  say so. If the authors later corrected or extended it, say so.
- Credit what the original work got right; most famous claims were reasonable given the evidence then.
- Never attribute to a paper a claim it does not make. Every number attributed to a paper must be
  checked against the paper (or its abstract, tables or a reliable secondary source, and then say which).
- A verdict applies to a claim, not to a person or a paper. Write "the claim did not survive", not "the
  authors were wrong".

### 5.3 Proofs (Halmos rules — run the linter; see §8)
1. State the result; then prove it. Open every proof longer than a paragraph with its plan in one sentence
   ("The plan is to …").
2. Write forward, from hypotheses to conclusion. Motivate a choice where it is made.
3. Before a chain of (in)equalities, say in words what it does. Split chains that do more than two things.
4. Name trivial and degenerate cases and dispose of them.
5. Never repeat an argument: extract a lemma instead.
6. Cite every hypothesis where it is used, including parameter ranges.
7. Number only displays that are cited later (`equation` + `\label`); use `\[ \]` otherwise. Punctuate
   displays as part of the sentence.
8. Proofs end with the tombstone (automatic); heuristic derivations end with ◇ (automatic). If either ends
   in a display, put `\qedhere` in the display.

### 5.4 Words and symbols
- Never start a sentence with a symbol.
- Pair every "if" with "then". Write "every", "each", or "some" — never "any".
- Banned: clearly, obviously, trivially, evidently, "it is easy to see", "it is straightforward",
  "significant" in the everyday sense (reserve it for statistical significance, and then give the test
  and the $p$-value or interval).
- Define symbols before a display, not in a trailing "where".
- One word, one meaning: "run", "trial" (one hyperparameter configuration, possibly with several runs),
  "seed", "arm", "budget", "protocol" keep their meanings throughout.
- Every reported number carries its uncertainty or the reason it has none: write `\pmval{71.3}{0.4}`
  (mean and 95\% half-width) and say what the interval is the first time in a paragraph.
- No ∀, ∃, ⇒ in prose. American spelling. Serial comma.

## 6. Laboratory scripts and figures

- Scripts: `figures/scripts/NN-key-name.py`, using numpy, scipy, scikit-learn, torch and the shared kit:
  ```python
  import os, sys
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  import labkit as lk
  ```
  Deterministic seeds; at most about five minutes per script on a laptop CPU with two threads (labkit sets
  this); run from the volume root (`python3 figures/scripts/NN-key-name.py`). Each script writes
  whitespace-separated data with a header row to `figures/data/NN-key-name*.dat` with `lk.write_dat`, and
  prints the numbers quoted in the text. Keep row counts modest (at most a few thousand rows per file).
  Do not modify `labkit.py`; define what you need in your script.
- A lab sheet names its script by base name: `\begin{lab}{Budget curves}{04-budgets-curves}`; the margin
  shows `04-budgets-curves.py`.
- Plot with pgfplots inside the chapter, with the styles of `laboratory.sty`:
  ```
  \begin{figure}[t]
  \centering
  \begin{tikzpicture}
  \begin{axis}[lplot,xlabel={trials $k$},ylabel={expected best test error}]
  \addplot[run] table[x=k,y=err] {figures/data/04-budgets-runs.dat};
  \addplot[mean,ebars] table[x=k,y=mean,y error=hw] {figures/data/04-budgets-curve.dat};
  \addplot[fit] table[x=k,y=pred] {figures/data/04-budgets-curve.dat};
  \end{axis}
  \end{tikzpicture}
  \caption{... what was run: sizes, number of runs, seeds, what the bars are.}\label{fig:key:name}
  \end{figure}
  ```
  Styles: `lplot` (axis on graph paper), `run` (individual runs, hollow ink
  marks), `mean` (filled marks), `ebars` / `xebars` (95\% intervals from an error column), `runB`,
  `meanB`, `ebarsB` (the same in vermilion, for a second arm), `fit`
  (vermilion curve), `extrapolation` (dashed vermilion), `ciband` (filled band; use with `fill between` or
  closed paths), `reference` (dashed grey), `armA` / `armB` (two arms of a comparison, ink blue and
  vermilion). Log axes: pgfplots accepts `xmode` and `ymode` only as direct axis options, never
  inside a style, so write `\begin{loglogaxis}[lplot,lloglog,...]` (or `semilogxaxis`,
  `semilogyaxis`, or `\begin{axis}[lplot,lloglog,xmode=log,...]`); `lloglog` gives the log-paper
  minor grid. Use `fullwidth` for wide figures. Colours: only the palette (`inkblue`, `rubric`,
  `rubricdeep`, `gridline`, `paper`, `paperline`, `inkwash`, `ink`, `muted`, and tints of them such as
  `inkblue!60`).
- Every caption says what was run (sizes, number of runs, seeds) and what the intervals are. Run every
  script yourself and check that the figure shows what the caption and the text say.
- Scratch work goes in `build/`, not in `figures/`.

## 7. LaTeX conventions

- **Engine:** pdfLaTeX + biber. Build your chapter with `tools/build_chapter.sh chapters/NN-key.tex` (or
  `tools/build_chapter.sh NN`) and fix every error it reports. Do not edit the class, `laboratory.sty`,
  `main.tex`, other chapters, `labkit.py`, or the tools. If you need a feature they lack, say so in your
  report.
- **Labels** carry your chapter key: `ch:key`, `sec:key:...`, `thm:key:...`, `lem:key:...`, `prop:key:...`,
  `cor:key:...`, `def:key:...`, `hyp:key:...`, `ass:key:...`, `fd:key:...` (findings), `eq:key:...`,
  `fig:key:...`, `tab:key:...`, `lab:key:...`, `prot:key:...`, `ex:key:...`. Refer with `\cref{...}`
  (sentence start `\Cref{...}`). Other chapters: `\cref{ch:key}` with the keys in OUTLINE.md; appendix
  results with the fixed `toolkit` labels listed there.
- **Environments (from the class):** `theorem lemma proposition corollary conjecture claim` (ink-wash box,
  italic); `definition problem` (cream box); `example remark` (unboxed); starred versions; `proof`,
  `proofsketch`; `chaptersummary`; `aside[Title]` (grey box for a digression); `fullwidth`; `exercises`
  with `\consolidation`, `\research`, `\hard`, `\harder`, `\hint{...}`.
- **Environments (from `laboratory.sty`):**
  - `finding[source]` — an empirical result as reported (left ink rule): `\begin{finding}[\textcite{Key}]`.
  - `hypothesis`, `assumption` — cream boxes.
  - `heuristic` — a derivation that is not a proof (ends with ◇).
  - `experiment[whose]` — the record of an experimental design, one `\field{label}{text}` per row. Useful
    labels: question, units, treatment, levels, tuned, held fixed, varied, response, replicates, budget,
    analysis. Keep each field to one or two lines.
  - `verdict{word}` — closes the post-mortem; the word is stamped in the margin. Words: reversed,
    vanished, confounded, artifact, narrowed, misattributed, unreplicated, open.
  - `lab{Title}{script-base-name}` — a lab sheet on graph paper with `\labpart{Setup}`,
    `\labpart{Predict first}`, `\labpart{Look for}`. Numbered "Lab N.k"; label `lab:key:name`.
  - `protocol{Title}` — a numbered checklist (`\item` per step) on a cream card with a vermilion header.
    Numbered "Protocol N.k"; label `prot:key:name`.
  - Exercise tags `\labwork`, `\design`, `\critique` (in addition to `\consolidation`, `\research`).
  - Inline: `\pmval{mean}{halfwidth}`, `\script{file.py}` (a file name in the text),
    `\verdictword{reversed}` (a verdict word in running text).
- **Margin notes:** `\sidenote{...}` (numbered) for history, cross-references and side comments;
  `\marginaside{...}` for unnumbered ones. Use 6–15 per chapter. Never put essential content in a side
  note. **Side notes only in running text** — not inside boxes, theorem-like environments, displays,
  captions, tables, lab sheets, protocols, experiment records or verdicts (LaTeX will error). Do **not**
  use `\footnote`.
- **Terms:** `\term{...}` at the defining occurrence and `\index{...}` for 8–20 key terms per chapter
  (plain words; subentries with `!`).
- **Macros available:** from the class: `\R \N \Z \Q \C \F \E \eps \ind \dd \defeq \Var \Cov \argmin \argmax
  \rank \tr \sgn \supp \poly \conv \diam \dist \abs{} \norm{} \ip{} \ceil{} \floor{}` and `\cA … \cZ`; from
  `laboratory.sty`: `\Loss \Lhat \Prob \iid \SE \SD \Bias \MSE \Corr \logit \Unif \Bin \Bern \IQM \med \diag
  \softmax \relu`. Need more? Define it at the top of your chapter with a name that starts with your
  chapter key (e.g. `\newcommand{\budgetsbest}{...}`), never a generic name.
- **Tables:** booktabs rules; numbers aligned (`siunitx` `S` columns are available); intervals as
  `\pmval{..}{..}` or as an `S` column with uncertainty `71.3(4)`.
- **Citations:** keys `AuthorYear`, `AuthorAuthorYear` or `AuthorEtAlYear` (e.g. `Henderson2018` is wrong
  for a six-author paper: use `HendersonEtAl2018`). Your bib file is `chapters/NN-key.bib`. Every entry
  must be real and verified with a web search (authors, title, venue, year; volume and pages only when
  verified). Give arXiv numbers only if verified. Mark unrefereed preprints as such in the text when a
  claim rests on them. 15–40 entries per chapter.
- **No** `\usepackage`, no changes to page layout, no `\newpage` tricks, no colours outside the palette, no
  `\footnote`.

## 8. Quality checklist before you report back

- [ ] `tools/build_chapter.sh` runs with zero LaTeX errors, zero biber errors, no undefined references or
      citations belonging to your chapter, and no overfull box over 3pt.
- [ ] The Halmos linter (run by the build script) shows no WARN, or each remaining one is deliberate.
- [ ] Every claim attributed to a paper is checked; the claim and the verdict are fair and accurate.
- [ ] Every theorem is true as stated, with all hypotheses; every proof is complete or explicitly a sketch
      or a heuristic; constants are right (checked numerically where possible).
- [ ] Every script runs within its budget from the volume root, writes its data, and the figures show
      what the captions and text say; every number quoted from a lab matches the script's output.
- [ ] Every historical and bibliographic claim is verified; no invented quotations or anecdotes.
- [ ] The anatomy is followed, including the experiment record, the verdict box, lab sheets with
      "Predict first", the protocol and its limits, the exercise mix, and "Translating the literature".
- [ ] Length is 15–20 pages.

When you finish, report: the page count; the main results, saying which have complete proofs, which are
sketched and which are heuristic; the labs, their runtimes and what they showed (including any failure to
reproduce); any facts you could not verify and what you did about them; and anything the editor should
check.
