# Dissolving Problems — Writing Guide for Chapter Authors

*Dissolving Problems: A Casebook of Theory-Building* is a graduate textbook in English. It teaches one
skill: finding the setting in which a hard problem stops being hard. Each chapter is a **case**. It
shows a problem being fought with hammer and chisel, then the single reframing that dissolved it, then
how short the old proofs became and what new results came within reach, and finally what the reframing
cost and where it fails.

Read this guide and `OUTLINE.md` (the book plan and your chapter specs) before writing. Look at
`../template/sample.tex` (a tiny case built with every macro of the class) and, once it exists,
`chapters/00-prologue.tex` (the book's voice).

---

## 1. Audience and level

Readers are graduate students and researchers in mathematics, theoretical computer science, machine
learning theory, and statistics. Assume undergraduate analysis, linear algebra, probability and
algorithms, and the mathematical maturity to read proofs. Do **not** assume the specialist background of
your chapter: define what the chapter needs. A strong student from a neighbouring field must be able to
read every chapter on its own.

## 2. The case template (every chapter follows it)

```
\casechapter{Title}{Subtitle}{Move}{ch:key}
\begin{refsection}[chapters/NN-key.bib]

\begin{chaptersummary} 3–5 sentences: the problem, the move, what dissolved. \end{chaptersummary}

\section{Before: <specific title>}        % ~20–25% of the chapter
\section{The move: <specific title>}      % ~15–20%
\section{After: <specific title>}         % ~35–40%  (may be split into two "After" sections)
\section{Costs and limits}                % ~10%
\section{The move, distilled}             % ~5%
\begin{exercises} ... \end{exercises}
\notesandrefs ... \printchapterbib
\end{refsection}
```

**Before.** Present the problem in its original setting, with enough of the old mathematics that the
reader feels the difficulty: state the old results precisely, show an old proof or the skeleton of one
(an `aside` box is good for a skeleton), and point out where it is ad hoc, where hypotheses look
artificial, and which questions it could not answer. Be historically accurate.

**The move.** Introduce the reframing: the new object, definition, or viewpoint. Explain why it is
natural in hindsight and what a reader would have had to notice to find it. Grothendieck's advice was to
*stoop low*: the key object is often humble (a noodle instead of a needle, a function on the unit square
instead of a graph, a tree of labels, a symmetry the data has for free). Say so when it is.

**After.** Re-prove the old results in the new setting, visibly shorter, with complete proofs. Then
prove (or carefully state, with a proof sketch) results that the old setting could not reach. This is
the heart of the chapter: the reader should feel the "of course".

**Costs and limits.** Three paragraphs are mandatory, each opened with `\paragraph{...}`:
- `\paragraph{Where did the difficulty go?}` Difficulty is conserved. Say exactly where it was paid
  (a construction, an existence theorem, a compactness argument, a running time) and why paying it once
  is a bargain. If it must be paid again in every application, the reframing is a renaming — say so.
- `\paragraph{What did not dissolve.}` Problems the move does not solve, settings where it fails,
  open questions, known barriers. Be specific and honest.
- `\paragraph{Other seas.}` (optional) Competing frameworks or later refinements.

**The move, distilled.** A `move` box with four labelled parts — `\movepart{In one sentence}`,
`\movepart{Triggers}` (signals that a problem is ripe for this move), `\movepart{Recipe}` (the steps),
`\movepart{Anti-pattern}` (what a fake version of this move looks like) — followed by
`\dissolutiontest{...}{...}{...}{...}` filled concretely for this chapter (shorter proofs / fewer
assumptions / unification / new reach; write "none" where honest), and one or two paragraphs relating
the move to other chapters (use `\cref{ch:key}`).

**Exercises.** 9–13 items: about 5 `\consolidation` (re-derive or check results of the chapter in the
new setting), 3–5 `\transfer` (apply *the same move* to a new problem — this is the point of the book),
1–2 `\research` (open-ended but well posed; may point to open problems or to ML/TCS/statistics
questions). Mark difficulty with `\hard` or `\harder` after the category tag. Give `\hint{...}` for
starred items. Every exercise must be correct and solvable as stated (except research items, which must
say what is open).

**Notes and references.** Two to five paragraphs of history and pointers, with author–year citations
(`\textcite`, `\parencite`), then `\printchapterbib`.

## 3. Length

Target **13–18 compiled pages** per chapter (about 6,000–8,500 words including mathematics). The
"After" part must contain at least two complete proofs that are visibly shorter than their "Before"
counterparts.

## 4. Voice and proof style

### 4.1 Voice
- Precise, warm, and economical: the register of a good graduate textbook with an essayist's sense of
  story (think O'Donnell's *Analysis of Boolean Functions*, Aigner–Ziegler, Tao's expository writing).
- "We" means the author and the reader. Prefer plain declarative sentences and the imperative.
- Tell the history as a story with real dates and people, but never invent anecdotes, motives, or
  quotations. Paraphrase. Do not quote more than a short phrase (under 15 words) from any source, and
  attribute it. Never quote Grothendieck's *Récoltes et Semailles* beyond a short phrase.
- No hype ("revolutionary", "beautiful theory", "magic"). Let the mathematics impress.
- Be honest about status: what is proved, what is conjectured, what is folklore.

### 4.2 Proofs (Halmos rules — run the linter; see §6)
1. State the result; then prove it. Open every proof longer than a paragraph with its plan in one
   sentence ("The plan is to …").
2. Write forward, from hypotheses to conclusion. Motivate a choice where it is made.
3. Before a chain of (in)equalities, say in words what it does. Split chains that do more than two things.
4. Name trivial and degenerate cases and dispose of them.
5. Never repeat an argument: extract a lemma instead.
6. Cite every hypothesis where it is used, including parameter ranges.
7. Number only displays that are cited later (`equation` + `\label`); use `\[ \]` otherwise. Punctuate
   displays as part of the sentence.
8. End proofs with the class's tombstone (automatic). If a proof ends in a display, put `\qedhere` in it.
9. Use `proofsketch` for sketches and say what is omitted and where to find it.

### 4.3 Words and symbols
- Never start a sentence with a symbol ("The set $A$ is finite", not "$A$ is finite").
- Pair every "if" with "then". Write "every", "each", or "some" — never "any".
- Banned: clearly, obviously, trivially, evidently, "it is easy to see", "it is straightforward".
  If something is easy, show it in one line.
- Define symbols before a display, not in a trailing "where".
- One word, one meaning. No ∀, ∃, ⇒ in prose.
- American spelling. Serial comma.

## 5. LaTeX conventions (the class is `../template/missingbook.cls`)

- **Engine:** pdfLaTeX + biber. Build your chapter with `tools/build_chapter.sh chapters/NN-key.tex`
  and fix every error it reports. Do not edit the class, `main.tex`, other chapters, or the tools.
- **Labels:** every label carries your chapter key: `ch:key`, `sec:key:...`, `thm:key:...`,
  `lem:key:...`, `def:key:...`, `eq:key:...`, `fig:key:...`. Refer with `\cref{...}` (sentence start:
  `\Cref{...}`); it prints the right name for lemmas, corollaries, definitions and exercises.
  In theorem notes, `[\textcite{Key}]` prints "(Author Year)". Other chapters: `\cref{ch:vc}` → "Chapter 11" (keys in OUTLINE.md).
- **Environments:** `theorem lemma proposition corollary conjecture claim` (boxed, italic),
  `definition problem` (sand box), `example remark` (unboxed), starred versions for unnumbered,
  `proof`, `proofsketch`, `chaptersummary`, `move`, `aside[Title]`, `fullwidth` (spans the margin
  column — use for wide displays/figures/tables), `exercises`.
- **Margin notes:** `\sidenote{...}` (numbered) for historical asides, cross-references, and side
  comments; `\marginaside{...}` for unnumbered ones. Use 6–15 per chapter. Never put essential
  mathematics in a side note. **Side notes only in running text** — not inside boxes, theorem
  environments, displays, captions, or tables (LaTeX will error). Do **not** use `\footnote`.
- **Terms:** `\term{...}` at the defining occurrence (italic) and `\index{...}` for 8–20 key terms per
  chapter (plain words; subentries with `!`; avoid special characters `@ | !` inside terms otherwise).
- **Macros available:** `\R \N \Z \Q \C \F \E \eps \ind \dd \defeq \Var \Cov \argmin \argmax \rank \tr
  \sgn \supp \poly \conv \diam \dist \abs{} \norm{} \ceil{} \floor{}`, `\ip{f,g}` (or `\ip{f}{g}`; `\ip*{..}` auto-sizes), `\cA … \cZ`
  (calligraphic). Need more? Define it at the top of your chapter with a name that starts with your
  chapter key (e.g. `\newcommand{\vcGrowth}{\Pi}`), never a generic name.
- **Figures:** TikZ drawn inline, simple and clean, colours `sea`, `seadeep`, `seafoam`, `seamist`,
  `sand`, `sandline`, `muted`. 1–3 figures per chapter where they genuinely help.
- **Citations:** keys `AuthorYear` or `AuthorAuthorYear` (e.g. `Lebesgue1902`, `FreundSchapire1997`).
  Your bib file is `chapters/NN-key.bib`. Every entry must be real and verified: check authors, title,
  venue, year (and volume/pages when you include them) with web search when you are not certain. Omit a
  field rather than guess it. No URLs or DOIs unless verified. 8–25 entries per chapter.
- **No** `\usepackage`, no changes to page layout, no `\newpage` tricks, no colour other than the class
  palette, no `\footnote`.

## 6. Quality checklist before you report back

- [ ] `tools/build_chapter.sh` runs with zero LaTeX errors, zero biber errors, no undefined references
      or citations belonging to your chapter, and no overfull box over 3pt.
- [ ] The Halmos linter (run by the build script) shows no WARN, or each remaining one is deliberate.
- [ ] Every theorem is true as stated, with all hypotheses; every proof is complete or explicitly a
      sketch; constants are right; edge cases are named.
- [ ] Every historical claim (who, when, where) is verified; no invented quotations or anecdotes.
- [ ] The template is followed: summary, Before, The move, After, Costs and limits (with the two
      mandatory paragraphs), The move distilled (move box + dissolution test), Exercises, Notes and
      References.
- [ ] The chapter reads on its own, and its cross-references to other chapters use the keys in OUTLINE.md.
- [ ] Length is 13–18 pages.

When you finish, report: the page count, the list of main theorems with which ones have full proofs,
any facts you could not verify (and what you did about them), and anything you think the editor should
check.
