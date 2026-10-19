# Verifiable Intelligence — Writing Guide for Chapter Authors

*Verifiable Intelligence: Interactive Proofs as a Theory of AI Oversight* is a graduate textbook in
English, volume 5 of *The Missing Textbooks*. When an AI system is more capable than the people
who oversee it, they can no longer check its answers by finding the answers themselves. What can a
weak verifier learn from a strong prover that may lie? Complexity theory has studied exactly this
question since 1985, and AI-safety research has rediscovered pieces of it since 2018. This book
teaches the theory systematically for the oversight problem. **Each chapter teaches one oversight
protocol, proves its main theorem completely, runs the protocol against honest and cheating
provers, and answers the same three questions:**

1. **Is honesty an equilibrium?** (Is telling the truth a best response for the prover — in what
   game, under what equilibrium notion — and can the honest prover afford to play it?)
2. **What does the verifier spend?** (Time, randomness, rounds, queries to a proof, samples, calls
   to a human judge.)
3. **How often is a lie caught?** (The probability that a cheating prover is rejected, against
   which class of cheaters, per round and after repetition.)

Read this guide and `OUTLINE.md` (the plan, the global notation, the fixed appendix labels, and
your chapter's specification) before writing. For voice and LaTeX quality read one finished chapter
of volume 3, `../optimizers-in-the-wild/chapters/05-momentum.tex` or any other there; this volume
keeps that voice and those conventions, with its own chapter anatomy and design (`verifiable.sty`).
Read the docstring of `figures/scripts/protokit.py`.

---

## 1. Audience and level

Graduate students and researchers in machine learning and AI safety who want the theory behind
scalable oversight, and theoretical computer scientists who want to see where their theory meets
AI oversight. Assume undergraduate discrete mathematics, probability, linear algebra, and algorithms
(big-O, polynomial time, reductions at the level of a first algorithms course). Do **not** assume a
course in complexity theory or cryptography: a strong student who has never seen an interactive
proof must be able to read your chapter on its own, with the appendix (`\cref{ch:toolkit}`) for
shared tools. Define every complexity class you use when you first use it in the chapter (one
sentence, with a pointer to the appendix). Earlier chapters' results may be cited by `\cref` with
the labels listed in OUTLINE.md; also say in prose what you are citing.

## 2. The chapter anatomy (every chapter follows it)

```
\protocolchapter{Title}{Subtitle}{Protocol name}{band}{ch:key}
\begin{refsection}[chapters/NN-key.bib]

\begin{chaptersummary} 3–5 sentences: the oversight problem, the protocol, what was proved, the
three answers in a phrase each, what the model leaves out. \end{chaptersummary}

\section{The oversight problem: <specific title>}  % ~10–12%: an oversight box; who proposed what
\section{The protocol: <specific title>}          % ~10–12%: definitions; protocol box(es); a worked run; dictionary
\section{The theory: <specific title>}            % ~35–40%; may be split into two "The theory" sections
\section{Running the protocol}                    % ~8–10%: a figure from a script: honest vs cheating provers against the bounds
\section{The three questions}                     % ~10–12%: \questionone \questiontwo \questionthree, then \protocolcard
\section{What the model leaves out}               % ~10–12%: three mandatory paragraphs
\begin{exercises} ... \end{exercises}
\notesandrefs ... \printchapterbib
\end{refsection}
```

**The opening.** `\protocolchapter` prints the title, the subtitle, "the protocol — *name*", two
rails, and at the foot of the page the *band*: the message pattern of the chapter's protocol, drawn
between a prover rail and a verifier rail. The proposed values of all five arguments are in
OUTLINE.md; keep the band unless the protocol you write differs. The band is a comma-separated list
of tokens without spaces: `P` prover message, `p` short prover message, `A`/`a` second prover or
debater, `V` verifier message, `S` proof string (first), `Q` query to it, `K` hidden cell, `C`
commitment, `H` call to a human judge, `X` samples from the world, `R` payment, `D` verdict, `o` gap.
Do not put `\cite` in the opening arguments.

**The oversight problem.** Start concretely: a situation in which a person (or a weaker system)
must accept or reject the output of a stronger AI system that may be wrong or deceptive, and cannot
produce the answer itself. State it in an `oversight` box:
```
\begin{oversight}[a patch that claims to be safe]
A model submits a 40{,}000-line patch and claims that it introduces no memory-safety bug. ...
\end{oversight}
```
Then say who proposed the protocol for oversight (with citations), what the complexity-theory
ancestor is, and why the problem is hard: why the overseer cannot simply check the answer. End the
section with the question that the protocol answers. Scenarios may be invented (say so if a reader
could mistake one for a real event); claims about real systems, papers and events must be sourced.

**The protocol.** Define the setting completely: who the parties are, what they can compute, what
they know, what is exchanged, how the verifier decides; state completeness and soundness (or the
chapter's analogues) as definitions. Write the protocol in a `protocol` box as a transcript:
```
\begin{protocol}{Sumcheck}\label{prot:sumcheck:main}
\item[\msg{P}{V}] Send the claimed value $C$ and the polynomial $g_1(X_1)$.
\item[\act{V}] Check that $g_1(0)+g_1(1)=C$; otherwise \rej. Pick $r_1\in\F$ uniformly at random.
\phase{Round $j=2,\dots,\ell$}
\item[\msg{V}{P}] Send $r_{j-1}$.
...
\end{protocol}
```
Then a worked run on a small instance (numbers the reader can check by hand), and a `dictionary`
box translating the protocol into oversight (left: the protocol; right: oversight):
```
\begin{dictionary}
verifier $\V$ & the overseer: a person, a weaker model, or a test harness \\
soundness error $s$ & the chance that a false claim survives the check \\
\end{dictionary}
```
Say what one had to notice to see the oversight problem this way.

**The theory.** The heart of the chapter. At least **one main theorem with a complete proof** — a
proof that a careful graduate student can check line by line, with every hypothesis used and every
constant right. Then further results: complete proofs where they fit, `proofsketch` (saying what is
omitted and where to find it) where they do not, and `heuristic` derivations (ending with ◇) where
the honest status is "argued, not proved". Never let a heuristic look like a proof, and never call a
sketch a proof. State completeness and soundness separately and prove both. When a result holds
only against a restricted class of cheaters (computationally bounded provers, provers that cannot
communicate, provers that follow a learned policy), say so in the statement.

**Running the protocol.** Implement the protocol in a script (§6): the honest prover and **at least
one cheating prover** — the natural lie and, where possible, the best cheating strategy the theory
allows — interact with the verifier many times. Show at least **one figure that compares measured
acceptance rates (points with 95% intervals) with the bounds or exact values of the theory
(curves)**, and explain every curve, where the bound is tight, and where it is loose and why. A
measured rate that violates a proved bound is a bug in the script or the theorem: find it. A small
table of costs (verifier time or operations, prover time, communication) measured on the
implementation is welcome.

**The three questions.** Answer the book's three questions for this protocol, each in a paragraph
opened by its macro:
```
\questionone   % prints "(1) Is honesty an equilibrium?"
\questiontwo   % prints "(2) What does the verifier spend?"
\questionthree % prints "(3) How often is a lie caught?"
```
Each answer is precise and refers to the theorems: name the game and the equilibrium notion (a
one-prover protocol is a game in which the prover is paid when the verifier accepts; say what
completeness and soundness make the honest strategy), the honest prover's cost, the verifier's
resources with their dependence on the input size and the security or error parameters, and the
catch probability against each class of cheaters, per round and after repetition. When the answer is
"no", "unknown" or "only under an assumption", say so plainly. Close the section with the card:
```
\protocolcard{what it checks}{the parties}{honesty}{verifier's cost}{catching a lie}{what it rests on}
```
one or two lines per field.

**What the model leaves out.** Three mandatory paragraphs, each opened with `\paragraph{...}`:
- `\paragraph{What was idealized away.}` The gap between the protocol's model and real oversight:
  the verifier is a person or a model, not a polynomial-time machine with perfect judgment; the
  prover is a trained network, neither unbounded nor a worst-case adversary; the claim must first be
  stated formally; the human's errors may be systematic. Be concrete about what each idealization
  costs here.
- `\paragraph{Where it breaks.}` Specific, cited results — theorems, attacks or experiments — where
  the protocol's guarantee fails, does not apply, or has not been observed in practice. This is the
  most valuable paragraph of the chapter; make it specific, sourced and fair.
- `\paragraph{Open problems.}` What is open, precisely stated, with citations where someone has
  posed it.

**Exercises.** 10–14 items: about 4–5 `\consolidation` (re-derive or check results of the chapter),
2–3 `\variation` (change the protocol or the model and re-analyze: another field size, more rounds,
a noisy verifier, a bounded prover), 2–3 `\attack` (design a cheating prover against a stated
protocol and compute its acceptance probability, or find the flaw in a proposed oversight protocol;
proposed protocols are invented and must not be attributed to real papers unless they accurately
describe them), 1–2 `\simulation` (run or modify a script; say what to observe and why), 1–2
`\research` (open-ended but well posed; say what is open). Mark difficulty with `\hard` or `\harder`
after the tag; give `\hint{...}` for starred items. Every exercise must be correct and solvable as
stated. Label exercises `\label{ex:key:name}` when you refer to them.

**Notes and references.** Three to six paragraphs of history and pointers with author–year
citations, ending with `\paragraph{Translating the literature.}`, which maps the notation and terms
of the two or three main sources onto this book's (e.g. "Irving et al.\ call the verifier the judge
and the provers agents; Brown-Cohen et al.\ write $\mathcal{O}$ for the human oracle, our
$\party{H}$"). Then `\printchapterbib`.

## 3. Length

Target **16–20 compiled pages** per chapter (about 7,500–9,500 words including mathematics, figures
and boxes). The theory sections must contain at least one complete proof of a nontrivial theorem.

## 4. Notation

Use the global notation of OUTLINE.md. In brief:
- Parties are sans-serif: verifier `\V` ($\mathsf V$; "the judge" in debate), prover `\Prov`
  ($\mathsf P$, the honest prover's strategy), an arbitrary — possibly cheating — prover `\Prov^*`,
  provers `\Prov_1,\Prov_2`, debaters `\party{A}`, `\party{B}`, the human judge consulted as an
  oracle `\party{H}`. Never write a bare $P$ for the prover.
- Complexity classes are bold, always through `\cc{...}`: `\cc{P}`, `\cc{NP}`, `\cc{coNP}`,
  `\cc{BPP}`, `\cc{MA}`, `\cc{AM}`, `\cc{IP}`, `\cc{PSPACE}`, `\cc{EXP}`, `\cc{NEXP}`, `\cc{MIP}`,
  `\cc{PCP}(r,q)`, `\cc{\#P}`. 
- Input $x$ of length $n$; language $L$; the verifier's output `\interact{\Prov}{\V}(x)` $\in\{$`\acc`,
  `\rej`$\}$; completeness $c$, soundness error $s$, catch probability $1-s$; rounds $k$; queries
  $q$; random bits $\rho$ (or $r$ for a random field element); repetitions $t$.
- `\F` a finite field, $\abs{\F}$ its size; $\ell$ variables; individual degree $d$; `\tilde f` the
  multilinear extension; circuits of size $S$ and depth $D$; running time $T$.
- Security parameter $\lambda$, `\negl`$(\lambda)$; samples $m$, hypothesis class $\cH$,
  distribution $\cD$; `\Prob`, `\E`, `\bits`.
If your chapter needs a symbol that clashes with this list, keep the global meaning and choose
another symbol for the local one; translate source notation in "Translating the literature".

## 5. Voice, fairness and proofs

### 5.1 Voice
- Precise, warm and economical: a good graduate textbook with an essayist's sense of story
  (Arora and Barak's clarity, O'Donnell's *Analysis of Boolean Functions*, Goldreich's conceptual
  care). "We" means the author and the reader.
- The oversight frame is the point of the book, not a decoration: every chapter moves from an
  oversight problem, to a protocol on paper, to a theorem, back to the problem. Do not oversell the
  bridge: say where the analogy between a complexity-theoretic verifier and a human overseer is
  exact, where it is a useful idealization, and where it fails.
- Tell the history with real dates and people; never invent anecdotes, motives or quotations.
  Paraphrase; quote at most a short phrase (under 15 words), attributed.
- No hype ("revolutionary", "magic", "mysterious", "beautiful theory", "secretly", "dramatically",
  "solves alignment"). Let the mathematics impress. Do not oversell a protocol or a paper; do not
  dismiss one either.
- Be explicit about status: **proved** (theorem with proof), **sketched**, **argued** (heuristic,
  ◇), **observed** (an experiment, with its setting and size), **conjectured**, **proposed** (an
  oversight protocol suggested but not analyzed), **open**. Much of the AI-safety side is proposed or
  observed in small experiments; say so.
- Date every claim about current practice or capability ("as of 2025, ...") and give a source.

### 5.2 Fairness
- State each proposal as its authors stated it, including its hedges, before examining it.
- Never attribute to a paper a claim it does not make. Check every theorem you attribute against the
  paper (its statement, hypotheses and constants), and every number against its tables or text.
- Credit the complexity-theory origin of an idea when an AI-safety paper rediscovers it, and credit
  the AI-safety paper for what it adds (a new model, a new question, an experiment).

### 5.3 Proofs (Halmos rules — the build script runs a linter)
1. State the result; then prove it. Open every proof longer than a paragraph with its plan in one
   sentence ("The plan is to …").
2. Write forward, from hypotheses to conclusion. Motivate a choice where it is made.
3. Before a chain of (in)equalities, say in words what it does. Split chains that do more than two
   things.
4. Name trivial and degenerate cases and dispose of them (an empty proof, a zero polynomial, a
   field too small, a verifier that rejects everything).
5. Never repeat an argument: extract a lemma instead.
6. Cite every hypothesis where it is used, including parameter ranges.
7. Number only displays that are cited later (`equation` + `\label`); use `\[ \]` otherwise.
   Punctuate displays as part of the sentence.
8. Proofs end with the tombstone (automatic); heuristic derivations end with ◇ (automatic). If
   either ends in a display, put `\qedhere` in the display.
9. `proofsketch` for sketches: say what is omitted and where to find it.
10. Probability statements say over what the probability is taken (the verifier's coins, the
    prover's coins, the input distribution) and which quantifier comes first ("for every prover
    $\Prov^*$, with probability at least $1-s$ over $\V$'s coins").

### 5.4 Words and symbols
- Never start a sentence with a symbol ("The verifier $\V$ ...", not "$\V$ ...").
- Pair every "if" with "then". Write "every", "each", or "some" — never "any".
- Banned: clearly, obviously, trivially, evidently, "it is easy to see", "it is straightforward".
  If something is easy, show it in one line.
- Define symbols before a display, not in a trailing "where".
- One word, one meaning: "verifier" (the party in a protocol) and "overseer" (the person or system
  in the oversight problem) are kept apart; "prover" and "model" likewise; "sound" and "complete"
  only in their technical senses; "honest" means "follows the prescribed strategy".
- No ∀, ∃, ⇒ in prose. American spelling. Serial comma.

## 6. Running the protocol: scripts and figures

- Scripts: `figures/scripts/NN-key-name.py`, using numpy and scipy (torch only if your chapter
  trains a network; call `torch.set_num_threads(2)`), and the shared kit:
  ```python
  import os, sys
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  import protokit as pk
  ```
  Deterministic seeds; at most about two minutes per script on a laptop CPU; run from the volume root
  (`python3 figures/scripts/NN-key-name.py`). Each script writes whitespace-separated data with a
  header row to `figures/data/NN-key-name*.dat` with `pk.write_dat`, and prints every number the text
  quotes. Keep files small (at most a few thousand rows). Do not modify `protokit.py`; define what
  you need in your script. Several authors share one machine: keep scripts light.
- Plot with pgfplots in the chapter, with the styles of `verifiable.sty`:
  ```
  \begin{figure}[t]
  \centering
  \begin{tikzpicture}
  \begin{axis}[vplot,xlabel={field size $\abs{\F}$},ylabel={acceptance rate},xmode=log,ymode=log]
  \addplot[bound] table[x=F,y=bound] {figures/data/05-sumcheck-rates.dat};
  \addplot[cheat,ebars] table[x=F,y=cheat,y error plus=cheathi,y error minus=cheatlo] {figures/data/05-sumcheck-rates.dat};
  \addplot[honest] table[x=F,y=honest] {figures/data/05-sumcheck-rates.dat};
  \legend{bound $\ell d/\abs{\F}$, cheating prover, honest prover}
  \end{axis}
  \end{tikzpicture}
  \caption{... say what was run: sizes, number of trials, seeds, what the intervals are.}\label{fig:key:name}
  \end{figure}
  ```
  Styles: `vplot` (axis), `theory`, `theory2` (exact predictions), `bound` (a proved bound, dashed),
  `honest` (filled indigo marks), `cheat` (hollow ochre marks), `cheat2` (hollow teal squares),
  `ebars` (95% intervals from error columns), `guide` (dashed reference lines), `band` (filled
  region). `xmode`/`ymode` are given directly as axis options. Use `fullwidth` for wide figures.
  Colours: only the palette — `prover`, `proverdeep`, `proverB`, `verifier`, `oracle`, `vellum`,
  `vellumline`, `indigowash`, `ochrewash`, `ink`, `muted`, and tints such as `prover!40`.
- Every caption says what was run (sizes, number of trials, seeds) and what the intervals are
  (Clopper–Pearson 95% from `pk.acceptance`). Run every script yourself and check that the figure
  shows what the caption and the text say. Use numerics also to check theorems, constants and
  exercise answers. Scratch work goes in `build/`, not in `figures/`.
- TikZ diagrams (game trees, circuits, protocol flows, the layers of GKR) are welcome; draw them in
  the palette.

## 7. LaTeX conventions

- **Engine:** pdfLaTeX + biber. Build your chapter with `tools/build_chapter.sh NN` and fix every
  error it reports. Do not edit the class, `verifiable.sty`, `main.tex`, other chapters,
  `protokit.py`, or the tools. If you need a feature they lack, say so in your report.
- **Labels** carry your chapter key: `ch:key`, `sec:key:...`, `thm:key:...`, `lem:key:...`,
  `prop:key:...`, `cor:key:...`, `def:key:...`, `ass:key:...`, `obs:key:...`, `eq:key:...`,
  `fig:key:...`, `tab:key:...`, `prot:key:...`, `ex:key:...`. Refer with `\cref{...}` (sentence start
  `\Cref{...}`). Other chapters: `\cref{ch:key}` with the keys in OUTLINE.md, and their results with
  the labels OUTLINE.md fixes; appendix results with the fixed `toolkit` labels.
- **Environments (class):** `theorem lemma proposition corollary conjecture claim` (boxed, italic);
  `definition problem` (cream box); `example remark` (unboxed); starred versions; `proof`,
  `proofsketch`; `chaptersummary`; `aside[Title]` (grey box for a digression); `fullwidth`;
  `exercises` with `\consolidation`, `\research`, `\hard`, `\harder`, `\hint{...}`.
- **Environments (volume):**
  - `oversight[short title]` — the oversight problem (ochre rule, cream tint).
  - `protocol{Name}` — a protocol as a transcript; steps `\item[\msg{P}{V}]` (a message from P to V,
    coloured by sender), `\item[\act{V}]` (a local step of V), `\phase{Round $j$}` (a group heading).
    Parties: `V` verifier/judge (ochre), `P`, `P_1`, `A` (indigo), `P_2`, `B` (teal), `H` human
    oracle (sage). An optional colour must be braced inside `\item[...]`:
    `\item[{\msg[oracle]{X}{V}}]`. Numbered "Protocol N.k"; label `prot:key:name`.
  - `dictionary` — rows `protocol side & oversight side \\`.
  - `\protocolcard{checks}{parties}{honesty}{verifier's cost}{catching a lie}{rests on}`.
  - `\questionone`, `\questiontwo`, `\questionthree` — the paragraph heads of "The three questions".
  - `assumption` (cream box), `observation[source]` (an empirical result as reported, with a left
    rule: `\begin{observation}[\textcite{KhanEtAl2024}]`), `heuristic` (◇).
  - Exercise tags `\variation`, `\attack`, `\simulation` (with `\consolidation`, `\research`).
- **Margin notes:** `\sidenote{...}` (numbered) for history, cross-references and side comments;
  `\marginaside{...}` for unnumbered ones. Use 6–15 per chapter. Never put essential mathematics in
  a side note. **Side notes only in running text** — not inside boxes, theorem-like environments,
  displays, captions, tables, protocols, the dictionary, the card or proofs' displays (LaTeX will
  error). Do **not** use `\footnote`.
- **Terms:** `\term{...}` at the defining occurrence and `\index{...}` for 8–20 key terms per
  chapter (plain words; subentries with `!`, e.g. `\index{soundness!computational}`).
- **Macros available:** class: `\R \N \Z \Q \C \F \E \eps \ind \dd \defeq \Var \Cov \argmin \argmax
  \rank \tr \sgn \supp \poly \conv \diam \dist \abs{} \norm{} \ip{} \ceil{} \floor{}` and
  `\cA … \cZ`; volume: `\V \Prov \party{} \cc{} \Prob \bits \iid \acc \rej \interact{}{} \negl \Unif
  \Bern \diag \polylog \maj`. Need more? Define it at the top of your chapter with a name that
  starts with your chapter key (e.g. `\newcommand{\sumcheckclaim}{...}`), never a generic name.
- **Tables:** booktabs rules; numbers aligned (`siunitx` `S` columns are available).
- **Citations:** keys `AuthorYear`, `AuthorAuthorYear` or `AuthorEtAlYear` (three or more authors:
  `IrvingEtAl2018`). Your bib file is `chapters/NN-key.bib`. Every entry must be real and verified
  with a web search (authors in order, title, venue, year; volume and pages only when verified).
  For arXiv preprints give `eprint = {2311.14125}`, `eprinttype = {arxiv}` (printed by this volume);
  give the refereed venue when there is one and it is verified. Mark unrefereed preprints as such in
  the text when a claim rests on them. 15–40 entries per chapter. Use the same key for the same work
  as other chapters do (OUTLINE.md lists the shared keys).
- **No** `\usepackage`, no changes to page layout, no `\newpage` tricks, no colours outside the
  palette, no `\footnote`.

## 8. Quality checklist before you report back

- [ ] `tools/build_chapter.sh NN` runs with zero LaTeX errors, zero biber errors, no undefined
      references or citations belonging to your chapter, and no overfull box over 3pt.
- [ ] The style lint and the Halmos lint show no warnings, or each remaining one is deliberate.
- [ ] Every theorem is true as stated, with all hypotheses; every proof is complete or explicitly a
      sketch or a heuristic; constants are right (checked numerically where possible); completeness
      and soundness are both proved.
- [ ] Every script runs within its budget from the volume root and writes its data; every figure
      shows what its caption and the text say; every number quoted from a script matches its output;
      no measured rate contradicts a proved bound.
- [ ] Every claim attributed to a paper is checked; every historical and bibliographic claim is
      verified; no invented quotations or anecdotes.
- [ ] The anatomy is followed, including the oversight box, a protocol box, the dictionary, the
      three questions with the card, the three paragraphs of "What the model leaves out", the
      exercise mix, and "Translating the literature".
- [ ] Length is 16–20 pages.

When you finish, report: the page count; the main results, saying which have complete proofs,
which are sketched and which are argued; the scripts, their runtimes and what they showed; your
answers to the three questions in one line each; any facts you could not verify and what you did
about them; and anything the editor should check.
