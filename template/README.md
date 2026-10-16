# missingbook — a LaTeX class for "The Missing Textbooks"

`missingbook.cls` is a textbook design built on KOMA-Script (`scrbook`) for pdfLaTeX. It has a
70-character text column with a margin column for side notes, Palatino text and mathematics, and a
sea-blue palette. Chapter openings carry a large chapter number and a wave band whose height rises with
the chapter number (the "rising sea"). It also provides boxed theorems, signature boxes for case-style
chapters, per-chapter bibliographies, and an index.

`sample.tex` is a short demonstration that uses every feature; `sample.pdf` is its output. The book
`../dissolving-problems/` is a full-length example.

## Build

```bash
./build.sh sample.tex        # pdflatex → biber → pdflatex → pdflatex (makeindex runs automatically)
```

Requirements: TeX Live 2023 or later with `newpx`, `sourcesans`, `tcolorbox`, `biblatex`/`biber`,
`imakeidx`, `cleveref`, `tikz`. Engine: **pdfLaTeX**.

*Biber on Apple Silicon.* On some Macs the TeX Live biber (a universal binary) fails with
"extracting arm64 binary with lipo failed". The workaround used here is to extract the arm64 slice
once and call it directly:

```bash
mkdir -p ../.tools && /usr/bin/lipo -thin arm64 -output ../.tools/biber /Library/TeX/texbin/biber
```

`build.sh` uses `../.tools/biber` automatically when it exists.

## Class options

| option | effect |
|--------|--------|
| `palatino` (default) | newpx text and math |
| `cochineal` | Cochineal + newtxmath |
| `garamond` | EB Garamond + newtxmath |
| `charter` | XCharter + newtxmath |
| `libertinus` | Libertinus + libertinust1math |

The page is 210 × 280 mm: a 120 mm text block plus a 44 mm margin column. To change it, edit the
`geometry` call near the top of the class.

## Structure commands

| command | purpose |
|---------|---------|
| `\mbtitlepage{Title}{Subtitle}{Author}{Imprint}` | title page with the soaking nut |
| `\mbtableofcontents` | contents in ink-coloured links |
| `\casepart{Title}{Blurb}{tide mm}` | part page with an italic blurb and a wave band |
| `\casechapter{Title}{Subtitle}{Move}{label}` | numbered chapter with subtitle, "the move" tag and wave rule; the tide rises with the chapter number |
| `\specialchapter{Title}{Subtitle}{tide mm}` | unnumbered chapter (prologue, epilogue) |
| `\addsec{...}` | unnumbered section that appears in the contents |

## Environments

- Theorem-like, boxed with a sea rule: `theorem lemma proposition corollary conjecture claim` (and starred).
- Sand box: `definition problem` (and starred).
- Unboxed: `example remark` (and starred).
- `proof` (small-caps head, Halmos tombstone) and `proofsketch`. Use `\qedhere` when a proof ends in a display.
- `chaptersummary` — sand box opened by "In this chapter."
- `\begin{move}{Name}` — the signature box "The Move · Name"; inside it use `\movepart{In one sentence}`,
  `\movepart{Triggers}`, `\movepart{Recipe}`, `\movepart{Anti-pattern}`.
- `\dissolutiontest{shorter proofs}{fewer assumptions}{unification}{new reach}` — a four-row table.
- `aside[Title]` — grey side box for an old proof's skeleton or a historical digression.
- `fullwidth` — spans the text block and the margin column (wide tables, figures, displays).
- `exercises` — numbered N.k, with tags `\consolidation`, `\transfer`, `\research`, difficulty marks
  `\hard` and `\harder`, and `\hint{...}`.

## Notes, references, index

- `\sidenote{...}` numbered margin note; `\marginaside{...}` unnumbered. Margin notes go in running text
  only (not inside boxes, theorem environments, displays or tables).
- Per-chapter bibliographies: wrap a chapter in `\begin{refsection}[chapters/file.bib] … \end{refsection}`
  and print with `\printchapterbib`. Citations are author–year (`\textcite`, `\parencite`).
- `\notesandrefs` starts the "Notes and References" section.
- `\term{...}` for a defined term; `\index{...}` for index entries; `\printindex` in the back matter.
- Cross-references with `\cref`/`\Cref` (cleveref): "Theorem 3.2", "Chapter 11", equations as "(3.4)".

## Maths macros

`\R \N \Z \Q \C \F \E \eps \ind \dd \defeq \Var \Cov \argmin \argmax \rank \tr \sgn \supp \poly \conv
\diam \dist \abs{} \norm{} \ip{}{} \ceil{} \floor{}` and calligraphic letters `\cA … \cZ`.

## Palette

`sea` (accent), `seadeep`, `seafoam` (theorem boxes), `seamist`, `sand` (definition boxes), `sandline`,
`ink`, `muted`.
