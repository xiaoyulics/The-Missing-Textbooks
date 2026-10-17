# The Missing Textbooks

A series of textbooks that should exist but don't.

*Series edited by Xiaoyu Li.* Each volume is written end to end by Claude, and conceived and edited by Xiaoyu Li. A colophon at the end of each book records the model, settings and process used. All volumes share one LaTeX design, `template/missingbook.cls`.

## Volumes

| # | Title | Sources | PDF |
|---|-------|---------|-----|
| 1 | *Dissolving Problems: A Casebook of Theory-Building* (332 pp.) | [`dissolving-problems/`](dissolving-problems/) | [`Dissolving-Problems.pdf`](dissolving-problems/Dissolving-Problems.pdf) |
| 2 | *The Hydrogen Atoms of Deep Learning: Solvable Models and What They Get Wrong* (396 pp.) | [`hydrogen-atoms/`](hydrogen-atoms/) | [`Hydrogen-Atoms-of-Deep-Learning.pdf`](hydrogen-atoms/Hydrogen-Atoms-of-Deep-Learning.pdf) |

## Layout

```
template/              the series class (missingbook.cls), a short sample, build script, README
dissolving-problems/   volume 1: main.tex, chapters/, frontmatter/, tools/, STYLE.md, OUTLINE.md
hydrogen-atoms/        volume 2: main.tex, hydrogen.sty, chapters/, figures/ (scripts and data), tools/, STYLE.md, OUTLINE.md
```

A new volume is a sibling folder with its own `main.tex` (`\documentclass{missingbook}`), per-chapter `.tex` and `.bib` files, and a copy of `dissolving-problems/tools/`.

## Building

You need TeX Live 2023 or later, with pdfLaTeX and biber.

```bash
cd dissolving-problems
tools/build_book.sh                 # whole book -> Dissolving-Problems.pdf
tools/build_chapter.sh 05           # one chapter, with error / reference / overfull / style checks
```

*Biber on Apple Silicon.* The TeX Live biber is a universal binary that may fail with "extracting arm64 binary with lipo failed". The fix is to extract the arm64 slice once:

```bash
mkdir -p .tools && /usr/bin/lipo -thin arm64 -output .tools/biber /Library/TeX/texbin/biber
```

The build scripts use `.tools/biber` when it exists. The folder is ignored by git.

*Source Sans on older TeX Live.* The class loads the `sourcesans` font package. Some distributions (for example Debian's and Ubuntu's TeX Live 2023) ship only the older `sourcesanspro`. A local wrapper fixes this without touching the class; put it in your personal tree, for example `~/texmf/tex/latex/sourcesans/sourcesans.sty`:

```latex
\ProvidesPackage{sourcesans}
\DeclareOption*{\PassOptionsToPackage{\CurrentOption}{sourcesanspro}}
\ProcessOptions\relax
\RequirePackage{sourcesanspro}
```

## How a volume is made

1. A plan, `OUTLINE.md`, gives the thesis, the chapter list and a specification for each chapter.
2. A writing guide, `STYLE.md`, sets the case template, voice, proof style and LaTeX conventions.
3. The chapters are drafted in parallel and built with the chapter tool.
4. Each chapter gets an independent referee pass. The referee re-derives proofs, checks constants and exercises, and checks historical claims and references wherever sources can be reached.
5. The whole book is built and checked for layout, cross-references and margin overflow.
