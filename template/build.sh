#!/bin/bash
# Build a document that uses missingbook.cls:  ./build.sh mybook.tex
# Runs pdflatex, biber, pdflatex, pdflatex. Uses ../.tools/biber if the system biber cannot start
# (on some Macs the universal biber binary fails at its lipo step; see README).
set -u
here="$(cd "$(dirname "$0")" && pwd)"
doc="${1%.tex}"
BIBER="$here/../.tools/biber"; [[ -x "$BIBER" ]] || BIBER=biber
export TEXINPUTS="$here:"
pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null
"$BIBER" "$doc" >/dev/null || echo "biber failed: see $doc.blg"
pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null
pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null
grep -n -A5 "^!" "$doc.log" | head -30
echo "done: $doc.pdf ($(pdfinfo "$doc.pdf" 2>/dev/null | awk '/^Pages/{print $2}') pages)"
