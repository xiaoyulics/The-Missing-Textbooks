#!/bin/bash
# Build and check one chapter on its own.
# Usage: tools/build_chapter.sh chapters/05-signalprop.tex     (or just: tools/build_chapter.sh 05)
# Output: build/<chapter>/wrapper.pdf plus a report: LaTeX errors, biber problems,
# undefined references belonging to this chapter, overfull lines, page count, style lint.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERIES="$(cd "$ROOT/../template" && pwd)"
BIBER="$ROOT/../.tools/biber"; [[ -x "$BIBER" ]] || BIBER=biber
HALMOS="$HOME/.claude/skills/halmos-style/scripts/halmos_check.py"
arg="$1"
if [[ "$arg" =~ ^[0-9]+$ ]]; then
  f=$(ls "$ROOT"/chapters/$(printf "%02d" "$((10#$arg))")-*.tex 2>/dev/null | head -1)
else
  f="$arg"; [[ "$f" = /* ]] || f="$ROOT/$f"
fi
[[ -f "$f" ]] || { echo "No such chapter file: $arg"; exit 2; }
base=$(basename "$f" .tex); num=${base%%-*}; key=${base#*-}; n=$((10#$num))
out="$ROOT/build/$base"; mkdir -p "$out"; cd "$out" || exit 2
ln -sfn ../../chapters chapters
ln -sfn ../../figures figures 2>/dev/null
if [[ "$key" == "toolkit" ]]; then pre='\appendix'; cnt=0; else pre=''; cnt=$((n>0 ? n-1 : 0)); fi
cat > wrapper.tex <<TEX
\\documentclass{missingbook}
\\usepackage{hydrogen}
\\begin{document}
$pre
\\setcounter{chapter}{$cnt}
\\input{chapters/$base}
\\end{document}
TEX
export TEXINPUTS="$SERIES:$ROOT:"
rm -f wrapper.aux wrapper.bbl wrapper.bcf wrapper.run.xml
pdflatex -interaction=nonstopmode -halt-on-error wrapper.tex > pass1.out 2>&1; s1=$?
if [[ $s1 -eq 0 ]]; then "$BIBER" wrapper > biber.out 2>&1; sb=$?; else sb=-; fi
pdflatex -interaction=nonstopmode -halt-on-error wrapper.tex > pass2.out 2>&1
pdflatex -interaction=nonstopmode -halt-on-error wrapper.tex > pass3.out 2>&1; s3=$?
echo "== $base: pdflatex exit $s1/$s3, biber exit $sb =="
if grep -q "^!" wrapper.log; then echo "-- LaTeX errors --"; grep -n -A7 "^!" wrapper.log | head -60; fi
if [[ -f biber.out ]]; then grep -E "ERROR|WARN" biber.out | grep -v "Overriding locale" | head -15; fi
echo "-- undefined references/citations in this chapter --"
{ grep -oE "Reference \`[^']+' on page [0-9]+ undefined" wrapper.log; grep -oE "Citation \`[^']+' on page [0-9]+ undefined" wrapper.log; } | sed -E "s/ on page [0-9]+ undefined//" | sort -u | grep -E ":$key|ch:$key|Citation" || echo "(none)"
echo "-- cross-chapter references (expected to be undefined in a single-chapter build) --"
grep -oE "Reference \`[^']+' on page [0-9]+ undefined" wrapper.log | sed -E "s/ on page [0-9]+ undefined//" | sort -u | grep -vE ":$key|ch:$key" | tr '\n' ' '; echo
grep -E "multiply defined" wrapper.log | sort -u
echo "-- overfull boxes: $(grep -c '^Overfull' wrapper.log) total; those over 3pt: --"
grep -E "^Overfull \\\\[hv]box \(([3-9]|[1-9][0-9]+)\.[0-9]+pt too (wide|high)\)" wrapper.log | head -15
[[ -f wrapper.pdf ]] && echo "-- pages: $(pdfinfo wrapper.pdf 2>/dev/null | awk '/^Pages/{print $2}') --"
echo "-- banned words (STYLE.md 5.3) --"
grep -nE "\b(clearly|obviously|trivially|evidently|it is easy to see|it is straightforward)\b|\bany\b" "$f" | head -25 || true
echo "-- leftovers (TODO, XXX, ??) --"
grep -nE "TODO|XXX|\?\?" "$f" | head -10 || true
if [[ -f "$HALMOS" ]]; then echo "-- Halmos lint (proofs) --"; python3 "$HALMOS" "$f" 2>/dev/null | tail -25; fi
exit $s3
