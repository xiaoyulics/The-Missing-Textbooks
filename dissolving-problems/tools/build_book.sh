#!/bin/bash
# Build the whole book: tools/build_book.sh   ->  Dissolving-Problems.pdf
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERIES="$(cd "$ROOT/../template" && pwd)"
BIBER="$ROOT/../.tools/biber"; [[ -x "$BIBER" ]] || BIBER=biber
cd "$ROOT" || exit 2
export TEXINPUTS="$SERIES:$ROOT:"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build/main1.out 2>&1; s1=$?
"$BIBER" main > build/biber.out 2>&1; sb=$?
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build/main2.out 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build/main3.out 2>&1; s3=$?
echo "== book: pdflatex exit $s1/$s3, biber exit $sb =="
grep -n -A7 "^!" main.log | head -60
grep -E "ERROR|WARN" build/biber.out | grep -v "Overriding locale" | head -20
echo "-- undefined references/citations --"
{ grep -oE "Reference \`[^']+' on page [0-9]+ undefined" main.log; grep -oE "Citation \`[^']+' on page [0-9]+ undefined" main.log; } | sed -E "s/ on page [0-9]+ undefined//" | sort -u | head -40
grep -E "multiply defined" main.log | sort -u | head -20
echo "-- overfull boxes: $(grep -c '^Overfull' main.log) total; over 3pt: --"
grep -E "^Overfull \\\\[hv]box \(([3-9]|[1-9][0-9]+)\.[0-9]+pt too (wide|high)\)" main.log | head -30
if [[ -f main.pdf ]]; then echo "-- pages: $(pdfinfo main.pdf | awk '/^Pages/{print $2}') --"; cp main.pdf "$ROOT/Dissolving-Problems.pdf"; fi
exit $s3
