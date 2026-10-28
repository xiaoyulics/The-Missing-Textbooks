#!/usr/bin/env python3
"""Style lint for a chapter of Singular Learning Theory for the Rest of Us (STYLE.md sections 5 and 7).

Usage: python3 tools/lint.py chapters/NN-key.tex
Prints one line per warning: line number, rule, and the offending text. Exit status is 0.
The checks are heuristics; a warning that is deliberate may stay.
"""
import re
import sys

BANNED = [r"\bclearly\b", r"\bobviously\b", r"\btrivially\b", r"\bevidently\b",
          r"it is easy to see", r"it is straightforward", r"\bany\b"]
HYPE = [r"\brevolutionary\b", r"\bmagic(al)?\b", r"\bmysterious(ly)?\b", r"\bsecretly\b",
        r"\bdramatic(ally)?\b", r"\bbeautiful\b", r"\bstunning\b", r"\bincredibl[ey]\b"]
NO_SIDENOTE_ENVS = {"theorem", "lemma", "proposition", "corollary", "conjecture", "claim",
                    "definition", "problem", "assumption", "example", "remark", "observation",
                    "theorem*", "lemma*", "proposition*", "corollary*", "definition*", "remark*",
                    "example*", "regularanswer", "computation", "chaptersummary", "aside",
                    "figure", "table", "equation", "equation*", "align", "align*", "gather",
                    "gather*", "multline", "tabular", "tikzpicture", "heuristic", "proofsketch"}


def strip_comment(line):
    out, i = [], 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2]); i += 2; continue
        if c == "%":
            break
        out.append(c); i += 1
    return "".join(out)


def prose_only(line):
    """Remove inline math, \\cite-like commands and labels, so that word checks see prose only."""
    line = re.sub(r"\$[^$]*\$", " MATH ", line)
    line = re.sub(r"\\(label|cref|Cref|ref|eqref|textcite|parencite|cite|index|term)\*?(\[[^\]]*\])?\{[^}]*\}", " ", line)
    return line


def main(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    warnings = []
    stack = []
    in_display = False
    proof_start = None
    proof_text = []
    prev_blank = True
    for no, raw in enumerate(lines, 1):
        line = strip_comment(raw)
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", line):
            kind, env = m.group(1), m.group(2)
            if kind == "begin":
                stack.append(env)
                if env in ("proof", "proofsketch"):
                    proof_start, proof_text = no, []
            else:
                if stack and stack[-1] == env:
                    stack.pop()
                elif env in stack:
                    while stack and stack.pop() != env:
                        pass
                if env in ("proof",) and proof_start is not None:
                    body = " ".join(proof_text)
                    if len(body) > 900 and not re.search(r"\b(plan|idea|strategy)\b", body[:600], re.I):
                        warnings.append((proof_start, "proof-plan", "long proof without a one-sentence plan near its start"))
                    proof_start = None
        if proof_start is not None:
            proof_text.append(line)
        if "\\[" in line:
            in_display = True
        if "\\]" in line:
            in_display = False
        if "\\sidenote" in line or "\\marginaside" in line:
            bad = [e for e in stack if e in NO_SIDENOTE_ENVS]
            if bad or in_display:
                warnings.append((no, "sidenote-placement", "margin note inside " + (bad[-1] if bad else "display")))
        if re.search(r"\\footnote(?![A-Za-z])", line):
            warnings.append((no, "footnote", "use \\sidenote, never \\footnote"))
        prose = prose_only(line)
        low = prose.lower()
        for pat in BANNED:
            for m in re.finditer(pat, low):
                warnings.append((no, "banned-word", m.group(0)))
        for pat in HYPE:
            for m in re.finditer(pat, low):
                warnings.append((no, "hype", m.group(0)))
        if re.search(r"(?<![.]{2})[.!?]\s+\$", line) and "\\label" not in line:
            warnings.append((no, "symbol-starts-sentence", line.strip()[:80]))
        if prev_blank and line.lstrip().startswith("$"):
            warnings.append((no, "symbol-starts-paragraph", line.strip()[:80]))
        if re.search(r"\\\]\s*where\b", line) or (no > 1 and re.match(r"\s*where\b", line) and "\\]" in lines[no - 2]):
            warnings.append((no, "trailing-where", "define symbols before a display, not in a trailing 'where'"))
        if re.search(r"TODO|XXX|\?\?", line):
            warnings.append((no, "leftover", line.strip()[:80]))
        if re.search("[\u2200\u2203\u21d2]", line):
            warnings.append((no, "logic-symbol", "no \u2200, \u2203, \u21d2 in prose"))
        prev_blank = (line.strip() == "")
    for no, rule, text in sorted(warnings):
        print(f"WARN line {no}: {rule}: {text}")
    if not warnings:
        print("lint: no warnings")


if __name__ == "__main__":
    main(sys.argv[1])
