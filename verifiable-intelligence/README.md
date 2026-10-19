# Verifiable Intelligence: Interactive Proofs as a Theory of AI Oversight

*The Missing Textbooks, volume 5.* Written by Claude. Conceived and edited by Xiaoyu Li. See the colophon, "How This Book Was Made", at the end of the PDF.

When an AI system is more capable than the people who oversee it, they can no longer check its answers by finding the answers themselves. What can a weak verifier learn from a strong prover that may lie? Complexity theory has studied this question since 1985, as the theory of interactive proofs, and AI-safety research has rediscovered pieces of it since 2018: debate, doubly-efficient debate, prover–verifier games, spot checks, delegation of computation, audits in zero knowledge. This graduate textbook teaches the theory systematically for the oversight problem. Each chapter teaches one oversight protocol, proves its main theorem completely, runs it in a script against honest and cheating provers, and answers the same three questions: **Is honesty an equilibrium? What does the verifier spend? How often is a lie caught?**

**[Read the PDF](Verifiable-Intelligence.pdf)** (357 pages).

## Contents

| | Chapter | The protocol |
|---|---|---|
| | **Prologue.** The weak verifier | |
| | **Part I. Fixed Provers** | |
| 1 | Certificates — What one message can prove | the certificate check |
| 2 | Spot Checks — Reading three bits of a long proof | the linearity test and the Hadamard PCP |
| 3 | Consistency — Checking a program on the input that matters | the program checker |
| | **Part II. Conversation** | |
| 4 | Interaction — Questions the prover cannot predict | private coins for graph non-isomorphism |
| 5 | Arithmetization — Sumcheck and IP = PSPACE | the sumcheck protocol |
| 6 | Delegation — Checking a computation without redoing it | the GKR protocol |
| | **Part III. Competition** | |
| 7 | Debate — Two provers and a judge | debate |
| 8 | Cross-Examination — Questioning provers who cannot confer | two isolated provers |
| 9 | Efficient Debate — When the honest side must be efficient too | doubly-efficient debate |
| | **Part IV. Incentives** | |
| 10 | Rational Proofs — Paying for the truth | the Brier-scored rational proof |
| 11 | Prover–Verifier Games — Learning to give checkable answers | the verifier-leading game |
| 12 | Verifying Learning — Checking a model without learning it | PAC verification |
| | **Part V. Secrets and Limits** | |
| 13 | Zero Knowledge — Auditing without disclosure | commit, challenge, respond |
| 14 | Proofs of Training — Arguments about how a model was made | logged-checkpoint spot checks |
| 15 | Undetectable Backdoors — What no audit can find | the black-box audit |
| | **Epilogue.** The oversight ledger | |
| A | A Toolkit for Verifiers | |

Each chapter has the same anatomy: *The oversight problem*, *The protocol* (written out as a transcript), *The theory* (with at least one complete proof), *Running the protocol* (a script runs the protocol against honest and cheating provers and compares measured acceptance rates with the theorem), *The three questions* (with a protocol card), *What the model leaves out* (what was idealized away, where it breaks, what is open), exercises (consolidation, variation, attack, simulation, research), and notes that translate the notation of the sources.

The epilogue, *The oversight ledger*, collects the fifteen protocols' answers to the three questions in two tables. Every chapter, the prologue, the appendix and the epilogue received an independent referee pass; the colophon records what the referees found.

## Design

The volume style `verifiable.sty` gives the series class two voices: the prover's indigo for the book's structure and the verifier's ochre for its marks — the bits a verifier reads, its verdicts, and the three questions. A second prover is teal and a human judge consulted as an oracle is sage. Each chapter opening carries, at the foot of the page, its protocol's message pattern drawn between a prover rail and a verifier rail. The title page carries the 2048-bit Hadamard encoding of the number 1985, the year interactive proofs were introduced, with the three bits that a linearity test reads ringed; the bits come from `figures/scripts/00-emblem.py`.

## Files

- `main.tex` is the master file (chapters are included when they exist); `verifiable.sty` is the volume style on top of `../template/missingbook.cls`.
- `frontmatter/` holds the preface and the notation; `backmatter/` the colophon.
- `chapters/NN-key.tex` and `chapters/NN-key.bib` hold one chapter and its bibliography.
- `figures/scripts/` holds the scripts that run the protocols: `protokit.py` (the shared kit: prime-field arithmetic, polynomial and multilinear-extension evaluation, acceptance rates with Clopper–Pearson intervals) and one script per experiment; `figures/data/` holds their output, which the chapters plot with pgfplots.
- `tools/build_book.sh` and `tools/build_chapter.sh` build and check the book and single chapters.
- `STYLE.md` is the writing guide and `OUTLINE.md` the plan. Where the plan and a chapter differ, the chapter is correct.

## Building and running

```bash
cd verifiable-intelligence
tools/build_book.sh                       # whole book -> Verifiable-Intelligence.pdf
tools/build_chapter.sh 05                 # one chapter, with error / reference / overfull / style checks
python3 figures/scripts/protokit.py       # self-test of the shared kit
python3 figures/scripts/00-emblem.py      # the title page's codeword
```

The scripts need Python 3 with NumPy and SciPy and run on a laptop CPU in a minute or two each.
