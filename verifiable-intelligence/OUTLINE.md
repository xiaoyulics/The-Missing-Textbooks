# Verifiable Intelligence — Book Plan and Chapter Specifications

*The Missing Textbooks, volume 5.* Working title: **Verifiable Intelligence: Interactive Proofs as a
Theory of AI Oversight** (Chinese working title 可验证的智能). Written by Claude; conceived and
edited by Xiaoyu Li, from the editor's proposal "T-5" (described by the editor as the most
imaginative volume of the series).

## The thesis

When an AI system is more capable than the people who oversee it, they can no longer check its
answers by producing the answers themselves. A model proposes a proof, a patch, a plan, a forecast,
or the result of a long computation, and claims that it is correct; the overseer can afford to
look at a small part of it. What can a weak verifier learn from a strong prover that may lie?

This is the question that complexity theory has studied since 1985. Goldwasser, Micali and Rackoff,
and independently Babai, defined interactive proofs: a verifier limited to polynomial time and
allowed to flip coins exchanges messages with an all-powerful prover, and must accept true claims
and reject false ones with high probability, whatever the prover does. Over forty years the theory
answered many versions of the question. A verifier that may ask questions can check claims as hard
as polynomial space (IP = PSPACE); one that questions two provers apart can check more (MIP = NEXP);
one that reads three bits of a suitably written proof can check any NP statement (the PCP theorem);
a verifier can check a long computation in time nearly linear in its input while the honest prover
does only polynomial work (delegation, GKR); a prover can convince a verifier without revealing why
(zero knowledge); a prover who is paid by a proper scoring rule maximizes its pay by telling the
truth (rational proofs); a program can be checked on the one input that matters by querying it on
others (program checking).

Since 2018, AI-safety research has rediscovered pieces of this theory: debate as an interactive
proof whose power is PSPACE (Irving, Christiano and Amodei 2018), doubly-efficient debate, in which
the honest debater's cost is bounded too (Brown-Cohen, Irving and Piliouras 2023), prover–verifier
games for learning checkable answers (Anil, Zhang, Wu and Grosse 2021; Kirchner et al. 2024),
neural interactive proofs, self-proving models, interactive proofs for verifying machine learning,
zero-knowledge proofs of model inference and evaluation, proofs of training and compute monitoring,
and the cryptographic impossibility of detecting planted backdoors. The results are scattered
across papers, written in two vocabularies, and nobody has taught the complexity theory
systematically for the oversight problem.

This book does. **Each chapter teaches one oversight protocol.** It opens with an oversight
problem; defines the protocol completely; proves its main theorem with a complete proof; runs the
protocol, in a script, against an honest prover and against cheating provers and compares the
measured acceptance rates with the theorem; answers the book's **three fixed questions**; and says
what the complexity-theoretic model leaves out of the real problem.

## The three questions

Every chapter answers the same three questions, in a section of that name, and records the answers
on a protocol card. The prologue states them precisely; chapter authors use these readings.

1. **Is honesty an equilibrium?** Read a protocol as a game. In a single-prover protocol, the
   prover is paid when the verifier accepts. Completeness says that on a true claim the honest
   strategy is accepted with probability at least $c$ (usually $1$, which makes it a best response);
   soundness says that on a false claim no strategy is accepted with probability more than $s$.
   When the prover first chooses which answer to assert and the protocol can prove both answers
   (as for a class closed under complement), asserting the truth and proving it honestly is a best
   response, and every lie earns at most $s$. Debate is a zero-sum game and the honest debater's
   strategy is a minimax strategy; prover–verifier games ask for Stackelberg equilibria; rational
   proofs ask for strictly dominant truthful reports; multi-prover protocols ask about profiles of
   strategies and collusion. Each chapter names its game and its equilibrium notion, says whether
   honesty is an equilibrium, and — the part that matters most for AI — **what the honest strategy
   costs**: an equilibrium that the honest prover cannot compute is no protection.
2. **What does the verifier spend?** Time, randomness, rounds, communication, queries to a proof
   or a program, samples from a distribution, calls to a human judge. State the dependence on the
   input size and on the error and security parameters.
3. **How often is a lie caught?** The probability $1-s$ that a cheating prover is rejected, per
   run and after $t$ repetitions, and **against which class of cheaters**: every prover
   (information-theoretic soundness), provers bounded in computation (computational soundness,
   arguments), provers that cannot communicate (multi-prover), rational provers (a lie is not
   caught but costs in expectation), or the learned provers seen in training (empirical soundness).

## Positioning (verified 2026-10-19; digest in `build/research/classics-positioning.md`)

No textbook, monograph, survey, or full course (as of October 2026) teaches interactive proofs, PCPs,
zero knowledge and delegation systematically as a theory of AI oversight, or analyzes oversight
protocols one by one for equilibrium, verifier cost and soundness. The closest materials:
- The Iliad Intensive (April 2026), module E.1 "Scalable Oversight and Debate": one day of a month-long
  alignment course, on debate only ($\cc{PSPACE}$, cross-examination, obfuscated arguments,
  prover–estimator debate, the UK AISI debate safety case).
- MIT 6.S976/18.S996, *Cryptography and Machine Learning: Foundations and Frontiers* (Goldwasser and
  Vaikuntanathan, Spring 2026): about five of 24 lectures on verification (interactive proofs and zero
  knowledge, PAC verification, self-proving models, Lean); no debate.
- The UK AISI alignment team's research agenda (Alignment Forum, May 2025) and the Alignment Project's
  "Computational Complexity Theory" research area: open-problem lists, not teaching material.
- Talks at the Simons Institute (Brown-Cohen 2024; Irving 2025); conceptual chapters on scalable
  oversight (AI Safety Atlas); Hendrycks, *Introduction to AI Safety, Ethics, and Society* (2025);
  surveys of alignment (Ji et al., ACM Computing Surveys 2025) that treat scalable oversight
  qualitatively.
- The complexity and cryptography textbooks this book builds on: Arora and Barak (2009), Goldreich
  (2008; 2001), Thaler (2022), Goldreich's survey of doubly-efficient interactive proofs (2018),
  O'Donnell (2014), Vadhan (2012). None is written for the oversight problem.

## Structure, keys, and files

| # | key | file | Title — Subtitle | Protocol | Band |
|---|-----|------|------------------|----------|------|
| – | prologue | 00-prologue.tex | Prologue — The weak verifier | (special chapter) | `V,P,V,P,D` |
| | | | **Part I. Fixed Provers** | | `S,Q,Q,Q,D` |
| 1 | certificates | 01-certificates.tex | Certificates — What one message can prove | the certificate check | `P,D` |
| 2 | spotchecks | 02-spotchecks.tex | Spot Checks — Reading three bits of a long proof | the linearity test and the Hadamard PCP | `S,Q,Q,Q,D` |
| 3 | consistency | 03-consistency.tex | Consistency — Checking a program on the input that matters | the program checker | `S,Q,Q,Q,Q,Q,D` |
| | | | **Part II. Conversation** | | `P,V,P,V,P,V,D` |
| 4 | interaction | 04-interaction.tex | Interaction — Questions the prover cannot predict | private coins for graph non-isomorphism | `V,P,V,P,V,P,D` |
| 5 | sumcheck | 05-sumcheck.tex | Arithmetization — Sumcheck and IP = PSPACE | the sumcheck protocol | `P,V,P,V,P,V,P,V,P,V,D` |
| 6 | delegation | 06-delegation.tex | Delegation — Checking a computation without redoing it | the GKR protocol | `P,V,P,V,P,o,P,V,P,V,P,o,P,V,P,V,P,D` |
| | | | **Part III. Competition** | | `P,A,P,A,P,A,D` |
| 7 | debate | 07-debate.tex | Debate — Two provers and a judge | debate | `P,A,P,A,P,A,P,A,D` |
| 8 | crossexam | 08-crossexam.tex | Cross-Examination — Questioning provers who cannot confer | two isolated provers | `V,V,P,A,o,V,V,P,A,D` |
| 9 | efficientdebate | 09-efficientdebate.tex | Efficient Debate — When the honest side must be efficient too | doubly-efficient debate | `P,A,P,A,P,A,P,A,H,D` |
| | | | **Part IV. Incentives** | | `P,X,R,o,P,V,P,D` |
| 10 | rational | 10-rational.tex | Rational Proofs — Paying for the truth | the Brier-scored rational proof | `P,X,R` |
| 11 | pvg | 11-pvg.tex | Prover–Verifier Games — Learning to give checkable answers | the verifier-leading game | `P,V,P,V,P,V,D` |
| 12 | pacverify | 12-pacverify.tex | Verifying Learning — Checking a model without learning it | PAC verification | `X,V,P,D` |
| | | | **Part V. Secrets and Limits** | | `C,V,P,o,S,K,Q,Q,D` |
| 13 | zeroknowledge | 13-zeroknowledge.tex | Zero Knowledge — Auditing without disclosure | commit, challenge, respond | `C,V,P,o,C,V,P,o,C,V,P,D` |
| 14 | training | 14-training.tex | Proofs of Training — Arguments about how a model was made | logged-checkpoint spot checks | `C,S,Q,Q,Q,D` |
| 15 | backdoors | 15-backdoors.tex | Undetectable Backdoors — What no audit can find | the black-box audit | `S,K,Q,Q,Q,Q,Q,D` |
| – | epilogue | 16-epilogue.tex | Epilogue — The oversight ledger | (special chapter) | `P,V,P,V,D` |
| A | toolkit | 17-toolkit.tex | A Toolkit for Verifiers (appendix) | | rails only |

Files: `chapters/NN-key.tex` and `chapters/NN-key.bib`; scripts `figures/scripts/NN-key-name.py`;
data `figures/data/NN-key-name.dat`. Chapter labels `ch:key`. The opening of each chapter is

```
\protocolchapter{Title}{Subtitle}{protocol name}{band}{ch:key}
```

with the values in the table (the protocol name is printed in small capitals; write it in lower
case). The appendix opens with `\appendixchapter{A Toolkit for Verifiers}{Classes, probability,
polynomials, games, and cryptography}{ch:toolkit}`.

## Global notation

All chapters use this notation (also in `frontmatter/notation.tex`). A chapter that needs a clashing
local symbol keeps the global meaning and chooses another symbol; a source paper's notation is
translated in "Translating the literature".

**Parties.** Verifier $\V$ (`\V`; in a debate "the judge"); prover $\Prov$ (`\Prov`), which also
names the honest prover's strategy; an arbitrary, possibly cheating, prover $\Prov^*$; two provers
questioned separately $\Prov_1,\Prov_2$; debaters $\party{A}$ and $\party{B}$ (`\party{A}`); a human
judge consulted as an oracle $\party{H}$. "Verifier" and "prover" are the parties of a protocol;
"overseer" and "model" are the people and systems of the oversight problem. Never write a bare $P$
for the prover (it is easily confused with the class $\cc{P}$).

**Protocols.** Input $x\in\bits^*$, $n=\abs{x}$; language $L$ (the claim "$x\in L$"); the
verifier's output $\interact{\Prov}{\V}(x)\in\{\acc,\rej\}$; completeness $c$ and soundness error
$s$ (soundness error is the largest acceptance probability of a false claim; the catch probability
is $1-s$); rounds $k$ (a round is one message from each side; say "messages" when counting
messages); queries $q$; the verifier's random bits $\rho$, a random field element $r$; repetitions
$t$; the running time of the computation a claim is about $T$; security parameter $\lambda$ and
$\negl(\lambda)$.

**Classes.** Bold via `\cc{}`: $\cc{P}$, $\cc{NP}$, $\cc{coNP}$, $\cc{BPP}$, $\cc{MA}$, $\cc{AM}$,
$\cc{IP}$, $\cc{PSPACE}$, $\cc{EXP}$, $\cc{NEXP}$, $\cc{MIP}$, $\cc{\#P}$, $\cc{PCP}(r(n),q(n))$.

**Algebra.** Finite field $\F$ of size $\abs{\F}$ (a prime field $\F_p$ in scripts); $\ell$ variables;
degree $d$ in each variable (total degree written $\deg$); the multilinear extension $\tilde f$ of
$f\colon\bits^\ell\to\F$; arithmetic circuits of size $S$ and depth $D$; layers indexed by $i$.
Fourier analysis on $\bits^n$: $\chi_S(x)=(-1)^{\sum_{i\in S}x_i}$, coefficients $\hat F(S)$ of
$F=(-1)^f$.

**Learning and probability.** Samples $m$; distribution $\cD$; hypothesis class $\cH$; accuracy
$\varepsilon$ (`\varepsilon`) and confidence $\delta$ in learning statements (and nowhere else
without saying so); $\Prob$, $\E$, always with the randomness stated.

**Games.** Strategies $\sigma$; payoffs (utilities) $u$; the value of a game $\mathrm{val}$.

**Asymptotics.** $O,\Omega,\Theta,o$ in the input length $n$ or the security parameter $\lambda$;
$\poly$, $\polylog$; logarithms base $2$ unless written $\ln$.

## The appendix and its fixed labels

The appendix (`ch:toolkit`, `17-toolkit.tex`) collects what several chapters use, with proofs where
they are short. Chapters cite these labels and may prove what they need inline when it is short.

| label | content |
|---|---|
| `sec:toolkit:machines` | Turing machines, time and space, oracle machines, uniform circuits |
| `def:toolkit:classes` | $\cc{P},\cc{NP},\cc{coNP},\cc{BPP},\cc{PSPACE},\cc{EXP},\cc{NEXP},\cc{\#P}$, the polynomial hierarchy; complete problems (SAT, TQBF, \#SAT) |
| `def:toolkit:ip` | interactive proof system: verifier, prover, completeness $c$, soundness error $s$, rounds; $\cc{IP}$; public coins, $\cc{AM}$ and $\cc{MA}$ |
| `prop:toolkit:amplification` | sequential repetition: one-sided error $s^t$; majority vote for two-sided error (with proof) |
| `thm:toolkit:chernoff` | Chernoff–Hoeffding bounds (statement and proof of one form) |
| `def:toolkit:pairwise` | pairwise-independent hash families; the affine family over $\F_2$ (with proof) |
| `lem:toolkit:schwartzzippel` | Schwartz–Zippel lemma (with proof) |
| `lem:toolkit:lagrange` | Lagrange interpolation; a polynomial of degree $d$ is fixed by $d+1$ values |
| `lem:toolkit:mle` | multilinear extensions: existence, uniqueness, the evaluation formula, cost $O(2^\ell)$ |
| `thm:toolkit:parseval` | Fourier analysis on $\bits^n$: characters, Plancherel and Parseval |
| `thm:toolkit:minimax` | von Neumann's minimax theorem (statement); values of finite zero-sum games |
| `thm:toolkit:backward` | backward induction for finite games of perfect information (with proof) |
| `def:toolkit:nash` | Nash equilibrium, dominant strategies, $\varepsilon$-equilibria |
| `def:toolkit:stackelberg` | Stackelberg (leader–follower) equilibrium; subgame perfection |
| `def:toolkit:negl` | negligible functions; computational indistinguishability |
| `def:toolkit:owf` | one-way functions, pseudorandom generators and functions |
| `def:toolkit:commitment` | commitment schemes: hiding and binding (perfect, statistical, computational) |
| `def:toolkit:crhf` | collision-resistant hash families; Merkle trees (definition) |
| `def:toolkit:signature` | digital signatures; existential unforgeability under chosen-message attack |
| `def:toolkit:rom` | the random-oracle model (used for Fiat–Shamir) |
| `sec:toolkit:intervals` | Clopper–Pearson intervals for measured acceptance rates (used by the scripts) |

## Results that other chapters cite (fixed labels)

Authors of the cited chapters must use exactly these labels for these results; authors of citing
chapters may rely on them (and should also say in prose what they cite).

| label | result |
|---|---|
| `thm:certificates:np` | $\cc{NP}$ is the class of claims with polynomial-size certificates checkable in polynomial time |
| `thm:spotchecks:blr` | the linearity test (Blum–Luby–Rubinfeld): rejection probability at least the distance to linear |
| `thm:spotchecks:hadamardpcp` | $\cc{NP}\subseteq\cc{PCP}(\poly(n),O(1))$ via the Hadamard code |
| `thm:spotchecks:pcp` | the PCP theorem, $\cc{NP}=\cc{PCP}(O(\log n),O(1))$ (statement) |
| `thm:consistency:freivalds` | Freivalds's check of a matrix product |
| `prop:consistency:selfcorrect` | self-correction of a function close to linear: $g(x+r)-g(r)$ |
| `thm:consistency:blumkannan` | program checkers and function-restricted interactive proofs |
| `thm:interaction:gni` | graph non-isomorphism has a private-coin interactive proof |
| `thm:interaction:setlower` | the Goldwasser–Sipser set lower bound protocol |
| `thm:interaction:ipinpspace` | $\cc{IP}\subseteq\cc{PSPACE}$: the optimal prover is computable in polynomial space |
| `thm:sumcheck:sumcheck` | the sumcheck protocol: perfect completeness, soundness error at most $\ell d/\abs{\F}$ |
| `thm:sumcheck:ippspace` | $\cc{IP}=\cc{PSPACE}$ |
| `thm:delegation:gkr` | the GKR protocol for layered arithmetic circuits |
| `thm:debate:pspace` | debate with a polynomial-time judge and optimal play decides exactly $\cc{PSPACE}$ |
| `thm:debate:bisection` | the bisection debate: the judge checks one step of a computation |
| `thm:crossexam:oracle` | two provers questioned apart simulate a fixed oracle (Fortnow–Rompel–Sipser) |
| `thm:crossexam:nexp` | $\cc{MIP}=\cc{NEXP}$ (statement) |
| `thm:efficientdebate:main` | the main theorem of doubly-efficient debate |
| `thm:rational:brier` | the Brier score is strictly proper |
| `thm:rational:sharpp` | a one-round rational proof for $\cc{\#P}$ |
| `thm:pvg:stackelberg` | verifier-leading equilibria of prover–verifier games are sound and complete proof systems |
| `thm:pacverify:main` | query delegation with hidden checks (GRSY): few trusted labels suffice to verify an ERM output; the separation from learning is a corollary |
| `thm:zeroknowledge:gi` | perfect zero knowledge for graph isomorphism |
| `thm:zeroknowledge:np` | every language in $\cc{NP}$ has a computational zero-knowledge proof, given commitments |
| `thm:training:merkle` | Merkle commitments are binding if the hash family is collision resistant |
| `prop:training:spotcheck` | spot checks of a committed log: detection probability $1-(1-\varepsilon)^k$ |
| `thm:backdoors:blackbox` | a signature-keyed backdoor is undetectable by black-box queries |
| `thm:backdoors:mitigation` | backdoors in functions close to linear are removed by self-correction, without detection |

## Shared citation keys

Chapters cite the same work with the same key (each chapter's `.bib` holds its own copy of the
entry; the entries must agree). Verify every entry before use; venue and year below are the version
to cite.

| key | work |
|---|---|
| `AroraBarak2009` | Arora and Barak, *Computational Complexity: A Modern Approach*, Cambridge University Press, 2009 |
| `Goldreich2008` | Goldreich, *Computational Complexity: A Conceptual Perspective*, Cambridge University Press, 2008 |
| `Goldreich2001` | Goldreich, *Foundations of Cryptography, Volume I: Basic Tools*, Cambridge University Press, 2001 |
| `Thaler2022` | Thaler, *Proofs, Arguments, and Zero-Knowledge*, Foundations and Trends in Privacy and Security, 2022 |
| `ODonnell2014` | O'Donnell, *Analysis of Boolean Functions*, Cambridge University Press, 2014 |
| `GoldwasserMicaliRackoff1989` | "The knowledge complexity of interactive proof systems", SIAM J. Comput. 1989 (STOC 1985) |
| `Babai1985` | "Trading group theory for randomness", STOC 1985 |
| `BabaiMoran1988` | "Arthur–Merlin games: a randomized proof system ...", JCSS 1988 |
| `GoldreichMicaliWigderson1991` | "Proofs that yield nothing but their validity ...", JACM 1991 (FOCS 1986) |
| `GoldwasserSipser1986` | "Private coins versus public coins in interactive proof systems", STOC 1986 |
| `LundEtAl1992` | Lund, Fortnow, Karloff, Nisan, "Algebraic methods for interactive proof systems", JACM 1992 |
| `Shamir1992` | "IP = PSPACE", JACM 1992 |
| `Shen1992` | "IP = PSPACE: simplified proof", JACM 1992 |
| `BenOrEtAl1988` | Ben-Or, Goldwasser, Kilian, Wigderson, "Multi-prover interactive proofs ...", STOC 1988 |
| `BabaiFortnowLund1991` | "Non-deterministic exponential time has two-prover interactive protocols", Computational Complexity 1991 |
| `FortnowRompelSipser1994` | "On the power of multi-prover interactive protocols", TCS 1994 (Structures 1988) |
| `AroraSafra1998`, `AroraEtAl1998` | the PCP theorem, JACM 1998 |
| `Dinur2007` | "The PCP theorem by gap amplification", JACM 2007 |
| `BlumLubyRubinfeld1993` | "Self-testing/correcting with applications to numerical problems", JCSS 1993 |
| `BlumKannan1995` | "Designing programs that check their work", JACM 1995 |
| `GoldwasserKalaiRothblum2015` | "Delegating computation: interactive proofs for muggles", JACM 2015 (STOC 2008) |
| `IrvingEtAl2018` | Irving, Christiano, Amodei, "AI safety via debate", arXiv 1805.00899 |
| `BrownCohenEtAl2024` | Brown-Cohen, Irving, Piliouras, "Scalable AI safety via doubly-efficient debate" (ICML 2024; arXiv 2311.14125) |
| `AnilEtAl2021` | Anil, Zhang, Wu, Grosse, "Learning to give checkable answers with prover-verifier games", arXiv 2108.12099 |
| `KirchnerEtAl2024` | Kirchner et al., "Prover-verifier games improve legibility of LLM outputs", arXiv 2407.13692 |
| `GoldwasserEtAl2021` | Goldwasser, Rothblum, Shafer, Yehudayoff, "Interactive proofs for verifying machine learning", ITCS 2021 |
| `GoldwasserEtAl2022` | Goldwasser, Kim, Vaikuntanathan, Zamir, "Planting undetectable backdoors in machine learning models", FOCS 2022 |
| `AzarMicali2012` | "Rational proofs", STOC 2012 |
| `AmitEtAl2025` | Amit, Goldwasser, Paradise, Rothblum, "A theory for worst-case vs. average-case guarantees for LLMs", NeurIPS 2025 (arXiv 2405.15722, "Models that prove their own correctness") |
| `FluriEtAl2024` | Fluri, Paleka, Tramèr, "Evaluating superhuman models with consistency checks", SaTML 2024 (verify) |
| `LightmanEtAl2024` | Lightman et al., "Let's verify step by step", ICLR 2024 (verify) |

## Chapter specifications

Each specification gives the opening, the oversight problem, the protocol, the theory (main results
with proof plans), the script for "Running the protocol", the expected answers to the three
questions, what the model leaves out, exercise ideas, cross-links and key references. "Verify" marks
a fact that the chapter author must check against the source before printing. The author may
improve every proposal here; where the chapter and the plan differ, the chapter is correct, and the
author reports the difference.

Where a chapter's protocol makes acceptance rates uninteresting (a certificate check never accepts a
false claim), "Running the protocol" measures the quantity the chapter's theorem is about instead
(certificate sizes, sample counts, costs), always against the theory's curve.

---

### Prologue — The weak verifier (key: prologue; special chapter)

Written by the coordinating author. Opens with an oversight scene and with Babai's Arthur and Merlin
(1985): a king who can verify but not compute, a wizard who can compute but not be trusted. The
question of the book; the formal model of 1985 (GMR; Babai), and the analogy that AI-safety research
drew in 2018 (Irving, Christiano, Amodei). The three questions, stated precisely as above, with why
each matters for AI (training finds equilibria, if anything; human time is the scarce resource; the
soundness error is the residual risk). The ladder of what a weak verifier can check with each
resource (a figure): certificates ($\cc{NP}$), spot checks ($\cc{PCP}$), interaction and private
coins ($\cc{IP}=\cc{PSPACE}$), two isolated provers ($\cc{MIP}=\cc{NEXP}$), and the efficiency
constraint on the honest prover (delegation, doubly-efficient debate). The large dictionary between
interactive proofs and oversight, and where the analogy is exact, idealized, or broken. How to read
the book: the anatomy of a chapter; the band at the foot of each opening page (its tokens); the
scripts; prerequisites and the appendix; a map of dependencies between chapters.

---

### Part I. Fixed Provers

*The simplest prover is a fixed object: a certificate, a proof written out in full, a program. The
verifier reads it, reads a few bits of it, or queries it on inputs of its choosing, and the object
cannot change its answers in response.*

---

### Chapter 1. Certificates — What one message can prove (key: certificates)

`\protocolchapter{Certificates}{What one message can prove}{the certificate check}{P,D}{ch:certificates}`

**The oversight problem.** A model returns an answer with a certificate that a small, trusted
program can check. Positive claims are easy to certify (a satisfying assignment, a schedule that
meets every constraint, a formal proof checked by a proof assistant's kernel); negative and
optimality claims are not ("no schedule exists", "this plan is optimal", "this formula is
unsatisfiable"). Real cases: proof-carrying code (Necula 1997); certifying algorithms (McConnell,
Mehlhorn, Näher, Schweitzer 2011), which return a witness with every answer; SAT solvers that emit
DRAT refutations checked by an independent checker, and the Boolean Pythagorean triples problem,
whose refutation was about 200 terabytes (Heule, Kullmann, Marek 2016; verify size and checker);
AI systems that produce formal proofs checked by Lean (AlphaProof, Hubert et al., Nature 2025 —
verify; DeepSeek-Prover — verify), and the de Bruijn criterion (Barendregt and Wiedijk 2005): trust
only a small kernel. The question: which true claims have short certificates, and what does a
certificate guarantee?

**The protocol.** The prover sends an answer $y$ and a certificate $w$; the verifier runs a
deterministic checker $R(x,y,w)$ in time polynomial in $\abs{x}$ and accepts if and only if it
accepts. Perfect completeness and perfect soundness, relative to the checker's specification.
Worked runs: a 3-coloring; a pair of primal and dual solutions of a small linear program; a Pratt
certificate of primality.

**The theory.**
1. **Theorem (`thm:certificates:np`).** $\cc{NP}$ is exactly the class of languages with
   polynomial-time checkers and polynomial-length certificates (the definition, made into a
   statement about protocols: one message, deterministic verifier); a one-message protocol with a
   deterministic verifier gains nothing from longer messages than poly. Short proof.
2. **Certificates of optimality: linear-programming duality.** The prover claims that $x^*$ is
   optimal and sends a dual solution $y^*$; the verifier checks primal feasibility, dual feasibility
   and equality of objectives. Soundness is weak duality (complete one-line proof); completeness is
   strong duality (complete proof from Farkas's lemma, or cite with a precise statement). Remark on
   why this does not extend to integer programs unless $\cc{NP}=\cc{coNP}$.
3. **Ask for the witness, not the verdict.** Search reduces to decision for SAT (self-reducibility,
   complete proof): a model that can only be trusted to answer yes/no questions can be made to
   produce a certificate with $n$ more questions, which the overseer checks.
4. **Proposition (Cook and Reckhow 1979).** There is a propositional proof system with
   polynomial-size proofs of every tautology if and only if $\cc{NP}=\cc{coNP}$ (complete proof).
5. **Main theorem: a lower bound on certificate length (complete proof).** Tree-like resolution
   refutations of the pigeonhole principle $\mathrm{PHP}^{n+1}_n$ have size $2^{\Omega(n)}$, via the
   Prover–Delayer game of Pudlák and Impagliazzo (2000; verify the reference and the exact bound
   the game gives): if the Delayer has a strategy that scores $p$ points against every Prover, every
   tree-like refutation has at least $2^p$ leaves. The book's theme in miniature: the lower bound is
   proved by a game. Then state, with a proof sketch, the general-resolution bound (Haken 1985;
   Ben-Sasson and Wigderson 2001's size–width relation) and say that DPLL without clause learning
   produces tree-like refutations and CDCL produces general resolution refutations (verify).

**Running the protocol.** `01-certificates-dpll.py`: a small DPLL solver that records its search
tree, run on $\mathrm{PHP}^{n+1}_n$ for $n=2,\dots,9$ (or as far as two minutes allow) and on
satisfiable random 3-CNFs of matching size; plot refutation size (number of leaves, log scale)
against $n$ with the lower bound from the theorem, and certificate size for satisfiable instances
(linear). Also time the checker against the solver. A cheating prover (a wrong assignment, a
refutation with one bad resolution step) is always rejected; report it in a sentence.

**The three questions.** (1) A dishonest assertion is never accepted, so asserting the truth with a
valid certificate is a best response for every prover that wants acceptance, and the only
alternative is to withhold; but the honest prover must find the certificate (search can be
$\cc{NP}$-hard), and for negative claims a short one may not exist at all. (2) Polynomial time in
the length of the certificate, which may be exponential in the claim. (3) Probability $1$, relative
to the formal statement and the checker; a lie about the intended property survives when the
formal statement does not express it.

**What the model leaves out.** The specification gap (the formal statement may not say what the
overseer means: misformalized competition problems, specification gaming); the trusted computing
base (checker and kernel bugs; the de Bruijn criterion); certificate length (200 TB); a certificate
convinces the checker, not the person, who must still understand what was proved; models may
produce plausible-looking but invalid informal proofs, which is why formal checking matters.

**Exercises.** Pratt certificates (PRIMES in $\cc{NP}$); certificates for maximum matching (König)
and max-flow (min-cut); self-reducibility for 3-coloring; an attack on a checker that checks only a
random subset of constraints; tree-like refutation of $\mathrm{PHP}^3_2$ by hand; research:
certificates for properties of neural networks (robustness certificates as an example).

**Cross-links.** `\cref{ch:spotchecks}` (reading a few bits of a certificate), `\cref{ch:interaction}`
(why interaction helps for negative claims), `\cref{ch:sumcheck}` (\cc{coNP} claims with
interaction).

**Key references.** Cook 1971; Levin 1973; Cook and Reckhow 1979; Haken 1985; Ben-Sasson and Wigderson
2001; Pudlák and Impagliazzo 2000; Necula 1997; McConnell et al. 2011; Heule, Kullmann, Marek 2016;
Barendregt and Wiedijk 2005; Wetzler, Heule, Hunt 2014 (DRAT-trim; verify); AlphaProof (verify);
Arora and Barak 2009.

---

### Chapter 2. Spot Checks — Reading three bits of a long proof (key: spotchecks)

`\protocolchapter{Spot Checks}{Reading three bits of a long proof}{the linearity test and the Hadamard PCP}{S,Q,Q,Q,D}{ch:spotchecks}`

**The oversight problem.** A model writes a ten-thousand-step derivation; the overseer can check
three steps. If one step is wrong, checking $k$ random steps of an ordinary proof finds it with
probability about $k/L$. Process supervision checks every step (Lightman et al. 2023, "Let's verify
step by step"; PRM800K — verify numbers). Can a proof be written so that every error shows up
almost everywhere? The PCP theorem says yes, for formal proofs.

**The protocol.** The Hadamard PCP for quadratic equations over $\F_2$ (QUADEQ, which is
$\cc{NP}$-complete): the proof is the Hadamard encoding of a solution $u$ and of $u\otimes u$; the
verifier runs linearity tests, a consistency (tensor) test with local decoding, and a satisfiability
test on a random combination of the equations. Worked example with $n=3$.

**The theory.**
1. Fourier analysis on $\bits^n$ (cite the appendix; recall what is used).
2. **Theorem (`thm:spotchecks:blr`, Blum–Luby–Rubinfeld; Bellare et al. 1996 for the Fourier
   proof — verify).** For every $f\colon\F_2^n\to\F_2$, the probability that the test
   $f(x)+f(y)=f(x+y)$ rejects is at least $\operatorname{dist}(f,\text{linear})$. Complete proof via
   $\Prob[\acc]=\tfrac12+\tfrac12\sum_S\hat F(S)^3$.
3. Local decoding of the Hadamard code with two queries (complete).
4. **Main theorem (`thm:spotchecks:hadamardpcp`).** $\cc{NP}\subseteq\cc{PCP}(\poly(n),O(1))$:
   completeness $1$ and soundness error at most a stated constant (compute it). Complete proof
   following Arora and Barak, §11.5 (verify), with every constant.
5. **The PCP theorem (`thm:spotchecks:pcp`).** $\cc{NP}=\cc{PCP}(O(\log n),O(1))$ (Arora and Safra
   1998; Arora, Lund, Motwani, Sudan, Szegedy 1998), statement; Dinur's gap amplification (2007),
   sketch; Håstad's three-query PCP with soundness $1/2+\varepsilon$ (statement); the link to
   hardness of approximation (one paragraph).
6. Why naive spot checks fail: a proposition computing the detection probability of $k$ random
   step checks on a proof of $L$ steps with $b$ bad steps ($1-\binom{L-b}{k}/\binom{L}{k}$), and
   the observation that a PCP is an error-correcting encoding of the proof.

**Running the protocol.** `02-spotchecks-blr.py`: the linearity test on functions at controlled
distance from linear (random corruptions of a linear function, and the worst cases the theory
allows); plot rejection probability against distance with the bound. `02-spotchecks-pcp.py`: the
Hadamard PCP on small QUADEQ instances, with the proof given as an oracle computed on demand (the
table for $u\otimes u$ has $2^{n^2}$ entries); cheating oracles: a valid encoding of a non-solution,
a corrupted encoding, an inconsistent pair. Compare naive step-checking of an $L$-step chain with one
bad step.

**The three questions.** (1) The proof is a fixed string; there is nothing to adapt, and soundness
holds against every string. The honest proof for the Hadamard PCP is exponentially long; the PCP
theorem gives polynomial (and quasi-linear) proofs, at a large constant cost to the honest prover.
(2) A constant number of queries; $\poly(n)$ random bits here and $O(\log n)$ with the PCP theorem.
(3) A constant per run, amplified by repetition.

**What the model leaves out.** A chain of thought in natural language is not an error-correcting
code; a PCP is unreadable to people; the verifier must know the encoding and the statement; the
chain of thought may not be the computation that produced the answer (Turpin et al. 2023; Lanham
et al. 2023 — verify). Open: proofs that people can spot-check; local checkability of informal
arguments.

**Exercises.** The linearity test over $\F_p$; the acceptance probability of a random function;
soundness of the satisfiability test; an attack on a verifier that skips the consistency test;
simulation of naive spot checks; research: error-correcting arguments in natural language.

**Cross-links.** `\cref{ch:certificates}`; `\cref{ch:consistency}` (self-correction uses
`thm:spotchecks:blr`); `\cref{ch:crossexam}` (PCPs and two-prover games); `\cref{ch:training}`
(spot checks of a training log; Kilian's argument commits to a PCP).

**Key references.** Blum, Luby, Rubinfeld 1993; Bellare, Coppersmith, Håstad, Kiwi, Sudan 1996;
Arora and Safra 1998; Arora et al. 1998; Babai, Fortnow, Levin, Szegedy 1991; Feige et al. 1996;
Dinur 2007; Håstad 2001; Arora and Barak 2009; O'Donnell 2014; Lightman et al. 2023.

---

### Chapter 3. Consistency — Checking a program on the input that matters (key: consistency)

`\protocolchapter{Consistency}{Checking a program on the input that matters}{the program checker}{S,Q,Q,Q,Q,Q,D}{ch:consistency}`

**The oversight problem.** A superhuman chess engine evaluates a position, a forecasting model gives
probabilities, a model answers a legal question; no one can check the answer, but the overseer can
ask related questions and check that the answers fit together. Fluri, Paleka and Tramèr (2023/2024,
"Evaluating superhuman models with consistency checks") find violations in chess engines and
forecasters (verify details). When is consistency enough to guarantee correctness? The theory of
program checking (Blum and Kannan 1989/1995; Blum, Luby, Rubinfeld 1990/1993) answers this.

**The protocol.** A program checker: on input $x$, run the program on $x$ and on related inputs
chosen at random; output the program's answer if the answers are consistent, and BUGGY otherwise.
The program is a fixed function (queried in separate contexts).

**The theory.**
1. **Theorem (`thm:consistency:freivalds`, Freivalds 1977).** Checking $AB=C$ with a random vector:
   one-sided error at most $1/2$ per trial with a random $0/1$ vector, or $1/\abs{S}$ with entries
   from a set $S$ of field elements. Complete proof. Time $O(n^2)$ against $O(n^\omega)$.
2. **Proposition (`prop:consistency:selfcorrect`).** If $g$ agrees with a linear function $f$ on a
   $1-\delta$ fraction of inputs, then for every $x$, $\Prob_r[g(x+r)-g(r)=f(x)]\ge1-2\delta$.
   Complete proof. Self-testing with the linearity test of `thm:spotchecks:blr`.
3. **The Blum–Kannan checker for graph isomorphism** (complete): if the program says
   "isomorphic", use it to find an isomorphism by self-reduction and check it; if "not isomorphic",
   run the private-coin test of `\cref{ch:interaction}` with the program as the prover.
4. **Main theorem (`thm:consistency:blumkannan`, Blum and Kannan).** A decision problem has an
   efficient program checker if and only if it has a function-restricted interactive proof (the
   prover answers as a fixed function of the question). Complete proof.
5. **Consistent liars.** Proposition: for a fixed finite family of consistency relations without
   random self-reducibility, there are wrong programs that satisfy every relation (example: coherent
   but wrong probabilities pass every probabilistic-coherence check). Random self-reducibility
   (Lipton 1991 for the permanent — verify; linear functions) is what turns consistency into
   correctness.

**Running the protocol.** `03-consistency-freivalds.py` (catch rate of a wrong product against
trials, with the bound $2^{-t}$; costs against size); `03-consistency-selfcorrect.py` (error of the
self-corrected value against the corruption rate $\delta$, with the bound $2\delta$, including
adversarially placed corruptions); `03-consistency-forecast.py` (a coherent but miscalibrated
forecaster passes coherence checks; an incoherent one is caught).

**The three questions.** (1) The program is fixed, so there is no game; the checker never rejects a
correct program (perfect completeness), and it needs no cooperation. (2) The checker runs in
$o(\text{program time})$ (the little-oh rule of Blum and Kannan — verify), plus a few calls to the
program. (3) For a buggy output on $x$, at least $1-2^{-t}$ (or the chapter's bound), against every
fixed program; against a coherent liar, zero for checks that lack random self-reducibility.

**What the model leaves out.** A model is not a fixed function: it may depend on context, on sampling
temperature, or on detecting that it is being checked (evaluation awareness — verify a source); most
oversight tasks lack random self-reducibility; consistency of probabilities says nothing about
calibration. Open: consistency checks with soundness guarantees for natural-language tasks.

**Cross-links.** `\cref{ch:spotchecks}` (linearity), `\cref{ch:interaction}` (the GNI test used as a
checker), `\cref{ch:crossexam}` (a second prover forces a fixed function), `\cref{ch:backdoors}`
(self-correction removes backdoors).

**Key references.** Freivalds 1977 (verify); Blum and Kannan 1995; Blum, Luby, Rubinfeld 1993; Lipton
1991; Rubinfeld and Sudan 1996 (verify); Fluri, Paleka, Tramèr 2024; Paleka et al. on consistency of
LLM forecasters (verify); Arora and Barak 2009.

---

### Part II. Conversation

*A prover that answers questions can adapt its answers to them. In exchange the verifier gains a
resource it did not have: questions the prover cannot predict. With interaction and private coins,
a verifier that can barely compute can check claims about computations of exponential length.*

---

### Chapter 4. Interaction — Questions the prover cannot predict (key: interaction)

`\protocolchapter{Interaction}{Questions the prover cannot predict}{private coins for graph non-isomorphism}{V,P,V,P,V,P,D}{ch:interaction}`

**The oversight problem.** A model claims that two objects differ in a way the overseer cannot see
(two molecules, two programs' behaviors, two signals); or claims an ability that the overseer cannot
test directly. The classical illustrations: a taster who says two drinks differ is tested on
secretly shuffled cups; Goldreich's color-blind friend and two balls (verify the source of the
illustration). In AI evaluation, the overseer's advantage is a secret, randomized test; when the
model can predict the test (contamination, leaked benchmarks), the advantage disappears.

**The protocol.** Graph non-isomorphism (Goldreich, Micali, Wigderson 1986/1991): the verifier picks
$b\in\bits$ and a random permutation $\pi$, sends $H=\pi(G_b)$, and accepts if the prover returns $b$.

**The theory.**
1. Definitions (cite `def:toolkit:ip`): interactive proofs, private and public coins, $\cc{IP}$,
   $\cc{AM}$, $\cc{MA}$.
2. **Theorem (`thm:interaction:gni`).** GNI has a private-coin interactive proof with perfect
   completeness and soundness error $1/2$ per round (complete proof), and $2^{-t}$ after $t$
   sequential or parallel rounds (prove that parallel repetition works for single-prover
   interactive proofs with this structure, or cite `prop:toolkit:amplification` for sequential).
3. **Theorem (`thm:interaction:setlower`, Goldwasser–Sipser).** The set lower bound protocol with
   pairwise-independent hashing (complete proof, with the constants of Arora and Barak §8.2 —
   verify). Corollary: GNI $\in\cc{AM}$ (complete or sketch).
4. Private versus public coins: $\cc{IP}[k]\subseteq\cc{AM}[k+2]$ (Goldwasser and Sipser 1986,
   statement); Babai–Moran collapse $\cc{AM}[k]=\cc{AM}[2]$ for constant $k$ (statement); the
   oversight reading: secrecy can be replaced by a stronger prover and more rounds.
5. **Theorem (`thm:interaction:ipinpspace`).** $\cc{IP}\subseteq\cc{PSPACE}$: the maximum
   acceptance probability over all provers is computable in polynomial space by recursion over the
   transcript tree (complete proof). The honest prover's strategy, in general, needs this power.

**Running the protocol.** `04-interaction-gni.py`: random graph pairs on $n\approx8$ vertices,
honest prover by brute-force canonical forms; the best cheating prover on isomorphic pairs (it can
do no better than guess); acceptance against rounds with the bound $2^{-t}$; and a prover that sees
the verifier's coins (the protocol fails). `04-interaction-setlower.py`: the Goldwasser–Sipser
protocol for sets of controlled size; acceptance against $\abs{S}/K$ with the proved bounds.

**The three questions.** (1) In the game where the prover is paid on acceptance, the honest prover is
always accepted on a true claim and no prover beats $1/2$ per round on a false one; the honest
prover must solve graph isomorphism, and in general needs $\cc{PSPACE}$. (2) Polynomial time; one
random permutation per round; two messages per round. (3) $1/2$ per round, $1-2^{-t}$ after $t$
rounds, against every prover that cannot see the verifier's coins.

**What the model leaves out.** Secrecy of an evaluator's randomness is hard to keep when models are
trained on the internet and tested repeatedly; a model may recognize test conditions; private coins
fail against a prover that sees them. Where it breaks: benchmark contamination studies (verify a
standard source, e.g. Sainz et al. 2023 or Golchin and Surdeanu 2024).

**Cross-links.** `\cref{ch:consistency}` (the GNI test as a checker), `\cref{ch:sumcheck}` (public
coins suffice for $\cc{PSPACE}$), `\cref{ch:zeroknowledge}` (graph isomorphism in zero knowledge).

**Key references.** Goldwasser, Micali, Rackoff 1989; Babai 1985; Babai and Moran 1988; Goldreich,
Micali, Wigderson 1991; Goldwasser and Sipser 1986; Arora and Barak 2009; Goldreich 2008.

---

### Chapter 5. Arithmetization — Sumcheck and IP = PSPACE (key: sumcheck)

`\protocolchapter{Arithmetization}{Sumcheck and IP = PSPACE}{the sumcheck protocol}{P,V,P,V,P,V,P,V,P,V,D}{ch:sumcheck}`

**The oversight problem.** A model claims a fact about every one of $2^\ell$ cases: the number of
configurations of a system that violate a safety property, the total of a huge table, the value of
a game ("White wins from here"). The overseer can evaluate the property at a few points. The
history: Lund, Fortnow, Karloff and Nisan (1990) proved $\cc{coNP}\subseteq\cc{IP}$ and Shamir
(1990) $\cc{IP}=\cc{PSPACE}$, in a few weeks of results exchanged by email (Babai 1990, "E-mail and
the unexpected power of interaction" — verify).

**The protocol.** Sumcheck for $\sum_{b\in\bits^\ell}g(b)=C$, with $g$ of degree at most $d$ in each
variable over $\F$; the verifier needs one evaluation of $g$ at a random point.

**The theory.**
1. Schwartz–Zippel (`lem:toolkit:schwartzzippel`), recalled; the univariate case proved inline.
2. **Theorem (`thm:sumcheck:sumcheck`).** Perfect completeness; for every prover, soundness error
   at most $\ell d/\abs{\F}$. Complete proof by induction on the rounds (the "lie must continue"
   argument), with the probability space stated.
3. Arithmetization of 3-CNF; **#SAT $\in\cc{IP}$** (complete); $\cc{coNP}\subseteq\cc{IP}$.
4. **Main theorem (`thm:sumcheck:ippspace`).** $\cc{IP}=\cc{PSPACE}$: TQBF $\in\cc{IP}$ by Shen's
   degree-reduction (linearization) operators (complete proof with the soundness bound), and
   $\cc{IP}\subseteq\cc{PSPACE}$ from `thm:interaction:ipinpspace`.
5. Costs: prover time (exponential in general), verifier time $O(\ell d)$ plus one evaluation,
   communication $O(\ell d\log\abs{\F})$; the honest prover's sums.

**Running the protocol.** `05-sumcheck-rates.py`: sumcheck over $\F_p$ for the arithmetization of
random 3-CNFs with $\ell\approx10$–$14$; the honest prover; a cheating prover that claims a wrong
count and plays the best continuation (choose each $g_j$ consistent with the claim, so that it
survives only if the verifier's point is a root of the difference); acceptance against $\abs{\F}$
(small primes, so that cheating sometimes succeeds) with the bound $\ell d/\abs{\F}$; prover and
verifier operation counts against $\ell$.

**The three questions.** (1) The honest prover is always accepted, but it must compute sums over
$2^{\ell-j}$ points in round $j$ — exponential work: honesty is a best response the honest prover
may not afford. This motivates `\cref{ch:delegation}` and `\cref{ch:efficientdebate}`. (2)
$O(\ell d)$ field operations and one evaluation of $g$; $\ell$ rounds. (3) At least
$1-\ell d/\abs{\F}$ against every prover, unbounded; amplify by enlarging the field.

**What the model leaves out.** The claim must be arithmetized: oversight claims are rarely low-degree
polynomials; the verifier must evaluate $g$ at a non-Boolean point, which needs a circuit for the
property; the honest prover's exponential cost.

**Cross-links.** `\cref{ch:interaction}`; `\cref{ch:delegation}` (sumcheck inside GKR);
`\cref{ch:debate}` (PSPACE again, from two provers); `\cref{ch:crossexam}` (multilinearity tests in
MIP).

**Key references.** Lund, Fortnow, Karloff, Nisan 1992; Shamir 1992; Shen 1992; Babai 1990; Schwartz
1980; Zippel 1979; DeMillo and Lipton 1978; Arora and Barak 2009; Thaler 2022.

---

### Chapter 6. Delegation — Checking a computation without redoing it (key: delegation)

`\protocolchapter{Delegation}{Checking a computation without redoing it}{the GKR protocol}{P,V,P,V,P,o,P,V,P,V,P,o,P,V,P,V,P,D}{ch:delegation}`

Research digest: `build/research/verifiable-computation.md` (sections 1–2).

**The oversight problem.** A provider claims that an output is what a specified model computes on a
given input, or that an evaluation score is what a specified benchmark computes; the overseer cannot
afford to rerun it. Verifiable inference and zero-knowledge machine learning: zkCNN (Liu, Xie, Zhang,
CCS 2021: VGG16 proved in 88.3 s with a 341 KB proof verified in 59.3 ms, in the final version; the 2021
preprint's numbers differ), zkLLM (Sun, Li, Zhang, CCS 2024: LLaMa-2-13B with 2048 tokens, about 803 s
to prove, a 188 kB proof, 3.95 s to verify; zero knowledge only against a semi-honest verifier), and
2025–2026 systems at transformer scale (zkGPT, zkPyTorch, DeepProve — preprints or recent venues;
verify before citing). The theory: Goldwasser, Kalai and Rothblum, "Delegating computation:
interactive proofs for muggles" (STOC 2008; JACM 2015); Goldreich's survey of doubly-efficient
interactive proofs (Foundations and Trends in TCS, 2018).

**The protocol.** GKR for layered arithmetic circuits: the claim about the output layer is reduced,
layer by layer, to a claim about the input, each step by a sumcheck over the wiring predicates and
a "two points to one" reduction on a line.

**The theory.**
1. Multilinear extensions (`lem:toolkit:mle`); the identity expressing $\tilde W_i$ through
   $\tilde W_{i+1}$ and the wiring predicates $\widetilde{\mathrm{add}}_{i+1},
   \widetilde{\mathrm{mult}}_{i+1}$ (complete proof).
2. **Main theorem (`thm:delegation:gkr`).** For a layered arithmetic circuit of size $S$ and depth
   $D$ over $\F$ with $n$ inputs, whose wiring predicates' multilinear extensions the verifier can
   evaluate in time $t_w$: perfect completeness; soundness error at most $O(D\log S/\abs{\F})$
   (state the exact sum); verifier time $O(n+D(\log S+t_w))$ field operations up to the evaluation
   of the input's extension; honest prover time $\poly(S)$ (state the bound the chapter proves,
   e.g. $O(S\log S)$ or $O(S)$ as in Thaler 2013 — verify). Complete proof of completeness and
   soundness; costs proved for the implementation described.
3. Doubly-efficient interactive proofs: the definition; GKR for log-space uniform circuits of depth
   $D$ (statement as in GKR 2015 — verify); Reingold, Rothblum and Rothblum 2016 for bounded-space
   computations in constant rounds (statement).
4. From interactive proofs to practice: the Fiat–Shamir transform and succinct arguments
   (statement; see `\cref{ch:training}`); quantization of a network into field arithmetic.

**Running the protocol.** `06-delegation-gkr.py`: GKR over $\F_p$ for a small layered circuit that
computes a two-layer network with square activations on a quantized input; honest prover; a
cheating prover that changes the output and the best continuation; acceptance against $\abs{\F}$
with the bound; operation counts of prover and verifier against circuit size (log-log), with the
predicted slopes.

**The three questions.** (1) The honest prover runs in time polynomial (near-linear) in the circuit:
doubly efficient, so honesty is affordable. (2) Near-linear in the input, plus
$D\cdot\polylog(S)$; $O(D\log S)$ rounds. (3) At least $1-O(D\log S/\abs{\F})$ against every prover.

**What the model leaves out.** Floating-point networks must be quantized into field arithmetic; the
protocol checks the computation, not whether the model's answer is right; sampling makes outputs
nondeterministic (the randomness must be committed); prover overheads in practice (zkLLM numbers —
verify); the verifier must know which model was run (commitment to weights, `\cref{ch:zeroknowledge}`).

**Cross-links.** `\cref{ch:sumcheck}`, `\cref{ch:efficientdebate}` (efficiency of the honest side
again), `\cref{ch:zeroknowledge}`, `\cref{ch:training}`.

**Key references.** Goldwasser, Kalai, Rothblum 2008/2015; Cormode, Mitzenmacher, Thaler 2012; Thaler
2013; Reingold, Rothblum, Rothblum 2016/2021; Goldreich 2018 (doubly-efficient interactive proofs);
Kalai, Raz, Rothblum 2014 (no-signaling delegation); Thaler 2022; Liu, Xie, Zhang 2021; Sun, Li, Zhang
2024; Kang et al. 2022 (preprint).

---

### Part III. Competition

*When the verifier cannot check one prover, it can set two against each other. Debate gives a judge
who reads a single step of an argument the power of polynomial space, and provers questioned apart,
who cannot confer, give more. What the honest side must compute decides which of these can be used.*

---

### Chapter 7. Debate — Two provers and a judge (key: debate)

`\protocolchapter{Debate}{Two provers and a judge}{debate}{P,A,P,A,P,A,P,A,D}{ch:debate}`

Research digest (verified by a research assistant; re-check what you use): `build/research/debate.md`.

**The oversight problem.** Irving, Christiano and Amodei (2018) proposed training AI systems by
self-play in a zero-sum debate: two copies argue for opposite answers in front of a judge, who decides
which argued better. The hope: it is harder to lie than to refute a lie, so at equilibrium the
debaters tell the truth. Their Table 1 matches the number of debate steps with complexity classes: a
judge acting alone corresponds to $\cc{P}$ (supervised learning), one step to $\cc{NP}$
(reinforcement learning with human judgments), $k$ steps to $\Sigma_k^p$, and polynomially many steps
to $\cc{PSPACE}$. Their Theorem 1 states only the inclusion $\cc{PSPACE}\subseteq$ debate (via
TQBF); the converse is the easy half of Chandra, Kozen and Stockmeyer's alternation theorem, which they
do not argue. The model assumes judges make no errors and debaters are unbounded (their §8). The
book's version of the story should say all of this precisely.

**The protocol.** Debate: a question, two answers, alternating statements by $\party{A}$ and
$\party{B}$ for polynomially many rounds, and a polynomial-time judge who reads the transcript and
declares a winner.

**The theory.**
1. Debate as a finite two-player zero-sum game of perfect information; backward induction
   (`thm:toolkit:backward`); value; what "optimal play" means (minimax play; Irving et al. use Nash
   equilibria for variants).
2. **Main theorem (`thm:debate:pspace`).** A language is decided by debate with a polynomial-time
   judge, polynomially many rounds and optimal play if and only if it is in $\cc{PSPACE}$. Complete
   proof of both directions: TQBF as a debate ($\party{A}$ picks existential variables, $\party{B}$
   universal ones); evaluation of the game tree in polynomial space. Attribute the first direction to
   Irving et al.'s Theorem 1 and the characterization to Chandra, Kozen, Stockmeyer (1981:
   $\cc{APTIME}=\cc{PSPACE}$).
3. **Theorem (`thm:debate:bisection`).** The bisection debate for a deterministic computation of
   length $T$ in space $s$: the debaters disagree about the configuration at the midpoint; the judge
   recurses into one half; after $\lceil\log_2T\rceil$ rounds the judge checks a single step. The honest
   debater always wins; the judge reads $O(s\log T)$ bits and checks one transition. Complete proof.
   This makes precise the informal claim that the judge needs to check only one step of the argument
   (Irving et al.'s examples: prime counting by bisection; simulation by bisection on midpoints, which
   works for deterministic simulations). Note what the honest debater must compute (configurations at
   arbitrary midpoints of a possibly exponential computation).
4. **Probabilistically checkable debates** (Condon, Feigenbaum, Lund, Shor 1995, Chicago J. Theor.
   Comput. Sci.): $\cc{PSPACE}=\cc{PCD}(\log n,1)$, the judge reads $O(1)$ bits of the debate
   (statement and idea); the recent result that debate query complexity $O(\log n)$ characterizes
   $\cc{PSPACE}/\mathrm{poly}$ (Brown-Cohen, Irving, Marshall, Newman, Piliouras, Szegedy, "Debate is
   efficient with your time", arXiv 2602.08630, 2026 — a preprint; verify).
5. Debates with imperfect information: refereed games. Feige, Shamir, Tennenholtz (CRYPTO '88, "The
   noisy oracle problem") introduced competing provers; Koller and Megiddo (1992) give the upper bound;
   Feigenbaum, Koller, Shor (1995) the classification (imperfect information with perfect recall: $\cc{EXP}$;
   imperfect recall of the yes-player: $\cc{NEXP}$); Feige and Kilian (1997, "Making games short"):
   one round with private channels gives $\cc{PSPACE}$, polynomially many rounds give $\cc{EXP}$ (verify
   each against the digest and the sources). Statements only.
6. Training and equilibria: zero-sum self-play and no-regret dynamics (Chen, Chen, Foster, Hazan 2023,
   "Playing large games with oracles and AI debate" — statement); what is not known.
7. Experiments, as observations with their settings: Irving et al.'s MNIST debate (a sparse-pixel judge
   with accuracy 59.4% from 6 random pixels; the honest debater wins 88.9% of debates with
   precommitment — this is a win rate, not a judge accuracy); Michael et al. 2023 (human debate 84%
   versus consultancy 74%); Khan et al. 2024 (ICML 2024; debate 76% and 88% for LLM and human judges
   against naive baselines of 48% and 60%); Kenton et al. 2024 (NeurIPS 2024; debate beats consultancy
   on every task, beats direct question answering only with information asymmetry); Arnesen, Rein,
   Michael 2024 (self-play training improves judge accuracy); Engels et al. 2025 (scaling laws for
   scalable oversight, NeurIPS 2025). Verify every number.

**Running the protocol.** `07-debate-trees.py`: debates on random game trees (or random TQBF
instances) between depth-limited minimax debaters; the judge evaluates a leaf; plot the judge's
accuracy against the lookahead advantage of the dishonest debater, and confirm that with full lookahead
the honest side always wins. `07-debate-bisection.py`: bisection debates over a recorded deterministic
computation with a liar who corrupts one step; the judge's cost (rounds, bits read) against $T$.

**The three questions.** (1) Zero-sum: the honest debater has a winning strategy, so honest play is a
minimax strategy and every equilibrium gives the honest side the win; but computing it can require
$\cc{PSPACE}$, and against a stronger liar the guarantee is gone. (2) The judge runs in polynomial time
and, in the bisection debate, checks one step after $O(\log T)$ rounds; with probabilistically
checkable debates it reads $O(1)$ bits. (3) Under optimal play every lie loses; with bounded debaters
this theorem gives no guarantee (see `\cref{ch:efficientdebate}`).

**What the model leaves out.** Optimal play; persuasion versus truth (human judges are swayed by
rhetoric and confidence; withheld evidence); the obfuscated-arguments problem (Barnes 2020, Alignment
Forum; formalized in `\cref{ch:efficientdebate}`); debaters that are copies of one model may tacitly
collude; the judge's systematic errors become the ground truth.

**Cross-links.** `\cref{ch:sumcheck}` ($\cc{PSPACE}$ from one prover), `\cref{ch:crossexam}`,
`\cref{ch:efficientdebate}`.

**Key references.** Irving, Christiano, Amodei 2018; Chandra, Kozen, Stockmeyer 1981; Stockmeyer and
Meyer 1973; Condon, Feigenbaum, Lund, Shor 1995; Feige, Shamir, Tennenholtz 1988; Koller and Megiddo
1992; Feigenbaum, Koller, Shor 1995; Feige and Kilian 1997; Brown-Cohen et al. 2026 (debate query
complexity); Chen, Chen, Foster, Hazan 2023; Michael et al. 2023; Khan et al. 2024; Kenton et al.
2024; Arnesen, Rein, Michael 2024; Engels et al. 2025; Barnes 2020; Irving and Askell 2019.

---

### Chapter 8. Cross-Examination — Questioning provers who cannot confer (key: crossexam)

`\protocolchapter{Cross-Examination}{Questioning provers who cannot confer}{two isolated provers}{V,V,P,A,o,V,V,P,A,D}{ch:crossexam}`

Research digests: `build/research/debate.md`, `build/research/classics-positioning.md`.

**The oversight problem.** Investigators question two suspects in separate rooms; an overseer asks two
instances of a model, in separate contexts, questions whose answers must fit together. In the debate
literature, *cross-examination* was introduced in the Alignment Forum writeup "Progress on AI safety
via debate" (Barnes and Christiano, February 2020): a debater may question a snapshot of its opponent
taken when a claim was written, and the writeup states that in theory this raises debate from
$\cc{PSPACE}$ to $\cc{NEXP}$, with the argument in a linked document that is not public. It is not in
Irving et al. 2018 (a common misattribution, for example in Buhl et al. 2025 — say so neutrally).
Doubly-efficient debate uses cross-examination of independent copies (`\cref{ch:efficientdebate}`).
The theory behind the claim: multi-prover interactive proofs (Ben-Or, Goldwasser, Kilian, Wigderson
1988), whose power is $\cc{NEXP}$ (Babai, Fortnow, Lund 1991).

**The protocol.** Oracularization: the verifier sends a full set of queries to $\Prov_1$ and one of
them, at random, to $\Prov_2$, and checks the answers against each other and against its oracle
verifier.

**The theory.**
1. Definitions: $\cc{MIP}$; strategies as functions of a prover's own questions; the no-communication
   assumption.
2. **Main theorem (`thm:crossexam:oracle`, Fortnow, Rompel, Sipser 1988/1994).** A language has a
   multi-prover interactive proof if and only if it has a probabilistic oracle verifier sound against
   every oracle. Complete proof of both directions, with the soundness bound $1-(1-s)/q$ for the
   consistency check (derive it, and show it is attained in some example). Corollary:
   $\cc{MIP}\subseteq\cc{NEXP}$.
3. Shared randomness does not help classical provers: the value of a two-prover game is attained by
   deterministic strategies (complete, averaging).
4. **Theorem (`thm:crossexam:nexp`).** $\cc{MIP}=\cc{NEXP}$ (Babai, Fortnow, Lund 1991): statement and
   a sketch of the multilinearity test; the 1989–1990 e-mail race that produced it (Babai 1990).
5. Repetition: the value of a parallel-repeated two-prover game need not multiply (give a verified
   counterexample, e.g. Feige's or Fortnow's, from the literature); Raz's parallel repetition theorem
   (statement as restated by Holenstein — verify).
6. Collusion resources: entanglement and $\cc{MIP}^*=\cc{RE}$ (Ji, Natarajan, Vidick, Wright, Yuen;
   arXiv 2001.04383; version 4 of October 2026 is accepted to the Annals of Mathematics),
   statement; the CHSH game as the smallest example of entangled provers beating classical ones.
7. Memory as a resource. The book's own remark (say it is ours): Feigenbaum, Koller and Shor's result
   that a refereed game whose yes-player has imperfect recall captures $\cc{NEXP}$ mirrors
   cross-examination of memoryless copies; and Irving and Marshall's 2025 Alignment Forum note that
   oversight results must relativize because humans are oracles.

**Running the protocol.** `08-crossexam-clause.py`: the clause–variable game for 3-CNF formulas (the
verifier asks $\Prov_1$ for the values of a random clause's variables and $\Prov_2$ for one of them);
isolated cheating strategies on unsatisfiable formulas (the best fixed assignment and randomized ones)
against a prover pair that can communicate; acceptance against rounds with the bounds; and the CHSH
game's classical value $3/4$ with an exhaustive check of deterministic strategies.

**The three questions.** (1) Honest provers answering from one fixed witness are always accepted, and
no isolated pair beats the soundness bound; communication between provers destroys soundness, shared
randomness does not help classical provers, entanglement can. (2) Polynomial time, one round, short
questions; small error needs repetition (and parallel repetition needs Raz's theorem). (3) $1-s$ per
round against isolated provers (constant with a PCP, $\Omega(1/m)$ for the naive clause game), none
against provers that can confer.

**What the model leaves out.** Copies of one model share weights and can coordinate on focal answers
without communicating; isolating agentic systems is hard (shared memory, tools, the internet);
steganography in outputs; what "cannot confer" means for models trained together. Open: the right model
of collusion between copies of a model.

**Cross-links.** `\cref{ch:spotchecks}` (PCPs), `\cref{ch:consistency}` (a fixed function),
`\cref{ch:debate}`, `\cref{ch:efficientdebate}`.

**Key references.** Ben-Or, Goldwasser, Kilian, Wigderson 1988; Fortnow, Rompel, Sipser 1994; Babai,
Fortnow, Lund 1991; Babai 1990; Raz 1998; Holenstein 2009 (verify); Ji et al. 2020/2021; Clauser, Horne,
Shimony, Holt 1969 (CHSH; verify); Barnes and Christiano 2020; Feigenbaum, Koller, Shor 1995; Irving
and Marshall 2025; Arora and Barak 2009.

---

### Chapter 9. Efficient Debate — When the honest side must be efficient too (key: efficientdebate)

`\protocolchapter{Efficient Debate}{When the honest side must be efficient too}{doubly-efficient debate}{P,A,P,A,P,A,P,A,H,D}{ch:efficientdebate}`

Research digest: `build/research/debate.md` (sections 2–5 and 9).

**The oversight problem.** The $\cc{PSPACE}$ theorem of `\cref{ch:debate}` lets the honest debater be
all-powerful; a model is not. If the honest model must out-compute its opponent to win, debate
protects only against weaker liars. Brown-Cohen, Irving and Piliouras ("Scalable AI Safety via
Doubly-Efficient Debate", ICML 2024) ask for protocols in which the honest debater's work is close to
the work of the computation itself, the judge's work is small, and the judge consults the human
($\party{H}$, modeled as an oracle) a constant number of times, against a dishonest debater with
unlimited computation. The obfuscated-arguments problem (Barnes, Alignment Forum, December 2020) shows
what goes wrong when the honest debater cannot find the flaw in a long argument; prover–estimator
debate (Brown-Cohen, Irving, Piliouras, Chen, Li, Xun, arXiv 2506.13609 v2, 2026; v1's main theorem
was wrong — cite v2 and say so) and "How to have a sensitive debate" (Li, Xun, Chen, Brown-Cohen, arXiv
2610.02557, October 2026) respond to it. A single-prover alternative: Chen, Kalai, Xi, "How to avoid
debate" (ICML 2026).

**The protocol.** The doubly-efficient debate for a deterministic oracle machine $M^{\party{H}}$ that
runs in time $T$ and space $S$: $\party{A}$ commits to the computation by bisection over configurations
(with cross-examination, by a transcript that the judge reads selectively), $\party{B}$ points to a
disputed half, and after $O(\log T)$ rounds the judge checks one step, consulting $\party{H}$ at most
once.

**The theory.**
1. The model (verify against the paper): oracle Turing machines with an oracle $\party{H}\colon\bits^l\to\bits$
   (their $\mathcal{O}$) modeling human judgment; deterministic and stochastic oracles; a $(P_{\mathrm{time}},
   V_{\mathrm{time}}, q)$-protocol; doubly efficient means polynomial honest-debater time,
   verifier time linear in $l$ and ideally sublinear in $n$, $q=O(1)$, against an unbounded opponent;
   the equivalence with a guaranteed value in the zero-sum game (their Appendix C).
2. **Main theorem (`thm:efficientdebate:main`, their Theorem 5.1).** If $L$ is decided by a
   deterministic oracle machine in time $T$ and space $S$, then $L$ has an
   $(O(T\log T),O(S\log T),O(1))$ debate protocol. Complete proof (bisection on configurations; the
   honest debater simulates once and answers midpoint queries from its record; the judge checks one
   transition with at most one oracle query). With cross-examination (their Theorem 5.3):
   $(O(T\log T),O(l\log T),O(1))$, statement and proof idea.
3. **Stochastic oracles (their Theorem 6.2).** $K$-Lipschitz machines; the protocol with
   $(O(K^2T\log T),O(K^2+l\log T),O(K^2))$, completeness $3/5$ and soundness $2/5$; nontrivial when
   $K=o(\sqrt T)$. Proof sketch with the key backward-induction lemma proved if it fits (their
   formalization in Lean 4 is public — verify and mention).
4. **Obfuscated arguments.** State the problem precisely and prove a small proposition that captures
   it: a family of claims (for example "no factor below $X$" and "no factor above $X$" for an RSA
   modulus, or a hash-based family) on which a dishonest debater's decomposition has a false leaf that
   an efficient honest debater cannot locate except with small probability, under a stated
   assumption. Present Barnes's RSA illustration as she presents it (a toy illustration).
5. **Prover–estimator debate** (v2): the model (stability $\rho$, decomposition depth $d$ and width $q$),
   the protocol, and Theorems 6.1 and 6.2 as statements with the proof idea (growing rewards; the
   estimator need not find the flaw); the instance-optimal "sensitive debate" protocol (Theorems
   4.1, 4.2 and the $\rho^d$ lower bound 5.1), statements.
6. **Without debate.** Chen, Kalai, Xi (ICML 2026): single-prover doubly-efficient proofs for
   computations that call an oracle, when the computation tolerates a small fraction of wrong oracle
   answers or the oracle is a low-degree polynomial (statements); the relativization barrier
   (IP = PSPACE and the PCP theorem do not relativize) and why oversight results must relativize when
   the human is an oracle.
7. Human error: Irving's "Dodging systematic human errors in scalable oversight" (Alignment Forum
   2025): if the human errs on an $\varepsilon$ fraction of questions, the honest side wins only with
   an argument robust to $\varepsilon$ errors (state as a model, and prove a simple version if
   possible); the safety case of Buhl et al. 2025 and its four claims (exploration, equilibria, online
   training, low stakes).

**Running the protocol.** `09-efficientdebate-bisection.py`: a long deterministic oracle computation
whose oracle is a simulated human judgment (a fixed function, then a noisy one); the honest debater's
work, the judge's work and the number of oracle calls against $T$ (log scale), with a liar who lies at
one step (always caught with a deterministic oracle); with a stochastic oracle, the judge's error
against the number of samples. `09-efficientdebate-obfuscation.py`: an argument tree with one hidden
false leaf among $N$; an honest debater who can check $k$ leaves finds it with probability $k/N$;
plot against $k$.

**The three questions.** (1) Zero-sum with an honest strategy that runs in $O(T\log T)$: honesty is
affordable and wins against every opponent, for deterministic oracle computations; for stochastic
ones with constant probability gap; obfuscated arguments break it when the honest side must find a
needle; prover–estimator debate restores average-case guarantees under stability. (2) $O(S\log T)$
or $O(l\log T)$ time and $O(1)$ calls to the human. (3) Deterministic oracle: every lie loses;
stochastic: soundness $2/5$ per run, amplified by repetition; obfuscation: no guarantee without
stability.

**What the model leaves out.** The human as a fixed function (consistency across queries, independent
errors); systematic human errors become "truth"; the computation $M$ must be specified (which program
of human judgments decides the question?); stability is an assumption about the world.

**Cross-links.** `\cref{ch:debate}`, `\cref{ch:delegation}` (efficiency of the honest prover),
`\cref{ch:crossexam}` (cross-examination), `\cref{ch:pvg}` (training to equilibria).

**Key references.** Brown-Cohen, Irving, Piliouras 2024; Brown-Cohen et al. 2026 (prover–estimator
debate v2); Li, Xun, Chen, Brown-Cohen 2026; Chen, Kalai, Xi 2026; Barnes 2020; Brown-Cohen and Irving
2024 (Alignment Forum, "Debate, oracles, and obfuscated arguments"); Irving 2025; Irving and Marshall
2025; Buhl, Pfau, Hilton, Irving 2025; Canetti, Riva, Rothblum 2013 (refereed delegation); Kol and Raz
2013; Goldwasser, Kalai, Rothblum 2015.

---

### Part IV. Incentives

*A trained model is not a worst-case adversary. It is the outcome of an optimization, and what it
learns depends on what it is paid for. When does payment make the truth the best policy, which
equilibria does training find, and can a learned model be checked with fewer samples than it took to
learn?*

Research digest for this part (verified by a research assistant; re-check what you use):
`build/research/learning-incentives.md`.

---

### Chapter 10. Rational Proofs — Paying for the truth (key: rational)

`\protocolchapter{Rational Proofs}{Paying for the truth}{the Brier-scored rational proof}{P,X,R}{ch:rational}`

**The oversight problem.** Much of AI training pays a model rather than checking it: a reward model,
a forecasting score, a grader that sees one outcome. A model reports the probability that a
deployment will fail, or the fraction of a huge space of inputs on which a property holds; the overseer
can observe one outcome or evaluate one random input, and pays. When does the payment make the
truthful report the model's best policy? Proper scoring rules (Brier 1950; Gneiting and Raftery
2007) answer this for forecasts; Azar and Micali (2012) turned them into *rational proofs*, in which
one random sample pays for a whole $\cc{\#P}$ computation.

**The protocol.** The rational proof for \#SAT: the prover reports $y$, the claimed number of
satisfying assignments of a CNF $\varphi$ on $n$ variables; the verifier draws one uniformly random
assignment, evaluates $\varphi$ on it, and pays by the (shifted) Brier score of the forecast
$p=y/2^n$ for the event "the random assignment satisfies $\varphi$".

**The theory.**
1. Proper and strictly proper scoring rules; **Theorem (`thm:rational:brier`).** The Brier score and
   the logarithmic score are strictly proper (complete proofs; the expected Brier penalty of
   reporting $p$ when the truth is $q$ exceeds the honest one by exactly $(p-q)^2$ per outcome
   coordinate — state the constant for the binary case). The characterization by convex functions
   (McCarthy 1956; Savage 1971; Gneiting and Raftery 2007; statement — verify which to cite).
2. **Main theorem (`thm:rational:sharpp`, Azar and Micali 2012).** $\#\mathrm{SAT}$, hence every
   function in $\cc{\#P}$, has a one-round rational proof: with $q=\#\varphi/2^n$,
   $\E[R]=2-2q(1-q)-2(p-q)^2$, uniquely maximized at $p=q$ (complete proof). Corollary:
   $\cc{PP}\subseteq\cc{DRMA}[1]$ (complete, short). Statements of the class results: constant rounds
   capture the counting hierarchy, polynomially many rounds capture $\cc{PSPACE}$ (verify in the
   paper).
3. **The reward gap.** A lie of size $\abs{y-\#\varphi}=\Delta$ costs $2(\Delta/2^n)^2$ in expectation,
   which can be $2\cdot4^{-n}$: exponentially small. Definition of a noticeable reward gap (Guo,
   Hubáček, Rosen, Vald 2014), and why a risk-neutral prover with a small cost of computation may
   prefer to guess. Sequential composability (Campanelli and Gennaro 2015): a prover with a budget can
   earn more by answering many questions cheaply and wrongly than one question correctly; prove a
   small proposition exhibiting this for the \#SAT protocol (a lazy report $p=1/2$ earns
   $2-2q(1-q)-2(1/2-q)^2$, which is near-optimal when $q$ is near $1/2$).
4. Rational proofs with several provers (Chen, McCauley, Singh 2016), statement; the oversight reading:
   competition can buy back a noticeable gap.
5. From payments to training: a reward model is a scoring rule only when it is evaluated on observed
   outcomes; RLHF rewards are human judgments, not outcomes. Eliciting what the model knows versus what
   it predicts the overseer will believe (Christiano, Cotra, Xu 2021, the ELK report); the recent
   impossibility claim of Friedl et al. (2026 preprint — verify, and say it is a preprint).

**Running the protocol.** `10-rational-brier.py`: random CNFs with $n\approx12$; the exact expected
reward against the report (a parabola) with Monte Carlo points from repeated runs; the reward gap
against the size of the lie; the number of runs an observer would need to tell an honest prover from a
liar by average reward (against a formula); and the lazy prover against the honest one when
computation has a cost.

**The three questions.** (1) For a risk-neutral prover that maximizes expected payment, the truthful
report is the unique best response, a strictly dominant strategy; but the honest report requires
counting, which may be infeasible, and the gap that rewards it can be exponentially small, so a
prover that pays for computation may rationally guess. (2) One random sample, one evaluation of
$\varphi$, $O(n)$ arithmetic. (3) A lie is never caught: it only costs $2(\Delta/2^n)^2$ in
expectation, and telling a liar from an honest prover from payments alone takes on the order of
$\Delta^{-4}4^{2n}$ runs (derive the exact statement).

**What the model leaves out.** Risk attitudes and objectives other than payment; outcomes the
overseer never observes (long horizons, counterfactuals); beliefs that are honest but wrong; budgets
and many questions; reward hacking of learned reward models (verify a standard source, e.g. Gao,
Schulman, Hilton 2023 on reward-model overoptimization).

**Exercises.** The logarithmic rule's gap; a scoring rule that is proper but not strictly; the
rational proof for a statistic of a distribution; an attack on a reward that is not proper (absolute
error elicits the median, not the mean); the sequential-composability counterexample with numbers;
simulation of detection by averaging; research: rational proofs with noticeable gaps for oversight
questions.

**Cross-links.** `\cref{ch:sumcheck}` (\#SAT by interaction instead of payment), `\cref{ch:pvg}`
(equilibria of trained provers), `\cref{ch:debate}` (zero-sum payments).

**Key references.** Azar and Micali 2012, 2013; Guo et al. 2014; Campanelli and Gennaro 2015, 2017;
Chen, McCauley, Singh 2016; Brier 1950; Gneiting and Raftery 2007; Christiano, Cotra, Xu 2021.

---

### Chapter 11. Prover–Verifier Games — Learning to give checkable answers (key: pvg)

`\protocolchapter{Prover--Verifier Games}{Learning to give checkable answers}{the verifier-leading game}{P,V,P,V,P,V,D}{ch:pvg}`

**The oversight problem.** Instead of designing a protocol by hand, train both parties: a strong
prover and a weak verifier, rewarded so that the verifier learns to accept only what it can check and
the prover learns to produce answers the verifier can check (Anil, Zhang, Wu, Grosse 2021). The idea
has been applied to make language-model solutions legible to people (Kirchner et al. 2024: helpful
and sneaky provers, a "legibility tax"), generalized to neural interactive proofs with several
protocols (Hammond and Adam-Day, ICLR 2025), and turned around into models trained to prove their own
answers to a fixed verifier (Amit, Goldwasser, Paradise, Rothblum: self-proving models). The question:
which equilibria of the training game are sound and complete proof systems, and does training find
them?

**The protocol.** The prover–verifier game: a verifier from a class $\cV$ and a prover from a class
$\cP$ (say which symbols the chapter uses; avoid clashing with $\Prov$), a distribution of instances,
the losses of Anil et al.; the order of moves (simultaneous, verifier-leading, prover-leading) and
the moment the instance is revealed.

**The theory.**
1. Equilibrium notions (`def:toolkit:nash`, `def:toolkit:stackelberg`) for the eight formulations of
   Anil et al. (Table 1 — verify).
2. **Main theorem (`thm:pvg:stackelberg`, Anil et al. 2021, Theorem 2).** In the verifier-leading game
   in which the instance is revealed after the verifier commits, and provided the classes contain a
   complete and sound pair, a strategy pair is an equilibrium if and only if it is a complete and sound
   proof system. Complete proof in the paper's model (zero loss at a complete and sound pair; positive
   loss against the follower's best response otherwise).
3. The other orders fail (complete proofs of the small examples): a verifier who sees the instance
   first can ignore the prover (Proposition 1); prover-leading equilibria can be uninformative
   ("flood the zone", Theorem 1); simultaneous Nash equilibria are necessary but not sufficient
   (Theorem 3). Every verifier-leading equilibrium is a Nash equilibrium (Corollary 1).
4. Neural interactive proofs (Hammond and Adam-Day 2025): $(\varepsilon_c,\varepsilon_s)$-validity,
   worst-case losses, and their Theorem 8 (a valid system is an approximate verifier-leading
   Stackelberg equilibrium of the nip game, and conversely, when a valid system exists): statement and
   proof sketch; their counterexample for the adp game (Proposition 7).
5. Self-proving models (Amit et al.): verifiability, the union-bound link to correctness, Transcript
   Learning and its convergence theorem under concavity assumptions (statement with all assumptions;
   proof sketch).
6. Dynamics: Anil et al.'s results on the binary erasure channel (simultaneous two-timescale gradient
   descent converges to the ideal protocol; prover-first descent converges to an uninformative prover),
   statement and sketch.
7. Observations, with numbers and settings: Kirchner et al. (legibility tax; human accuracy on helpful
   and sneaky solutions; verifier compute fractions); Hammond and Adam-Day's experiments; Amit et al.'s
   GCD experiment (verify each number).

**Running the protocol.** `11-pvg-bec.py`: the binary erasure channel game with softmax prover and
verifier trained by gradient dynamics, simultaneous two-timescale versus prover-first; plot verifier
loss, completeness and soundness against training steps; and an exact enumeration of the equilibria
of a tiny finite game showing a bad Nash equilibrium and the good verifier-leading one. NumPy only.

**The three questions.** (1) In the verifier-leading game with the instance revealed after
commitment, every equilibrium is complete and sound, so at equilibrium the prover's best response is
to argue honestly; in other orders bad equilibria exist; whether gradient training reaches the
Stackelberg equilibrium is known only for toy games. (2) A small verifier (state the compute fractions
reported); its inference cost. (3) Worst-case soundness holds only at an exact equilibrium with a
verifier class that contains a sound verifier; in practice soundness is measured against the provers
seen in training and against optimization attacks (Anil et al.'s L-BFGS attack; Kirchner et al.'s
sneaky provers).

**What the model leaves out.** Equilibria versus training dynamics; capacity (a sound verifier may not
exist in the class); adversarial examples against the verifier; the legibility tax; human legibility
versus model legibility; the verifier must already be able to recognize a correct argument.

**Cross-links.** `\cref{ch:interaction}` (the protocols that games try to learn), `\cref{ch:rational}`
(payments), `\cref{ch:debate}` (zero-sum self-play), `\cref{ch:pacverify}`.

**Key references.** Anil et al. 2021; Kirchner et al. 2024; Hammond and Adam-Day 2025; Amit et al.
2024/2025; Kim and Lee 2026 (decoupled games — verify); Irving et al. 2018; Burns et al. 2024.

---

### Chapter 12. Verifying Learning — Checking a model without learning it (key: pacverify)

`\protocolchapter{Verifying Learning}{Checking a model without learning it}{PAC verification}{X,V,P,D}{ch:pacverify}`

**The oversight problem.** A developer claims that its model is nearly as accurate as the best model
in a class on a task; an auditor can draw inputs from the task distribution but has few trusted
labels, because labels come from people. Must the auditor collect as many labels as it would take to
train such a model itself? Goldwasser, Rothblum, Shafer and Yehudayoff (ITCS 2021) defined PAC
verification and showed that, in some settings, verifying needs far fewer samples than learning.
The same question for claims about data: a developer claims that its training distribution has a
property (support size, entropy, distance from a reference).

**The protocol.** Query delegation with hidden checks (GRSY, Claim 5.2 of the full version): the
verifier draws $k$ labeled and many unlabeled inputs, shuffles them, asks the prover to label all of
them, rejects if a check label disagrees, and otherwise runs empirical risk minimization on the
returned labels.

**The theory.**
1. Definitions: $\alpha$-PAC verification (completeness, soundness against unbounded provers,
   efficiency of both sides; proper and improper); the comparison with PAC learning and with testing.
2. **Main theorem (`thm:pacverify:main`).** Under GRSY's conditions (a fixed marginal on inputs,
   labels given by a function, finite VC dimension, an ERM learner), $\cH$ is $1$-PAC verifiable with
   $k=\lceil4\ln(2/\delta)/\varepsilon\rceil$ labeled samples (check the logarithm's base in the
   source) and $m'=m^{\mathrm{UC}}_\cH(\varepsilon/4,\delta/2)$ unlabeled ones; complete proof (the
   spot-check bound $(1-t/m')^k$ and the six-step uniform-convergence chain). Corollary: the
   separation in labeled samples against agnostic learning, $\Theta((d+\log(1/\delta))/\varepsilon^2)$,
   is unbounded.
3. When labels are noisy: the lower bound $\Omega(\sqrt d/\varepsilon^2)$ for every PAC verifier of a
   class of VC dimension $d$ (Mutreja and Shafer, COLT 2023), statement and proof idea; multi-thresholds
   with $\tilde O(\sqrt d)$ samples by certificates of loss (GRSY Lemma 3.8): prove the certificate-of-
   loss claims (a pair of weighted events certifies a lower bound on the optimal loss; every
   distribution has a certificate for half its loss), state the rest.
4. Interactive Goldreich–Levin (GRSY Lemma 2.3) and its improvement (Gur et al., STOC 2024):
   statements; the oversight reading — a prover with stronger access (queries) helps a verifier with
   weaker access (samples).
5. Verifying claims about distributions: Chiesa and Gur 2018; Herman and Rothblum 2022, 2023, 2025
   (statements: $\tilde O(\sqrt N)$ against $\Theta(N/\log N)$ for testing; a doubly-efficient version;
   and when proving is harder than testing).
6. Refereed learning (Canetti, Linder, Wagaman, preprint — verify): choosing between two models with
   one honest prover.

**Running the protocol.** `12-pacverify-delegation.py`: a simple class (thresholds on a line, or
halfspaces in two or three dimensions) under a fixed input distribution; an honest prover; cheating
provers that mislabel a chosen fraction of the points, at random and adversarially (to shift the ERM
output); acceptance against the fraction mislabeled with the bound $(1-t/m')^k$; the excess loss of
the output when the prover is accepted; labeled samples used by the verifier against those a learner
needs for the same accuracy.

**The three questions.** (1) Truthful labeling is always accepted; a lie is either caught or too small
to move the output by more than $\varepsilon$, so lying gains nothing; the honest prover's cost is
labeling $O(m)$ points (queries to the labeling function). (2) $O(\log(1/\delta)/\varepsilon)$ labeled
samples, $O(m)$ unlabeled ones, one ERM computation, two messages. (3) At least $1-\delta/2$ against
every prover that mislabels more than an $\varepsilon/4$ fraction; the guarantee is about the output's
loss, not about catching every lie.

**What the model leaves out.** Deterministic labels and a known marginal (human labels disagree; the
prover may control the distribution); hidden checks must be indistinguishable from the other queries
(models can recognize test items); the verifier still runs ERM; "near-optimal in $\cH$" is not "safe".

**Cross-links.** `\cref{ch:spotchecks}` (hidden spot checks), `\cref{ch:interaction}` (private coins:
the positions of the checks), `\cref{ch:pvg}`, `\cref{ch:training}`.

**Key references.** Goldwasser, Rothblum, Shafer, Yehudayoff 2021; Mutreja and Shafer 2023; Gur et al.
2024; Chiesa and Gur 2018; Herman and Rothblum 2022, 2023, 2024, 2025; Canetti, Linder, Wagaman
(preprint); Shalev-Shwartz and Ben-David 2014 (uniform convergence; verify).

---

### Part V. Secrets and Limits

*Some claims are about objects the prover will not show: its weights, its data, the record of its
training. Cryptography lets a verifier check such claims without seeing the objects. It also shows
that some properties of a model can be hidden from every audit.*

---

### Chapter 13. Zero Knowledge — Auditing without disclosure (key: zeroknowledge)

`\protocolchapter{Zero Knowledge}{Auditing without disclosure}{commit, challenge, respond}{C,V,P,o,C,V,P,o,C,V,P,D}{ch:zeroknowledge}`

Research digests: `build/research/verifiable-computation.md` (section 2),
`build/research/classics-positioning.md`, `build/research/learning-incentives.md` (section 11).

**The oversight problem.** A developer must convince an auditor that its secret model has a property
— it computed this output, it scores this on a held-out evaluation, it has this architecture, it was
trained under a stated constraint — without revealing the weights. Zero-knowledge proofs of model
inference and evaluation: zkCNN, zkLLM (zero knowledge only against a semi-honest verifier),
verifiable evaluation attestations (South et al. 2024, preprint), trustless audits (ZkAudit,
Waiwitlikhit et al., ICML 2024: audits for as little as \$10 to \$108), fairness certificates for trees
(Confidential-PROFITT, ICLR 2023). Data-forging attacks on fixed-audit-set certificates (Luck et al.
2026, preprint: certified accuracy above 99% but under 30% on fresh samples — verify) show what a
certificate about a fixed set does not say. The question of whether oversight itself can be zero
knowledge when the verifier consults a human oracle: Chiesa, Guan, Yildiz, "Can AI oversight be zero
knowledge?" (arXiv 2610.01995, October 2026: impossible for all oracle-aided computations in the
random-oracle model; possible if the oracle signs its answers). The theory: Goldwasser, Micali,
Rackoff (1985/1989); Goldreich, Micali, Wigderson (1986/1991).

**The protocol.** Commit, challenge, respond: for graph isomorphism (the prover commits to a random
isomorphic copy, the verifier asks for one of two openings) and for 3-coloring (the prover commits to
a randomly permuted coloring, the verifier opens one edge).

**The theory.**
1. Definitions: honest-verifier and general zero knowledge; perfect, statistical and computational;
   the simulator; why "the verifier learns nothing it could not have computed".
2. **Theorem (`thm:zeroknowledge:gi`).** Graph isomorphism has a perfect zero-knowledge proof:
   completeness $1$, soundness error $1/2$ per round, and a simulator for every verifier that runs
   in expected polynomial time by rewinding (complete proof).
3. Commitments (`def:toolkit:commitment`); a commitment from a one-way permutation or from a
   collision-resistant hash (statement).
4. **Main theorem (`thm:zeroknowledge:np`).** Given a perfectly binding, computationally hiding
   commitment scheme, 3-coloring — hence every language in $\cc{NP}$ — has a computational
   zero-knowledge proof with soundness error $1-1/\abs{E}$ per round (complete proof, including the
   hybrid argument for the simulator).
5. Non-interactive and succinct: Fiat–Shamir (statement, random-oracle model), zkSNARKs (statement);
   costs reported for zkML systems (verify every number).

**Running the protocol.** `13-zeroknowledge-coloring.py`: the 3-coloring protocol with hash
commitments on small graphs; a cheating prover with an invalid coloring (one bad edge, and the best
cheating strategy); acceptance against rounds with $(1-1/\abs{E})^t$; the simulator's transcripts
against real ones (the distribution of revealed color pairs, a statistical distance with an interval).

**The three questions.** (1) The honest prover always convinces and reveals nothing beyond the claim,
so a developer can afford honesty without disclosure; soundness holds against every prover (for
3-coloring with perfectly binding commitments). (2) Polynomial time; many rounds for small error in
the interactive version, succinct verification with SNARKs. (3) $1/\abs{E}$ per round for
3-coloring, $1/2$ for graph isomorphism; $1-(1-1/\abs{E})^t$ after $t$ rounds.

**What the model leaves out.** A proof about a committed model says nothing about the deployed one
unless deployment is bound to the commitment; the property must be formalized; prover costs for
large models; trusted setup and implementation bugs; zero knowledge protects the prover, not the
public.

**Cross-links.** `\cref{ch:interaction}`, `\cref{ch:delegation}`, `\cref{ch:training}`.

**Key references.** Goldwasser, Micali, Rackoff 1989; Goldreich, Micali, Wigderson 1991; Fiat and
Shamir 1986; Goldreich 2001 (Foundations of Cryptography I); Thaler 2022; Liu, Xie, Zhang 2021; Sun, Li,
Zhang 2024; South et al. 2024; Waiwitlikhit et al. 2024.

---

### Chapter 14. Proofs of Training — Arguments about how a model was made (key: training)

`\protocolchapter{Proofs of Training}{Arguments about how a model was made}{logged-checkpoint spot checks}{C,S,Q,Q,Q,D}{ch:training}`

Research digest: `build/research/verifiable-computation.md` (sections 3 and 7, including two
proposed textbook theorems, A and B, which are the research assistant's synthesis and not quoted
results).

**The oversight problem.** A regulator wants to check claims about how a model was trained: that it
used less than a compute threshold, that it was trained on the claimed data, that a published model
came from the reported procedure. Proof-of-learning (Jia et al., IEEE S&P 2021) logs checkpoints and
data and re-executes the segments with the largest updates, accepting within a distance tolerance; its
security argument is heuristic. Spoofing attacks (Zhang et al., S&P 2022; Fang et al., EuroS&P 2023)
produce passing logs at less than the cost of training. Shavit (2023) proposes compute monitoring with
randomly timed on-chip weight snapshots and a detection bound (his Eq. 1; note the typo in his
Appendix B); Choi, Shavit, Duvenaud (NeurIPS 2023) propose proof-of-training-data; zero-knowledge proofs
of training exist for small networks at high cost (Abbaszadeh et al., CCS 2024); verifiable training by
controlling hardware nondeterminism with an auditor who replicates and challenges (Srivastava, Arora,
Boneh, NeurIPS 2024); refereed delegation with two trainers, one honest (Canetti, Riva, Rothblum 2011/2013).

**The protocol.** The prover commits to the training log (a Merkle root over checkpoints and batch
indices); the verifier samples segments, the prover opens them, and the verifier re-executes each
segment and checks that the result is within a tolerance of the next checkpoint.

**The theory.**
1. Collision-resistant hashing (`def:toolkit:crhf`) and **Theorem (`thm:training:merkle`).** Merkle
   commitments are binding: an opening of two different values at one position yields a collision
   (complete proof); opening cost $O(\log N)$ hashes.
2. **Proposition (`prop:training:spotcheck`).** If a committed log has at least $\varepsilon N$
   invalid segments, then $k$ uniformly random checks (with or without replacement — state which)
   detect one with probability at least $1-(1-\varepsilon)^k$ (complete proof).
3. Kilian's succinct argument (1992): commit to a PCP with a Merkle tree and open the queried bits;
   computational soundness against bounded provers (statement and proof sketch); Micali's CS proofs
   and Fiat–Shamir (statement).
4. **Why proof-of-learning lacks soundness.** In a toy model (for example, gradient descent on a
   convex quadratic) prove a proposition: with re-execution tolerance $\tau$, there is a log that
   passes every check but whose segments were not produced by training on the claimed data (an
   explicit spoofing construction), and quantify the work it saves. Say what assumptions would
   restore soundness (exact reproducibility, committed randomness, hardware attestation).
5. **Refereed training (complete proof).** With deterministic, reproducible steps, binding
   commitments, a common initialization and committed data, and two trainers of whom at least one is
   honest, bisection over the committed checkpoints finds a disagreement in $\lceil\log_2T\rceil$
   rounds and the referee re-executes one step; the referee outputs the honest result except with
   negligible probability (Canetti, Riva, Rothblum specialized to training; compare the bisection debate
   of `\cref{ch:debate}`).
6. Compute monitoring as an oversight protocol (Shavit 2023), with its detection bound and its trust
   assumptions.

**Running the protocol.** `14-training-pol.py`: a small training run (logistic regression or a tiny
network in numpy) with logged checkpoints; honest logs, logs with a fraction of forged segments, and
an interpolation spoof; detection against the number of checked segments with the bound
$1-(1-\varepsilon)^k$, and spoof acceptance against the tolerance $\tau$; honest logs rejected
because of nondeterminism at small $\tau$.

**The three questions.** (1) An honest trainer passes if training is reproducible within the
tolerance; a spoofer may pass too, so honesty is not enforced without further assumptions. (2) $k$
re-executed segments: a $k/N$ fraction of the training compute, plus $O(k\log N)$ hashes. (3) At least
$1-(1-\varepsilon)^k$ for logs with an $\varepsilon$ fraction of detectably invalid segments; no
guarantee against spoofs that are valid within the tolerance.

**What the model leaves out.** Hardware and nondeterminism; data provenance (the log names data that
the verifier cannot see); the gap between "a valid log exists" and "this model came from it"; the
cost of re-execution for frontier runs.

**Cross-links.** `\cref{ch:spotchecks}`, `\cref{ch:delegation}`, `\cref{ch:zeroknowledge}`.

**Key references.** Merkle 1987/1989 (verify); Kilian 1992; Micali 2000; Fiat and Shamir 1986; Jia et
al. 2021; Zhang et al. 2022; Fang et al. 2023; Shavit 2023; Choi, Shavit, Duvenaud 2023; Thaler 2022.

---

### Chapter 15. Undetectable Backdoors — What no audit can find (key: backdoors)

`\protocolchapter{Undetectable Backdoors}{What no audit can find}{the black-box audit}{S,K,Q,Q,Q,Q,Q,D}{ch:backdoors}`

**The oversight problem.** An auditor receives a model from a developer it does not trust and must
decide whether the model behaves differently on inputs carrying a hidden trigger. Sleeper agents in
language models persist through safety training (Hubinger et al. 2024 — verify). Goldwasser, Kim,
Vaikuntanathan and Zamir (2022) prove that some backdoors cannot be detected at all.

**The protocol.** The black-box audit: the auditor queries the model on inputs of its choice and
decides; the white-box audit sees the weights.

Research digest: `build/research/verifiable-computation.md` (section 4).

**The theory.**
1. Definitions (GKVZ, FOCS 2022): a $\gamma$-backdoor (Backdoor, Activate); white-box and black-box
   undetectability (Definition 4.6); non-replicability (Definition 4.7, a real/ideal comparison).
2. **Main theorem (`thm:backdoors:blackbox`).** The signature-keyed backdoor $\tilde h(w,y,\sigma)=
   (-1)^y$ if the signature verifies, $h$ otherwise, is black-box undetectable: for every efficient
   oracle distinguisher, the advantage is at most the advantage of a forger against the signature
   scheme under a no-message attack (complete proof: the two oracles differ only on validly signed
   inputs; the forger simulates training and outputs the first query that verifies). GKVZ's Theorem 5.4
   assumes strongly unforgeable signatures, which non-replicability needs; the sharpening to
   no-message unforgeability for undetectability alone is the book's (say so). Non-replicability:
   statement and proof idea.
3. White-box undetectable backdoors for random Fourier features under the hardness of continuous
   LWE (statement and sketch, as in the paper); consequences for robustness certification.
4. The other direction: some properties can always be read from code (Barak et al. 2001/2012,
   impossibility of virtual black-box obfuscation; statement and what it does and does not say about
   white-box audits).
5. **Theorem (`thm:backdoors:mitigation`).** Mitigation without detection: if $\tilde f$ agrees with a
   linear function $L$ on at least a $1-\varepsilon$ fraction of inputs, $\varepsilon<1/4$, then for
   each fixed $x$ the majority of $s$ self-corrected values $\tilde f(x+r_j)+\tilde f(r_j)$ equals
   $L(x)$ with probability at least $1-e^{-2s(1/2-2\varepsilon)^2}$ — every planted trigger is removed
   without being detected (complete proof, from `prop:consistency:selfcorrect` and Hoeffding). This is
   the base case of Goldwasser, Shafer, Vafa, Vaikuntanathan, "Oblivious defense in ML models"
   (STOC 2025): state their model (a mitigator with black-box queries and samples; security as
   closeness to an ideal output), their global result for Fourier-heavy functions, their local results
   for affine functions and low-degree polynomials, and their limits (mitigation cannot be
   distribution-free; for random functions it is no better than relearning).
6. Defense, learnability and obfuscation (Christiano, Hilton, Lecomte, Xu, ITCS 2025): PAC
   learnability implies efficient defendability; under one-way functions and indistinguishability
   obfuscation, polynomial-size circuits are not efficiently defendable (statements). Unelicitable
   backdoors (Draguns et al., NeurIPS 2024; note that titles and author lists differ between the
   proceedings and arXiv); white-box undetectable backdoors in deep networks (Bogdanov, Rosen, Vafa,
   ICML 2026 — verify); detection versus mitigation for generative models (Gluch and Goldwasser 2025,
   preprint).

**Running the protocol.** `15-backdoors-audit.py`: a classifier with a keyed trigger (an HMAC
truncated to $\lambda$ bits as a stand-in for a signature, said so plainly); black-box audits with
random and adversarially structured queries; detection rate against the number of queries and
$\lambda$, with the bound (queries)$\cdot2^{-\lambda}$; self-correction of a linear model with a
planted trigger set: error on triggered inputs before and after, against $2\delta$.

**The three questions.** (1) Nothing in the audit makes an honest model the developer's best
response when backdoors are undetectable; mitigation changes the game by making a backdoor useless.
(2) Polynomially many queries; (3) negligible in $\lambda$ for signature-keyed backdoors against
every efficient auditor, under the cryptographic assumption.

**What the model leaves out.** Backdoors found in practice are not cryptographic, and many are
detectable; the assumption is about the developer's intent; detection of naive triggers works;
white-box access changes the picture in some settings. Open: undetectability for trained language
models; mitigation beyond linear and low-degree classes.

**Cross-links.** `\cref{ch:consistency}`, `\cref{ch:zeroknowledge}`, `\cref{ch:training}`.

**Key references.** Goldwasser, Kim, Vaikuntanathan, Zamir 2022; Goldwasser, Shafer, Vafa,
Vaikuntanathan 2024/2025; Barak et al. 2012; Hubinger et al. 2024; Goldwasser, Micali, Rivest 1988
(signatures — verify); Draguns et al. 2024 (verify); Christiano et al. 2024 on backdoor defense and
learnability (verify).
