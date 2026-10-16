# Dissolving Problems — Book Plan and Chapter Specifications

## The thesis

Grothendieck contrasted two ways of opening a nut: strike it with hammer and chisel, or soak it and let
time soften it until it opens under the hand; at a larger scale, a sea that rises unnoticed until the
problem is submerged and dissolved in a theory that goes well beyond it. This book is a casebook of such
dissolutions across analysis, probability, combinatorics, optimization, theoretical computer science,
learning theory, and statistics. Each chapter isolates one **move** — a transferable reframing — and
teaches it through one historical case.

A reframing counts as a dissolution only if it pays at least one toll (the **dissolution test**):
1. **shorter proofs** — old theorems get proofs that are shorter for a reason, not just in notation;
2. **fewer assumptions** — hypotheses that looked necessary become unnecessary;
3. **unification** — previously separate results become one statement;
and its dividend is 4. **new reach** — results the old setting could not reach.
Difficulty is conserved: it is paid once, up front (in a definition, a construction, an existence
theorem), and then amortized over many problems. A reframing that must pay the difficulty again in every
application is a renaming, not a dissolution. Abstraction that does not pay is forbidden.

## Structure, keys, and files

| # | key | file | Title — Subtitle | Move |
|---|-----|------|------------------|------|
| – | prologue | 00-prologue.tex | What It Means to Dissolve a Problem — The needle, the noodle, and the rising sea | — |
| | | | **Part I. Building the Right Object** | |
| 1 | lebesgue | 01-lebesgue.tex | Completing the Integral — Lebesgue and the escaping limits | Completion |
| 2 | graphon | 02-graphon.tex | Completing Graphs — From the regularity lemma to graph limits | Completion |
| 3 | kolmogorov | 03-kolmogorov.tex | Axiomatizing Chance — Kolmogorov and the hidden parameter | Axiomatization |
| 4 | matroid | 04-matroid.tex | Why Greed Works — Matroids and the exchange axiom | Axiomatization |
| | | | **Part II. Changing the Battlefield** | |
| 5 | lpduality | 05-lpduality.tex | All Minimax Theorems Are One — Linear programming duality | Duality |
| 6 | regret | 06-regret.tex | The Dynamics Behind Duality — No-regret learning | Replace existence by dynamics |
| 7 | kneser | 07-kneser.tex | Topology for Combinatorics — Kneser's conjecture and the Borsuk–Ulam theorem | Transfer |
| 8 | fourier | 08-fourier.tex | Change of Basis — Fourier analysis on the Boolean cube | Diagonalize the symmetry |
| 9 | dimension | 09-dimension.tex | Combinatorics as Dimension — From Oddtown to cap sets | Linearization |
| 10 | transport | 10-transport.tex | Inequalities as Maps — Optimal transport | Geometrization |
| | | | **Part III. Finding the Right Measure** | |
| 11 | vc | 11-vc.tex | From Finite to Infinite — The Vapnik–Chervonenkis dimension | The right invariant |
| 12 | littlestone | 12-littlestone.tex | The Adversary's Best Tree — Littlestone dimension, online learning, and privacy | The right invariant |
| 13 | lecam | 13-lecam.tex | Every Regular Problem Is Locally Gaussian — Le Cam's theory of experiments | Localize to a limit object |
| 14 | conformal | 14-conformal.tex | Symmetry for Free — From permutation tests to conformal prediction | Exploit a free symmetry |
| | | | **Part IV. From Existence to Construction** | |
| 15 | compression | 15-compression.tex | If It Failed, You Could Compress Randomness — Entropy compression and the algorithmic local lemma | Encoding |
| 16 | sos | 16-sos.tex | Proofs Are Algorithms — The sum-of-squares method | Proofs to algorithms |
| 17 | pseudorandom | 17-pseudorandom.tex | Four Objects Are One — Expanders, extractors, codes, and samplers | Unification |
| – | epilogue | 18-epilogue.tex | A Grammar of Moves — When the sea rises, and when it cannot | — |

Chapter labels are `ch:<key>`; `\cref{ch:<key>}` prints "Chapter N". Each chapter starts with
`\casechapter{Title}{Subtitle}{Move}{ch:key}` using exactly the strings in the table.

Cross-chapter threads worth pointing out (with `\cref`), when natural:
- Completion: Lebesgue (1) ↔ graph limits (2) ↔ pseudo-distributions in SoS (16) ↔ Prokhorov/OT (10).
- Duality: LP duality (5) → no-regret dynamics (6) → matroid polytope (4) → Kantorovich duality (10).
- Compactness/regularity: graph limits (2) ↔ stable regularity via Littlestone (12).
- Compression/encoding: entropy compression (15) ↔ reconstruction arguments for extractors (17) ↔
  sample compression (11).
- Invariants: VC (11) ↔ Littlestone (12) ↔ dimension arguments (9) ↔ Cover's counting function.
- Symmetry: Fourier (8) ↔ exchangeability (14) ↔ symmetrization in VC (11).
- Lower bounds via minimax: Yao's principle (5) ↔ local minimax (13).

---

## Chapter specifications

Facts marked **[verify]** must be checked with a web search before they go into the text.

### Chapter 1 — Completing the Integral (key `lebesgue`) — Move: Completion

**Before.** The Riemann integral and three leaks.
(a) *Limits escape.* Enumerate the rationals of [0,1]; indicators of finite sets are Riemann integrable,
their pointwise limit (Dirichlet's function) is not. Interchanging limit and integral needs uniform
convergence. Arzelà's bounded convergence theorem for Riemann integrals (1885): if Riemann-integrable
f_n are uniformly bounded and converge pointwise to a Riemann-integrable f, then ∫f_n → ∫f — true, but
every elementary proof is long; see Luxemburg, Amer. Math. Monthly 1971 [verify]. Show the shape of an
elementary proof in an aside (why it is hard: the bad sets are not intervals).
(b) *Spaces are incomplete.* R[a,b] with the L¹ (or L²) seminorm is not complete: indicators of
step-function approximations of a fat Cantor set form a Cauchy sequence with no Riemann-integrable limit.
Consequence: Riesz–Fischer cannot be stated for Riemann functions (square-summable Fourier coefficients
need not come from a Riemann-integrable function).
(c) *The fundamental theorem of calculus fails.* Volterra (1881): a differentiable function with bounded
derivative that is not Riemann integrable [verify date/details].

**The move.** Measure first, then integrate by partitioning the *range* (Lebesgue's own analogy of
counting money by sorting coins — paraphrase, attribute). Equivalent view (Daniell, also "L¹ is the
completion of continuous functions in the L¹ norm"): the Lebesgue integral is the unique continuous
extension of the elementary integral to the completion. Emphasize: the move is completion; the limit
theorems become continuity statements.

**After.** Full proofs: monotone convergence theorem (from the definition), Fatou's lemma, dominated
convergence. Arzelà's theorem as a two-line corollary of DCT. Completeness of L^p (Riesz–Fischer) via a
rapidly Cauchy subsequence + monotone convergence. Fourier series in L²: orthonormal basis ⇒ Parseval
and Riesz–Fischer for coefficients (cite density of trigonometric polynomials). FTC for derivatives
bounded everywhere: if F is differentiable on [a,b] with |F'| ≤ M, then F' is Lebesgue integrable and
∫ₐᵇ F' = F(b) − F(a) — prove via difference quotients + DCT (extend F past b linearly).

**Costs and limits.** Difficulty went into constructing Lebesgue measure (Carathéodory extension) and
into measurability. What did not dissolve: non-measurable sets (Vitali; needs the axiom of choice;
Solovay's model shows choice is needed [verify statement]); conditionally convergent improper integrals
such as ∫ sin x / x are not Lebesgue integrable; not every derivative is Lebesgue integrable — the
Henstock–Kurzweil integral integrates every derivative (so "complete in which norm?" is a choice). Other
seas: Daniell integral, Bochner integral, distributions.

**Move distilled.** "When limits keep escaping your space, put the limits into the space." Triggers:
theorems that need uniform convergence or ad hoc conditions to swap limits; natural Cauchy sequences
without limits; existence proofs that need compactness that fails. Recipe: find the norm/topology in
which your operations are continuous; complete; check old objects are dense; prove by density +
continuity. Anti-pattern: completing in a norm in which the operations you care about are not continuous.

**Exercises (ideas).** Consolidation: Arzelà from DCT; non-completeness of R[a,b]; Fatou's lemma
sharpness examples (moving bump); Riesz–Fischer in ℓ²; FTC example where F' unbounded. Transfer:
reals as completion of rationals (which theorems of calculus become free); Sobolev space H¹ as a
completion and the Dirichlet principle (existence of minimizers via Riesz representation); weak
convergence of probability measures and Prokhorov (existence of optimal couplings, link to
\cref{ch:transport}); completing the space of graphs (\cref{ch:graphon}). Research: infinite-width
limits of neural networks as measures over neurons (mean-field) — what is the right completion and in
which topology is training continuous?

**References (core).** Lebesgue 1902 (thesis "Intégrale, longueur, aire", Annali di Matematica);
Riesz 1907; Fischer 1907; Arzelà 1885; Volterra 1881; Luxemburg 1971; Daniell 1918; Bressoud, *A Radical
Approach to Lebesgue's Theory of Integration* (2008); Hawkins, *Lebesgue's Theory of Integration: Its
Origins and Development* (1970/1975); Folland *Real Analysis*; Bartle *A Modern Theory of Integration*
(Henstock–Kurzweil).

### Chapter 2 — Completing Graphs (key `graphon`) — Move: Completion

**Before.** Extremal and dense graph theory with ε's: Mantel/Turán; Goodman's bound (1959): triangle
density ≥ p(2p−1) at edge density p; quasirandomness (Chung–Graham–Wilson 1989) as a list of equivalent
properties of *sequences* of graphs; Szemerédi's regularity lemma (state it; tower-type bounds; Gowers
1997 showed towers are necessary); triangle removal lemma ⇒ Roth's theorem (state). The pain: optimizers
of extremal problems are sequences, not objects (minimizing triangles at density 1/2: quasirandom
sequences, no finite optimizer); proofs full of "for n sufficiently large".

**The move.** Lovász–Szegedy (2006): graphons W:[0,1]²→[0,1] symmetric measurable; homomorphism
densities t(F,W) = ∫ ∏_{ij∈E(F)} W(x_i,x_j) dx; a graph G is the step graphon W_G; a sequence converges
iff all t(F,G_n) converge; limits exist and are graphons; cut norm and cut distance δ_□; the space of
graphons modulo measure-preserving relabelling is compact (Lovász–Szegedy 2007) [verify citations].
Stooping low: a graph is just a function on the unit square.

**After.** Full proofs: (i) Goodman's bound in graphon language in three lines:
∫W(x,z)W(y,z)dz ≥ d(x)+d(y)−1 since ab ≥ a+b−1 on [0,1]; hence t(K₃,W) ≥ 2∫d² − p ≥ 2p² − p.
(ii) t(C₄,W) ≥ t(K₂,W)⁴ via Cauchy–Schwarz twice; and quasirandomness: t(C₄,W)=p⁴ with t(K₂,W)=p
forces W ≡ p a.e. (spectral proof: T_W Hilbert–Schmidt, t(C₄)=Σλᵢ⁴ ≥ λ₁⁴ ≥ p⁴; equality analysis).
State and explain the Chung–Graham–Wilson equivalences as "the limit object is the constant graphon".
(iii) Weak regularity (Frieze–Kannan 1999) as L² energy increment: projection onto a partition,
≤ 1/ε² refinements; full proof. (iv) Counting lemma |t(F,U) − t(F,W)| ≤ e(F)·‖U−W‖_□ (prove).
(v) Compactness theorem (statement + proof sketch via weak regularity and martingale convergence) and
consequences: existence of extremal graphons; flag algebras (Razborov 2007) turn extremal problems into
semidefinite programs — state Razborov's 2008 triangle-density result without proof [verify].
Optionally: exchangeable random graphs and Aldous–Hoover ↔ graphons (Diaconis–Janson 2008).

**Costs and limits.** Difficulty: measure theory + compactness; compactness proofs are ineffective (no
bounds), whereas the combinatorial regularity lemma gives (tower) bounds; Fox's 2011 improvement of the
removal lemma's bounds needed combinatorics [verify]. Sparse graphs converge to 0 — sparse limit theories
(Benjamini–Schramm local limits; L^p graphons of Borgs–Chayes–Cohn–Zhao) are less complete. Sidorenko's
conjecture stays open despite its clean integral form.

**Exercises (ideas).** Consolidation: Goodman for finite graphs via W_G; t(F,p); K₂ and C₄ densities
of a step graphon; energy increment lemma; counting lemma for triangles. Transfer: permutons (limits of
permutations as probability measures on [0,1]² with uniform marginals; compactness via Prokhorov);
limits of hypergraphs (what goes wrong); Research: limits of neural networks / matrices (mean-field
measures over neurons) — what plays the role of cut distance?

**References.** Lovász–Szegedy 2006 (JCTB); Borgs–Chayes–Lovász–Sós–Vesztergombi 2008 (Adv. Math.);
Lovász 2012 *Large Networks and Graph Limits*; Lovász–Szegedy 2007 (GAFA, "Szemerédi's lemma for the
analyst"); Frieze–Kannan 1999; Goodman 1959; Chung–Graham–Wilson 1989; Szemerédi 1978; Gowers 1997;
Razborov 2007, 2008; Diaconis–Janson 2008; Fox 2011; Zhao, *Graph Theory and Additive Combinatorics*
(2023).

### Chapter 3 — Axiomatizing Chance (key `kolmogorov`) — Move: Axiomatization

**Before.** Bertrand's paradox (1889): three "random chords", three answers 1/2, 1/3, 1/4 (compute all
three). Infinite coin tossing: what is "the probability that the frequency of heads tends to 1/2"?
Borel's 1909 strong law for normal numbers relied on notions of infinitely many independent trials not
yet founded [verify how to phrase]. The Borel–Kolmogorov paradox: a uniform point on the sphere
conditioned on a great circle is uniform (equator) or has density ∝ cos(latitude) (meridian). Hilbert's
sixth problem (1900) asked for axioms for probability [verify wording].

**The move.** Kolmogorov 1933 (*Grundbegriffe*): probability space (Ω, F, P); random variables are
measurable functions; expectation is the Lebesgue integral (\cref{ch:lebesgue}); independence is product
measure; infinite products exist (Kolmogorov extension); conditional expectation given a σ-algebra
(Radon–Nikodym / L² projection). The hidden parameter: every paradox above hides an unstated choice
(a measure, a σ-algebra, a limiting procedure); the axioms force it to be declared.

**After.** Full proofs: strong law for bounded i.i.d. variables via fourth moment + Borel–Cantelli;
Borel's normal numbers theorem as a corollary; Kolmogorov's 0–1 law (tail σ-algebra independent of
itself); conditional expectation as orthogonal projection in L² (existence and the tower property);
dissolution of Borel–Kolmogorov: compute both conditional densities as conditional expectations given
different sub-σ-algebras (latitude vs longitude), showing the answers are both correct for different
questions. Bertrand dissolved as underspecification (mention Jaynes's invariance argument as a different
move).

**Costs and limits.** Difficulty went into measure theory and extension theorems. Not dissolved: which
model is right (Bertrand is underspecified, not solved); interpretation (frequency vs belief);
conditioning on null events needs regular conditional probabilities, which exist on Polish spaces but
not in general [verify]; de Finetti's finitely additive alternative; game-theoretic probability
(Shafer–Vovk) as another sea in which many limit theorems hold without measure.

**Exercises (ideas).** Consolidation: three Bertrand answers; meridian density computation; SLLN via
4th moment; 0–1 law examples; tower property. Transfer: Shannon entropy from axioms (Faddeev's theorem —
prove the uniform case: H(1/n,…,1/n) = c log n); differential privacy as an axiomatization that settled
the anonymization debate (Dinur–Nissim 2003 reconstruction; Dwork–McSherry–Nissim–Smith 2006);
axiomatizing "random sequence" (Martin-Löf); Research: "flat minima" in deep learning are not
reparametrization invariant (Dinh et al. 2017) — a paradox signalling a hidden parameter (the metric);
propose an axiomatization.

**References.** Kolmogorov 1933 (and English translation 1950/1956); Bertrand 1889; Borel 1909; Shafer
& Vovk 2006 "The sources of Kolmogorov's Grundbegriffe" (Statistical Science); Jaynes 1973 "The
well-posed problem"; Williams *Probability with Martingales* (1991); Durrett; Faddeev 1956; Dwork et al.
2006; Dinur–Nissim 2003; Dinh et al. 2017; Shafer–Vovk 2019 *Game-Theoretic Foundations for Probability
and Finance*.

### Chapter 4 — Why Greed Works (key `matroid`) — Move: Axiomatization

**Before.** Three greedy algorithms with three exchange-style proofs: Kruskal (1956) minimum spanning
tree (cut property); maximum-weight linearly independent set of vectors; scheduling unit-time jobs with
deadlines and profits. Show one proof in detail and note the shared lemma.

**The move.** Whitney (1935) abstracted linear independence; the exchange axiom (I3). Rado (1957),
Gale (1968), Edmonds (1971): greedy and matroids.

**After.** Full proofs: Rado–Edmonds theorem — for a nonempty hereditary family, the greedy algorithm
returns a maximum-weight independent set for every nonnegative weight function iff the family is a
matroid (both directions). Corollaries: graphic, linear, and the scheduling (transversal) matroid —
verify exchange for each. Matroid polytope (Edmonds 1970/71): greedy constructs a dual solution, proving
LP integrality (link \cref{ch:lpduality}). Matroid intersection theorem (state; Edmonds 1970) with König
as a corollary (two partition matroids). Rank function submodular; greedy for monotone submodular
maximization under a cardinality constraint gives 1 − 1/e (Nemhauser–Wolsey–Fisher 1978) — full proof:
the axiom weakened, the guarantee weakened.

**Costs and limits.** Difficulty moved into verifying the exchange axiom for each new structure.
Not dissolved: intersection of three matroids is NP-hard (3-dimensional matching); greedy works for
interval scheduling and Huffman coding without a matroid (greedoids, Korte–Lovász); greedy for
non-monotone submodular functions needs other ideas.

**Exercises (ideas).** Consolidation: verify graphic matroid axioms; uniform/partition matroids; rank
submodularity; greedy fails on a non-matroid (example); scheduling via matroid. Transfer: Steinitz exchange
⇒ dimension is well defined; derive Hall's theorem from matroid intersection (or König); weak
submodularity / submodularity ratio for forward stepwise selection (Das–Kempe 2011); Research: when
does greedy token/feature selection provably work in ML pipelines?

**References.** Whitney 1935; Rado 1957; Gale 1968; Edmonds 1970, 1971; Kruskal 1956; Nemhauser,
Wolsey, Fisher 1978; Korte–Lovász 1981 (greedoids) [verify]; Oxley *Matroid Theory*; Schrijver
*Combinatorial Optimization* (2003); Das–Kempe 2011.

### Chapter 5 — All Minimax Theorems Are One (key `lpduality`) — Move: Duality

**Before.** König (1931): bipartite max matching = min vertex cover; Menger (1927); max-flow min-cut
(Ford–Fulkerson 1956; Elias–Feinstein–Shannon 1956); Dilworth (1950); Birkhoff–von Neumann (1946);
von Neumann's minimax theorem (1928, original proof topological). Show one combinatorial proof
(augmenting paths for König) and the zoo.

**The move.** LP duality (via Farkas's lemma / separating hyperplanes) + integrality (total
unimodularity, Hoffman–Kruskal 1956). A min–max theorem is LP duality of an integral pair.

**After.** Full proofs: Farkas's lemma from the separating hyperplane theorem; strong LP duality from
Farkas; incidence matrices of bipartite graphs and network matrices are TU (prove bipartite case);
König, max-flow min-cut (integral), Birkhoff–von Neumann in a few lines each; Dilworth via König;
von Neumann's minimax theorem from LP duality. New reach: Yao's principle (randomized lower bounds =
distributional lower bounds) with one application; boosting is possible iff weak learnability, via
minimax (weak learner for every distribution ⇒ a mixture of hypotheses with margin on every point ⇒
majority vote is correct; Freund–Schapire 1996 view); Impagliazzo's hardcore lemma via minimax (Nisan's
proof) — statement and proof sketch.

**Costs and limits.** Difficulty: convexity (separation theorem) and integrality. Not dissolved:
non-bipartite matching needs Edmonds's odd-set constraints (1965); duality gives certificates, not
algorithms (ellipsoid 1979 / interior point 1984 do; \cref{ch:regret} gives dynamics); nonconvex problems
have duality gaps (max-cut SDP 0.878).

**Exercises (ideas).** Consolidation: Hall from König; Birkhoff–von Neumann via TU; Farkas variants;
Dilworth from König; weak duality for hard-margin SVM and support vectors as dual variables. Transfer:
Kantorovich duality for discrete optimal transport (\cref{ch:transport}); Yao's principle for searching an
unsorted array; LP duality for the value of a two-player game with constraints; Research: regularity
lemmas and hardness amplification as min–max statements (Trevisan–Tulsiani–Vadhan 2009).

**References.** von Neumann 1928; König 1931; Menger 1927; Ford–Fulkerson 1956; Dilworth 1950;
Birkhoff 1946; Hoffman–Kruskal 1956; Gale–Kuhn–Tucker 1951; Farkas 1902; Yao 1977; Freund–Schapire 1996
(COLT "Game theory, on-line prediction and boosting") [verify]; Impagliazzo 1995; Schrijver *Theory of
Linear and Integer Programming* (1986); Matoušek–Gärtner *Understanding and Using Linear Programming*.

### Chapter 6 — The Dynamics Behind Duality (key `regret`) — Move: Replace existence by dynamics

**Before.** Approximately solving games and LPs; boosting designed by hand (Schapire 1990 recursive
majority; Freund 1995 boost-by-majority); Plotkin–Shmoys–Tardos packing/covering; each with its own
analysis. Von Neumann's theorem as pure existence.

**The move.** Prediction with expert advice: Weighted Majority (Littlestone–Warmuth 1994), Hedge
(Freund–Schapire 1997). If two players run no-regret algorithms against each other, their time-averaged
strategies form an approximate equilibrium. Multiplicative weights as a meta-algorithm (Arora–Hazan–Kale
2012).

**After.** Full proofs: Hedge regret ≤ √(2T ln n) (or (ln n)/η + ηT/8 form) via the potential
Φ_t = Σ w_i; von Neumann's minimax theorem from no-regret (both players run Hedge); AdaBoost as Hedge on
the training examples with training error ≤ exp(−2γ²T); approximate LP feasibility with an oracle
(width-dependent bound); online-to-batch conversion: SGD/online gradient descent regret ⇒ excess risk
bound for convex Lipschitz losses (Cesa-Bianchi–Conconi–Gentile 2004). State online gradient descent
regret O(GD√T) with proof.

**Costs and limits.** Difficulty: in the regret analysis (potential functions) and in the 1/ε² iteration
count (width dependence) — slow for high accuracy versus interior-point methods. Not dissolved: last-iterate
convergence (simultaneous gradient play cycles on bilinear games; optimistic methods fix it — Daskalakis
et al. 2018) [verify]; nonconvex games; swap regret and correlated equilibria need more.

**Exercises (ideas).** Consolidation: Hedge regret with general η; AdaBoost bound; online-to-batch;
regret lower bound Ω(√(T log n)) idea. Transfer: Blackwell approachability via no-regret (Abernethy–
Bartlett–Hazan 2011) [verify]; calibration via approachability (Foster–Vohra 1998); max-flow
approximation via MW (Christiano et al. 2011) [verify]; set cover / covering LPs via MW. Research: GAN
training as a game — show gradient play diverges on min_x max_y xy and optimistic GD converges.

**References.** Littlestone–Warmuth 1994; Freund–Schapire 1997 (JCSS), 1999 (Games and Economic Behavior)
[verify]; Arora–Hazan–Kale 2012 (Theory of Computing); Plotkin–Shmoys–Tardos 1995; Cesa-Bianchi–Lugosi
2006 *Prediction, Learning, and Games*; Hazan *Introduction to Online Convex Optimization*;
Cesa-Bianchi–Conconi–Gentile 2004; Schapire 1990; Blackwell 1956; Daskalakis et al. 2018.

### Chapter 7 — Topology for Combinatorics (key `kneser`) — Move: Transfer to a new setting

**Before.** Kneser graph KG(n,k) (k-subsets of [n], adjacent when disjoint). Kneser (1955) conjectured
χ = n − 2k + 2. Upper bound coloring (prove). Lower bound open for 23 years; Petersen graph.

**The move.** Lovász (1978): a topological lower bound via the Borsuk–Ulam theorem; Bárány's short proof
(1978); Greene's proof (2002, Amer. Math. Monthly) [verify].

**After.** Full proofs: Lyusternik–Shnirelmann form of Borsuk–Ulam (state Borsuk–Ulam, derive the cover
version); Greene's proof of Kneser's conjecture (points on S^d in general position, d = n − 2k + 1;
open hemispheres; antipodal pair; general position contradiction). Necklace splitting for two thieves
(Goldberg–West 1985; Alon–West 1986) via Borsuk–Ulam — proof. Ham sandwich theorem. Matoušek's (2004)
combinatorial proof via Tucker's lemma: once topology found the reason, combinatorics followed (state).
Generalizations: Alon–Frankl–Lovász (1986) Kneser hypergraphs (state) [verify].

**Costs and limits.** Difficulty: the Borsuk–Ulam theorem (algebraic topology or Tucker's lemma). Not
dissolved: existence only — necklace splitting is PPA-complete (Filos-Ratsikas–Goldberg 2018/2019)
[verify]; topological lower bounds are weak for most graphs (e.g., random graphs).

**Exercises (ideas).** Consolidation: Petersen graph; check the upper-bound coloring; derive
Lyusternik–Shnirelmann from Borsuk–Ulam; Schrijver graphs. Transfer: ham sandwich in the plane; Sperner ⇒
Brouwer and fair rent division (Su 1999); discrete ham sandwich for point sets. Research: topological
obstructions in learning theory — replicability lower bounds via Borsuk–Ulam / Sperner–KKM (Chase,
Chornomaz, Moran, Yehudayoff 2024 "Local Borsuk–Ulam, stability, and replicability"; Dixon, Pavan, Vander
Woude, Vinodchandran 2023) [verify].

**References.** Kneser 1955; Lovász 1978; Bárány 1978; Greene 2002; Matoušek 2003 *Using the Borsuk–Ulam
Theorem*; Matoušek 2004 (Combinatorica) [verify]; Alon 1987 (necklace); Goldberg–West 1985; Alon–West
1986; Alon–Frankl–Lovász 1986; Filos-Ratsikas–Goldberg 2018 (STOC) [verify]; Su 1999; Chase et al. 2024.

### Chapter 8 — Change of Basis (key `fourier`) — Move: Diagonalize the symmetry

**Before.** Linearity testing: Blum–Luby–Rubinfeld (1990/1993) and its combinatorial self-correction
analysis with weak constants; Arrow's impossibility theorem (1951) and its combinatorial proof; the
Condorcet paradox (Guilbaud 1952: ≈ 8.77% under impartial culture for majority) [verify].

**The move.** The Fourier–Walsh expansion f = Σ_S f̂(S)χ_S on {−1,1}^n; Parseval; the noise operator
T_ρ is diagonal; probabilities of tests become weighted sums of squared (or cubed) coefficients.

**After.** Full proofs: Parseval/Plancherel; BLR analysis (Bellare–Coppersmith–Håstad–Kiwi–Sudan 1996):
Pr[accept] = 1/2 + 1/2 Σ f̂(S)³ ≤ 1/2 + 1/2 max f̂(S), and dist(f, χ_S) = (1 − f̂(S))/2; Kalai's Fourier
proof of Arrow's theorem (2002) for three candidates when one function f is used for all three pairwise
contests: Pr[cyclic] = 1/4 + 3/4·Σ_S (−1/3)^{|S|} f̂(S)²; if never cyclic, all Fourier weight sits on
level 1, so f = ±χ_i, and unanimity gives a dictator (prove that a Boolean function with all weight on
level 1 is ±x_i); Guilbaud's number for majority via Stab_ρ(Maj_n) → (2/π) arcsin ρ (state Sheppard's
formula); Maj₃ gives exactly 1/18 (compute). Poincaré inequality Var f ≤ Σ Inf_i f. State KKL (1988)
and explain why its proof (hypercontractivity) lives only in the Fourier world; state quantitative Arrow
(Mossel 2012) [verify].

**Costs and limits.** Difficulty: in setting up the Fourier dictionary and, for KKL, hypercontractivity
(Bonami–Beckner). Not dissolved: non-product measures, slices, non-abelian settings need more; general
Arrow (different functions per pair) needs extra work (Kalai handles it — state); the method says little
about computational questions beyond structure.

**Exercises (ideas).** Consolidation: Fourier expansions of Maj₃, AND, Tribes; Parseval; Poincaré;
exact BLR constants; Pr[cyclic] for Maj₃ = 1/18. Transfer: Meshulam's cap-set bound O(3^n/n) via Fourier
on F₃^n (link \cref{ch:dimension}); Goldreich–Levin / Kushilevitz–Mansour: learning = finding heavy
Fourier coefficients; Linial–Mansour–Nisan: AC⁰ has Fourier concentration (state) ⇒ quasi-polynomial
learning; spectral graph theory of the hypercube as a Fourier computation. Research: what is the right
basis for neural networks on Boolean inputs (staircase / leap complexity, Abbe et al.) [verify].

**References.** Blum–Luby–Rubinfeld 1993 (JCSS); Bellare et al. 1996 (IEEE Trans. Inf. Theory);
Kahn–Kalai–Linial 1988; Kalai 2002 (Adv. Appl. Math.); Mossel 2012; Arrow 1951; Guilbaud 1952;
O'Donnell 2014 *Analysis of Boolean Functions*; Linial–Mansour–Nisan 1993; Kushilevitz–Mansour 1993;
Goldreich–Levin 1989.

### Chapter 9 — Combinatorics as Dimension (key `dimension`) — Move: Linearization

**Before.** Oddtown (odd sets with even pairwise intersections; Berlekamp 1969); Fisher's inequality
(1940); Borsuk's conjecture (1933) believed for 60 years; finite-field Kakeya (Wolff 1999) with
combinatorial bounds of the form q^{cn}, c<1; cap sets in F₃^n: Meshulam 1995 O(3^n/n) by Fourier
density increment, Bateman–Katz 2012 O(3^n/n^{1+ε}); lower bounds ≈ 2.2^n (Edel 2004) [verify].

**The move.** Map objects to vectors or polynomials; independence (or vanishing) forces count ≤
dimension.

**After.** Full proofs: Oddtown (≤ n) via F₂ independence; Fisher's inequality via rank of MM^T;
a Frankl–Wilson-type theorem with the polynomial method (state precisely; prove the version you use);
Kahn–Kalai (1993) disproof of Borsuk's conjecture — present the construction (vectors from ±1
tensor squares, diameter attained exactly at intersections of size p, Frankl–Wilson bound on families
avoiding that intersection) and the resulting count; later dimension improvements down to 64
(Jenrich–Brouwer 2014) [verify]. Dvir (2008/2009) finite-field Kakeya: |K| ≥ C(q+n−1, n) — full proof
via a vanishing polynomial and its top homogeneous part. Cap set bound (Croot–Lev–Pach 2017;
Ellenberg–Gijswijt 2017): |A| ≤ 3N, N = #monomials with exponents in {0,1,2} and degree ≤ 2n/3 —
full proof via Tao's slice-rank formulation (prove the diagonal slice-rank lemma), and estimate
N ≤ (2.756)^n.

**Costs and limits.** Difficulty: finding the right vector space; the method needs algebraic structure.
Not dissolved: Euclidean Kakeya (dimension 3 announced by Wang–Zahl 2025 with very different tools)
[verify]; the gap between 2.2^n and 2.756^n for cap sets; slice-rank limitations (Blasiak et al. 2017
on the group-theoretic approach to matrix multiplication) [verify].

**Exercises (ideas).** Consolidation: Eventown ≤ 2^{⌊n/2⌋}; Schwartz–Zippel; Fisher; bound N
numerically; Ray-Chaudhuri–Wilson. Transfer: two-distance sets in R^d have ≤ C(d+2,2) points
(Larman–Rogers–Seidel / Blokhuis); joints problem (Guth–Katz 2010) [verify]; Cauchy–Davenport via the
Combinatorial Nullstellensatz (Alon 1999); Cover's function-counting theorem (1965) for linear
dichotomies (link \cref{ch:vc}). Research: dimension arguments for memorization capacity of networks.

**References.** Babai–Frankl *Linear Algebra Methods in Combinatorics*; Berlekamp 1969; Fisher 1940;
Frankl–Wilson 1981; Kahn–Kalai 1993 (Bull. AMS); Dvir 2009 (JAMS); Croot–Lev–Pach 2017 (Annals);
Ellenberg–Gijswijt 2017 (Annals); Tao 2016 blog; Guth *Polynomial Methods in Combinatorics* (2016);
Meshulam 1995; Bateman–Katz 2012; Edel 2004; Alon 1999; Jenrich–Brouwer 2014.

### Chapter 10 — Inequalities as Maps (key `transport`) — Move: Geometrization

**Before.** Brunn–Minkowski (boxes, then unions of boxes by induction — Hadwiger–Ohmann, then
approximation); the isoperimetric inequality and Steiner symmetrization (with Weierstrass's objection
about existence of a minimizer); Prékopa–Leindler; Gross's Gaussian log-Sobolev inequality (1975);
Talagrand's transportation inequality (1996).

**The move.** Transport one measure to another: Knothe (1957) triangular maps; Brenier's theorem (1987/
1991): the optimal map for quadratic cost is the gradient of a convex function; McCann's displacement
interpolation and displacement convexity (1995/1997); Otto calculus (2001) and the JKO scheme (1998):
Fokker–Planck as the gradient flow of free energy in Wasserstein space.

**After.** Full proofs: 1-D optimal transport is the monotone rearrangement (quantile coupling) for
convex costs; Brunn–Minkowski via the Knothe map (fully elementary: triangular Jacobian + AM–GM) for
nice sets; isoperimetric inequality from Brunn–Minkowski; Prékopa–Leindler in one dimension via
transport. Talagrand's T₂ for the standard Gaussian, W₂²(μ,γ) ≤ 2H(μ|γ), via the Brenier map and
convexity (Cordero-Erausquin 2002) — full proof in dimension one, sketch in general. HWI (Otto–Villani
2000) ⇒ log-Sobolev ⇒ T₂ (state; derive LSI from HWI by AM–GM). Langevin dynamics as Wasserstein
gradient flow of KL; LSI ⇒ exponential decay of KL (formal derivation, clearly labelled); link to
sampling and to mean-field two-layer neural networks (Chizat–Bach 2018; Mei–Montanari–Nguyen 2018).
Kantorovich duality (discrete case from LP duality, \cref{ch:lpduality}).

**Costs and limits.** Difficulty: existence and regularity of optimal maps (Caffarelli's regularity
theory), and the rigorous Otto calculus (Ambrosio–Gigli–Savaré). Not dissolved: high-dimensional
statistical estimation of OT (curse of dimensionality, W₁ rates n^{-1/d}); entropic regularization
(Cuturi 2013) changes the object; transport proofs often need convexity or log-concavity.

**Exercises (ideas).** Consolidation: monotone rearrangement optimality; Knothe map Jacobian;
isoperimetry from BM; Prékopa–Leindler in 1-D; Gaussian T₂ in 1-D. Transfer: Marton's argument —
T₂ ⇒ Gaussian concentration of Lipschitz functions; Kantorovich–Rubinstein duality for W₁ and Wasserstein
GANs; continuity equation for mean-field two-layer networks. Research: transport-information inequalities
for generalization bounds.

**References.** Monge 1781; Kantorovich 1942; Knothe 1957; Brenier 1991 (CPAM); McCann 1995, 1997;
Jordan–Kinderlehrer–Otto 1998; Otto 2001; Otto–Villani 2000; Cordero-Erausquin 2002; Cordero-Erausquin–
Nazaret–Villani 2004; Gross 1975; Talagrand 1996; Bakry–Émery 1985; Villani 2003 *Topics in Optimal
Transportation*, 2009 *Optimal Transport: Old and New*; Santambrogio 2015; Ambrosio–Gigli–Savaré 2008;
Chewi *Log-Concave Sampling* (draft); Chizat–Bach 2018; Mei–Montanari–Nguyen 2018; Cuturi 2013.

### Chapter 11 — From Finite to Infinite (key `vc`) — Move: The right invariant

**Before.** Finite classes: union bound + Hoeffding (Occam; Blumer et al. 1987); Glivenko–Cantelli
(1933) with its ad hoc grid proof; intervals and half-planes handled case by case.

**The move.** Vapnik–Chervonenkis (1968/1971): growth function, shattering, VC dimension; symmetrization
with a ghost sample reduces a supremum over an infinite class to finitely many behaviours on 2m points;
Sauer–Shelah–Perles lemma.

**After.** Full proofs: Sauer–Shelah (shifting or induction); uniform convergence via symmetrization and
random swaps, Pr[sup_h |L_S(h) − L_D(h)| > ε] ≤ 4Π_H(2m)e^{−mε²/8}; the fundamental theorem of
statistical learning for binary classification (finite VC ⇔ uniform convergence ⇔ ERM learns ⇔ PAC
learnable), with the no-free-lunch lower bound from a shattered set (proof sketch acceptable for the
sharp constants); Glivenko–Cantelli as the VC-dimension-1 case; VC dimension of half-spaces in R^d is
d+1 (Radon's theorem).

**Costs and limits.** Difficulty: Sauer–Shelah and symmetrization. Not dissolved: distribution-free worst
case does not explain deep learning (Zhang et al. 2017; Nagarajan–Kolter 2019); the equivalences fail
beyond binary classification: multiclass with infinitely many labels (Natarajan dimension insufficient;
DS dimension, Brukhim–Carmon–Dinur–Moran–Yehudayoff 2022), stochastic convex optimization (learnable
without uniform convergence, Shalev-Shwartz–Shamir–Srebro–Sridharan 2010), and EMX learnability is
independent of ZFC (Ben-David–Hrubeš–Moran–Shpilka–Yehudayoff 2019).

**Exercises (ideas).** Consolidation: VC of intervals, rectangles, half-spaces; Sauer–Shelah by
shifting; Glivenko–Cantelli rate. Transfer: Rademacher complexity via symmetrization; ε-nets (Haussler–
Welzl 1987); fat-shattering dimension for real-valued classes (Alon–Ben-David–Cesa-Bianchi–Haussler
1997); Cover's counting function. Research: the "combinatorial shadow" of overparameterized networks;
sample compression (Littlestone–Warmuth conjecture; Moran–Yehudayoff 2016 exponential bound) [verify].

**References.** Vapnik–Chervonenkis 1971 (Theory Probab. Appl.); Sauer 1972; Shelah 1972; Blumer,
Ehrenfeucht, Haussler, Warmuth 1989 (JACM); Glivenko 1933; Cantelli 1933; Shalev-Shwartz & Ben-David
2014; Haussler–Welzl 1987; Zhang et al. 2017; Nagarajan–Kolter 2019; Brukhim et al. 2022; Shalev-Shwartz
et al. 2010; Ben-David et al. 2019; Moran–Yehudayoff 2016.

### Chapter 12 — The Adversary's Best Tree (key `littlestone`) — Move: The right invariant

**Before.** Online learning (mistake bounds): the halving algorithm makes ≤ log₂|H| mistakes; thresholds
on [0,1] have VC dimension 1 but no finite mistake bound (adversary's binary search). Private learning
of thresholds over [N] needs ω(1) samples (Bun–Nissim–Stemmer–Vadhan 2015: Ω(log* N) for proper private
learning) [verify].

**The move.** Littlestone (1988): mistake trees, Littlestone dimension, the Standard Optimal Algorithm.

**After.** Full proofs: optimal mistake bound = Ldim (adversary from a shattered tree; SOA decreases Ldim
of the version space on each mistake); VC ≤ Ldim ≤ log₂|H|; thresholds over [N] have Ldim ⌊log₂ N⌋.
Agnostic online learning regret Θ̃(√(Ldim·T)) (Ben-David–Pál–Shalev-Shwartz 2009): statement + proof
sketch (experts built from SOA runs). The hub (statements, careful attribution): private PAC learnable
⇔ finite Ldim (Alon–Livni–Malliaris–Moran 2019; Bun–Livni–Moran 2020); replicability (Impagliazzo–Lei–
Pitassi–Sorrell 2022) equivalent to private learnability (Bun et al. 2023 "Stability is stable") [verify];
Ldim ↔ thresholds (Shelah; Hodges): Ldim ≥ d ⇒ thresholds of size ⌊log₂ d⌋ — prove the easy direction
(thresholds of size 2^d ⇒ Ldim ≥ d), state the other; model theory: finite Ldim ⇔ stability (Shelah's
2-rank; Chase–Freitag 2019) and the Malliaris–Shelah stable regularity lemma (2014) — connect to
\cref{ch:graphon}. Universal learning rates trichotomy (Bousquet–Hanneke–Moran–van Handel–Yehudayoff
2021) [verify].

**Costs and limits.** Difficulty: Ramsey-theoretic arguments behind the equivalences; original bounds
for private learning were tower-type, later polynomial (Ghazi–Golowich–Kumar–Manurangsi 2021) [verify].
Not dissolved: Ldim is infinite for many natural classes (thresholds, half-spaces over R^d), so online and
private learning need margins or other structure; the exact sample complexity of private learning.

**Exercises (ideas).** Consolidation: Ldim of singletons, thresholds over [N]; Ldim ≤ log₂|H|;
VC ≤ Ldim; SOA by hand. Transfer: perceptron mistake bound 1/γ² (Novikoff 1962) as margin replacing Ldim;
Littlestone trees in universal learning rates; stable graphs and half-graphs. Research: find another
stability notion and test whether it falls into the privacy–online equivalence class.

**References.** Littlestone 1988 (Machine Learning); Ben-David–Pál–Shalev-Shwartz 2009 (COLT);
Bun–Nissim–Stemmer–Vadhan 2015 (FOCS); Alon–Livni–Malliaris–Moran 2019 (STOC) / JACM 2022; Bun–Livni–
Moran 2020 (FOCS); Ghazi et al. 2021; Shelah 1978 *Classification Theory*; Hodges 1997; Chase–Freitag 2019;
Malliaris–Shelah 2014 (Trans. AMS); Impagliazzo et al. 2022 (STOC); Bun et al. 2023; Bousquet et al. 2021
(STOC); Novikoff 1962.

### Chapter 13 — Every Regular Problem Is Locally Gaussian (key `lecam`) — Move: Localize to a limit object

**Before.** Fisher's claim that the MLE is asymptotically efficient; Cramér–Rao (1945–46) restricted to
unbiased estimators; Hodges's superefficient estimator (1951, published via Le Cam 1953): T_n = X̄ if
|X̄| ≥ n^{−1/4}, else 0, for N(θ,1), with asymptotic variance 0 at θ = 0.

**The move.** Le Cam (1953–1972) and Hájek (1970, 1972): localize, θ = θ₀ + h/√n; local asymptotic
normality (LAN); the sequence of experiments converges to the Gaussian shift experiment; optimality
questions are solved in the limit and transferred back.

**After.** Full proofs: Hodges's local risk blows up — under θ_n = h/√n, √n(T_n − θ_n) → −h in
probability, so the maximal local risk over |h| ≤ c is ≥ c² eventually; minimax risk of the Gaussian
shift X ~ N(h,1) over |h| ≤ c tends to 1 as c → ∞ (Bayes argument with a uniform prior or normal
prior); LAN for a smooth one-parameter exponential family (full) and statement for models
differentiable in quadratic mean. Statements with proof sketches: Hájek's convolution theorem; the local
asymptotic minimax theorem (Hájek–Le Cam); Le Cam's theorem that superefficiency happens only on a
Lebesgue-null set; Le Cam's third lemma and contiguity (state). Mention semiparametric efficiency in one
paragraph; modern echo in stochastic optimization (Duchi–Ruan 2021) [verify].

**Costs and limits.** Difficulty: contiguity, LAN expansions (DQM). Not dissolved: non-regular models
(uniform(0,θ): rate n, limit experiment not Gaussian — compute it), high-dimensional regimes where p
grows with n, finite-sample questions.

**Exercises (ideas).** Consolidation: Fisher information and LAN for Bernoulli, Poisson; Hodges local
risk; regularity of X̄. Transfer: the limit experiment for Uniform(0,θ); James–Stein (global dominance
in d ≥ 3) versus local optimality; local minimax lower bounds in stochastic convex optimization.
Research: limit experiments for proportional-asymptotics regression.

**References.** Fisher 1922, 1925; Cramér 1946; Rao 1945; Le Cam 1953, 1960, 1972, 1986 *Asymptotic
Methods in Statistical Decision Theory*; Le Cam–Yang 2000; Hájek 1970, 1972; van der Vaart 1998
*Asymptotic Statistics*; van der Vaart 1997 "Superefficiency" (Festschrift for Le Cam) [verify];
Duchi–Ruan 2021 (Ann. Statist.).

### Chapter 14 — Symmetry for Free (key `conformal`) — Move: Exploit a free symmetry

**Before.** Prediction intervals need models (Gaussian linear model interval); black-box predictors have
no finite-sample guarantees; two-sample problems with unknown distributions; Fisher's (1935) and Pitman's
(1937) permutation tests as the early instance of the move.

**The move.** Exchangeability: the rank of a score among n+1 exchangeable scores is uniform (up to
ties). Any statistic computed symmetrically yields a finite-sample-valid procedure.

**After.** Full proofs: rank lemma with ties (coverage ≥ 1 − α with the ⌈(1−α)(n+1)⌉ quantile; ≤ 1 − α
+ 1/(n+1) without ties); permutation test validity; split conformal (Papadopoulos et al. 2002; Lei et al.
2018); full conformal (Vovk–Gammerman–Shafer 2005) with a symmetric fitting algorithm; conformal
classification (adaptive prediction sets, Romano–Sesia–Candès 2020); weighted conformal under covariate
shift (Tibshirani–Barber–Candès–Ramdas 2019); de Finetti's theorem as context; jackknife+ (Barber et al.
2021) coverage ≥ 1 − 2α (state, maybe sketch).

**Costs and limits.** Difficulty: none paid in the proof — paid in assumptions (exchangeability) and in
efficiency (length depends on the model). Not dissolved: conditional coverage is impossible
distribution-free (Vovk 2012; Lei–Wasserman 2014; Barber–Candès–Ramdas–Tibshirani 2021) — state and
explain; time series and shift need other tools (Gibbs–Candès 2021 adaptive conformal; Barber et al.
2023 beyond exchangeability) [verify].

**Exercises (ideas).** Consolidation: coverage upper bound; permutation-test p-value; split conformal on
a toy dataset; ties. Transfer: model-X knockoffs (Barber–Candès 2015; Candès et al. 2018) — the swap
symmetry gives FDR control; conformal risk control (Angelopoulos et al. 2022/2024) [verify]; rank tests
(Wilcoxon) as conformal statements; conformal guarantees for language-model outputs (Quach et al. 2024)
[verify]. Research: what symmetry does an LLM benchmark have for free?

**References.** Fisher 1935 *The Design of Experiments*; Pitman 1937; Vovk–Gammerman–Shafer 2005 (2nd
ed. 2022); Papadopoulos et al. 2002; Lei et al. 2018 (JASA); Romano–Sesia–Candès 2020; Tibshirani et al.
2019; Barber et al. 2021 (Ann. Statist.); Barber et al. 2021 (Information and Inference, limits);
Angelopoulos–Bates 2023 (Foundations and Trends); Angelopoulos–Barber–Bates 2024 *Theoretical
Foundations of Conformal Prediction* [verify]; Barber–Candès 2015; Candès et al. 2018; de Finetti 1937.

### Chapter 15 — If It Failed, You Could Compress Randomness (key `compression`) — Move: Encoding

**Before.** Lovász Local Lemma (Erdős–Lovász 1975; symmetric form ep(d+1) ≤ 1): nonconstructive.
Algorithmic versions with weaker parameters: Beck 1991, Alon 1991, … [verify list].

**The move.** Moser (2009, STOC) — resample violated clauses; analysis by entropy compression: a long
run would let us describe the random bits used in fewer bits than were drawn. Moser–Tardos (2010) general
version.

**After.** Full proof: k-SAT where each clause shares variables with at most d other clauses and d is
below 2^{k−c} for an explicit constant c: the recursive Fix algorithm terminates quickly with high
probability — complete entropy-compression argument with explicit encoding and counting (invariant that
Fix preserves satisfied clauses; reconstruction of the random bits from the log; the incompressibility
bound Pr[compressible by s bits] ≤ 2^{−s}). Statement of Moser–Tardos with expected resamplings ≤
Σ x_i/(1 − x_i) and the witness-tree idea (sketch). Entropy compression beyond LLL: nonrepetitive
sequences from lists of size 4 (Grytczuk–Kozik–Micek 2013) — state and sketch; Kolmogorov
incompressibility method: infinitely many primes (with a quantitative bound) and an average-case lower
bound for comparison sorting.

**Costs and limits.** Difficulty: in designing a lossless encoding (the reconstruction step). Not
dissolved: the general (non-variable) LLL and lopsided versions need other frameworks (Harvey–Vondrák
2015; Achlioptas–Iliopoulos 2016) [verify]; Shearer's bound (1985) is the limit of LLL-type criteria
(Kolipaka–Szegedy 2011 reach it algorithmically) [verify].

**Exercises (ideas).** Consolidation: check the reconstruction step; compute constants; prove the
incompressibility bound. Transfer: sorting lower bound by incompressibility; primes with p_k ≤ k(log k)^{2+ε};
an entropy-compression proof for a coloring problem; Research: an encoding argument for a stochastic
optimization method's convergence (e.g., Schöning's random walk for 3-SAT).

**References.** Erdős–Lovász 1975; Spencer 1977; Beck 1991; Alon 1991; Moser 2009; Moser–Tardos 2010
(JACM); Fortnow 2009 blog; Tao 2009 blog; Grytczuk–Kozik–Micek 2013; Li–Vitányi *An Introduction to
Kolmogorov Complexity and Its Applications*; Shearer 1985; Kolipaka–Szegedy 2011; Harvey–Vondrák 2015;
Achlioptas–Iliopoulos 2016; Alon–Spencer *The Probabilistic Method*.

### Chapter 16 — Proofs Are Algorithms (key `sos`) — Move: Proofs to algorithms

**Before.** Robust mean estimation in high dimension: Tukey median achieves O(ε) error but is hard to
compute; coordinate-wise median and geometric median are efficient with error O(ε√d); efficient
dimension-free algorithms only in 2016 (Diakonikolas–Kamath–Kane–Li–Moitra–Stewart; Lai–Rao–Vempala),
via ad hoc spectral filtering. Tensor decomposition, dictionary learning, clustering mixtures each with
custom algorithms.

**The move.** Pseudo-distributions and sum-of-squares proofs (Shor, Nesterov, Parrilo, Lasserre;
Grigoriev); the proofs-to-algorithms paradigm (Barak–Steurer; Barak–Kelner–Steurer; Hopkins;
Kothari–Steinhardt–Steurer; Hopkins–Li): a low-degree SoS proof of identifiability yields an algorithm.

**After.** Full proofs: definitions of SoS proofs and pseudo-expectations; Cauchy–Schwarz for
pseudo-expectations; duality (degree-d refutation exists iff no pseudo-distribution — state, prove the
easy direction); Goemans–Williamson as degree-2 SoS with hyperplane rounding (0.878); the robust mean
estimation pipeline under bounded covariance: identifiability proof (two distributions with bounded
covariance sharing a 1−ε fraction have means within O(√ε)) written as a low-degree SoS proof, then the
algorithm by rounding μ̂ = Ẽ[μ'] — present carefully (a fully rigorous SoS derivation, or a rigorous
statement with an honest sketch of the SoS certificate). Mention certifiable hypercontractivity and
better rates for Gaussians (state).

**Costs and limits.** Difficulty: finding low-degree certificates; running time n^{O(d)} (spectral
algorithms that emulate SoS, Hopkins–Schramm–Shi–Steurer 2016); bit complexity (O'Donnell 2017;
Raghavendra–Weitz 2017) [verify]. Not dissolved: planted clique — degree-o(log n) SoS fails below
n^{1/2−o(1)} (Barak et al. 2016), marking a boundary; some true statements lack low-degree proofs
(Grigoriev 2001).

**Exercises (ideas).** Consolidation: SoS proof of simple polynomial inequalities; pseudo-expectation
Cauchy–Schwarz; GW rounding constant; robust mean in one dimension (median) vs. high dimension.
Transfer: Jennrich's algorithm and SoS tensor decomposition; robust linear regression; clustering
mixtures via SoS identifiability (state). Research: which identifiability proofs in ML (ICA,
mixtures, sparse dictionary learning) are low-degree?

**References.** Shor 1987; Parrilo 2000; Lasserre 2001; Grigoriev 2001; Goemans–Williamson 1995; Barak–
Steurer 2014 (ICM); Barak–Kelner–Steurer 2015; Hopkins 2018 (thesis); Kothari–Steinhardt–Steurer 2018;
Hopkins–Li 2018; Diakonikolas et al. 2016; Lai–Rao–Vempala 2016; Barak et al. 2016 (planted clique);
Fleming–Kothari–Pitassi 2019; Diakonikolas–Kane 2023 *Algorithmic High-Dimensional Robust Statistics*;
Hopkins–Schramm–Shi–Steurer 2016; O'Donnell 2017; Raghavendra–Weitz 2017.

### Chapter 17 — Four Objects Are One (key `pseudorandom`) — Move: Unification

**Before.** Expander graphs (Pinsker 1973; Margulis 1973; Lubotzky–Phillips–Sarnak 1988); randomness
extractors (Nisan–Zuckerman 1996); list-decodable codes (Sudan 1997; Guruswami–Sudan 1999); averaging
samplers; hardness-based pseudorandom generators (Nisan–Wigderson 1994). Separate definitions,
constructions, communities.

**The move.** Write each object as a function Γ: [N] × [D] → [M] (a bipartite graph with left degree D)
and each property as "for every test of a certain kind, neighbourhoods behave" (Vadhan's unified view;
Trevisan 1999/2001; Zuckerman 1997; Ta-Shma–Zuckerman–Safra; Guruswami–Umans–Vadhan 2007/2009).

**After.** Full proofs: expander mixing lemma; extractors ⇔ averaging samplers (Zuckerman 1997, with
parameter translation); a random function is an extractor with seed length log(n−k) + 2 log(1/ε) + O(1)
(probabilistic method); the leftover hash lemma (Impagliazzo–Levin–Luby 1989) via collision probability.
List-decodable codes as Γ(x,y) = (y, Enc(x)_y): list decoding ⇔ few x have many neighbours in a set
(prove the translation). Trevisan's extractor from the Nisan–Wigderson generator and a list-decodable
code: the reconstruction paradigm (if Ext(x,·) fails a test, x has a short description, so few x are bad
— compression again, \cref{ch:compression}) — proof sketch. Vadhan's parameter table. New reach:
zig-zag product (Reingold–Vadhan–Wigderson 2002) and SL = L (Reingold 2005/2008) — state; GUV
condensers from Parvaresh–Vardy codes — state.

**Costs and limits.** Difficulty: the definitions must be tuned (parameter regimes differ: codes want
large alphabets, expanders constant degree); unification at the level of definitions does not
automatically transfer constructions. Not dissolved: explicit constant-degree lossless expanders needed
separate ideas (Capalbo–Reingold–Vadhan–Wigderson 2002) [verify]; optimal explicit two-source extractors
(Chattopadhyay–Zuckerman 2016 — state as later progress) [verify].

**Exercises (ideas).** Consolidation: mixing lemma; extractor ⇒ sampler parameters; random extractor;
leftover hash lemma; condensers. Transfer: Yao's XOR lemma via the hardcore lemma (\cref{ch:lpduality});
privacy amplification in cryptography as an extractor statement; samplers from expander walks (Chernoff
bound for expander walks, Gillman 1998 — state). Research: subsampling and shuffling amplification in
differential privacy — are they extractor statements?

**References.** Vadhan 2012 *Pseudorandomness* (Foundations and Trends); Nisan–Zuckerman 1996;
Zuckerman 1997 (Random Structures & Algorithms); Trevisan 2001 (JACM); Ta-Shma–Zuckerman–Safra 2006;
Guruswami–Umans–Vadhan 2009 (JACM); Hoory–Linial–Wigderson 2006 (Bull. AMS); Reingold–Vadhan–Wigderson
2002 (Annals); Reingold 2008 (JACM); Alon–Chung 1988; Impagliazzo–Levin–Luby 1989; Nisan–Wigderson 1994;
Sudan 1997; Guruswami–Sudan 1999; Gillman 1998; Capalbo et al. 2002; Chattopadhyay–Zuckerman 2016.
