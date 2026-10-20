import { mulberry32 } from "./math";

// Fragments of what the five books are made of, plus the unresolved references
// ("??") that LaTeX prints for something that is cited but does not exist yet.
const WORDS = [
  "dissolution test", "Lebesgue integral", "the rising sea", "completion", "matroid", "exchange axiom",
  "LP duality", "no-regret", "Kneser graph", "Borsuk–Ulam", "Fourier on the cube", "Oddtown", "cap sets",
  "optimal transport", "VC dimension", "Littlestone", "Le Cam", "conformal prediction", "entropy compression",
  "sum-of-squares", "expanders", "extractors", "double descent", "deep linear networks", "implicit bias",
  "neural tangent kernel", "information exponent", "committee machine", "sparse parities", "grokking",
  "superposition", "neural collapse", "in-context learning", "induction heads", "edge of stability",
  "scaling laws", "steepest descent", "sign descent", "Muon", "Newton–Schulz", "µP", "momentum", "Adam",
  "Shampoo", "K-FAC", "SOAP", "warmup", "gradient clipping", "critical batch size", "cosine schedule",
  "weight decay", "river valley", "seed 3407", "bootstrap CI", "tuning budget", "random search", "confounding",
  "ablation", "flat minima", "Chinchilla", "emergent abilities", "replication", "post-mortem", "protocol",
  "interactive proof", "sumcheck", "PCP theorem", "linearity test", "zero knowledge", "GKR", "soundness", "completeness",
  "prover", "verifier", "Hadamard code", "rational proofs", "undetectable backdoor", "oversight",
];
const MATH = [
  "∫ f dμ", "E‖x‖²", "∇L(θ)", "argmin_θ", "Σᵢ₌₁ⁿ xᵢ", "lim sup", "O(1/√n)", "λ_max(H) < 2/η", "R(γ) = σ²/(γ−1)",
  "θ ← θ − η g", "‖·‖∞", "σ_max(W) ≤ 1", "p < 0.05", "H = −Σ p log p", "Pr[A | B]", "∂L/∂w", "e^{iπ} + 1 = 0",
  "Theorem 3.2.", "Lemma 7.", "Proof. ∎", "Definition 1.", "Corollary.", "Exercise 4.6", "Fig. 2.3", "§ 11.4",
  "L(N,D)", "N^{−0.34}", "tr(A⁻¹)", "∀ε>0 ∃δ>0", "x ∈ ℝᵈ", "w ∼ 𝒩(0, I)",
  "f(x) ⊕ f(y) = f(x ⊕ y)", "Pr[accept] ≥ 2/3", "g(0) + g(1) = H", "IP = PSPACE", "Hadamard: f(x) = ⟨u, x⟩",
];
const MISSING = ["[?]", "??", "Fig. ??", "Theorem ??", "[missing]", "see Chapter ??", "[citation needed]", "(??)", "[?, p. ?]"];

/**
 * Draws a seamlessly tiling wall of tiny text. Red channel = ordinary text coverage,
 * green channel = "missing" tokens (shaded vermilion by the shader).
 */
export function makeWallCanvas(seed = 11, W = 4096, H = 2048) {
  const cv = document.createElement("canvas");
  cv.width = W;
  cv.height = H;
  const ctx = cv.getContext("2d")!;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = "lighter";
  ctx.textBaseline = "alphabetic";
  const rnd = mulberry32(seed);
  const pick = <T,>(a: T[]) => a[Math.floor(rnd() * a.length)];
  const rows = 64;
  const lh = H / rows;
  const fonts = [
    `italic ${lh * 0.82}px Palatino, serif`,
    `${lh * 0.78}px Palatino, serif`,
    `${lh * 0.72}px Menlo, monospace`,
    `${lh * 0.8}px "Helvetica Neue", sans-serif`,
  ];
  for (let r = 0; r < rows; r++) {
    let x = -rnd() * 400;
    const rowAlpha = 0.25 + 0.75 * Math.pow(rnd(), 1.6);
    while (x < W) {
      const kind = rnd();
      let text: string;
      let miss = false;
      if (kind < 0.07) {
        text = pick(MISSING);
        miss = true;
      } else if (kind < 0.5) text = pick(WORDS);
      else text = pick(MATH);
      ctx.font = pick(fonts);
      const w = ctx.measureText(text).width;
      const a = Math.min(1, rowAlpha * (0.45 + 0.75 * rnd()));
      ctx.fillStyle = miss ? `rgba(0,255,0,${Math.min(1, a + 0.35)})` : `rgba(255,0,0,${a})`;
      for (const dx of [0, -W]) ctx.fillText(text, x + dx, (r + 0.78) * lh);
      x += w + lh * (0.8 + 2.2 * rnd());
    }
  }
  return cv;
}
