export type Book = {
  n: number;
  roman: string;
  title: string;
  lines: string[]; // title broken for display
  subtitle: string;
  pages: number;
  accent: string; // label colour on dark
  cover: string; // staticFile path
  spine: string; // spine colour
  spine2: string;
};

export const BOOKS: Book[] = [
  {
    n: 1, roman: "I", title: "Dissolving Problems", lines: ["Dissolving", "Problems"],
    subtitle: "A Casebook of Theory-Building", pages: 333, accent: "#6fd4f0",
    cover: "covers/v1.jpg", spine: "#0c5f7d", spine2: "#063a52",
  },
  {
    n: 2, roman: "II", title: "The Hydrogen Atoms of Deep Learning", lines: ["The Hydrogen Atoms", "of Deep Learning"],
    subtitle: "Solvable Models and What They Get Wrong", pages: 394, accent: "#ff7a66",
    cover: "covers/v2.jpg", spine: "#2b2f7a", spine2: "#14183f",
  },
  {
    n: 3, roman: "III", title: "Optimizers in the Wild", lines: ["Optimizers", "in the Wild"],
    subtitle: "Steepest Descent and the Dynamics of Training", pages: 369, accent: "#f1c75f",
    cover: "covers/v3.jpg", spine: "#4d6b2f", spine2: "#2b3d19",
  },
  {
    n: 4, roman: "IV", title: "Deep Learning as an Experimental Science", lines: ["Deep Learning as an", "Experimental Science"],
    subtitle: "A Laboratory Manual", pages: 365, accent: "#ff7a55",
    cover: "covers/v4.jpg", spine: "#b03a24", spine2: "#6e2113",
  },
  {
    // prover indigo / verifier ochre (verifiable.sty); the cover carries the Hadamard codeword of 1985
    n: 5, roman: "V", title: "Verifiable Intelligence", lines: ["Verifiable", "Intelligence"],
    subtitle: "Interactive Proofs as a Theory of AI Oversight", pages: 357, accent: "#ffb347",
    cover: "covers/v5.jpg", spine: "#2e3476", spine2: "#1b1e48",
  },
];
export const TOTAL_PAGES = BOOKS.reduce((a, b) => a + b.pages, 0); // 1818
