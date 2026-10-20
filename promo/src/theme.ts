// Palette and type stacks. Colours are taken from the books themselves
// (missingbook.cls: sea, seadeep, seafoam, sand; laboratory.sty: rubric, inkblue, paper).
export const C = {
  night: "#03060b",
  navy: "#07101c",
  deep: "#0a1a2b",
  sea: "#005c7a",
  seaDeep: "#003a50",
  seaFoam: "#eaf4f7",
  seaMist: "#d6eaf0",
  sand: "#f9f5ec",
  sandLine: "#cdba8e",
  nut: "#6b4423",
  ink: "#222222",
  white: "#f4f7fb",
  rubric: "#b03a24",
  rubricHot: "#e8482b",
  inkBlue: "#1c365e",
  gridline: "#de7c54",
  paper: "#faf7f0",
  paperLine: "#d6c6aa",
  // Balmer series, as seen in the volume-2 emblem: H-alpha .. H-delta
  balmer: ["#ff4b3e", "#2fe0ff", "#5b7bff", "#a35cff"],
  gold: "#ffd98a",
  cyan: "#5fe3ff",
  amber: "#ffb454",
  moss: "#8bd48a",
};

export const F = {
  serif: '"Palatino", "Palatino Linotype", "Book Antiqua", "Songti SC", serif',
  sans: '"Helvetica Neue", "SF Pro Text", "PingFang SC", system-ui, sans-serif',
  mono: 'ui-monospace, "SF Mono", Menlo, "PingFang SC", monospace',
  cn: '"Songti SC", "STSong", "Palatino", serif',
  cnSans: '"PingFang SC", "Hiragino Sans GB", sans-serif',
};
