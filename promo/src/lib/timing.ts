// One beat grid for picture and score: 120 BPM, 30 fps -> 15 frames per beat, 60 per bar.
export const FPS = 30;
export const BPM = 120;
export const BEAT = 15;
export const BAR = 60;

export const bars = (n: number) => n * BAR;
export const beats = (n: number) => n * BEAT;
export const secs = (s: number) => Math.round(s * FPS);

// Scenes, in bars. Eight bars (16 s) per volume.
export const SCENE = {
  missing: { from: bars(0), dur: bars(4) },
  title: { from: bars(4), dur: bars(4) },
  v1: { from: bars(8), dur: bars(8) },
  v2: { from: bars(16), dur: bars(8) },
  v3: { from: bars(24), dur: bars(8) },
  v4: { from: bars(32), dur: bars(8) },
  v5: { from: bars(40), dur: bars(8) },
  finale: { from: bars(48), dur: bars(8) },
} as const;

export const TOTAL = bars(56); // 3360 frames = 112 s

/** 1 right on the beat, decaying to ~0 before the next one. `k` is the decay rate. */
export const pulse = (frame: number, k = 7) => Math.exp(-k * ((frame % BEAT) / BEAT));
/** Same, but only on every `every` beats. */
export const pulseEvery = (frame: number, every: number, k = 7) =>
  Math.exp(-k * ((frame % (BEAT * every)) / BEAT / every));
