import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Canvas2D } from "../components/Canvas2D";
import { RevealText } from "../components/RevealText";
import { VolumeCard } from "../components/VolumeCard";
import { bloom } from "../lib/bloom";
import { BOOKS } from "../data";
import { F } from "../theme";
import { E, clamp, hash1, smoothstep, TAU } from "../lib/math";
import { Keys, sample } from "../world/camera";
import { COLS, ROWS, N, cheatedBit, honestBit, isLie, tests, where, HONEST, FIRST_REJECT } from "../world/codeword";

/**
 * Volume 5, "Verifiable Intelligence". The wall is the book's own emblem: the 2048-bit Hadamard codeword of 1985,
 * 32 rows of 64 digits in the order of the title page. The verifier reads three bits at a time (the BLR linearity test)
 * and accepts when f(x) + f(y) = f(x XOR y). The tests below are run on the real bits (src/world/codeword.ts);
 * the prover's lie is a set of flipped positions, and the first rejection is the test the film cuts to.
 */
const W = 1920, H = 1080, FOCAL = 1500;
type RGB = [number, number, number];
const OCHRE: RGB = [255, 176, 64]; // the verifier's marks
const INDIGO: RGB = [138, 148, 255]; // the prover's ink
const CORAL: RGB = [255, 96, 82]; // a caught lie
const WARM: RGB = [255, 238, 205];
const rgba = (c: RGB, a: number) => `rgba(${c[0]},${c[1]},${c[2]},${(a < 0 ? 0 : a > 1 ? 1 : a).toFixed(3)})`;
const mix3 = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

// ------------------------------------------------------------------ the timeline (local frames; a bar is 60 frames)
const CHEAT = 244; // the prover tampers with its proof
const CATCH = 270; // first rejection
const Q0 = 315; // first question card
const SURGE = 420;

/** frame at which test i is read */
const TF: number[] = [];
(() => {
  TF[0] = 135; // the title page's triple: rings pop at 135, 150, 165; verdict at 180
  for (let i = 1; i <= 4; i++) TF[i] = 195 + (i - 1) * 6;
  for (let i = 5; i < HONEST; i++) TF[i] = Math.round(219 + 20 * Math.pow((i - 5) / (HONEST - 6), 0.9));
  const cheat = [252, 256, 260, 264, 267, CATCH]; // tests HONEST .. FIRST_REJECT; the last one is rejected
  cheat.forEach((f, j) => (TF[HONEST + j] = f));
  for (let i = HONEST + cheat.length; i < tests.length; i++) TF[i] = CATCH + 6 + (i - (HONEST + cheat.length)) * 5;
})();
const verdictFrame = (i: number) => (i === 0 ? 180 : TF[i]);
const lifeOf = (i: number) => (i === 0 ? 72 : i <= 4 ? 24 : i < HONEST ? 8 : i < FIRST_REJECT ? 16 : i === FIRST_REJECT ? 46 : 8);

const readsAt = (g: number) => {
  let n = (g >= 135 ? 1 : 0) + (g >= 150 ? 1 : 0) + (g >= 165 ? 1 : 0);
  for (let i = 1; i < tests.length; i++) if (TF[i] <= g) n += 3;
  return n;
};
const countVerdicts = (g: number, accepted: boolean) => {
  let n = 0;
  for (let i = 0; i < tests.length; i++) if (verdictFrame(i) <= g && tests[i].accepted === accepted) n++;
  return n;
};

// ------------------------------------------------------------------ the camera: an orbit around a point on the floor of digits
type Cam = { D: number; az: number; el: number; tx: number; tz: number };
const cellX = (c: number) => c - 31.5;
const cellZ = (r: number) => r - 15.5; // row 0 is the far edge of the floor (the top of the title page)
const cellOf = (x: number): [number, number] => { const p = where[x]; return [p % COLS, Math.floor(p / COLS)]; };
const TRI = [45, 673, 45 ^ 673].map((x) => cellOf(x)); // x, y, x XOR y of the title page
const wxs = TRI.map(([c]) => cellX(c)), wzs = TRI.map(([, r]) => cellZ(r));

const CK = {
  D: [[0, 54], [66, 50], [120, 46], [134, 13], [180, 13], [200, 46], [300, 44], [330, 58], [420, 60], [470, 15]] as Keys,
  az: [[0, -0.3], [66, -0.18], [120, -0.04], [134, 0], [180, 0], [200, 0.05], [300, 0.3], [330, 0.25], [420, 0], [470, 0]] as Keys,
  el: [[0, 0.6], [66, 0.62], [120, 0.95], [134, 1.15], [180, 1.15], [200, 0.85], [300, 0.75], [330, 1.1], [420, 1.15], [470, 1.4]] as Keys,
  tx: [[0, 0], [120, 0], [134, wxs[0]], [150, wxs[0]], [158, wxs[1]], [165, wxs[1]], [173, wxs[2]], [180, wxs[2]], [200, 6], [300, 2], [420, 0], [470, 0]] as Keys,
  tz: [[0, -2], [120, -2], [134, wzs[0]], [150, wzs[0]], [158, wzs[1]], [165, wzs[1]], [173, wzs[2]], [180, wzs[2]], [200, 2], [300, 0], [420, 0], [470, 0]] as Keys,
};
const camAt = (g: number): Cam => ({
  D: sample(CK.D, g),
  az: sample(CK.az, g, E.inOutSine),
  el: sample(CK.el, g, E.inOutSine),
  tx: sample(CK.tx, g),
  tz: sample(CK.tz, g),
});

type Proj = (x: number, y: number, z: number) => [number, number, number, number] | null;
function makeProjector(c: Cam, sx: number, sy: number): Proj {
  const cx = c.tx + c.D * Math.sin(c.az) * Math.cos(c.el);
  const cy = c.D * Math.sin(c.el);
  const cz = c.tz + c.D * Math.cos(c.az) * Math.cos(c.el);
  let fx = c.tx - cx, fy = -cy, fz = c.tz - cz;
  const fl = Math.hypot(fx, fy, fz);
  fx /= fl; fy /= fl; fz /= fl;
  let rx = -fz, rz = fx;
  const rl = Math.hypot(rx, rz);
  rx /= rl; rz /= rl;
  const ux = -rz * fy, uy = rz * fx - rx * fz, uz = rx * fy;
  return (x, y, z) => {
    const vx = x - cx, vy = y - cy, vz = z - cz;
    const zc = vx * fx + vy * fy + vz * fz;
    if (zc < 0.5) return null;
    const xc = vx * rx + vz * rz;
    const yc = vx * ux + vy * uy + vz * uz;
    const k = FOCAL / zc;
    return [960 + sx + xc * k, 540 + sy - yc * k, zc, k];
  };
}

// ------------------------------------------------------------------ the wall
const boost = new Float32Array(N); // brightness of cells that a test is reading right now
const FLIP_AT = new Float32Array(N).map((_, p) => CHEAT + Math.floor(hash1(p * 0.37) * 9));

function drawWall(ctx: CanvasRenderingContext2D, g: number) {
  // backdrop
  const bg = ctx.createRadialGradient(960, 500, 80, 960, 540, 1300);
  bg.addColorStop(0, "#0b0e38");
  bg.addColorStop(0.55, "#05061c");
  bg.addColorStop(1, "#02020a");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  const cam = camAt(g);
  const catchAge = g - CATCH;
  const shake = catchAge >= 0 && catchAge < 18 ? 16 * Math.exp(-catchAge * 0.28) : 0;
  const P = makeProjector(cam, (hash1(g * 1.7) - 0.5) * shake, (hash1(g * 2.9 + 4) - 0.5) * shake);
  const dim = (0.2 + 0.8 * smoothstep(0, 28, g)) * (1 - 0.62 * smoothstep(300, 316, g) + 0.62 * smoothstep(418, 440, g));
  const lieFlash = catchAge >= 0 ? Math.exp(-Math.max(0, catchAge - 2) / 13) : 0;
  const lieTint = smoothstep(CATCH, CATCH + 30, g) * 0.26 * (1 - 0.7 * smoothstep(Q0 - 6, Q0 + 12, g));

  // which cells are being read right now, and the rings/lines to draw for them
  boost.fill(0);
  type Active = { i: number; age: number; life: number };
  const active: Active[] = [];
  for (let i = 0; i < tests.length; i++) {
    const age = g - TF[i];
    if (age < 0 || age > lifeOf(i)) continue;
    active.push({ i, age, life: lifeOf(i) });
    const t = tests[i];
    const env = i === 0 ? 1 : Math.exp(-age / (i <= 4 ? 10 : 4));
    for (const x of [t.x, t.y, t.z]) {
      const p = where[x];
      if (i === 0) {
        const which = [t.x, t.y, t.z].indexOf(x);
        if (g >= [135, 150, 165][which]) boost[p] = Math.max(boost[p], 1 - smoothstep(190, 205, g));
      } else boost[p] = Math.max(boost[p], env);
    }
  }

  // ---- 2048 digits lying on the floor
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `500 40px ${F.mono}`;
  const S = 40, sc = 0.9 / S;
  const centre = [31.5, 15.5];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const p = r * COLS + c;
      const wx = cellX(c), wz = cellZ(r);
      const P0 = P(wx, 0, wz);
      if (!P0) continue;
      if (P0[0] < -80 || P0[0] > W + 80 || P0[1] < -80 || P0[1] > H + 80) continue;
      const d = Math.hypot(c - centre[0], (r - centre[1]) * 1.7);
      // decode: scrambled digits lock to their true values in a wave from the middle
      const lock = 6 + d * 0.92;
      const locked = g >= lock;
      let bit: number;
      if (!locked) bit = hash1(p * 0.713 + Math.floor(g / 2) * 1.91) > 0.5 ? 1 : 0;
      else bit = isLie(p) && g >= FLIP_AT[p] ? 1 - honestBit(p) : honestBit(p);
      const front = Math.exp(-Math.pow((g - lock) / 2.6, 2));
      const fog = Math.exp(-Math.max(0, P0[2] - cam.D * 0.8) / (cam.D * 0.62));
      let a = (bit ? 0.7 : 0.42) * fog * dim * (0.86 + 0.14 * Math.sin(g * 0.11 + p * 1.7));
      let col: RGB = INDIGO;
      // rendering flourishes: the wave front, the verifier's reads, the caught lie, the final surge
      if (front > 0.01) { col = mix3(col, WARM, front * 0.8); a += front * 0.35; }
      const b = boost[p];
      if (b > 0.01) { col = mix3(col, OCHRE, Math.min(1, b * 1.2)); a = Math.max(a, 0.35 + 0.65 * b); }
      if (g >= CATCH && isLie(p)) {
        const lf = Math.max(lieFlash, lieTint);
        col = mix3(col, CORAL, Math.min(1, lf * 1.6));
        a = Math.max(a, (0.18 + 0.82 * lieFlash) * fog + lieTint * 0.9 * fog);
      }
      const surge = smoothstep(SURGE, SURGE + 46, g - d * 0.7);
      if (surge > 0) { col = mix3(col, WARM, surge); a = Math.min(1, a + surge * 0.7); }
      if (a < 0.02) continue;

      const Px = P(wx + 1, 0, wz), Pz = P(wx, 0, wz + 1);
      if (!Px || !Pz) continue;
      const ex = Px[0] - P0[0], ey = Px[1] - P0[1], zx = Pz[0] - P0[0], zy = Pz[1] - P0[1];
      const hgt = Math.hypot(zx, zy) * 0.9;
      ctx.fillStyle = rgba(col, a);
      if (hgt < 3.2) {
        ctx.fillRect(P0[0] - 0.8, P0[1] - 0.8, 1.6, 1.6);
        continue;
      }
      ctx.setTransform(ex * sc, ey * sc, zx * sc, zy * sc, P0[0], P0[1]);
      ctx.fillText(bit ? "1" : "0", 0, 0);
    }
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  // ---- the edge of the proof string
  {
    const corners = [[-32.6, -16.2], [32.6, -16.2], [32.6, 16.2], [-32.6, 16.2]].map(([x, z]) => P(x, 0, z));
    if (corners.every(Boolean)) {
      ctx.strokeStyle = rgba(INDIGO, 0.22 * dim);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      corners.forEach((q, i) => (i ? ctx.lineTo(q![0], q![1]) : ctx.moveTo(q![0], q![1])));
      ctx.closePath();
      ctx.stroke();
    }
  }

  // ---- rings, lines and ripples for the tests being read
  const ringR = Math.min(2.3, Math.max(0.95, 0.9 + (cam.D - 12) * 0.032));
  const ringPath = (wx: number, wz: number, R: number) => {
    ctx.beginPath();
    for (let j = 0; j <= 44; j++) {
      const q = P(wx + R * Math.cos((j / 44) * TAU), 0, wz + R * Math.sin((j / 44) * TAU));
      if (!q) return false;
      if (j) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]);
    }
    return true;
  };
  const labels: { x: number; y: number; text: string; color: RGB; size: number; alpha: number }[] = [];
  const geoFade = 1 - 0.78 * smoothstep(Q0 - 6, Q0 + 12, g) + 0.78 * smoothstep(SURGE - 12, SURGE, g);
  for (const { i, age, life } of active) {
    const t = tests[i];
    const cells = [t.x, t.y, t.z].map((x) => cellOf(x));
    const wxy = cells.map(([c]) => cellX(c)), wzy = cells.map(([, r]) => cellZ(r));
    const rejected = !t.accepted;
    const col = rejected ? CORAL : OCHRE;
    const pops = i === 0 ? [135, 150, 165].map((f) => g - f) : [age, age, age];
    const fadeOut = (i === 0 ? 1 - smoothstep(192, 207, g) : 1 - smoothstep(life * 0.55, life, age)) * geoFade;
    for (let j = 0; j < 3; j++) {
      const a = pops[j];
      if (a < 0) continue;
      const pop = 1 + 1.3 * Math.exp(-a * 0.32);
      ctx.strokeStyle = rgba(col, fadeOut);
      ctx.lineWidth = 3.4;
      if (ringPath(wxy[j], wzy[j], ringR * pop)) ctx.stroke();
      if (i === 0 || i === FIRST_REJECT || i <= 4) {
        const q = P(wxy[j], 0, wzy[j]);
        if (q && i === 0) labels.push({ x: q[0], y: q[1] + ringR * q[3] * 1.5 + 30, text: ["f(x) = ", "f(y) = ", "f(x ⊕ y) = "][j] + t.bits[j], color: OCHRE, size: 26, alpha: fadeOut * smoothstep(0, 5, a) });
      }
    }
    // the triangle: drawn progressively for the first test, at once for the others
    const lineT = i === 0 ? clamp((g - 180) / 14) : 1;
    if (lineT > 0) {
      const pts = [0, 1, 2, 0].map((j) => P(wxy[j], 0, wzy[j]));
      if (pts.every(Boolean)) {
        ctx.strokeStyle = rgba(col, fadeOut * 0.85);
        ctx.lineWidth = i === 0 ? 3 : 2;
        ctx.beginPath();
        let remaining = lineT * 3;
        ctx.moveTo(pts[0]![0], pts[0]![1]);
        for (let j = 1; j <= 3 && remaining > 0; j++) {
          const seg = Math.min(1, remaining);
          const [ax, ay] = pts[j - 1]!, [bx, by] = pts[j]!;
          ctx.lineTo(ax + (bx - ax) * seg, ay + (by - ay) * seg);
          remaining -= 1;
        }
        ctx.stroke();
        // verdict next to the triangle
        if ((i === 0 && g >= 186) || i === FIRST_REJECT || (i >= 1 && i <= 4)) {
          const cx = (pts[0]![0] + pts[1]![0] + pts[2]![0]) / 3, cy = (pts[0]![1] + pts[1]![1] + pts[2]![1]) / 3;
          const [b0, b1, b2] = t.bits;
          const text = t.accepted ? `${b0} ⊕ ${b1} = ${b2}   ✓` : `${b0} ⊕ ${b1} ≠ ${b2}   ✗`;
          labels.push({ x: cx, y: cy - 6, text, color: col, size: i === 0 ? 46 : i === FIRST_REJECT ? 54 : 34, alpha: fadeOut * smoothstep(i === 0 ? 186 : TF[i], (i === 0 ? 186 : TF[i]) + 5, g) });
        }
      }
    }
    // a lie is caught: a ripple spreads from a flipped bit among the three
    if (rejected && i === FIRST_REJECT) {
      const lieCell = cells.find(([c, r]) => isLie(r * COLS + c)) ?? cells[0];
      const lwx = cellX(lieCell[0]), lwz = cellZ(lieCell[1]);
      for (let w = 0; w < 3; w++) {
        const aw = age - w * 5;
        if (aw < 0) continue;
        const R = 0.8 + aw * 1.9;
        ctx.strokeStyle = rgba(CORAL, Math.max(0, 1 - aw / 34) * 0.9);
        ctx.lineWidth = 4 - w;
        if (ringPath(lwx, lwz, R)) ctx.stroke();
      }
    }
  }

  bloom(ctx, W, H, { strength: 0.85, blur: 6, scale: 0.25, key: "s7" });

  // ---- text labels (after the bloom so that they stay sharp)
  ctx.textAlign = "center";
  for (const l of labels) {
    if (l.alpha <= 0.01) continue;
    ctx.font = `700 ${l.size}px ${F.mono}`;
    ctx.fillStyle = `rgba(2,3,16,${0.55 * l.alpha})`;
    const w = ctx.measureText(l.text).width + 30;
    ctx.fillRect(l.x - w / 2, l.y - l.size * 0.72, w, l.size * 1.34);
    ctx.fillStyle = rgba(l.color, l.alpha);
    ctx.fillText(l.text, l.x, l.y);
  }

  // vignette
  const vg = ctx.createRadialGradient(960, 540, 520, 960, 540, 1250);
  vg.addColorStop(0, "rgba(0,0,0,0)");
  vg.addColorStop(1, "rgba(0,0,6,0.55)");
  ctx.fillStyle = vg;
  ctx.fillRect(0, 0, W, H);

  drawBand(ctx, g);
}

/** the transcript band of a chapter opening: a prover rail above a verifier rail, one query/answer exchange per test */
function drawBand(ctx: CanvasRenderingContext2D, g: number) {
  const op = smoothstep(126, 146, g) * (1 - smoothstep(298, 314, g));
  if (op <= 0.01) return;
  const yP = 930, yV = 1000, x0 = 230, x1 = 1830;
  ctx.save();
  ctx.globalAlpha = op;
  ctx.fillStyle = "rgba(2,3,18,0.62)";
  ctx.fillRect(60, yP - 52, 1800, yV - yP + 96);
  ctx.lineWidth = 2;
  ctx.strokeStyle = rgba(INDIGO, 0.7);
  ctx.beginPath(); ctx.moveTo(x0, yP); ctx.lineTo(x1, yP); ctx.stroke();
  ctx.strokeStyle = rgba(OCHRE, 0.8);
  ctx.beginPath(); ctx.moveTo(x0, yV); ctx.lineTo(x1, yV); ctx.stroke();
  ctx.font = `700 21px ${F.sans}`;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = rgba(INDIGO, 1);
  ctx.fillText("PROVER", 92, yP);
  ctx.fillStyle = rgba(OCHRE, 1);
  ctx.fillText("VERIFIER", 92, yV);
  ctx.beginPath();
  ctx.rect(x0, yP - 40, x1 - x0, yV - yP + 80);
  ctx.clip();
  for (let i = 0; i < tests.length; i++) {
    const age = g - TF[i];
    if (age < 0 || age > 78) continue;
    const xi = x1 - age * 20;
    const fade = Math.min(1, (xi - x0) / 90) * (1 - age / 90);
    if (fade <= 0) continue;
    const t = tests[i];
    const c = t.accepted ? OCHRE : CORAL;
    ctx.globalAlpha = op * fade;
    ctx.strokeStyle = rgba(OCHRE, 0.95);
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xi, yV); ctx.lineTo(xi + 16, yP); ctx.stroke();           // query: x, y, x XOR y
    ctx.fillStyle = rgba(OCHRE, 0.95);
    ctx.beginPath(); ctx.moveTo(xi + 16, yP); ctx.lineTo(xi + 7, yP + 13); ctx.lineTo(xi + 22, yP + 14); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = rgba(INDIGO, 0.95);
    ctx.beginPath(); ctx.moveTo(xi + 30, yP); ctx.lineTo(xi + 46, yV); ctx.stroke();       // answer: three bits
    ctx.fillStyle = rgba(INDIGO, 0.95);
    ctx.beginPath(); ctx.moveTo(xi + 46, yV); ctx.lineTo(xi + 38, yV - 13); ctx.lineTo(xi + 53, yV - 12); ctx.closePath(); ctx.fill();
    if (t.accepted) {
      ctx.fillStyle = rgba(c, 1);
      ctx.beginPath(); ctx.arc(xi + 62, yV, 5.5, 0, TAU); ctx.fill();
    } else {
      ctx.strokeStyle = rgba(c, 1);
      ctx.lineWidth = 3.4;
      ctx.beginPath(); ctx.arc(xi + 62, yV, 10, 0, TAU); ctx.moveTo(xi + 56, yV - 6); ctx.lineTo(xi + 68, yV + 6); ctx.moveTo(xi + 68, yV - 6); ctx.lineTo(xi + 56, yV + 6); ctx.stroke();
    }
  }
  ctx.restore();
}

// ------------------------------------------------------------------ overlays
const QUESTIONS = [
  { n: "1", lines: ["Is honesty", "an equilibrium?"] },
  { n: "2", lines: ["What does the", "verifier spend?"] },
  { n: "3", lines: ["How often", "is a lie caught?"] },
];

const Counters: React.FC<{ g: number }> = ({ g }) => {
  const op = smoothstep(150, 170, g) * (1 - smoothstep(296, 312, g));
  if (op <= 0.01) return null;
  const rows: [string, number, RGB][] = [
    ["READS", readsAt(g), WARM],
    ["ACCEPTED", countVerdicts(g, true), OCHRE],
    ["REJECTED", countVerdicts(g, false), CORAL],
  ];
  return (
    <div style={{ position: "absolute", right: 96, top: 86, width: 330, opacity: op, fontFamily: F.mono }}>
      <div style={{ height: 3, background: rgba(OCHRE, 0.85), marginBottom: 14, width: 330 }} />
      {rows.map(([label, n, c]) => (
        <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
          <span style={{ fontFamily: F.sans, fontWeight: 700, fontSize: 21, letterSpacing: "0.22em", color: rgba(c, 0.9) }}>{label}</span>
          <span style={{ fontSize: 52, color: rgba(c, 1), fontVariantNumeric: "tabular-nums" }}>{n}</span>
        </div>
      ))}
    </div>
  );
};

const Cards: React.FC<{ g: number }> = ({ g }) => {
  const leave = E.inCubic(clamp((g - 416) / 18));
  return (
    <>
      {QUESTIONS.map((q, i) => {
        const t = E.outCubic(clamp((g - (Q0 + i * 15)) / 18));
        if (t <= 0 || leave >= 1) return null;
        const cx = 400 + i * 560;
        return (
          <div
            key={q.n}
            style={{
              position: "absolute", left: cx - 245, top: 300, width: 490, padding: "30px 40px 38px",
              background: "linear-gradient(180deg, rgba(16,18,62,0.82), rgba(5,6,28,0.88))",
              border: "1px solid rgba(255,176,64,0.4)", borderTop: "4px solid #ffb040", borderRadius: 6,
              boxShadow: "0 34px 90px rgba(0,0,0,0.55), 0 0 60px rgba(255,176,64,0.08)", backdropFilter: "blur(10px)",
              opacity: t * (1 - leave), transform: `translateY(${(1 - t) * 50 + leave * 36}px)`, filter: `blur(${(1 - t) * 10}px)`,
            }}
          >
            <div style={{ fontFamily: F.serif, fontSize: 110, lineHeight: 0.95, color: "#ffb040" }}>{q.n}</div>
            <div style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 52, lineHeight: 1.12, color: "#f4f7fb", marginTop: 16 }}>
              {q.lines[0]}
              <br />
              {q.lines[1]}
            </div>
          </div>
        );
      })}
    </>
  );
};

export const S7Verifiable: React.FC = () => {
  const g = useCurrentFrame();
  const drain = E.inOutCubic(clamp(g / 26));
  const flashOut = smoothstep(462, 479, g);
  const ital = { fontFamily: F.serif, fontStyle: "italic" as const };
  return (
    <AbsoluteFill style={{ background: "#02020a" }}>
      <Canvas2D draw={(ctx, frame) => drawWall(ctx, frame)} />

      <div style={{ position: "absolute", left: 110, top: 96, width: 1100 }}>
        <RevealText text="What can a weak verifier learn" start={74} stagger={4} dur={14} exit={134} exitDur={10}
          style={{ ...ital, fontSize: 68, color: "#f4f7fb", textShadow: "0 0 40px rgba(2,3,20,0.95)" }} />
        <div style={{ height: 6 }} />
        <RevealText text="from a strong prover that may lie?" start={96} stagger={4} dur={14} exit={134} exitDur={10}
          style={{ ...ital, fontSize: 68, color: "#ffb347", textShadow: "0 0 40px rgba(2,3,20,0.95)" }} />
      </div>

      <div style={{ position: "absolute", left: 110, top: 96 }}>
        <RevealText text="Read three bits." start={122} stagger={6} dur={14} exit={192} exitDur={10}
          style={{ ...ital, fontSize: 76, color: "#f4f7fb", textShadow: "0 0 40px rgba(2,3,20,0.95)" }} />
      </div>

      <div style={{ position: "absolute", left: 110, top: 96 }}>
        <RevealText text="Now the prover cheats." start={240} stagger={5} dur={14} exit={266} exitDur={6}
          style={{ ...ital, fontSize: 62, color: "#f4f7fb", textShadow: "0 0 40px rgba(2,3,20,0.95)" }} />
      </div>
      <div style={{ position: "absolute", left: 110, top: 96 }}>
        <RevealText text="Caught." start={CATCH} stagger={0} dur={8} rise={18} blur={6} exit={304} exitDur={10}
          style={{ ...ital, fontSize: 124, color: "#ff6a58", textShadow: "0 0 50px rgba(255,80,60,0.55), 0 0 30px rgba(2,3,20,0.95)" }} />
      </div>

      <Counters g={g} />
      <Cards g={g} />

      <div style={{ position: "absolute", left: 0, right: 0, bottom: 118, textAlign: "center" }}>
        <RevealText text="Fifteen protocols. The same three questions." start={396} stagger={3} dur={14} exit={426} exitDur={10}
          style={{ ...ital, fontSize: 48, color: "rgba(244,247,251,0.92)", textShadow: "0 0 30px rgba(2,3,20,0.95)" }} />
      </div>

      <VolumeCard book={BOOKS[4]} start={0} dur={66} titleSize={100} />

      {/* vermilion flood from the volume before, draining; and the white-out into the finale */}
      <AbsoluteFill style={{ background: "#d9482b", clipPath: `circle(${(1 - drain) * 1800}px at 960px 500px)` }} />
      <AbsoluteFill style={{ background: "#fff", opacity: flashOut, pointerEvents: "none" }} />
    </AbsoluteFill>
  );
};
