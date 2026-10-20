import React, { useMemo } from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Canvas2D } from "../components/Canvas2D";
import { RevealText } from "../components/RevealText";
import { VolumeCard } from "../components/VolumeCard";
import { bloom } from "../lib/bloom";
import { drawKernel } from "../components/kernel";
import { useZh } from "../lang";
import { BOOKS } from "../data";
import { C, F } from "../theme";
import { E, clamp, hash1, lerp, mulberry32, smoothstep, TAU } from "../lib/math";

export const S2_PRE = 90; // frames of pre-roll: this scene is already running while the title portal opens
const W = 1920, H = 1080;

// ---------- timeline (local frames; 0 = bar boundary) ----------
const HITS = [75, 105, 135];
const RISE0 = 165, RISE1 = 335;
const L0 = 770;
const L1 = -1000;
const riseP = (f: number) => Math.pow(clamp((f - RISE0) / (RISE1 - RISE0)), 2.15);
const seaLevel = (f: number) => lerp(L0, L1, riseP(f));
const wetFrame = (y: number) => {
  let lo = RISE0, hi = RISE1;
  for (let i = 0; i < 30; i++) {
    const m = (lo + hi) / 2;
    if (seaLevel(m) > y) lo = m; else hi = m;
  }
  return lo;
};

// world layout
const NUT = { x: 960, y: 590, rx: 100, ry: 108 };
const ROCK_TOP = 690;
const TIP_DIR = { x: 0.64, y: -0.77 }; // unit vector from the nut centre toward the chisel head
const CHISEL_LEN = 250;

// ---------- the nut: a sprite we can draw, dissolve and sample ----------
const SS = 3; // sprite scale
const SW = 660, SH = 700;
type NutData = { sprite: HTMLCanvas; px: Float32Array; py: Float32Array; col: Uint8Array; n: number; rel: Float32Array; ph: Float32Array };
type HTMLCanvas = HTMLCanvasElement;

function buildNut(): NutData {
  const cv = document.createElement("canvas");
  cv.width = SW; cv.height = SH;
  const ctx = cv.getContext("2d")!;
  const cx = SW / 2, cy = SH / 2 + 10, rx = NUT.rx * SS, ry = NUT.ry * SS;
  const rnd = mulberry32(21);
  const shape = () => {
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, TAU);
  };
  ctx.save();
  shape();
  ctx.clip();
  const g = ctx.createRadialGradient(cx - rx * 0.35, cy - ry * 0.4, rx * 0.1, cx, cy, rx * 1.15);
  g.addColorStop(0, "#c58c55");
  g.addColorStop(0.45, "#8a5a32");
  g.addColorStop(0.8, "#5a3519");
  g.addColorStop(1, "#2c190b");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, SW, SH);
  // grain along meridians
  for (let i = 0; i < 1400; i++) {
    const m = (rnd() - 0.5) * 2; // -1..1 meridian offset
    const t = rnd() * 2 - 1;
    const x = cx + m * rx * Math.sqrt(1 - t * t) * 0.98;
    const y = cy + t * ry;
    ctx.strokeStyle = rnd() < 0.5 ? `rgba(30,14,4,${0.08 + rnd() * 0.18})` : `rgba(235,185,125,${0.05 + rnd() * 0.12})`;
    ctx.lineWidth = 1 + rnd() * 2.2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    const len = 14 + rnd() * 40;
    ctx.lineTo(x + m * 6, y + len);
    ctx.stroke();
  }
  // seams
  for (const off of [-0.62, 0, 0.62]) {
    ctx.beginPath();
    ctx.moveTo(cx, cy - ry);
    ctx.quadraticCurveTo(cx + off * rx * 1.35, cy, cx, cy + ry);
    ctx.strokeStyle = "rgba(28,14,5,0.65)";
    ctx.lineWidth = 7;
    ctx.stroke();
    ctx.strokeStyle = "rgba(240,200,150,0.22)";
    ctx.lineWidth = 3;
    ctx.translate(4, 0);
    ctx.stroke();
    ctx.translate(-4, 0);
  }
  // gloss
  const gl = ctx.createRadialGradient(cx - rx * 0.4, cy - ry * 0.5, 2, cx - rx * 0.4, cy - ry * 0.5, rx * 0.5);
  gl.addColorStop(0, "rgba(255,238,205,0.55)");
  gl.addColorStop(1, "rgba(255,238,205,0)");
  ctx.fillStyle = gl;
  ctx.fillRect(0, 0, SW, SH);
  ctx.restore();
  // sample particles
  const im = ctx.getImageData(0, 0, SW, SH).data;
  const px: number[] = [], py: number[] = [], col: number[] = [], rel: number[] = [], ph: number[] = [];
  const step = 7;
  for (let y = 0; y < SH; y += step)
    for (let x = 0; x < SW; x += step) {
      const xx = x + (rnd() - 0.5) * step, yy = y + (rnd() - 0.5) * step;
      const ix = Math.max(0, Math.min(SW - 1, Math.round(xx))), iy = Math.max(0, Math.min(SH - 1, Math.round(yy)));
      const k = (iy * SW + ix) * 4;
      if (im[k + 3] < 200) continue;
      px.push(xx); py.push(yy);
      col.push(im[k], im[k + 1], im[k + 2]);
      rel.push(rnd()); ph.push(rnd() * TAU);
    }
  return { sprite: cv, px: Float32Array.from(px), py: Float32Array.from(py), col: Uint8Array.from(col), n: px.length, rel: Float32Array.from(rel), ph: Float32Array.from(ph) };
}

// ---------- tools ----------
type V = { x: number; y: number };
const U: V = TIP_DIR;
const PERP: V = { x: 0.77, y: 0.64 };
const TIP: V = { x: NUT.x + U.x * (NUT.rx - 4), y: NUT.y + U.y * (NUT.ry - 4) };
const HEAD: V = { x: TIP.x + U.x * CHISEL_LEN, y: TIP.y + U.y * CHISEL_LEN };
const HLEN = 560;
const Q: V = { x: HEAD.x - PERP.x * HLEN, y: HEAD.y - PERP.y * HLEN };

const hammerAngle = (f: number) => {
  // radians about Q; 0 = at the chisel head, negative = raised
  const d = (a: number) => (a * Math.PI) / 180;
  const keys: [number, number][] = [];
  const raiseIn = 62;
  keys.push([raiseIn - 8, d(-30)], [raiseIn + 4, d(-22)]);
  HITS.forEach((fh, i) => {
    keys.push([fh - 6, d(-22)], [fh, 0], [fh + 4, d(-8)]);
    if (i < HITS.length - 1) keys.push([fh + 16, d(-22)]);
  });
  keys.push([HITS[2] + 20, d(-30)], [HITS[2] + 40, d(-46)]);
  if (f <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    if (f <= keys[i][0]) {
      const [f0, a0] = keys[i - 1], [f1, a1] = keys[i];
      const u = clamp((f - f0) / (f1 - f0));
      // swing down accelerates; raising decelerates
      const e = a1 > a0 ? E.inQuad(u) : E.outCubic(u);
      return lerp(a0, a1, e);
    }
  }
  return keys[keys.length - 1][1];
};

const rotAbout = (p: V, c: V, a: number): V => ({
  x: c.x + (p.x - c.x) * Math.cos(a) - (p.y - c.y) * Math.sin(a),
  y: c.y + (p.x - c.x) * Math.sin(a) + (p.y - c.y) * Math.cos(a),
});

function steel(ctx: CanvasRenderingContext2D, a: V, b: V, w0: number, w1: number, light = 1) {
  const nx = -(b.y - a.y), ny = b.x - a.x, l = Math.hypot(nx, ny) || 1;
  const ux = nx / l, uy = ny / l;
  ctx.beginPath();
  ctx.moveTo(a.x + ux * w0, a.y + uy * w0);
  ctx.lineTo(b.x + ux * w1, b.y + uy * w1);
  ctx.lineTo(b.x - ux * w1, b.y - uy * w1);
  ctx.lineTo(a.x - ux * w0, a.y - uy * w0);
  ctx.closePath();
  const g = ctx.createLinearGradient(a.x + ux * w0, a.y + uy * w0, a.x - ux * w0, a.y - uy * w0);
  g.addColorStop(0, `rgb(${236 * light | 0},${242 * light | 0},${246 * light | 0})`);
  g.addColorStop(0.45, `rgb(${120 * light | 0},${134 * light | 0},${148 * light | 0})`);
  g.addColorStop(1, `rgb(${34 * light | 0},${44 * light | 0},${58 * light | 0})`);
  ctx.fillStyle = g;
  ctx.fill();
  ctx.strokeStyle = "rgba(8,24,38,0.7)";
  ctx.lineWidth = 2.5;
  ctx.stroke();
}

function drawChisel(ctx: CanvasRenderingContext2D, off: V, snap: number, f: number) {
  // before snapping: one piece; after: two pieces falling
  const tip = { x: TIP.x + off.x, y: TIP.y + off.y };
  const head = { x: HEAD.x + off.x, y: HEAD.y + off.y };
  const mid = { x: lerp(tip.x, head.x, 0.5), y: lerp(tip.y, head.y, 0.5) };
  if (snap <= 0) {
    steel(ctx, tip, head, 7, 15);
    // bevel highlight and mushroomed head
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(tip.x - PERP.x * 3, tip.y - PERP.y * 3);
    ctx.lineTo(head.x - PERP.x * 8, head.y - PERP.y * 8);
    ctx.stroke();
    ctx.fillStyle = "#4b5663";
    ctx.beginPath();
    ctx.ellipse(head.x, head.y, 20, 11, Math.atan2(PERP.y, PERP.x), 0, TAU);
    ctx.fill();
    return;
  }
  const t = snap; // frames since the snap
  const g = 0.55;
  // lower piece (tip): stays a moment, then slides and falls
  const lp = { x: lerp(tip.x, mid.x, 0.0) + 2.4 * t, y: tip.y + 0.5 * g * t * t * 0.55 };
  const la = 0.018 * t;
  const pieceA0 = rotAbout(tip, lp, la), pieceA1 = rotAbout(mid, lp, la);
  const pa0 = { x: pieceA0.x + (lp.x - tip.x), y: pieceA0.y + (lp.y - tip.y) };
  const pa1 = { x: pieceA1.x + (lp.x - tip.x), y: pieceA1.y + (lp.y - tip.y) };
  steel(ctx, pa0, pa1, 7, 11, 0.9);
  // upper piece flies up-right with the hammer's rebound
  const up = { x: 5.5 * t, y: -9 * t + 0.5 * g * t * t };
  const ub = 0.09 * t;
  const u0 = rotAbout({ x: mid.x + up.x, y: mid.y + up.y }, { x: mid.x + up.x, y: mid.y + up.y }, 0);
  const c = { x: lerp(mid.x, head.x, 0.5) + up.x, y: lerp(mid.y, head.y, 0.5) + up.y };
  const a1p = rotAbout({ x: mid.x + up.x, y: mid.y + up.y }, c, ub);
  const a2p = rotAbout({ x: head.x + up.x, y: head.y + up.y }, c, ub);
  steel(ctx, a1p, a2p, 11, 15, 0.95);
  ctx.fillStyle = "#4b5663";
  ctx.beginPath();
  ctx.ellipse(a2p.x, a2p.y, 20, 11, Math.atan2(PERP.y, PERP.x) + ub, 0, TAU);
  ctx.fill();
  void u0; void f;
}

function drawHammer(ctx: CanvasRenderingContext2D, ang: number, alpha = 1, x = 0) {
  const handleEnd = rotAbout(Q, Q, 0);
  void handleEnd;
  const headC = rotAbout({ x: HEAD.x + U.x * 78, y: HEAD.y + U.y * 78 }, Q, ang);
  const dirU = { x: U.x * Math.cos(ang) - U.y * Math.sin(ang), y: U.x * Math.sin(ang) + U.y * Math.cos(ang) };
  const dirP = { x: -dirU.y, y: dirU.x }; // perpendicular (handle axis)
  const sgn = dirP.x * PERP.x + dirP.y * PERP.y > 0 ? 1 : -1;
  const hp = { x: dirP.x * sgn, y: dirP.y * sgn }; // from head toward Q-ish: negative of handle direction
  ctx.save();
  ctx.globalAlpha = alpha;
  // handle: from the head centre toward Q
  const toQ = { x: Q.x - headC.x, y: Q.y - headC.y };
  const lq = Math.hypot(toQ.x, toQ.y);
  const hq = { x: toQ.x / lq, y: toQ.y / lq };
  const end = { x: headC.x + hq.x * (HLEN + 160), y: headC.y + hq.y * (HLEN + 160) };
  const nx = -hq.y, ny = hq.x;
  ctx.beginPath();
  ctx.moveTo(headC.x + nx * 15, headC.y + ny * 15);
  ctx.lineTo(end.x + nx * 20, end.y + ny * 20);
  ctx.lineTo(end.x - nx * 20, end.y - ny * 20);
  ctx.lineTo(headC.x - nx * 15, headC.y - ny * 15);
  ctx.closePath();
  const hg = ctx.createLinearGradient(headC.x + nx * 18, headC.y + ny * 18, headC.x - nx * 18, headC.y - ny * 18);
  hg.addColorStop(0, "#d99a55");
  hg.addColorStop(0.5, "#98602c");
  hg.addColorStop(1, "#4a2c10");
  ctx.fillStyle = hg;
  ctx.fill();
  ctx.strokeStyle = "rgba(30,14,4,0.65)";
  ctx.lineWidth = 2.5;
  ctx.stroke();
  // head: a block whose long axis is the strike direction; its thickness runs along the handle
  const ax = dirU.x, ay = dirU.y;
  const L = 78, Wd = 46;
  const p = [
    { x: headC.x - ax * L + hq.x * Wd, y: headC.y - ay * L + hq.y * Wd },
    { x: headC.x + ax * L + hq.x * Wd, y: headC.y + ay * L + hq.y * Wd },
    { x: headC.x + ax * L - hq.x * Wd, y: headC.y + ay * L - hq.y * Wd },
    { x: headC.x - ax * L - hq.x * Wd, y: headC.y - ay * L - hq.y * Wd },
  ];
  ctx.beginPath();
  ctx.moveTo(p[0].x, p[0].y);
  for (let i = 1; i < 4; i++) ctx.lineTo(p[i].x, p[i].y);
  ctx.closePath();
  const sg = ctx.createLinearGradient(headC.x + hq.x * Wd, headC.y + hq.y * Wd, headC.x - hq.x * Wd, headC.y - hq.y * Wd);
  sg.addColorStop(0, "#eef3f6");
  sg.addColorStop(0.3, "#8a98a6");
  sg.addColorStop(1, "#26313c");
  ctx.fillStyle = sg;
  ctx.fill();
  ctx.strokeStyle = "rgba(8,24,38,0.75)";
  ctx.lineWidth = 3;
  ctx.stroke();
  // striking face band at the end that meets the chisel
  ctx.fillStyle = "#59646f";
  ctx.beginPath();
  ctx.moveTo(p[1].x, p[1].y);
  ctx.lineTo(p[1].x - ax * 12, p[1].y - ay * 12);
  ctx.lineTo(p[2].x - ax * 12, p[2].y - ay * 12);
  ctx.lineTo(p[2].x, p[2].y);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
  void hp; void x;
}

// ---------- sea ----------
const waveY = (x: number, t: number, base: number, amp: number, k: number, w: number, ph: number) =>
  base + amp * Math.sin(x * k + t * w + ph) + amp * 0.45 * Math.sin(x * k * 2.3 + t * w * 1.6 + ph * 1.7) + amp * 0.2 * Math.sin(x * k * 5.1 - t * w * 0.9 + ph * 0.3);

function band(ctx: CanvasRenderingContext2D, t: number, base: number, amp: number, k: number, w: number, ph: number, c0: string, c1: string, alpha = 1, x0 = -500, x1 = W + 500) {
  ctx.beginPath();
  ctx.moveTo(x0, base + 3000);
  for (let x = x0; x <= x1; x += 20) ctx.lineTo(x, waveY(x, t, base, amp, k, w, ph));
  ctx.lineTo(x1, base + 3000);
  ctx.closePath();
  const g = ctx.createLinearGradient(0, base - amp * 2, 0, base + 600);
  g.addColorStop(0, c0);
  g.addColorStop(1, c1);
  ctx.globalAlpha = alpha;
  ctx.fillStyle = g;
  ctx.fill();
  ctx.globalAlpha = 1;
}

export const S2Sea: React.FC = () => {
  const frame = useCurrentFrame();
  const zh = useZh();
  const f = frame - S2_PRE; // local frame, negative during the pre-roll
  const nut = useMemo(buildNut, []);
  const rel = useMemo(() => {
    const r = new Float32Array(nut.n);
    for (let i = 0; i < nut.n; i++) {
      const wy = NUT.y + (nut.py[i] - SH / 2 - 10) / SS;
      r[i] = wetFrame(Math.min(Math.max(wy, 560), 700)) + 12 + nut.rel[i] * 46;
    }
    return r;
  }, [nut]);
  const tmp = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = SW; c.height = SH;
    return c;
  }, []);
  const inkCanvas = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = W / 2; c.height = H / 2;
    return c;
  }, []);
  const inkCtx = inkCanvas.getContext("2d")!;
  const t = f / 30;

  // camera
  const sWide = 1.0, sMacro = 1.42, sMid = 1.2;
  const zoomIn = E.inOutCubic(clamp((f - 58) / 20));
  const pullBack = E.inOutCubic(clamp((f - 146) / 40));
  let cs = lerp(sWide, sMacro, zoomIn);
  cs = lerp(cs, sMid, pullBack);
  let ccx = lerp(960, 1100, zoomIn), ccy = lerp(540, 372, zoomIn);
  ccx = lerp(ccx, 960, pullBack);
  ccy = lerp(ccy, 520, pullBack);
  // follow the kernel down
  const kernelSink = Math.max(0, f - 300) * 1.9;
  const kernelY = NUT.y + kernelSink;
  const follow = E.inOutCubic(clamp((f - 285) / 80));
  cs = lerp(cs, 2.1, E.inOutCubic(clamp((f - 330) / 130)));
  ccx = lerp(ccx, NUT.x, follow);
  ccy = lerp(ccy, kernelY, follow);

  // shake
  let shake = 0;
  HITS.forEach((fh, i) => {
    const d = f - fh;
    if (d >= 0 && d < 14) shake = Math.max(shake, Math.exp(-d * 0.3) * (i === 2 ? 22 : 12 + i * 3));
  });
  const shx = (hash1(f * 3.1) - 0.5) * shake, shy = (hash1(f * 5.3 + 9) - 0.5) * shake;

  const draw = (ctx: CanvasRenderingContext2D, fr: number) => {
    const ff = fr - S2_PRE;
    const tt = ff / 30;
    const L = seaLevel(ff);
    ctx.save();
    ctx.translate(960 + shx, 540 + shy);
    ctx.scale(cs, cs);
    ctx.translate(-ccx, -ccy);

    // sky
    const horizon = L - 250;
    const sky = ctx.createLinearGradient(0, horizon - 700, 0, horizon + 30);
    sky.addColorStop(0, "#3d97c0");
    sky.addColorStop(0.45, "#76bcd8");
    sky.addColorStop(0.8, "#c3e4ef");
    sky.addColorStop(1, "#eef8fb");
    ctx.fillStyle = sky;
    ctx.fillRect(-1500, horizon - 3000, W + 3000, 3030);
    // sun haze
    const sun = ctx.createRadialGradient(1480, horizon - 300, 0, 1480, horizon - 300, 560);
    sun.addColorStop(0, "rgba(255,252,236,0.95)");
    sun.addColorStop(0.2, "rgba(255,248,225,0.45)");
    sun.addColorStop(1, "rgba(255,248,225,0)");
    ctx.fillStyle = sun;
    ctx.fillRect(850, horizon - 900, 1300, 1300);
    // soft clouds
    const cr = mulberry32(3);
    for (let i = 0; i < 7; i++) {
      const cx0 = ((cr() * 2400 - 300 + tt * (6 + i * 1.5)) % 2400) - 300;
      const cy0 = horizon - 130 - cr() * 380, rw = 260 + cr() * 340, rh = 30 + cr() * 38;
      const cg = ctx.createRadialGradient(cx0, cy0, 0, cx0, cy0, rw);
      cg.addColorStop(0, "rgba(255,255,255,0.55)");
      cg.addColorStop(1, "rgba(255,255,255,0)");
      ctx.save();
      ctx.translate(cx0, cy0); ctx.scale(1, rh / rw); ctx.translate(-cx0, -cy0);
      ctx.fillStyle = cg;
      ctx.fillRect(cx0 - rw, cy0 - rw, rw * 2, rw * 2);
      ctx.restore();
    }

    // far bands (they rise with the sea)
    band(ctx, tt, L - 250, 3, 0.011, 0.5, 0.2, "#cfe7ef", "#bfe0ea");
    band(ctx, tt, L - 195, 4, 0.0085, 0.65, 1.7, "#b5dae6", "#a3d0df");
    band(ctx, tt, L - 140, 5.5, 0.0072, 0.8, 3.1, "#97cadb", "#82bdd1");
    band(ctx, tt, L - 85, 7, 0.0061, 0.95, 4.9, "#79b7cd", "#62a6c0");

    // rock
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(560, 1500);
    ctx.bezierCurveTo(600, ROCK_TOP + 160, 700, ROCK_TOP + 20, 820, ROCK_TOP);
    ctx.lineTo(1100, ROCK_TOP);
    ctx.bezierCurveTo(1220, ROCK_TOP + 20, 1320, ROCK_TOP + 160, 1360, 1500);
    ctx.closePath();
    const rg = ctx.createLinearGradient(0, ROCK_TOP, 0, ROCK_TOP + 420);
    rg.addColorStop(0, "#a39a8a");
    rg.addColorStop(0.18, "#857b6d");
    rg.addColorStop(1, "#413b34");
    ctx.fillStyle = rg;
    ctx.fill();
    ctx.clip();
    const rr = mulberry32(8);
    for (let i = 0; i < 80; i++) {
      ctx.strokeStyle = `rgba(30,25,20,${0.06 + rr() * 0.12})`;
      ctx.lineWidth = 1 + rr() * 3;
      const x = 560 + rr() * 800, y = ROCK_TOP + rr() * 400;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + (rr() - 0.5) * 120, y + (rr() - 0.2) * 40); ctx.stroke();
    }
    const topl = ctx.createLinearGradient(0, ROCK_TOP - 6, 0, ROCK_TOP + 22);
    topl.addColorStop(0, "rgba(255,255,255,0.45)");
    topl.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = topl;
    ctx.fillRect(700, ROCK_TOP - 6, 520, 28);
    ctx.restore();

    // nut contact shadow
    ctx.fillStyle = "rgba(20,12,6,0.35)";
    ctx.beginPath();
    ctx.ellipse(NUT.x + 6, ROCK_TOP + 2, 96, 9, 0, 0, TAU);
    ctx.fill();

    // nut (with dissolving holes), wobble on hits
    let wob = 0, squash = 0;
    HITS.forEach((fh) => {
      const d = ff - fh;
      if (d >= 0 && d < 24) { wob += Math.sin(d * 0.9) * Math.exp(-d * 0.16) * 0.045; squash = Math.max(squash, Math.exp(-d * 0.35) * 0.035); }
    });
    const tctx = tmp.getContext("2d")!;
    tctx.setTransform(1, 0, 0, 1, 0, 0);
    tctx.clearRect(0, 0, SW, SH);
    tctx.globalCompositeOperation = "source-over";
    tctx.drawImage(nut.sprite, 0, 0);
    tctx.globalCompositeOperation = "destination-out";
    tctx.fillStyle = "#000";
    tctx.beginPath();
    for (let i = 0; i < nut.n; i++) if (rel[i] <= ff) { tctx.moveTo(nut.px[i] + 5.2, nut.py[i]); tctx.arc(nut.px[i], nut.py[i], 5.2, 0, TAU); }
    tctx.fill();
    tctx.globalCompositeOperation = "source-over";
    ctx.save();
    ctx.translate(NUT.x, ROCK_TOP);
    ctx.rotate(wob);
    ctx.scale(1 + squash, 1 - squash);
    ctx.translate(-NUT.x, -ROCK_TOP);
    ctx.drawImage(tmp, NUT.x - SW / 2 / SS, NUT.y - (SH / 2 + 10) / SS, SW / SS, SH / SS);
    ctx.restore();

    // cracks flash
    HITS.forEach((fh, hi) => {
      const d = ff - fh;
      if (d < 0 || d > 9) return;
      const r = mulberry32(100 + hi);
      ctx.strokeStyle = `rgba(255,${d < 3 ? 255 : 170},${d < 3 ? 255 : 120},${1 - d / 9})`;
      ctx.lineWidth = 2.4;
      for (let k = 0; k < 7; k++) {
        const ang = Math.atan2(-U.y, -U.x) + (r() - 0.5) * 2.2;
        let x = TIP.x, y = TIP.y;
        ctx.beginPath(); ctx.moveTo(x, y);
        const len = 24 + r() * 46;
        for (let s = 0; s < 4; s++) { x += Math.cos(ang + (r() - 0.5) * 0.9) * len / 4; y += Math.sin(ang + (r() - 0.5) * 0.9) * len / 4; ctx.lineTo(x, y); }
        ctx.stroke();
      }
    });

    // tools
    const toolIn = E.outCubic(clamp((ff - 54) / 12));
    const snapT = ff - HITS[2];
    if (ff > 50 && ff < 300) {
      const off = { x: U.x * (1 - toolIn) * 420, y: U.y * (1 - toolIn) * 420 };
      // chisel gets knocked in a little on each hit
      let pushIn = 0;
      HITS.forEach((fh) => { const d = ff - fh; if (d >= 0 && d < 8) pushIn += Math.sin((d / 8) * Math.PI) * 4; });
      const off2 = { x: off.x - U.x * pushIn, y: off.y - U.y * pushIn };
      drawChisel(ctx, off2, snapT >= 0 ? snapT : -1, ff);
      if (ff > 66 && ff < HITS[2] + 60) {
        // hammer motion blur: ghosts when it swings
        const a = hammerAngle(ff), a1 = hammerAngle(ff - 1);
        const sp = Math.abs(a - a1);
        if (sp > 0.03) for (let g = 3; g >= 1; g--) drawHammer(ctx, hammerAngle(ff - g * 0.8), 0.16);
        drawHammer(ctx, a, ff < HITS[2] + 36 ? 1 : 1 - clamp((ff - HITS[2] - 36) / 12));
      }
    }

    // impact flashes and sparks
    HITS.forEach((fh, hi) => {
      const d = ff - fh;
      if (d < 0 || d > 16) return;
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      const hx = HEAD.x - U.x * 8, hy = HEAD.y - U.y * 8;
      if (d < 6) {
        const k = 1 - d / 6;
        const rg2 = ctx.createRadialGradient(hx, hy, 0, hx, hy, 95 * (1 + hi * 0.4));
        rg2.addColorStop(0, `rgba(255,236,170,${0.85 * k})`);
        rg2.addColorStop(1, "rgba(255,170,70,0)");
        ctx.fillStyle = rg2;
        ctx.fillRect(hx - 220, hy - 220, 440, 440);
        ctx.strokeStyle = `rgba(255,240,200,${k})`;
        ctx.lineWidth = 3;
        for (let s = 0; s < 12; s++) {
          const a = (s / 12) * TAU + hi;
          const l0 = 20, l1 = 70 + (hash1(s + hi * 9) * 70) * (1 + hi * 0.3);
          ctx.beginPath(); ctx.moveTo(hx + Math.cos(a) * l0, hy + Math.sin(a) * l0); ctx.lineTo(hx + Math.cos(a) * l1, hy + Math.sin(a) * l1); ctx.stroke();
        }
      }
      const rs = mulberry32(300 + hi);
      for (let s = 0; s < 26; s++) {
        const a = -Math.PI / 2 + (rs() - 0.2) * 2.4 - 0.4;
        const v = 6 + rs() * 12;
        const life = 12 + rs() * 8;
        if (d > life) continue;
        const x = hx + Math.cos(a) * v * d, y = hy + Math.sin(a) * v * d + 0.6 * d * d;
        ctx.strokeStyle = `rgba(255,${190 + rs() * 50 | 0},120,${1 - d / life})`;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - Math.cos(a) * 8, y - Math.sin(a) * 8 - 2); ctx.stroke();
      }
      ctx.restore();
    });

    // water in front: surface band + volume
    // ripples from the hits (ellipses on the surface)
    HITS.forEach((fh, hi) => {
      const d = ff - fh;
      if (d < 0 || d > 40 || L < 400) return;
      const k = d / 40;
      ctx.strokeStyle = `rgba(255,255,255,${(1 - k) * 0.7})`;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.ellipse(NUT.x, L + 6, 60 + E.outCubic(k) * (360 + hi * 120), 6 + E.outCubic(k) * (24 + hi * 8), 0, 0, TAU);
      ctx.stroke();
    });
    // front surface: translucent so the submerged nut and rock are tinted
    const fy = (x: number) => waveY(x, tt, L, 5 + 9 * smoothstep(150, 260, ff), 0.0052, 1.1, 0.4);
    ctx.beginPath();
    ctx.moveTo(-500, L + 4000);
    for (let x = -500; x <= W + 500; x += 16) ctx.lineTo(x, fy(x));
    ctx.lineTo(W + 500, L + 4000);
    ctx.closePath();
    const vg = ctx.createLinearGradient(0, L - 20, 0, L + 1100);
    vg.addColorStop(0, "rgba(96,184,210,0.80)");
    vg.addColorStop(0.18, "rgba(40,126,160,0.9)");
    vg.addColorStop(1, "rgba(4,38,62,0.97)");
    ctx.fillStyle = vg;
    ctx.fill();
    // foam / glints along the surface
    ctx.strokeStyle = "rgba(255,255,255,0.75)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let x = -500; x <= W + 500; x += 16) { const y = fy(x); if (x === -500) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
    ctx.stroke();
    const gl = mulberry32(55);
    for (let i = 0; i < 70; i++) {
      const gx = gl() * (W + 600) - 300, off = gl() * 90 + 6, sp = 0.6 + gl();
      const a = 0.25 + 0.45 * Math.abs(Math.sin(tt * sp + i));
      ctx.strokeStyle = `rgba(255,255,255,${a * 0.6})`;
      ctx.lineWidth = 2;
      const y = fy(gx) + off * (0.4 + gl());
      ctx.beginPath(); ctx.moveTo(gx, y); ctx.lineTo(gx + 30 + gl() * 40, y); ctx.stroke();
    }

    // dissolving particles: an ink cloud (blurred pass) plus crisp flecks
    const ink = inkCtx;
    ink.setTransform(1, 0, 0, 1, 0, 0);
    ink.clearRect(0, 0, W / 2, H / 2);
    ink.translate(480 + shx / 2, 270 + shy / 2);
    ink.scale(cs / 2, cs / 2);
    ink.translate(-ccx, -ccy);
    for (let i = 0; i < nut.n; i++) {
      const r0 = rel[i];
      if (r0 > ff) continue;
      const age = ff - r0;
      const life = 70 + 90 * nut.rel[i];
      if (age > life) continue;
      const ox = NUT.x + (nut.px[i] - SW / 2) / SS, oy = NUT.y + (nut.py[i] - SH / 2 - 10) / SS;
      const dx = ox - NUT.x, dy = oy - NUT.y;
      const dl = Math.hypot(dx, dy) || 1;
      const spread = 1 - Math.exp(-age / 34);
      const x = ox + (dx / dl) * 70 * spread * (0.5 + nut.rel[i]) + Math.sin(age * 0.05 + nut.ph[i]) * 14 * spread;
      const y = oy + (dy / dl) * 50 * spread - 28 * spread * (0.4 + nut.rel[i]) + Math.cos(age * 0.045 + nut.ph[i] * 2) * 10 * spread;
      const k = age / life;
      const rr2 = nut.col[i * 3], gg = nut.col[i * 3 + 1], bb = nut.col[i * 3 + 2];
      const mixk = smoothstep(0, 0.75, k);
      const cr2 = lerp(rr2, 160, mixk) | 0, cg2 = lerp(gg, 210, mixk) | 0, cb2 = lerp(bb, 230, mixk) | 0;
      const a = (1 - k) * (1 - k);
      ink.fillStyle = `rgba(${cr2},${cg2},${cb2},${a * 0.55})`;
      const s2 = lerp(7, 15, k);
      ink.fillRect(x - s2 / 2, y - s2 / 2, s2, s2);
    }
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.filter = "blur(7px)";
    ctx.drawImage(inkCanvas, 0, 0, W, H);
    ctx.filter = "none";
    ctx.drawImage(inkCanvas, 0, 0, W, H);
    ctx.restore();

    ctx.restore();

    // ---------- screen-space: depth tint, shafts, bubbles, kernel ----------
    const deep = smoothstep(235, 430, ff);
    if (deep > 0) {
      ctx.fillStyle = `rgba(3,17,31,${deep * 0.9})`;
      ctx.fillRect(0, 0, W, H);
    }
    // light shafts fade with depth
    const shaftA = 0.2 * (1 - smoothstep(230, 420, ff)) * smoothstep(190, 250, ff) + 0.035;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < 9; i++) {
      const x0 = ((i * 263 + tt * 12 * (1 + (i % 3) * 0.4)) % (W + 500)) - 300;
      const w0 = 70 + (i % 4) * 40;
      const g = ctx.createLinearGradient(x0, 0, x0 + 400, H);
      g.addColorStop(0, `rgba(150,225,245,${shaftA})`);
      g.addColorStop(1, "rgba(150,225,245,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.moveTo(x0, 0); ctx.lineTo(x0 + w0, 0); ctx.lineTo(x0 + w0 + 380, H); ctx.lineTo(x0 + 120, H);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();

    // marine snow + bubbles streaming up as we sink
    const under = smoothstep(280, 340, ff);
    if (under > 0) {
      const br = mulberry32(77);
      for (let i = 0; i < 190; i++) {
        const size = 0.8 + br() * 3.2, sp = 0.4 + br() * 1.4 * (size / 3);
        const x = (br() * W + Math.sin(tt * 0.6 + i) * 14) % W;
        const rise = (ff - 280) * (1.4 + size * 0.5) * (ff > 300 ? 1 : 0.4);
        const y = ((br() * (H + 200) - rise * sp * 1.8) % (H + 200) + (H + 200)) % (H + 200) - 100;
        const bub = i % 7 === 0;
        ctx.globalAlpha = under * (bub ? 0.5 : 0.22 + br() * 0.3);
        if (bub) {
          ctx.strokeStyle = "rgba(210,245,255,0.9)";
          ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(x, y, size * 3.2, 0, TAU); ctx.stroke();
          ctx.fillStyle = "rgba(255,255,255,0.5)";
          ctx.beginPath(); ctx.arc(x - size, y - size, size * 0.7, 0, TAU); ctx.fill();
        } else {
          ctx.fillStyle = "#cfeaf5";
          ctx.beginPath(); ctx.arc(x, y, size * 0.8, 0, TAU); ctx.fill();
        }
      }
      ctx.globalAlpha = 1;
    }

    // the kernel: glows from inside the nut as the shell dissolves, then is all that is left
    const kGlow = smoothstep(190, 300, ff);
    if (kGlow > 0) {
      const kx = 960 + (NUT.x - ccx) * cs + shx, ky = 540 + (kernelY - ccy) * cs + shy;
      drawKernel(ctx, kx, ky, cs, kGlow, 0.85 + 0.15 * Math.sin(ff * 0.25));
    }
    bloom(ctx, W, H, { strength: 0.35, blur: 6 });
  };

  // captions
  const cap = (txt: string, at: number, color = "#0b2c3d") => (
    <RevealText
      text={txt}
      start={S2_PRE + at}
      stagger={0}
      dur={9}
      exit={S2_PRE + at + 26}
      exitDur={8}
      rise={30}
      blur={10}
      style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 118, color, textShadow: "0 0 30px rgba(255,255,255,0.7)" }}
    />
  );
  const night = smoothstep(230, 330, f);
  return (
    <AbsoluteFill style={{ background: "#dcedf3" }}>
      <Canvas2D draw={draw} />
      <AbsoluteFill style={{ left: 150, top: 200 }}>
        <div style={{ position: "absolute", left: 0, top: 0 }}>{cap("Hammer.", HITS[0] + 1)}</div>
        <div style={{ position: "absolute", left: 0, top: 0 }}>{cap("Chisel.", HITS[1] + 1)}</div>
        <div style={{ position: "absolute", left: 0, top: 0 }}>{cap("Nothing.", HITS[2] + 1)}</div>
      </AbsoluteFill>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", top: -260 }}>
        <RevealText text="Or raise the sea." start={S2_PRE + 168} stagger={7} dur={16} exit={S2_PRE + 232} exitDur={14}
          style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 104, color: "#0b2c3d", textShadow: "0 0 30px rgba(255,255,255,0.8)" }} />
      </AbsoluteFill>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "flex-end", paddingBottom: zh ? 150 : 190, opacity: night }}>
        <RevealText text="The problem doesn’t break. It dissolves." start={S2_PRE + 330} stagger={4} dur={14} exit={S2_PRE + 436} exitDur={14}
          style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 62, color: "rgba(244,247,251,0.92)" }} />
        {zh && (
          <>
            <div style={{ height: 16 }} />
            <RevealText text="不是敲开难题，而是让海面升高。" start={S2_PRE + 350} chars stagger={2} dur={14} blur={8} rise={10} exit={S2_PRE + 440} exitDur={14}
              style={{ fontFamily: F.cn, fontSize: 30, color: "rgba(244,247,251,0.6)", letterSpacing: "0.4em" }} />
          </>
        )}
      </AbsoluteFill>
      <VolumeCard book={BOOKS[0]} start={S2_PRE} dur={66} />
    </AbsoluteFill>
  );
};
