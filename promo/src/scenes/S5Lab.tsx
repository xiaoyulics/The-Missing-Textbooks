import React, { useMemo } from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Canvas2D } from "../components/Canvas2D";
import { RevealText } from "../components/RevealText";
import { VolumeCard } from "../components/VolumeCard";
import { BOOKS } from "../data";
import { F } from "../theme";
import { E, clamp, hash1, lerp, mulberry32, smoothstep, TAU } from "../lib/math";

const RUB = "#b03a24";
const RUB_HOT = "#d9482b";
const INK = "#1c2430";
const BLUE = "#1c365e";
const PAPER = "#faf7f0";
const GRID = "#de7c54";

// ------------------------------------------------------------------ paper
const Paper: React.FC<{ wash?: number }> = ({ wash = 0 }) => (
  <AbsoluteFill
    style={{
      background: PAPER,
      backgroundImage: [
        `repeating-linear-gradient(0deg, rgba(222,124,84,0.075) 0 1px, transparent 1px 12px)`,
        `repeating-linear-gradient(90deg, rgba(222,124,84,0.075) 0 1px, transparent 1px 12px)`,
        `repeating-linear-gradient(0deg, rgba(222,124,84,0.17) 0 1.5px, transparent 1.5px 60px)`,
        `repeating-linear-gradient(90deg, rgba(222,124,84,0.17) 0 1.5px, transparent 1.5px 60px)`,
        `radial-gradient(ellipse at 50% 40%, rgba(255,255,255,${0.5 - wash}) 0%, rgba(255,255,255,0) 70%)`,
      ].join(","),
    }}
  />
);

// ------------------------------------------------------------------ stamp
const InkFilter: React.FC = () => (
  <svg width={0} height={0} style={{ position: "absolute" }}>
    <defs>
      <filter id="ink" x="-8%" y="-12%" width="116%" height="124%">
        <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="3" seed="4" result="n1" />
        <feDisplacementMap in="SourceGraphic" in2="n1" scale="3.2" result="d" />
        <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" seed="9" result="n2" />
        <feColorMatrix in="n2" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 -16 12.6" result="m" />
        <feComposite in="d" in2="m" operator="in" />
      </filter>
    </defs>
  </svg>
);

const StampMark: React.FC<{ word: string; size?: number; color?: string }> = ({ word, size = 1, color = RUB_HOT }) => {
  const fs = 74 * size;
  const w = word.length * fs * 0.74 + 90 * size;
  const h = fs * 1.42;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ overflow: "visible", mixBlendMode: "multiply" }}>
      <g filter="url(#ink)" fill="none" stroke={color} strokeWidth={5 * size}>
        <rect x={4} y={4} width={w - 8} height={h - 8} rx={10 * size} />
        <rect x={16 * size} y={16 * size} width={w - 32 * size} height={h - 32 * size} rx={5 * size} strokeWidth={2.4 * size} />
        <text x={w / 2} y={h / 2 + fs * 0.34} textAnchor="middle" fontFamily={F.sans} fontWeight={800} fontSize={fs} letterSpacing={fs * 0.1} fill={color} stroke="none">
          {word}
        </text>
      </g>
    </svg>
  );
};

/** a stamp that slams down at frame `hit` (local g), positioned at (x, y) */
const Stamp: React.FC<{ g: number; hit: number; x: number; y: number; rot: number; word: string; size?: number; keep?: boolean }> = ({
  g, hit, x, y, rot, word, size = 1, keep = true,
}) => {
  const t = g - hit;
  if (t < -1) return null;
  const k = clamp(t / 5);
  const sc = lerp(2.1, 1, E.outExpo(k)) * (t >= 5 ? 1 + 0.025 * Math.exp(-(t - 5) * 0.35) * Math.sin((t - 5) * 1.6) : 1);
  const op = k < 1 ? E.outQuad(clamp((t + 1) / 3)) : 1;
  const sp = mulberry32(Math.floor(hit * 7 + x));
  return (
    <div style={{ position: "absolute", left: x, top: y, transform: `translate(-50%,-50%) rotate(${rot}deg) scale(${sc})`, opacity: keep ? op : op * (1 - smoothstep(30, 44, t)) }}>
      <StampMark word={word} size={size} />
      {/* ink spatter */}
      {t >= 2 && t < 40 &&
        Array.from({ length: 14 }, (_, i) => {
          const a = sp() * TAU, r = (60 + sp() * 90) * size * (1 - Math.exp(-t * 0.5));
          return (
            <div key={i} style={{ position: "absolute", left: "50%", top: "50%", width: 3 + sp() * 6, height: 3 + sp() * 6, borderRadius: "50%", background: RUB_HOT, opacity: 0.35 * (1 - t / 40), transform: `translate(${Math.cos(a) * r * 1.6}px, ${Math.sin(a) * r}px)` }} />
          );
        })}
    </div>
  );
};

// ------------------------------------------------------------------ ten runs
const RUNS = 10;
const N_PTS = 64;
function runCurve(seed: number, method: number): number[] {
  const r = mulberry32(seed * 131 + method * 7919 + 3);
  const off = (r() - 0.5) * 0.3;
  const ph = [r() * TAU, r() * TAU, r() * TAU, r() * TAU];
  const amp = [0.08, 0.055, 0.04, 0.025].map((a) => a * (0.7 + r() * 0.8));
  const out: number[] = [];
  for (let i = 0; i < N_PTS; i++) {
    const t = i / (N_PTS - 1);
    const base = 0.1 + 0.9 * Math.pow(1 + t * 14, -0.62);
    let n = 0;
    for (let k = 0; k < 4; k++) n += amp[k] * Math.sin(TAU * (0.9 * (k + 1)) * t + ph[k]);
    out.push(Math.max(0.03, base * (1 + off * (0.4 + 0.6 * t)) + n * (1 - 0.5 * t)));
  }
  return out;
}
const ENDV = (c: number[]) => c[N_PTS - 1];
function pickSingles() {
  // first run of each method such that "A beats B" on a single run, convincingly
  for (let s = 1; s < 400; s++) {
    const a = runCurve(s, 0), b = runCurve(s + 1000, 1);
    if (ENDV(b) - ENDV(a) > 0.03 && Math.abs(a[20] - b[20]) < 0.12) return [s, s + 1000];
  }
  return [1, 1001];
}

const RunsPlot: React.FC<{ g: number }> = ({ g }) => {
  const [sa, sb] = useMemo(pickSingles, []);
  const A = useMemo(() => Array.from({ length: RUNS }, (_, i) => runCurve(i === 0 ? sa : 50 + i, 0)), [sa]);
  const B = useMemo(() => Array.from({ length: RUNS }, (_, i) => runCurve(i === 0 ? sb : 90 + i, 1)), [sb]);
  const X0 = 250, X1 = 1500, Y0 = 290, Y1 = 830; // plot box
  const px = (t: number) => lerp(X0, X1, t);
  const py = (v: number) => lerp(Y1, Y0, clamp((v - 0.02) / 1.0));
  const path = (c: number[], upto: number) => {
    const n = Math.max(2, Math.floor(upto * (N_PTS - 1)) + 1);
    let d = "";
    for (let i = 0; i < n; i++) d += `${i ? "L" : "M"}${px(i / (N_PTS - 1))},${py(c[i])} `;
    return d;
  };
  const appear = E.outCubic(clamp((g - 62) / 14));
  const leave = E.inCubic(clamp((g - 158) / 16));
  const drawA = E.outCubic(clamp((g - 76) / 26)), drawB = E.outCubic(clamp((g - 88) / 26));
  const more = E.outCubic(clamp((g - 128) / 18));
  const merge = E.inOutCubic(clamp((g - 142) / 12));
  const op = appear * (1 - leave);
  if (op < 0.01) return null;
  return (
    <AbsoluteFill style={{ opacity: op, transform: `translateY(${leave * 40}px)` }}>
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{ position: "absolute", left: 0, top: 0 }}>
        {/* axes */}
        <g stroke={INK} strokeWidth={2.4} fill="none" opacity={0.85}>
          <path d={`M${X0},${Y0 - 20} L${X0},${Y1} L${X1 + 30},${Y1}`} />
        </g>
        <text x={X1 + 30} y={Y1 + 44} textAnchor="end" fontFamily={F.mono} fontSize={22} letterSpacing={2} fill="rgba(28,36,48,0.65)">STEPS →</text>
        <text x={X0 - 14} y={Y0 - 34} fontFamily={F.mono} fontSize={22} letterSpacing={2} fill="rgba(28,36,48,0.65)">LOSS</text>
        {/* nine more runs each */}
        {A.slice(1).map((c, i) => <path key={"a" + i} d={path(c, 1)} fill="none" stroke={RUB_HOT} strokeWidth={2.8} opacity={0.62 * more} />)}
        {B.slice(1).map((c, i) => <path key={"b" + i} d={path(c, 1)} fill="none" stroke={BLUE} strokeWidth={2.8} opacity={0.62 * more} />)}
        {/* the first run of each method: the one everyone would have published */}
        <path d={path(B[0], drawB)} fill="none" stroke={BLUE} strokeWidth={5.5} strokeLinecap="round" />
        <path d={path(A[0], drawA)} fill="none" stroke={RUB_HOT} strokeWidth={5.5} strokeLinecap="round" />
        <text x={px(1) + 22} y={py(ENDV(A[0])) + 8} fontFamily={F.sans} fontWeight={800} fontSize={30} letterSpacing={3} fill={RUB_HOT} opacity={drawA}>A</text>
        <text x={px(1) + 22} y={py(ENDV(B[0])) + 8} fontFamily={F.sans} fontWeight={800} fontSize={30} letterSpacing={3} fill={BLUE} opacity={drawB}>B</text>
        {/* the spread, once there are ten */}
        <g opacity={more * (1 - merge * 0.0)}>
          <line x1={px(1) + 70} y1={py(Math.min(...A.map(ENDV)))} x2={px(1) + 70} y2={py(Math.max(...B.map(ENDV)))} stroke={INK} strokeWidth={2.4} />
          <line x1={px(1) + 60} y1={py(Math.min(...A.map(ENDV)))} x2={px(1) + 80} y2={py(Math.min(...A.map(ENDV)))} stroke={INK} strokeWidth={2.4} />
          <line x1={px(1) + 60} y1={py(Math.max(...B.map(ENDV)))} x2={px(1) + 80} y2={py(Math.max(...B.map(ENDV)))} stroke={INK} strokeWidth={2.4} />
          <text x={px(1) + 92} y={(py(Math.min(...A.map(ENDV))) + py(Math.max(...B.map(ENDV)))) / 2 + 8} fontFamily={F.mono} fontSize={22} fill={INK}>seed</text>
        </g>
      </svg>
      <div style={{ position: "absolute", left: 250, top: 150 }}>
        {g < 118 ? (
          <RevealText text="Method A beats method B." start={92} stagger={4} exit={116} exitDur={8} style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 62, color: INK }} />
        ) : (
          <RevealText text="Ten runs each. Same two methods." start={122} stagger={3} dur={12} style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 62, color: INK }} />
        )}
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ cards
type Claim = { text: string[]; word: string; x: number; y: number; rot: number; hit: number; srot: number; sx: number; sy: number };
const CLAIMS: Claim[] = [
  { text: ["Adaptive methods generalize", "worse than SGD."], word: "REVERSED", x: 540, y: 330, rot: -4, hit: 195, srot: -8, sx: 40, sy: 70 },
  { text: ["Vision transformers beat", "convolutional networks."], word: "CONFOUNDED", x: 1380, y: 300, rot: 3, hit: 225, srot: 6, sx: -20, sy: 80 },
  { text: ["Abilities emerge abruptly", "and unpredictably."], word: "ARTIFACT", x: 520, y: 730, rot: 2.5, hit: 255, srot: -5, sx: 30, sy: 76 },
  { text: ["L(N,D) = 1.69 + 406.4/N^0.34", "+ 410.7/D^0.28"], word: "UNREPLICATED", x: 1390, y: 740, rot: -3, hit: 285, srot: 5, sx: -10, sy: 84 },
  { text: ["Batch normalization reduces", "internal covariate shift."], word: "MISATTRIBUTED", x: 960, y: 520, rot: 1.5, hit: 315, srot: -6, sx: 10, sy: 82 },
];
const CW = 600, CH = 330;

const CardBody: React.FC<{ c: Claim; t: number; idx: number }> = ({ c, t, idx }) => {
  const p = E.outCubic(clamp(t / 22)); // effect progress after the stamp
  const lineY = [92, 134];
  const text = c.text.map((l, i) => (
    <text key={i} x={36} y={lineY[i]} fontFamily={F.serif} fontSize={idx === 3 ? 31 : 34} fill={INK}>{l}</text>
  ));
  let extra: React.ReactNode = null;
  if (idx === 1) {
    // CONFOUNDED: two strands that were independent get braided together
    const pts = (ph: number) => {
      let d = "";
      for (let x = 0; x <= 520; x += 6) {
        const u = x / 520;
        const amp = lerp(2, 30, p) * (0.4 + 0.6 * Math.sin(Math.PI * u));
        const y = 232 + Math.sin(u * TAU * lerp(1.0, 5.5, p) + ph) * amp;
        d += `${x ? "L" : "M"}${40 + x},${y} `;
      }
      return d;
    };
    extra = (
      <>
        <path d={pts(0)} fill="none" stroke={BLUE} strokeWidth={5} strokeLinecap="round" />
        <path d={pts(Math.PI)} fill="none" stroke={RUB_HOT} strokeWidth={5} strokeLinecap="round" />
        <text x={40} y={206} fontFamily={F.mono} fontSize={19} fill={BLUE}>architecture</text>
        <text x={560} y={206} textAnchor="end" fontFamily={F.mono} fontSize={19} fill={RUB_HOT}>training recipe</text>
      </>
    );
  } else if (idx === 2) {
    // ARTIFACT: the cliff is the metric; change the metric and it is a smooth curve
    const pathAt = (m: number) => {
      let d = "";
      for (let x = 0; x <= 480; x += 5) {
        const u = x / 480;
        const smooth = 1 / (1 + Math.exp(-(u - 0.55) * 7));
        const step = u < 0.62 ? 0.03 : 0.03 + Math.min(1, (u - 0.62) * 14) * 0.97;
        const v = lerp(step, smooth, m);
        d += `${x ? "L" : "M"}${60 + x},${262 - v * 100} `;
      }
      return d;
    };
    extra = (
      <>
        <line x1={60} y1={266} x2={560} y2={266} stroke={INK} strokeWidth={2} opacity={0.6} />
        <path d={pathAt(p)} fill="none" stroke={BLUE} strokeWidth={5} strokeLinecap="round" />
        <text x={60} y={296} fontFamily={F.mono} fontSize={18} fill="rgba(28,36,48,0.7)">{p < 0.5 ? "metric: exact match" : "metric: log-likelihood"}</text>
      </>
    );
  } else if (idx === 3) {
    // UNREPLICATED: handled by clones outside the body
    extra = <text x={36} y={200} fontFamily={F.mono} fontSize={19} fill="rgba(28,36,48,0.6)">fit on a published curve</text>;
  } else if (idx === 4) {
    // MISATTRIBUTED: the effect is real, the explanation is swapped
    extra = (
      <>
        <text x={36} y={206} fontFamily={F.mono} fontSize={20} fill="rgba(28,36,48,0.7)">effect: faster training  ✓</text>
        <text x={36} y={246} fontFamily={F.mono} fontSize={20} fill={INK} opacity={1}>because: </text>
        <text x={142} y={246} fontFamily={F.mono} fontSize={20} fill={INK} opacity={1 - p}>reduces covariate shift</text>
        <line x1={140} y1={240} x2={140 + 292 * p} y2={240} stroke={RUB_HOT} strokeWidth={3} />
        <text x={142} y={246} fontFamily={F.mono} fontSize={20} fill={RUB_HOT} opacity={p} transform={`translate(0 ${(1 - p) * 10})`}>
          </text>
        <text x={142} y={278} fontFamily={F.mono} fontSize={20} fontWeight={700} fill={RUB_HOT} opacity={p}>smooths the loss landscape</text>
      </>
    );
  } else if (idx === 0) {
    extra = <text x={36} y={206} fontFamily={F.mono} fontSize={19} fill="rgba(28,36,48,0.6)">generalization gap, tuned fairly →</text>;
  }
  return (
    <svg width={CW} height={CH} viewBox={`0 0 ${CW} ${CH}`}>
      <text x={36} y={52} fontFamily={F.mono} fontSize={17} letterSpacing={4} fill={RUB}>THE CLAIM</text>
      {text}
      {extra}
    </svg>
  );
};

const Card: React.FC<{ c: Claim; idx: number; g: number }> = ({ c, idx, g }) => {
  const t = g - c.hit;
  const inT = clamp((g - (c.hit - 18)) / 14);
  if (inT <= 0) return null;
  const e = E.outBack(inT, 1.1);
  const fromX = idx % 2 === 0 ? -900 : 900;
  const fall = E.inCubic(clamp((g - (386 + idx * 3)) / 20));
  let rot = c.rot + (1 - e) * (idx % 2 ? 14 : -14);
  let sc = 1;
  let op = 1;
  if (idx === 0) rot += 180 * E.outBack(clamp(t / 20), 1.3);
  if (idx === 3 && t > 0) op = 1;
  // shadow lift before the stamp
  const lift = Math.exp(-Math.max(0, t) * 0.4);
  const clones = idx === 3 && t > 6 ? [0, 1, 2] : [];
  // no numbers here on purpose: the point is only that an independent refit does not land on the published constants
  const cloneText = [["independent refit", "constants differ"], ["independent refit", "constants differ"], ["independent refit", "constants differ"]];
  return (
    <>
      {clones.map((k) => {
        const u = E.outCubic(clamp((t - 6 - k * 4) / 16));
        return (
          <div key={k} style={{ position: "absolute", left: c.x + (k - 1) * 70 * u - 0, top: c.y + (130 + 54 * k) * u, width: CW * 0.82, height: 110, transform: `translate(-50%, -50%) rotate(${(k - 1) * 3}deg)`, opacity: u * (1 - fall) }}>
            <div style={{ width: "100%", height: "100%", background: "#fffdf6", border: "1.5px dashed rgba(28,36,48,0.35)", boxShadow: "0 6px 18px rgba(0,0,0,0.12)", padding: "14px 22px", fontFamily: F.serif, fontSize: 24, color: INK, lineHeight: 1.3, position: "relative" }}>
              {cloneText[k][0]}<br />{cloneText[k][1]}
              <span style={{ position: "absolute", right: 14, top: 8, fontFamily: F.sans, fontWeight: 800, fontSize: 26, color: RUB_HOT }}>≠</span>
            </div>
          </div>
        );
      })}
      <div style={{ position: "absolute", left: c.x, top: c.y, width: CW, height: CH, transform: `translate(calc(-50% + ${(1 - e) * fromX}px), calc(-50% + ${fall * 1300}px)) rotate(${rot}deg) scale(${sc})`, opacity: op }}>
        <div style={{ position: "absolute", inset: 0, background: "#fffdf8", borderRadius: 6, boxShadow: `0 ${10 + 18 * lift}px ${24 + 26 * lift}px rgba(40,24,10,${0.16 + 0.1 * lift}), 0 1px 0 rgba(0,0,0,0.05)`, backgroundImage: "repeating-linear-gradient(0deg, transparent 0 39px, rgba(90,130,190,0.16) 39px 40px)", borderTop: `6px solid ${RUB}` }} />
        <div style={{ position: "absolute", inset: 0, transform: idx === 0 && t > 0 ? undefined : undefined }}>
          <CardBody c={c} t={t} idx={idx} />
        </div>
      </div>
    </>
  );
};

// letters turning to dust: used for the VANISHED stamp on the runs plot
const VanishDust: React.FC<{ g: number }> = ({ g }) => {
  const t0 = 150;
  const parts = useMemo(() => {
    const r = mulberry32(31);
    return Array.from({ length: 900 }, () => ({ x: 250 + r() * 1250, y: 290 + r() * 540, vx: 1 + r() * 6, vy: -1.2 - r() * 3, s: 2 + r() * 4, d: r() * 10, c: r() < 0.5 ? RUB_HOT : BLUE }));
  }, []);
  return (
    <Canvas2D
      draw={(ctx, frame) => {
        const t = frame - t0;
        if (t < 0 || t > 50) return;
        for (const p of parts) {
          const a = t - p.d;
          if (a < 0) continue;
          const k = a / 34;
          if (k > 1) continue;
          ctx.globalAlpha = (1 - k) * 0.7;
          ctx.fillStyle = p.c;
          ctx.fillRect(p.x + p.vx * a * 1.4 + Math.sin(a * 0.3 + p.d) * 6, p.y + p.vy * a * 1.2 + a * a * 0.04, p.s, p.s);
        }
        ctx.globalAlpha = 1;
      }}
    />
  );
};

// ------------------------------------------------------------------ the extrapolation band
const Scaling: React.FC<{ g: number }> = ({ g }) => {
  const t0 = 396;
  const pIn = E.outCubic(clamp((g - t0) / 22));
  const leave = E.inCubic(clamp((g - 452) / 12));
  if (pIn <= 0 || leave >= 1) return null;
  const X0 = 240, X1 = 1680, Y0 = 250, Y1 = 830;
  const lx = (u: number) => lerp(X0, X1, u);
  const f = (u: number) => 700 - 400 * Math.pow(u, 0.55) * 1.0; // power law on log-log looks like a line
  const dataU = [0.02, 0.14, 0.26, 0.38, 0.5, 0.62];
  const fitEnd = 0.64;
  const bandAt = (u: number) => 8 + Math.max(0, u - fitEnd) * 340 + 6;
  const drawFit = E.outCubic(clamp((g - t0 - 6) / 22));
  const drawExt = E.inOutCubic(clamp((g - t0 - 20) / 26));
  let ext = "", up = "", dn = "";
  const nE = Math.floor(drawExt * 40);
  for (let i = 0; i <= nE; i++) {
    const u = fitEnd + ((1 - fitEnd) * i) / 40;
    ext += `${i ? "L" : "M"}${lx(u)},${f(u)} `;
    up += `${i ? "L" : "M"}${lx(u)},${f(u) - bandAt(u)} `;
  }
  for (let i = nE; i >= 0; i--) {
    const u = fitEnd + ((1 - fitEnd) * i) / 40;
    dn += `L${lx(u)},${f(u) + bandAt(u)} `;
  }
  let fit = "";
  for (let i = 0; i <= Math.floor(drawFit * 40); i++) {
    const u = (fitEnd * i) / 40;
    fit += `${i ? "L" : "M"}${lx(u)},${f(u)} `;
  }
  return (
    <AbsoluteFill style={{ opacity: pIn * (1 - leave) }}>
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{ position: "absolute", left: 0, top: 0 }}>
        <g stroke={INK} strokeWidth={2.4} fill="none"><path d={`M${X0},${Y0 - 30} L${X0},${Y1} L${X1 + 20},${Y1}`} /></g>
        <path d={`${up} ${dn} Z`} fill={RUB_HOT} opacity={0.14} />
        <path d={ext} fill="none" stroke={RUB} strokeWidth={4} strokeDasharray="14 10" />
        <path d={fit} fill="none" stroke={RUB} strokeWidth={5} />
        {dataU.map((u, i) => {
          const k = E.outBack(clamp((g - t0 - 4 - i * 3) / 10), 2);
          const y = f(u) + Math.sin(i * 7.3) * 9;
          return (
            <g key={i} opacity={clamp(k)} transform={`translate(${lx(u)} ${y})`}>
              <line x1={0} y1={-18} x2={0} y2={18} stroke={BLUE} strokeWidth={3} />
              <line x1={-8} y1={-18} x2={8} y2={-18} stroke={BLUE} strokeWidth={3} />
              <line x1={-8} y1={18} x2={8} y2={18} stroke={BLUE} strokeWidth={3} />
              <circle r={7} fill={BLUE} />
            </g>
          );
        })}
        <text x={lx(0.33)} y={f(0.33) + 112} fontFamily={F.mono} fontSize={21} letterSpacing={2} fill="rgba(28,36,48,0.7)" opacity={drawFit}>measured</text>
        <text x={lx(0.78)} y={f(0.78) + 130} fontFamily={F.mono} fontSize={21} letterSpacing={2} fill={RUB} opacity={drawExt}>extrapolated</text>
        {/* here be dragons */}
        <g opacity={smoothstep(0.8, 1, drawExt)} transform={`translate(${lx(0.88)} ${f(0.88) - 98})`}>
          <path d={Array.from({ length: 40 }, (_, i) => `${i ? "L" : "M"}${i * 4},${Math.sin(i * 0.5 + g * 0.18) * (4 + i * 0.4) * 0.8}`).join(" ")} fill="none" stroke={RUB} strokeWidth={4} strokeLinecap="round" />
          <circle cx={166} cy={Math.sin(39 * 0.5 + g * 0.18) * 14.6} r={9} fill={RUB} />
          <circle cx={169} cy={Math.sin(39 * 0.5 + g * 0.18) * 14.6 - 2} r={2.4} fill={PAPER} />
        </g>
        <text x={lx(0.88)} y={f(0.88) - 128} fontFamily={F.serif} fontStyle="italic" fontSize={34} fill={RUB} opacity={smoothstep(0.85, 1, drawExt)}>here be dragons</text>
      </svg>
      <div style={{ position: "absolute", left: 240, top: 96 }}>
        <RevealText text="Extrapolation is a claim, too." start={t0 + 30} stagger={4} exit={448} exitDur={10} style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 58, color: INK }} />
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ the scene
const STORM = [
  { w: "NARROWED", x: 700, y: 210, r: -9, s: 0.9, at: 330 },
  { w: "REVERSED", x: 1500, y: 560, r: 8, s: 0.8, at: 334 },
  { w: "VANISHED", x: 360, y: 560, r: 6, s: 1.0, at: 337 },
  { w: "CONFOUNDED", x: 1200, y: 840, r: -6, s: 0.9, at: 341 },
  { w: "ARTIFACT", x: 520, y: 880, r: 9, s: 0.8, at: 345 },
  { w: "UNREPLICATED", x: 1580, y: 230, r: -7, s: 0.8, at: 349 },
  { w: "MISATTRIBUTED", x: 960, y: 170, r: 4, s: 0.85, at: 352 },
  { w: "NARROWED", x: 1000, y: 700, r: -11, s: 1.1, at: 356 },
  { w: "VANISHED", x: 1700, y: 880, r: 7, s: 0.8, at: 360 },
  { w: "REVERSED", x: 260, y: 300, r: -5, s: 1.0, at: 364 },
  { w: "CONFOUNDED", x: 800, y: 540, r: 10, s: 1.15, at: 367 },
  { w: "ARTIFACT", x: 1400, y: 400, r: -8, s: 1.0, at: 371 },
];

export const S5Lab: React.FC = () => {
  const g = useCurrentFrame();
  // screen shake on every hit
  const hits = [150, ...CLAIMS.map((c) => c.hit), ...STORM.map((s) => s.at), 435];
  let shake = 0;
  hits.forEach((h) => {
    const d = g - h;
    if (d >= 0 && d < 10) shake = Math.max(shake, Math.exp(-d * 0.38) * (h === 435 ? 26 : 10));
  });
  const sx = (hash1(g * 2.3) - 0.5) * shake, sy = (hash1(g * 4.1 + 3) - 0.5) * shake;
  const fall = E.inCubic(clamp((g - 386) / 22));
  const flood = E.inQuart(clamp((g - 452) / 26));
  const bigIn = g >= 435;

  return (
    <AbsoluteFill style={{ background: PAPER }}>
      <InkFilter />
      <AbsoluteFill style={{ transform: `translate(${sx}px, ${sy}px)` }}>
        <Paper />
        <RunsPlot g={g} />
        <VanishDust g={g} />
        <Stamp g={g} hit={150} x={880} y={560} rot={-7} word="VANISHED" size={1.7} keep={false} />
        {CLAIMS.map((c, i) => <Card key={i} c={c} idx={i} g={g} />)}
        {CLAIMS.map((c, i) => (
          <div key={"s" + i} style={{ transform: `translateY(${fall * 1300}px)` }}>
            <Stamp g={g} hit={c.hit} x={c.x + c.sx + 80} y={c.y + c.sy + 50} rot={c.srot} word={c.word} size={0.9} />
          </div>
        ))}
        {STORM.map((s, i) => (
          <div key={"st" + i} style={{ transform: `translateY(${fall * 1500}px)` }}>
            <Stamp g={g} hit={s.at} x={s.x} y={s.y} rot={s.r} word={s.w} size={s.s} />
          </div>
        ))}
        <Scaling g={g} />
        {bigIn && <Stamp g={g} hit={435} x={960} y={500} rot={-5} word="OPEN" size={3.1} />}
      </AbsoluteFill>
      <AbsoluteFill style={{ background: RUB_HOT, opacity: flood, clipPath: `circle(${flood * 1700}px at 960px 500px)` }} />
      <VolumeCard book={BOOKS[3]} start={0} dur={66} light titleSize={88} />
    </AbsoluteFill>
  );
};
