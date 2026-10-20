import React, { useMemo } from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { LensCanvas, LensFrame } from "../components/LensCanvas";
import { Canvas2D } from "../components/Canvas2D";
import { RevealText } from "../components/RevealText";
import { MiniBooks } from "../components/MiniBooks";
import { useZh } from "../lang";
import { bloom } from "../lib/bloom";
import { fontOf, getLockup } from "../lib/lockup";
import { C, F } from "../theme";
import { E, clamp, hash1, mulberry32, smoothstep, TAU } from "../lib/math";

const TAN = Math.tan((38 * Math.PI) / 360);

type P = { tx: number; ty: number; kind: number; a: number; r: number; d: number; sw: number; col: number; vx: number; vy: number };

/** Particles sampled from the filled lockup. kind: 0 = THE/TEXTBOOKS, 1 = MISSING */
function buildParticles(): P[] {
  const L = getLockup();
  const cv = document.createElement("canvas");
  cv.width = 1920;
  cv.height = 1080;
  const ctx = cv.getContext("2d")!;
  const rnd = mulberry32(5);
  const out: P[] = [];
  const draw = (line: typeof L.the, kind: number) => {
    ctx.clearRect(0, 0, 1920, 1080);
    ctx.font = fontOf(line);
    (ctx as any).letterSpacing = `${line.ls}px`;
    ctx.textAlign = "center";
    ctx.fillStyle = "#fff";
    ctx.fillText(line.text, line.x + line.ls / 2, line.y);
    const im = ctx.getImageData(0, 0, 1920, 1080).data;
    const pts: [number, number][] = [];
    for (let y = 0; y < 1080; y += 1) for (let x = 0; x < 1920; x += 1) if (im[(y * 1920 + x) * 4 + 3] > 127) pts.push([x, y]);
    const want = kind === 1 ? 9000 : kind === 0 && line === L.tb ? 7000 : 1200;
    const p = Math.min(1, want / pts.length);
    for (const [x, y] of pts) {
      if (rnd() > p) continue;
      out.push({
        tx: x, ty: y, kind,
        a: rnd() * TAU, r: 30 + 1450 * Math.pow(rnd(), 1.6), d: rnd() * 10, sw: rnd() - 0.5,
        col: rnd(), vx: (rnd() - 0.5) * 1.6, vy: -(1.2 + rnd() * 3.4),
      });
    }
  };
  draw(L.the, 0);
  draw(L.tb, 0);
  draw(L.miss, 1);
  return out;
}

const DIVE0 = 150, DIVE1 = 240;
const BURST = 12;
const ARRIVE = 45;
const colorOf = (c: number, kind: number) =>
  c < 0.1 ? "255,120,86" : c < 0.62 ? "120,205,255" : c < 0.8 ? "236,248,255" : "255,226,170";

export const S1Title: React.FC = () => {
  const frame = useCurrentFrame();
  const zh = useZh();
  const t = frame / 30;
  const L = getLockup();
  const parts = useMemo(buildParticles, []);

  // backdrop: the wall again, calmer; drifts slowly forward
  const lf: LensFrame = {
    time: t,
    cam0: [0, 0, -t * 0.5],
    cam1: [0, 0, -t * 0.5 - 0.02],
    yaw: 0, pitch: 0, roll: 0, tanHalf: TAN, samples: 1,
    slabs: [],
    wallScroll: [t * 0.003 + 0.2, t * 0.0015],
    wallBright: 0.62 - 0.22 * smoothstep(ARRIVE, ARRIVE + 30, frame),
    light: 0.9 * (1 - smoothstep(0, 30, frame)),
    flash: 1 - smoothstep(0, 16, frame),
    shake: [0, 0],
  };

  // ---- the dive: fall into the second O of TEXTBOOKS; the world of volume 1 is inside it ----
  const O = L.oCenter;
  const du = clamp((frame - DIVE0) / (DIVE1 - DIVE0));
  const ZMAX = 46;
  const Z = Math.exp(Math.log(ZMAX) * Math.pow(du, 2.1));
  const wT = smoothstep(0.0, 0.6, du);
  const Tx = O.x + (960 - O.x) * wT, Ty = O.y + (540 - O.y) * wT;
  lf.hole = du > 0 ? [Tx, Ty, O.rx * Z * 1.09 + 2, O.ry * Z * 1.09 + 2] : undefined;
  const diving = du > 0;

  const impact = frame - ARRIVE;
  const shake = impact >= 0 && impact < 16 ? Math.exp(-impact * 0.28) : 0;
  const sx = (hash1(frame * 1.7) - 0.5) * 26 * shake;
  const sy = (hash1(frame * 2.9 + 4) - 0.5) * 20 * shake;

  const drain = clamp((frame - (ARRIVE + 8)) / 26);
  const missTop = L.miss.y - L.miss.size * 0.72;
  const missH = L.miss.size * 0.74;
  const titleIn = smoothstep(ARRIVE - 4, ARRIVE + 6, frame);

  const drawParticles = (ctx: CanvasRenderingContext2D, f: number, w: number, h: number) => {
    ctx.globalCompositeOperation = "lighter";
    const cx = w / 2, cy = h / 2 + 20;
    const fade = (1 - 0.82 * smoothstep(ARRIVE, ARRIVE + 14, f)) * (1 - smoothstep(DIVE0 - 6, DIVE0 + 6, f));
    for (const p of parts) {
      const burstE = E.outCubic(clamp(f / BURST));
      const bx = cx + Math.cos(p.a) * p.r * burstE, by = cy + Math.sin(p.a) * p.r * burstE * 0.7;
      const travel = clamp((f - BURST - p.d - (p.tx / w) * 6) / (ARRIVE - BURST - 12));
      const e = E.inOutCubic(travel);
      // curved flight: sideways bulge that vanishes on arrival
      const dx = p.tx - bx, dy = p.ty - by;
      const bulge = Math.sin(Math.PI * e) * p.sw * 0.7;
      let x = bx + dx * e - dy * bulge, y = by + dy * e + dx * bulge;
      if (f > ARRIVE) {
        x += (hash1(p.tx * 0.37 + f * 0.9) - 0.5) * 1.4;
        y += (hash1(p.ty * 0.91 + f * 1.3) - 0.5) * 1.4;
      }
      // MISSING drains into dust
      let alpha = fade;
      if (p.kind === 1 && f > ARRIVE + 8) {
        const release = ARRIVE + 8 + clamp((p.ty - missTop) / missH) * 26;
        const age = f - release;
        if (age > 0) {
          x += p.vx * age * 1.4;
          y += p.vy * age * 1.2;
          alpha = Math.max(0, 1 - age / 30) * 0.9;
        } else alpha = 0;
      }
      if (alpha <= 0.01) continue;
      // trail while flying
      const speedy = f < ARRIVE + 2 && travel < 1;
      const col = colorOf(p.col, p.kind);
      if (speedy) {
        const e2 = E.inOutCubic(clamp((f - 1 - BURST - p.d - (p.tx / w) * 6) / (ARRIVE - BURST - 12)));
        const burstE2 = E.outCubic(clamp((f - 1) / BURST));
        const bx2 = cx + Math.cos(p.a) * p.r * burstE2, by2 = cy + Math.sin(p.a) * p.r * burstE2 * 0.7;
        const bulge2 = Math.sin(Math.PI * e2) * p.sw * 0.7;
        const x2 = bx2 + (p.tx - bx2) * e2 - (p.ty - by2) * bulge2, y2 = by2 + (p.ty - by2) * e2 + (p.tx - bx2) * bulge2;
        ctx.strokeStyle = `rgba(${col},${0.2 * alpha})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x2 + (x2 - x) * 1.8, y2 + (y2 - y) * 1.8);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
      ctx.fillStyle = `rgba(${col},${alpha * 0.62})`;
      ctx.fillRect(x - 1, y - 1, 2.2, 2.2);
    }
    // shock ring on impact
    if (impact >= 0 && impact < 34) {
      const k = impact / 34;
      for (let i = 0; i < 3; i++) {
        const kk = clamp(k - i * 0.06);
        ctx.strokeStyle = `rgba(190,235,255,${(1 - kk) * 0.5 * (1 - i * 0.3)})`;
        ctx.lineWidth = 3 - i;
        ctx.beginPath();
        ctx.arc(cx, h * 0.43, 140 + E.outExpo(kk) * 1500, 0, TAU);
        ctx.stroke();
      }
    }
    bloom(ctx, w, h, { strength: f < ARRIVE ? 0.55 : 0.9, blur: 4 });
  };

  const marchOff = -frame * 0.9;
  const pulseK = Math.exp(-6 * ((frame % 15) / 15));
  const svgCommon = { fontFamily: F.serif, textAnchor: "middle" as const };
  return (
    <AbsoluteFill style={{ background: "transparent" }}>
      <LensCanvas frame={lf} />
      <AbsoluteFill style={{ transform: `translate(${sx}px, ${sy}px)` }}>
        <Canvas2D draw={drawParticles} />
        <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{ position: "absolute", left: 0, top: 0, opacity: titleIn, filter: diving ? undefined : "drop-shadow(0 0 28px rgba(110,190,255,0.35))" }}>
          <defs>
            <clipPath id="drain">
              <rect x={0} y={missTop + drain * missH} width={1920} height={missH * (1 - drain) + 80} />
            </clipPath>
          </defs>
          <g transform={diving ? `translate(${Tx} ${Ty}) scale(${Z}) translate(${-O.x} ${-O.y})` : undefined}>
          <text x={L.the.x + L.the.ls / 2} y={L.the.y} fontSize={L.the.size} fontWeight={L.the.weight} letterSpacing={L.the.ls} fill="rgba(244,247,251,0.9)" {...svgCommon}>THE</text>
          <g fill="#f4f7fb" fontSize={L.tb.size} fontWeight={L.tb.weight} fontFamily={F.serif}>
            <text x={L.tbParts.leftX} y={L.tb.y} textAnchor="start">TEXTBO</text>
            <text x={L.tbParts.oCell} y={L.tb.y} textAnchor="middle">O</text>
            <text x={L.tbParts.rightX} y={L.tb.y} textAnchor="start">KS</text>
          </g>
          {/* MISSING: filled, then drained into a dashed outline */}
          <g clipPath="url(#drain)">
            <text x={L.miss.x + L.miss.ls / 2} y={L.miss.y} fontSize={L.miss.size} fontWeight={L.miss.weight} letterSpacing={L.miss.ls} fill="#f4f7fb" {...svgCommon}>MISSING</text>
          </g>
          <text
            x={L.miss.x + L.miss.ls / 2} y={L.miss.y} fontSize={L.miss.size} fontWeight={L.miss.weight} letterSpacing={L.miss.ls}
            fill="none" stroke="#c9eeff" strokeWidth={4.2} strokeDasharray="18 11" strokeDashoffset={marchOff}
            opacity={smoothstep(ARRIVE + 4, ARRIVE + 20, frame) * (0.85 + 0.15 * pulseK)}
            style={{ filter: "drop-shadow(0 0 10px rgba(120,210,255,0.7))" }} {...svgCommon}
          >MISSING</text>
          </g>
        </svg>
      </AbsoluteFill>

      <MiniBooks start={75} y={zh ? 905 : 840} leave={DIVE0 - 8} />
      <AbsoluteFill style={{ alignItems: "center" }}>
        {zh && (
          <div style={{ position: "absolute", top: L.tb.y + 44 }}>
            <RevealText text="不存在的教科书" start={73} chars stagger={3} dur={16} blur={10} rise={14} exit={DIVE0 - 10} exitDur={8}
              style={{ fontFamily: F.cn, fontWeight: 900, fontSize: 78, letterSpacing: "0.5em", color: "#f4f7fb", paddingLeft: "0.5em" }} />
          </div>
        )}
        {/* the tagline rises into the Chinese line's place when there is none */}
        <div style={{ position: "absolute", top: L.tb.y + (zh ? 150 : 72) }}>
          <RevealText text="A series of textbooks that should exist, but don’t." start={zh ? 97 : 80} stagger={3} dur={14} exit={DIVE0 - 10} exitDur={8}
            style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: zh ? 38 : 44, color: "rgba(244,247,251,0.82)" }} />
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
