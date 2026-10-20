import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Canvas2D } from "../components/Canvas2D";
import { drawKernel } from "../components/kernel";
import { RevealText } from "../components/RevealText";
import { VolumeCard } from "../components/VolumeCard";
import { BOOKS } from "../data";
import { project, worldCam } from "../world/camera";
import { BALMER, curveX, curveY, CURVE_GMAX, SPEC_H, SPEC_Y0, specX } from "../world/shapes";
import { C, F } from "../theme";
import { E, clamp, hash1, smoothstep } from "../lib/math";
import { useZh } from "../lang";

const GREEK = ["α", "β", "γ", "δ"];
const LAM = BALMER.map((b) => b.lam);
// printed labels: the standard tabulated wavelengths of H-alpha ... H-delta (geometry still comes from the Balmer formula)
const LABEL_NM = [656.3, 486.1, 434.0, 410.2];

/** Overlays for volume 2 (world time f = frame - 0). */
export const S3Atom: React.FC = () => {
  const f = useCurrentFrame();
  const zh = useZh();
  const cam = worldCam(f);
  const pj = (x: number, y: number, z = 0) => project(x, y, z, f, cam);

  // ---- spectrum labels (frontal view) ----
  const specShow = smoothstep(228, 240, f) * (1 - smoothstep(296, 312, f));
  const heights = [1.0, 0.64, 0.46, 0.34];
  const specLabels = [0, 1, 2, 3].map((i) => {
    const [x, y] = pj(specX(LAM[i]), SPEC_Y0 + heights[i] * SPEC_H + 0.0);
    const [, yb] = pj(specX(LAM[i]), SPEC_Y0);
    const t = E.outBack(clamp((f - 236 - i * 8) / 14), 1.6);
    return (
      <div key={i} style={{ position: "absolute", left: x, top: y - 70, transform: `translate(-50%, ${(1 - t) * 14}px)`, opacity: clamp(t) * specShow, textAlign: "center" }}>
        <div style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 44, color: "#fff", textShadow: `0 0 18px ${C.balmer[i]}` }}>H<span style={{ fontSize: 44 }}>{GREEK[i]}</span></div>
        <div style={{ position: "absolute", left: "50%", top: yb - y + 70 + 14, transform: "translateX(-50%)", fontFamily: F.mono, fontSize: 19, letterSpacing: "0.08em", color: C.balmer[i], whiteSpace: "nowrap" }}>
          {LABEL_NM[i].toFixed(1)} nm
        </div>
      </div>
    );
  });
  const [limX, limY] = pj(specX(364.56), SPEC_Y0);

  // ---- double descent overlay ----
  const curveShow = smoothstep(338, 352, f) * (1 - smoothstep(428, 446, f));
  const gx = (g: number) => pj(curveX(g), curveY(g));
  const [ax0, ay0] = pj(curveX(0) - 0.35, -2.55);
  const [ax1] = pj(curveX(CURVE_GMAX) + 0.45, -2.55);
  const [, ay1] = pj(curveX(0) - 0.35, 2.55);
  const [thX, thY] = gx(1);
  const nDots = 34;
  const dots = Array.from({ length: nDots }, (_, i) => {
    const g = 0.12 + (i / (nDots - 1)) * 4.2;
    const base = curveY(g);
    // simulation: follows theory away from the threshold, is cut off (finite size) near it
    const near = Math.exp(-Math.pow((g - 1) / 0.16, 2));
    const y = base - near * 0.9 * (0.5 + hash1(i * 3.3)) + (hash1(i * 7.1) - 0.5) * 0.11;
    const [px, py] = pj(curveX(g) + (hash1(i * 1.9) - 0.5) * 0.04, y);
    const t = E.outBack(clamp((f - 360 - i * 0.9) / 10), 2);
    return { px, py, t };
  });

  const lines1 = (
    <div style={{ position: "absolute", left: 150, bottom: zh ? 150 : 176 }}>
      <RevealText text="Physics has its hydrogen atom." start={74} stagger={4} exit={150} exitDur={12}
        style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 64, color: "rgba(244,247,251,0.95)" }} />
      {zh && (
        <>
          <div style={{ height: 8 }} />
          <RevealText text="物理学有氢原子。" start={92} chars stagger={3} exit={150} exitDur={12} blur={8} rise={10}
            style={{ fontFamily: F.cn, fontSize: 28, color: "rgba(244,247,251,0.6)", letterSpacing: "0.4em" }} />
        </>
      )}
    </div>
  );
  const lines2 = (
    <div style={{ position: "absolute", left: 0, right: 0, bottom: 120, textAlign: "center" }}>
      <RevealText text="Simple enough to solve exactly." start={238} stagger={4} exit={300} exitDur={12}
        style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 58, color: "rgba(244,247,251,0.95)" }} />
      <div style={{ height: 6 }} />
      <RevealText text="Wrong in just the right ways." start={262} stagger={4} exit={300} exitDur={12}
        style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 58, color: C.balmer[0], textShadow: "0 0 24px rgba(255,75,62,0.45)" }} />
    </div>
  );
  const lines3 = (
    <div style={{ position: "absolute", left: 860, top: 110 }}>
      <RevealText text="Deep learning gets sixteen." start={348} stagger={5} dur={16} exit={428} exitDur={14}
        style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 70, color: "#fff" }} />
      <div style={{ height: 18 }} />
      <RevealText text="SIXTEEN SOLVABLE MODELS · ONE NOTATION" start={376} stagger={0} dur={18} exit={428} exitDur={14} rise={14} blur={6}
        style={{ fontFamily: F.sans, fontSize: 24, letterSpacing: "0.32em", color: "rgba(244,247,251,0.6)", fontWeight: 600 }} />
    </div>
  );

  return (
    <AbsoluteFill>
      <Canvas2D
        draw={(ctx, fr, w, h) => {
          // the kernel from the end of volume 1 carries over, then dissolves into the nucleus of the atom
          const k = 1 - smoothstep(6, 52, fr);
          if (k <= 0.001) return;
          ctx.fillStyle = `rgba(5,25,43,${0.9 * k})`;
          ctx.fillRect(0, 0, w, h);
          const halo = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, 760);
          halo.addColorStop(0, `rgba(30,110,160,${0.55 * k})`);
          halo.addColorStop(1, "rgba(30,110,160,0)");
          ctx.fillStyle = halo;
          ctx.fillRect(0, 0, w, h);
          drawKernel(ctx, w / 2, h / 2, 2.1, 1 - 0.55 * smoothstep(20, 52, fr), 0.85 + 0.15 * Math.sin((480 + fr) * 0.25));
        }}
      />
      {lines1}
      {lines2}
      {lines3}
      {specLabels}
      <div style={{ position: "absolute", left: limX - 8, top: limY + 66, transform: "translateX(0)", opacity: specShow * smoothstep(262, 276, f), fontFamily: F.mono, fontSize: 18, letterSpacing: "0.1em", color: "rgba(190,170,255,0.9)", whiteSpace: "nowrap" }}>
        364.6 nm · series limit
      </div>
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{ position: "absolute", left: 0, top: 0, opacity: curveShow }}>
        <g stroke="rgba(190,225,255,0.55)" strokeWidth={2} fill="none">
          <line x1={ax0} y1={ay0} x2={ax1} y2={ay0} />
          <line x1={ax0} y1={ay0} x2={ax0} y2={ay1} />
        </g>
        <path d={`M ${ax1} ${ay0} l -14 -7 l 0 14 z`} fill="rgba(190,225,255,0.7)" />
        <path d={`M ${ax0} ${ay1} l -7 14 l 14 0 z`} fill="rgba(190,225,255,0.7)" />
        <line x1={thX} y1={ay0} x2={thX} y2={thY} stroke="rgba(255,210,170,0.7)" strokeWidth={2} strokeDasharray="8 8" />
        <text x={thX + 18} y={thY + 150} fontFamily={F.mono} fontSize={20} letterSpacing={2} fill="rgba(255,210,170,0.95)">interpolation threshold</text>
        <text x={ax1 - 8} y={ay0 + 40} textAnchor="end" fontFamily={F.serif} fontStyle="italic" fontSize={30} fill="rgba(220,238,255,0.85)">model size  p / n</text>
        <text x={ax0 + 16} y={ay1 + 4} fontFamily={F.serif} fontStyle="italic" fontSize={30} fill="rgba(220,238,255,0.85)">test error</text>
        {dots.map((d, i) => (
          <g key={i} opacity={clamp(d.t)} transform={`translate(${d.px} ${d.py}) scale(${clamp(d.t, 0, 1.4)})`}>
            <circle r={14} fill="rgba(255,236,170,0.20)" />
            <circle r={6.2} fill="#ffe9a8" />
          </g>
        ))}
      </svg>
      <div style={{ position: "absolute", left: 860, top: 292, opacity: curveShow * smoothstep(372, 386, f), fontFamily: F.mono, fontSize: 23, letterSpacing: "0.08em", color: "rgba(255,233,168,0.95)" }}>
        ● simulation <span style={{ color: "rgba(120,225,255,0.95)", marginLeft: 18 }}>— theory</span>
      </div>
      <VolumeCard book={BOOKS[1]} start={0} dur={66} flip titleSize={82} />
    </AbsoluteFill>
  );
};
void E;
