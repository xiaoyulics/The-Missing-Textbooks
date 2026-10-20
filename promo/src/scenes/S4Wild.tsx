import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { RevealText } from "../components/RevealText";
import { VolumeCard } from "../components/VolumeCard";
import { BOOKS } from "../data";
import { project, worldCam } from "../world/camera";
import { RACERS, racerPos, RACE_F0, STEPS, stepAt } from "../world/terrain";
import { F } from "../theme";
import { E, clamp, smoothstep, TAU } from "../lib/math";

const CREAM = "#f3ead2";

const Legend: React.FC<{ g: number }> = ({ g }) => {
  const racers = RACERS();
  const fade = 1 - 0.7 * smoothstep(110, 130, g) - 0.3 * smoothstep(262, 282, g);
  const head = E.outCubic(clamp((g - 64) / 16));
  return (
    <div style={{ position: "absolute", left: 96, top: 215, width: 720, opacity: fade * head }}>
      <div style={{ fontFamily: F.mono, fontSize: 22, letterSpacing: "0.16em", color: "rgba(243,234,210,0.6)", marginBottom: 8 }}>
        FIG. 1 — FOUR OPTIMIZERS, ONE LANDSCAPE
      </div>
      <div style={{ height: 1, background: "rgba(243,234,210,0.28)", marginBottom: 26, width: 600 }} />
      {racers.map((r, i) => {
        const t = E.outCubic(clamp((g - 70 - i * 9) / 16));
        return (
          <div key={r.name} style={{ opacity: t, transform: `translateX(${(1 - t) * -26}px)`, marginBottom: 24, display: "flex", gap: 18, alignItems: "flex-start" }}>
            <div style={{ width: 14, height: 14, borderRadius: 7, background: r.color, boxShadow: `0 0 16px ${r.color}`, marginTop: 11 }} />
            <div>
              <div style={{ fontFamily: F.sans, fontWeight: 700, fontSize: 29, letterSpacing: "0.16em", color: r.color }}>{r.name.toUpperCase()}</div>
              <div style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 31, color: CREAM, marginTop: 1 }}>{r.latin}</div>
              <div style={{ fontFamily: F.serif, fontSize: 25, color: "rgba(243,234,210,0.66)", marginTop: 2 }}>{r.note}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

const Tags: React.FC<{ g: number }> = ({ g }) => {
  const f = g + 480;
  if (f < RACE_F0 + 4) return null;
  const cam = worldCam(f);
  const s = stepAt(f);
  const racers = RACERS();
  const offs: [number, number][] = [[-70, -58], [70, 54], [-84, 50], [60, -62]];
  const op = smoothstep(RACE_F0 + 4, RACE_F0 + 24, f) * (1 - smoothstep(740, 770, f));
  return (
    <>
      {racers.map((r, i) => {
        const p = racerPos(i, s);
        const [x, y] = project(p[0], p[1], p[2], f, cam);
        const [ox, oy] = offs[i];
        return (
          <React.Fragment key={r.name}>
            <svg width={1920} height={1080} style={{ position: "absolute", left: 0, top: 0, opacity: op * 0.8 }}>
              <line x1={x} y1={y} x2={x + ox} y2={y + oy} stroke={r.color} strokeWidth={1.6} />
            </svg>
            <div style={{ position: "absolute", left: x + ox, top: y + oy, transform: `translate(${ox < 0 ? "-100%" : "0"}, ${oy < 0 ? "-100%" : "0"})`, opacity: op, fontFamily: F.sans, fontWeight: 700, fontSize: 24, letterSpacing: "0.14em", color: r.color, textShadow: "0 0 14px rgba(0,0,0,0.8)", padding: "0 8px" }}>
              {r.name.toUpperCase()}
            </div>
          </React.Fragment>
        );
      })}
    </>
  );
};

const Counter: React.FC<{ g: number }> = ({ g }) => {
  const f = g + 480;
  const s = Math.round(stepAt(f));
  const op = smoothstep(RACE_F0 - 6, RACE_F0 + 8, f) * (1 - smoothstep(762, 786, f));
  return (
    <div style={{ position: "absolute", left: 96, bottom: 92, opacity: op, fontFamily: F.mono, fontSize: 30, letterSpacing: "0.12em", color: CREAM, fontVariantNumeric: "tabular-nums" }}>
      <span style={{ color: "rgba(243,234,210,0.5)" }}>STEP </span>
      {String(s).padStart(4, "0")}
      <span style={{ color: "rgba(243,234,210,0.5)" }}> / {STEPS}</span>
    </div>
  );
};

/** two unit balls: the l2 circle (SGD) and the l-infinity square (sign descent, Adam, Lion) */
const NormPanel: React.FC<{ g: number }> = ({ g }) => {
  const t0 = 292;
  const enter = E.outCubic(clamp((g - t0) / 22));
  const leave = E.inCubic(clamp((g - 428) / 16));
  const op = enter * (1 - leave);
  if (op <= 0.001) return null;
  const R = 112;
  const ang = 0.55 + ((g - t0) / 150) * TAU;
  const dx = Math.cos(ang), dy = Math.sin(ang);
  const trail = (kind: "c" | "s") => {
    const pts: string[] = [];
    for (let k = 0; k <= 26; k++) {
      const a = ang - (k / 26) * 0.9;
      const ex = Math.cos(a), ey = Math.sin(a);
      const x = kind === "c" ? ex * R : Math.tanh(7 * ex) * R / Math.tanh(7) ;
      const y = kind === "c" ? ey * R : Math.tanh(7 * ey) * R / Math.tanh(7);
      pts.push(`${x},${-y}`);
    }
    return pts.join(" ");
  };
  const sCx = kindX(dx), sCy = kindX(dy);
  function kindX(v: number) { return Math.tanh(7 * v) * R / Math.tanh(7); }
  const panel = (cx: number, kind: "c" | "s", color: string, title: string, sub: string) => (
    <g transform={`translate(${cx} 0)`}>
      <g transform="translate(0 0)">
        <line x1={-R * 1.35} y1={0} x2={R * 1.35} y2={0} stroke="rgba(243,234,210,0.16)" />
        <line x1={0} y1={-R * 1.35} x2={0} y2={R * 1.35} stroke="rgba(243,234,210,0.16)" />
        {kind === "c" ? (
          <circle r={R} fill={`${color}14`} stroke={color} strokeWidth={3} />
        ) : (
          <rect x={-R} y={-R} width={2 * R} height={2 * R} fill={`${color}14`} stroke={color} strokeWidth={3} />
        )}
        <polyline points={trail(kind)} fill="none" stroke={color} strokeWidth={3} opacity={0.5} strokeLinecap="round" />
        <line x1={0} y1={0} x2={dx * R * 0.82} y2={-dy * R * 0.82} stroke="rgba(243,234,210,0.9)" strokeWidth={2.4} />
        <circle cx={dx * R * 0.82} cy={-dy * R * 0.82} r={4} fill={CREAM} />
        <circle cx={kind === "c" ? dx * R : sCx} cy={kind === "c" ? -dy * R : -sCy} r={13} fill={color} opacity={0.25} />
        <circle cx={kind === "c" ? dx * R : sCx} cy={kind === "c" ? -dy * R : -sCy} r={7} fill="#fff" />
      </g>
      <text x={0} y={R + 58} textAnchor="middle" fontFamily={F.sans} fontWeight={700} fontSize={25} letterSpacing={3.5} fill={color}>{title}</text>
      <text x={0} y={R + 92} textAnchor="middle" fontFamily={F.serif} fontStyle="italic" fontSize={26} fill={CREAM}>{sub}</text>
    </g>
  );
  return (
    <AbsoluteFill style={{ opacity: op, transform: `translateY(${(1 - enter) * 60 + leave * 40}px)` }}>
      <div style={{ position: "absolute", right: 96, top: 120, width: 920, height: 800, borderRadius: 28, background: "linear-gradient(180deg, rgba(8,16,22,0.72), rgba(6,12,16,0.82))", border: "1px solid rgba(243,234,210,0.16)", backdropFilter: "blur(14px)", boxShadow: "0 30px 80px rgba(0,0,0,0.5)" }} />
      <div style={{ position: "absolute", right: 150, top: 160, width: 820 }}>
        <div style={{ fontFamily: F.mono, fontSize: 20, letterSpacing: "0.3em", color: "rgba(243,234,210,0.55)" }}>ONE RULE</div>
        <div style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 54, lineHeight: 1.12, color: "#fff", marginTop: 18 }}>
          Every optimizer is steepest descent—<br />under some norm.
        </div>
      </div>
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{ position: "absolute", left: 0, top: 0 }}>
        <g transform="translate(1450 660)">
          {panel(-210, "c", "#ff6b5e", "ℓ₂ BALL", "SGD")}
          {panel(210, "s", "#c78bff", "ℓ∞ BALL", "sign · Adam · Lion")}
        </g>
      </svg>
      <div style={{ position: "absolute", right: 150, bottom: 118, fontFamily: F.serif, fontSize: 24, color: "rgba(243,234,210,0.6)", opacity: smoothstep(t0 + 40, t0 + 60, g) }}>
        … and the spectral norm gives Muon and Shampoo.
      </div>
    </AbsoluteFill>
  );
};

/** Overlays for volume 3 (local frame g = frame in the volume; world time f = g + 480). */
export const S4Wild: React.FC = () => {
  const g = useCurrentFrame();
  return (
    <AbsoluteFill>
      <div style={{ position: "absolute", left: 0, top: 0, width: 900, height: 1080, background: "linear-gradient(90deg, rgba(2,8,12,0.78), rgba(2,8,12,0))", opacity: smoothstep(60, 80, g) * (1 - smoothstep(262, 290, g)) }} />
      <Legend g={g} />
      <Tags g={g} />
      <Counter g={g} />
      <NormPanel g={g} />
      <VolumeCard book={BOOKS[2]} start={0} dur={66} titleSize={104} />
      <AbsoluteFill style={{ background: "#faf7f0", opacity: smoothstep(466, 479, g) }} />
    </AbsoluteFill>
  );
};
void RevealText;
