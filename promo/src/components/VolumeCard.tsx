import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Book } from "../data";
import { Book3D } from "./Book3D";
import { F } from "../theme";
import { E, clamp, lerp } from "../lib/math";

/** Chapter-card for a volume: the real book spins in beside its title, over a dimmed world. */
export const VolumeCard: React.FC<{ book: Book; start?: number; dur?: number; flip?: boolean; dim?: number; titleSize?: number; light?: boolean }> = ({
  book,
  start = 0,
  dur = 64,
  flip = false,
  dim = 0.6,
  titleSize = 104,
  light = false,
}) => {
  const frame = useCurrentFrame();
  const f = frame - start;
  if (f < -2 || f > dur + 2) return null;
  const enter = E.outCubic(clamp(f / 24));
  const spin = E.outBack(clamp(f / 30), 1.1);
  const leave = E.inCubic(clamp((f - (dur - 16)) / 16));
  const op = enter * (1 - leave);
  const dir = flip ? -1 : 1;
  const bx = (flip ? 540 : 1380) + dir * (1 - enter) * 260 + dir * leave * 260;
  const tx = (flip ? 1010 : 150) - dir * (1 - enter) * 120 - dir * leave * 140;
  const row = (i: number) => {
    const t = E.outCubic(clamp((f - 4 - i * 5) / 18));
    return { opacity: t * (1 - leave), transform: `translateY(${(1 - t) * 34}px)`, filter: `blur(${(1 - t) * 8}px)` } as React.CSSProperties;
  };
  return (
    <AbsoluteFill style={{ opacity: 1 }}>
      <AbsoluteFill style={{ background: light ? `rgba(250,247,240,${0.62 * op})` : `rgba(3,16,28,${dim * op})`, backdropFilter: `blur(${(light ? 5 : 7) * op}px)` }} />
      <div style={{ position: "absolute", left: bx - 220, top: 545 - 293, opacity: op }}>
        {/* soft contact shadow */}
        <div style={{ position: "absolute", left: 40, top: 600, width: 400, height: 60, borderRadius: "50%", background: `radial-gradient(ellipse, rgba(0,0,0,${light ? 0.28 : 0.55}), rgba(0,0,0,0) 70%)`, filter: "blur(10px)" }} />
        <div style={{ perspective: 1800 }}>
          <Book3D
            book={book}
            rotY={lerp(flip ? 100 : -100, flip ? 26 : -26, spin)}
            rotX={lerp(14, 5, enter)}
            sheen={clamp((f - 18) / 40)}
          />
        </div>
      </div>
      <div style={{ position: "absolute", left: tx, top: 330, width: 900, textAlign: "left", color: light ? "#1c2430" : "#f4f7fb" }}>
        <div style={{ ...row(0), fontFamily: F.sans, fontSize: 25, letterSpacing: "0.46em", color: light ? "#b03a24" : book.accent, fontWeight: 600, marginBottom: 26 }}>
          VOLUME {book.roman}
        </div>
        {book.lines.map((l, i) => (
          <div key={i} style={{ ...row(1 + i), fontFamily: F.serif, fontSize: titleSize, lineHeight: 1.04, fontWeight: 400, letterSpacing: "-0.005em" }}>
            {l}
          </div>
        ))}
        <div style={{ ...row(3), fontFamily: F.serif, fontStyle: "italic", fontSize: 40, marginTop: 24, color: light ? "rgba(28,36,48,0.72)" : "rgba(244,247,251,0.72)" }}>
          {book.subtitle}
        </div>
        <div style={{ ...row(4), fontFamily: F.sans, fontSize: 22, letterSpacing: "0.2em", marginTop: 34, color: light ? "rgba(28,36,48,0.5)" : "rgba(244,247,251,0.5)" }}>
          {book.pages} PAGES · WRITTEN BY CLAUDE
        </div>
      </div>
    </AbsoluteFill>
  );
};
