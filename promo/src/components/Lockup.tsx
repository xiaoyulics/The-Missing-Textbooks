import React from "react";
import { getLockup } from "../lib/lockup";
import { F } from "../theme";
import { E, clamp } from "../lib/math";

/**
 * THE / MISSING (a dashed outline: the word is missing) / TEXTBOOKS
 * `t` is the local frame since the lockup started revealing.
 */
export const Lockup: React.FC<{ t: number; cx?: number; cy?: number; scale?: number; opacity?: number; march?: number }> = ({
  t, cx = 960, cy = 300, scale = 0.62, opacity = 1, march = 0,
}) => {
  const L = getLockup();
  const top = L.the.y - L.the.size * 0.8, bottom = L.tb.y + 20;
  const mid = (top + bottom) / 2;
  const reveal = (d: number, dur = 16) => E.outCubic(clamp((t - d) / dur));
  const rows = [
    { d: 0, k: reveal(0) },
    { d: 6, k: reveal(6) },
    { d: 12, k: reveal(12) },
  ];
  const lift = (k: number) => `translate(0 ${(1 - k) * 34})`;
  const common = { fontFamily: F.serif, textAnchor: "middle" as const };
  return (
    <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{ position: "absolute", left: 0, top: 0, opacity, filter: "drop-shadow(0 0 22px rgba(110,190,255,0.32))" }}>
      <g transform={`translate(${cx} ${cy}) scale(${scale}) translate(-960 ${-mid})`}>
        <g opacity={rows[0].k} transform={lift(rows[0].k)}>
          <text x={L.the.x + L.the.ls / 2} y={L.the.y} fontSize={L.the.size} fontWeight={L.the.weight} letterSpacing={L.the.ls} fill="rgba(244,247,251,0.9)" {...common}>THE</text>
        </g>
        <g opacity={rows[1].k} transform={lift(rows[1].k)}>
          <text
            x={L.miss.x + L.miss.ls / 2} y={L.miss.y} fontSize={L.miss.size} fontWeight={L.miss.weight} letterSpacing={L.miss.ls}
            fill="none" stroke="#c9eeff" strokeWidth={4.6} strokeDasharray="18 11" strokeDashoffset={-march * 0.9} {...common}
            style={{ filter: "drop-shadow(0 0 10px rgba(120,210,255,0.7))" }}
          >MISSING</text>
        </g>
        <g opacity={rows[2].k} transform={lift(rows[2].k)}>
          <text x={L.tb.x + L.tb.ls / 2} y={L.tb.y} fontSize={L.tb.size} fontWeight={L.tb.weight} letterSpacing={L.tb.ls} fill="#f4f7fb" {...common}>TEXTBOOKS</text>
        </g>
      </g>
    </svg>
  );
};
