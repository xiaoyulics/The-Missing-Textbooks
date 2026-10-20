import React from "react";
import { useCurrentFrame } from "remotion";
import { E, clamp } from "../lib/math";
import { C, F } from "../theme";

export const VOLS = [
  { n: 1, roman: "I", title: "Dissolving Problems", pages: 333, a: "#2aa3c4", b: "#0a5f7e" },
  { n: 2, roman: "II", title: "The Hydrogen Atoms of Deep Learning", pages: 394, a: "#ff5a45", b: "#7a4cff" },
  { n: 3, roman: "III", title: "Optimizers in the Wild", pages: 369, a: "#e8bd5a", b: "#5a8f4e" },
  { n: 4, roman: "IV", title: "Deep Learning as an Experimental Science", pages: 365, a: "#e8502f", b: "#9a2d18" },
  { n: 5, roman: "V", title: "Verifiable Intelligence", pages: 357, a: "#6a73ff", b: "#e0932a" },
];

/** The dashed "missing" books (one per volume) that fill with colour, one per beat. */
export const MiniBooks: React.FC<{ start: number; every?: number; y?: number; leave?: number }> = ({
  start,
  every = 15,
  y = 905,
  leave,
}) => {
  const frame = useCurrentFrame();
  const out = leave === undefined ? 1 : 1 - E.inCubic(clamp((frame - leave) / 10));
  const w = 84, h = 116, gap = 84;
  const x0 = 960 - (VOLS.length * w + (VOLS.length - 1) * gap) / 2;
  return (
    <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{ position: "absolute", left: 0, top: 0, opacity: out }}>
      <defs>
        {VOLS.map((v) => (
          <linearGradient key={v.n} id={`mb${v.n}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor={v.a} />
            <stop offset="1" stopColor={v.b} />
          </linearGradient>
        ))}
      </defs>
      {VOLS.map((v, i) => {
        const t0 = start + i * every;
        const k = E.outBack(clamp((frame - t0) / 12), 2.2);
        const fill = clamp((frame - t0) / 5);
        const x = x0 + i * (w + gap);
        const pop = 1 + 0.16 * Math.sin(Math.PI * clamp((frame - t0) / 12)) ;
        return (
          <g key={v.n} transform={`translate(${x + w / 2} ${y + h / 2}) scale(${pop}) translate(${-w / 2} ${-h / 2})`}>
            <rect x={0} y={0} width={w} height={h} rx={5} fill="none" stroke="rgba(190,230,255,0.55)" strokeWidth={2} strokeDasharray="7 6" opacity={1 - fill} />
            <rect x={0} y={0} width={w} height={h} rx={5} fill={`url(#mb${v.n})`} opacity={fill} />
            <rect x={0} y={0} width={9} height={h} rx={4} fill="rgba(0,0,0,0.25)" opacity={fill} />
            <text x={w / 2 + 3} y={h / 2 + 14} textAnchor="middle" fontFamily={F.serif} fontWeight={700} fontSize={40} fill="#fff" opacity={fill}>
              {v.roman}
            </text>
            <text x={w / 2} y={h + 34} textAnchor="middle" fontFamily={F.sans} fontSize={22} letterSpacing={2} fill="rgba(244,247,251,0.7)" opacity={clamp(k)}>
              {v.pages} pp.
            </text>
          </g>
        );
      })}
    </svg>
  );
};
export { C };
