import React from "react";
import { useCurrentFrame } from "remotion";
import { E, clamp } from "../lib/math";

type Props = {
  text: string;
  start: number; // local frame at which the first word starts
  stagger?: number; // frames between words
  dur?: number; // frames per word
  exit?: number; // local frame at which everything starts leaving (optional)
  exitDur?: number;
  rise?: number; // px
  blur?: number; // px
  style?: React.CSSProperties;
  wordStyle?: (i: number, w: string) => React.CSSProperties;
  /** characters instead of words */
  chars?: boolean;
};

/** Word-by-word (or letter-by-letter) reveal: rise + unblur + fade. */
export const RevealText: React.FC<Props> = ({
  text,
  start,
  stagger = 4,
  dur = 14,
  exit,
  exitDur = 10,
  rise = 26,
  blur = 14,
  style,
  wordStyle,
  chars = false,
}) => {
  const frame = useCurrentFrame();
  const parts = chars ? Array.from(text) : text.split(" ");
  const leave = exit === undefined ? 0 : E.inCubic(clamp((frame - exit) / exitDur));
  return (
    <div style={{ display: "inline-block", whiteSpace: "pre", ...style }}>
      {parts.map((w, i) => {
        const t = E.outCubic(clamp((frame - start - i * stagger) / dur));
        const o = t * (1 - leave);
        return (
          <span
            key={i}
            style={{
              display: "inline-block",
              opacity: o,
              transform: `translateY(${(1 - t) * rise - leave * rise * 0.6}px)`,
              filter: `blur(${(1 - t) * blur + leave * blur * 0.8}px)`,
              whiteSpace: "pre",
              ...(wordStyle ? wordStyle(i, w) : null),
            }}
          >
            {w}
            {!chars && i < parts.length - 1 ? " " : ""}
          </span>
        );
      })}
    </div>
  );
};
