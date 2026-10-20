import React from "react";
import { staticFile } from "remotion";
import { Book } from "../data";
import { F } from "../theme";
import { deg } from "../lib/math";

type V3 = [number, number, number];
const rot = (v: V3, rx: number, ry: number): V3 => {
  // rotateX then rotateY (css order: rotateX(rx) rotateY(ry) applies right-to-left to the vector => Ry first, then Rx)
  const [x, y, z] = v;
  const x1 = x * Math.cos(ry) + z * Math.sin(ry);
  const z1 = -x * Math.sin(ry) + z * Math.cos(ry);
  const y2 = y * Math.cos(rx) - z1 * Math.sin(rx);
  const z2 = y * Math.sin(rx) + z1 * Math.cos(rx);
  return [x1, y2, z2];
};
const LIGHT: V3 = (() => {
  const l: V3 = [-0.45, -0.7, 0.55];
  const n = Math.hypot(...l);
  return [l[0] / n, l[1] / n, l[2] / n];
})();
const lambert = (n: V3, rx: number, ry: number) => {
  const r = rot(n, rx, ry);
  return Math.max(0, r[0] * LIGHT[0] + r[1] * LIGHT[1] + r[2] * LIGHT[2]);
};

const pageEdge = (dir: "h" | "v") =>
  dir === "h"
    ? "repeating-linear-gradient(0deg, #f7f2e6 0px, #f7f2e6 1.4px, #dcd4c0 1.4px, #dcd4c0 2.4px)"
    : "repeating-linear-gradient(90deg, #f7f2e6 0px, #f7f2e6 1.4px, #dcd4c0 1.4px, #dcd4c0 2.4px)";

/**
 * A hardcover book as CSS 3D: real cover on the front, a lettered spine, a block of page edges.
 * Rotations in degrees. The wrapper is centred on its own origin.
 */
export const Book3D: React.FC<{
  book: Book;
  w?: number;
  h?: number;
  d?: number;
  rotX?: number;
  rotY?: number;
  rotZ?: number;
  style?: React.CSSProperties;
  sheen?: number; // 0..1 moving highlight across the cover
}> = ({ book, w = 440, h = 586, d = 64, rotX = 0, rotY = -24, rotZ = 0, style, sheen = 0 }) => {
  const rx = deg(rotX), ry = deg(rotY);
  const sh = (n: V3, base = 0.5) => `brightness(${(base + 0.75 * lambert(n, rx, ry)).toFixed(3)})`;
  const face: React.CSSProperties = { position: "absolute", backfaceVisibility: "hidden" };
  return (
    <div style={{ width: w, height: h, position: "relative", transformStyle: "preserve-3d", transform: `rotateZ(${rotZ}deg) rotateX(${rotX}deg) rotateY(${rotY}deg)`, ...style }}>
      {/* back cover */}
      <div style={{ ...face, left: 0, top: 0, width: w, height: h, transform: `rotateY(180deg) translateZ(${d / 2}px)`, background: book.spine2, filter: sh([0, 0, -1]) }} />
      {/* page block: right edge, top, bottom */}
      <div style={{ ...face, left: (w - d + 2) / 2, top: 5, width: d - 2, height: h - 10, transform: `rotateY(90deg) translateZ(${w / 2 - 3}px)`, background: pageEdge("h"), filter: sh([1, 0, 0], 0.62) }} />
      <div style={{ ...face, left: 5, top: (h - d + 2) / 2, width: w - 10, height: d - 2, transform: `rotateX(90deg) translateZ(${h / 2 - 3}px)`, background: pageEdge("v"), filter: sh([0, -1, 0], 0.62) }} />
      <div style={{ ...face, left: 5, top: (h - d + 2) / 2, width: w - 10, height: d - 2, transform: `rotateX(-90deg) translateZ(${h / 2 - 3}px)`, background: pageEdge("v"), filter: sh([0, 1, 0], 0.62) }} />
      {/* spine */}
      <div
        style={{
          ...face, left: (w - d) / 2, top: 0, width: d, height: h,
          transform: `rotateY(-90deg) translateZ(${w / 2}px)`,
          background: `linear-gradient(90deg, ${book.spine2}, ${book.spine} 40%, ${book.spine2})`,
          filter: sh([-1, 0, 0], 0.55),
          display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden",
        }}
      >
        <div style={{ writingMode: "vertical-rl", fontFamily: F.serif, fontWeight: 700, fontSize: Math.min(26, d * 0.4), color: "rgba(255,255,255,0.92)", letterSpacing: "0.04em", whiteSpace: "nowrap" }}>
          {book.title}
        </div>
        <div style={{ position: "absolute", bottom: 16, fontFamily: F.serif, fontSize: 18, color: "rgba(255,255,255,0.8)", fontWeight: 700 }}>{book.roman}</div>
      </div>
      {/* front cover: the real title page */}
      <div
        style={{
          ...face, left: 0, top: 0, width: w, height: h,
          transform: `translateZ(${d / 2}px)`,
          backgroundImage: `url(${staticFile(book.cover)})`, backgroundSize: "100% 100%",
          filter: sh([0, 0, 1], 0.78),
          boxShadow: "inset 6px 0 8px -4px rgba(0,0,0,0.35)",
        }}
      >
        <div style={{ position: "absolute", inset: 0, background: `linear-gradient(105deg, rgba(255,255,255,0) ${sheen * 140 - 40}%, rgba(255,255,255,0.55) ${sheen * 140 - 20}%, rgba(255,255,255,0) ${sheen * 140}%)`, mixBlendMode: "soft-light", opacity: sheen > 0 && sheen < 1 ? 1 : 0 }} />
        <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 16, background: "linear-gradient(90deg, rgba(0,0,0,0.18), rgba(0,0,0,0))" }} />
      </div>
    </div>
  );
};
