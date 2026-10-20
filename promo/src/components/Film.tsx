import React, { useMemo } from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { hash1, mulberry32 } from "../lib/math";

/** Animated film grain (a small noise tile that jumps every frame) and a soft vignette. */
export const Film: React.FC<{ grain?: number; vignette?: number }> = ({ grain = 0.09, vignette = 0 }) => {
  const frame = useCurrentFrame();
  const url = useMemo(() => {
    const cv = document.createElement("canvas");
    cv.width = cv.height = 256;
    const ctx = cv.getContext("2d")!;
    const im = ctx.createImageData(256, 256);
    const r = mulberry32(99);
    for (let i = 0; i < 256 * 256; i++) {
      const v = Math.floor(r() * 255);
      im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = v;
      im.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    return cv.toDataURL("image/png");
  }, []);
  const ox = Math.floor(hash1(frame * 1.37) * 256);
  const oy = Math.floor(hash1(frame * 2.11 + 5) * 256);
  return (
    <AbsoluteFill style={{ pointerEvents: "none" }}>
      <AbsoluteFill
        style={{
          backgroundImage: `url(${url})`,
          backgroundSize: "256px 256px",
          backgroundPosition: `${ox}px ${oy}px`,
          opacity: grain,
          mixBlendMode: "overlay",
        }}
      />
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse at 50% 50%, rgba(0,0,0,0) 55%, rgba(0,0,0,${vignette}) 120%)`,
        }}
      />
    </AbsoluteFill>
  );
};
