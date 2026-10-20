import React, { useLayoutEffect, useRef } from "react";
import { useCurrentFrame } from "remotion";

/** A full-frame 2D canvas that calls `draw` on every frame. */
export const Canvas2D: React.FC<{
  draw: (ctx: CanvasRenderingContext2D, frame: number, w: number, h: number) => void;
  width?: number;
  height?: number;
  style?: React.CSSProperties;
}> = ({ draw, width = 1920, height = 1080, style }) => {
  const ref = useRef<HTMLCanvasElement>(null);
  const frame = useCurrentFrame();
  useLayoutEffect(() => {
    const ctx = ref.current!.getContext("2d")!;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, width, height);
    draw(ctx, frame, width, height);
  });
  return <canvas ref={ref} width={width} height={height} style={{ position: "absolute", left: 0, top: 0, width, height, ...style }} />;
};
