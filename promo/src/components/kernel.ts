import { TAU } from "../lib/math";

/**
 * The kernel left when the shell has dissolved: a warm point with a cold halo and a streak.
 * Drawn identically at the end of volume 1 and at the start of volume 2, where it becomes the nucleus.
 */
export function drawKernel(ctx: CanvasRenderingContext2D, x: number, y: number, cs: number, glow: number, pulse: number) {
  if (glow <= 0) return;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  const R = 260 * cs * glow * pulse;
  const hg = ctx.createRadialGradient(x, y, 0, x, y, R);
  hg.addColorStop(0, "rgba(255,236,190,0.95)");
  hg.addColorStop(0.1, "rgba(255,214,150,0.55)");
  hg.addColorStop(0.35, "rgba(95,200,240,0.28)");
  hg.addColorStop(1, "rgba(60,160,220,0)");
  ctx.fillStyle = hg;
  ctx.fillRect(x - R, y - R, R * 2, R * 2);
  const sl = 520 * glow * cs;
  const sg = ctx.createLinearGradient(x - sl, y, x + sl, y);
  sg.addColorStop(0, "rgba(120,210,255,0)");
  sg.addColorStop(0.5, `rgba(255,240,210,${0.7 * glow})`);
  sg.addColorStop(1, "rgba(120,210,255,0)");
  ctx.fillStyle = sg;
  ctx.fillRect(x - sl, y - 2.2, sl * 2, 4.4);
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(x, y, 11 * cs * glow, 0, TAU);
  ctx.fill();
  ctx.restore();
}
