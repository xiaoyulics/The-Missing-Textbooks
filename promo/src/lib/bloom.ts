/** Cheap bloom for a 2D canvas: downscale, blur, add back. */
const tmp: Record<string, HTMLCanvasElement> = {};
export function bloom(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  opts: { scale?: number; blur?: number; strength?: number; key?: string } = {},
) {
  const { scale = 0.25, blur = 5, strength = 0.9, key = "a" } = opts;
  const tw = Math.round(w * scale), th = Math.round(h * scale);
  let a = tmp[key + "a"], b = tmp[key + "b"];
  if (!a || a.width !== tw) {
    a = tmp[key + "a"] = document.createElement("canvas");
    b = tmp[key + "b"] = document.createElement("canvas");
    a.width = b.width = tw;
    a.height = b.height = th;
  }
  const actx = a.getContext("2d")!, bctx = b.getContext("2d")!;
  actx.clearRect(0, 0, tw, th);
  actx.drawImage(ctx.canvas, 0, 0, tw, th);
  bctx.clearRect(0, 0, tw, th);
  bctx.filter = `blur(${blur}px)`;
  bctx.drawImage(a, 0, 0);
  bctx.filter = "none";
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.globalAlpha = strength;
  ctx.drawImage(b, 0, 0, w, h);
  ctx.restore();
}
