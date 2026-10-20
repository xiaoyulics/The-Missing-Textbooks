import { F } from "../theme";

export type Line = { text: string; x: number; y: number; size: number; ls: number; weight: number };
export type Lockup = {
  the: Line; miss: Line; tb: Line; width: number;
  oCenter: { x: number; y: number; rx: number; ry: number };
  /** TEXTBOOKS split around its second O so that the O's position is exact */
  tbParts: { leftX: number; oCell: number; rightX: number };
};

let cache: Lockup | null = null;

const font = (l: { size: number; weight: number }) => `${l.weight} ${l.size}px ${F.serif}`;

/** Geometry of the "THE / MISSING / TEXTBOOKS" lockup, measured with the real font. */
export function getLockup(): Lockup {
  if (cache) return cache;
  const cv = document.createElement("canvas");
  const ctx = cv.getContext("2d")!;
  const measure = (text: string, size: number, weight: number, ls = 0) => {
    ctx.font = `${weight} ${size}px ${F.serif}`;
    (ctx as any).letterSpacing = `${ls}px`;
    return ctx.measureText(text).width;
  };
  const cx = 960;
  const missSize = 292;
  const wMiss = measure("MISSING", missSize, 700, 4);
  const w300 = measure("TEXTBOOKS", 300, 700, 0);
  const tbSize = Math.floor((300 * wMiss) / w300);
  const theSize = 56;
  const theLS = 38;
  const miss: Line = { text: "MISSING", x: cx, y: 478, size: missSize, ls: 4, weight: 700 };
  const tb: Line = { text: "TEXTBOOKS", x: cx, y: 478 + 12 + Math.round(tbSize * 0.7), size: tbSize, ls: 0, weight: 700 };
  const the: Line = { text: "THE", x: cx, y: 478 - Math.round(missSize * 0.7) - 34, size: theSize, ls: theLS, weight: 400 };

  // counter of the second "O" of TEXTBOOKS: scan a rendering of the glyph
  const sz = 400;
  const s = document.createElement("canvas");
  s.width = s.height = sz * 1.4;
  const sc = s.getContext("2d")!;
  sc.font = `${tb.weight} ${sz}px ${F.serif}`;
  sc.textBaseline = "alphabetic";
  sc.fillStyle = "#000";
  sc.fillText("O", sz * 0.2, sz * 1.05);
  const id = sc.getImageData(0, 0, s.width, s.height);
  const W = s.width, H = s.height;
  const solid = (x: number, y: number) => id.data[(y * W + x) * 4 + 3] > 127;
  // outer bbox
  let minx = W, maxx = 0, miny = H, maxy = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (solid(x, y)) { if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y; }
  const ccx = Math.round((minx + maxx) / 2), ccy = Math.round((miny + maxy) / 2);
  // inner hole bbox by flood fill from the centre
  let hx0 = ccx, hx1 = ccx, hy0 = ccy, hy1 = ccy;
  const seen = new Uint8Array(W * H);
  const stack = [ccx, ccy];
  while (stack.length) {
    const y = stack.pop()!, x = stack.pop()!;
    if (x < 0 || y < 0 || x >= W || y >= H || seen[y * W + x] || solid(x, y)) continue;
    seen[y * W + x] = 1;
    if (x < hx0) hx0 = x; if (x > hx1) hx1 = x; if (y < hy0) hy0 = y; if (y > hy1) hy1 = y;
    stack.push(x + 1, y, x - 1, y, x, y + 1, x, y - 1);
  }
  const k = tb.size / sz;
  // position of the 2nd O: width of "TEXTBOO" prefix + half of the O advance, relative to the line centre
  const prefix = measure("TEXTBOO", tb.size, tb.weight, tb.ls);
  const full = measure("TEXTBOOKS", tb.size, tb.weight, tb.ls);
  const oAdv = measure("O", tb.size, tb.weight, 0);
  const left = cx - full / 2;
  const oX = left + prefix - oAdv / 2 + ((minx + maxx) / 2 - (sz * 0.2 + (measure("O", sz, tb.weight, 0)) / 2)) * k;
  const oY = tb.y + ((hy0 + hy1) / 2 - sz * 1.05) * k;
  const oCell = left + prefix - oAdv / 2; // centre of the second O's advance cell
  const counterDx = ((hx0 + hx1) / 2 - (sz * 0.2 + measure("O", sz, tb.weight, 0) / 2)) * k;
  void oX;
  cache = {
    the, miss, tb, width: wMiss,
    oCenter: { x: oCell + counterDx, y: oY, rx: ((hx1 - hx0) / 2) * k, ry: ((hy1 - hy0) / 2) * k },
    tbParts: { leftX: left, oCell, rightX: left + prefix },
  };
  return cache;
}

export const fontOf = font;
