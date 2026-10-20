// Render a handful of frames of a composition in one browser session.
// usage: node scripts/stills.mjs Promo 0,30,60 [--scale 0.5] [--out out/stills] [--jpeg]
import path from "node:path";
import fs from "node:fs";
import { bundle } from "@remotion/bundler";
import { renderStill, selectComposition, openBrowser } from "@remotion/renderer";

const argv = process.argv.slice(2);
const comp = argv[0];
const spec = argv[1];
const opt = (name, d) => {
  const i = argv.indexOf("--" + name);
  return i >= 0 ? argv[i + 1] : d;
};
const scale = parseFloat(opt("scale", "0.5"));
const out = path.resolve(opt("out", "out/stills"));
const png = !argv.includes("--jpeg");
fs.mkdirSync(out, { recursive: true });

const frames = [];
for (const part of spec.split(",")) {
  const m = part.match(/^(\d+)-(\d+)(?:\/(\d+))?$/);
  if (m) for (let f = +m[1]; f <= +m[2]; f += +(m[3] || 1)) frames.push(f);
  else frames.push(+part);
}

const t0 = Date.now();
const serveUrl = await bundle({ entryPoint: path.resolve("src/index.ts"), onProgress: () => {} });
console.log("bundled in", ((Date.now() - t0) / 1000).toFixed(1), "s");
const chromiumOptions = { gl: "angle" };
const browser = await openBrowser("chrome", { chromiumOptions });
const composition = await selectComposition({ serveUrl, id: comp, puppeteerInstance: browser, chromiumOptions });
for (const f of frames) {
  const t1 = Date.now();
  const output = path.join(out, `${comp}-${String(f).padStart(5, "0")}.${png ? "png" : "jpg"}`);
  await renderStill({
    composition,
    serveUrl,
    output,
    frame: f,
    scale,
    imageFormat: png ? "png" : "jpeg",
    puppeteerInstance: browser,
    chromiumOptions,
    overwrite: true,
    timeoutInMS: 120000,
  });
  console.log("frame", f, ((Date.now() - t1) / 1000).toFixed(1) + "s");
}
await browser.close({ silent: true });
