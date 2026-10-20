import React, { useLayoutEffect, useRef, useState } from "react";
import { continueRender, delayRender, staticFile } from "remotion";
import { compile, fullscreenQuad, FULLSCREEN_VS, loadImage, texFromSource } from "../lib/gl";
import { makeWallCanvas } from "../lib/wall";
import { Mat3, rotYXZ } from "../lib/math";

export type SlabParams = {
  c: [number, number, number];
  yaw: number;
  pitch: number;
  roll: number;
  hs: [number, number]; // half size, world units
  r: number; // corner radius
  cover: number; // 0..1 how much of the printed cover shows through the glass
  glow: number; // 0..1 glass strength (rim light, body tint)
  idx: number; // 0..3 which cover
  k: number; // refraction strength at the bevel
  appear: number; // 0..1 scale-in
};

export type LensFrame = {
  time: number;
  cam0: [number, number, number];
  cam1: [number, number, number];
  yaw: number;
  pitch: number;
  roll: number;
  tanHalf: number;
  samples: number;
  slabs: SlabParams[];
  wallScroll: [number, number];
  wallBright: number;
  light: number;
  flash: number;
  shake: [number, number];
  /** punch an elliptical hole (canvas px: cx, cy, rx, ry): transparent inside */
  hole?: [number, number, number, number];
};

const FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform vec2  uRes;
uniform float uTime;
uniform vec3  uCam0;
uniform vec3  uCam1;
uniform mat3  uCamRot;
uniform float uTanHalf;
uniform int   uSamples;
uniform sampler2D uWall;
uniform sampler2D uCovers;
uniform vec4  uSlabA[5]; // xyz centre, w cover alpha
uniform vec4  uSlabB[5]; // xy half size, z corner radius, w glow
uniform vec4  uSlabC[5]; // x atlas index, y refraction, z appear
uniform mat3  uSlabR[5];
uniform vec2  uWallScroll;
uniform float uWallBright;
uniform float uLight;
uniform float uFlash;
uniform vec2  uShake;
uniform vec4  uHole;

out vec4 outColor;

float sdRoundBox(vec2 p, vec2 b, float r){
  vec2 q = abs(p) - b + r;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

vec3 wallLayer(vec3 ro, vec3 rd, float Z, float scale, vec2 off, float bright){
  float t = (-Z - ro.z) / rd.z;
  if (t <= 0.0) return vec3(0.0);
  vec2 p = ro.xy + rd.xy * t;
  vec4 s = texture(uWall, vec2(p.x, -p.y) * scale + off);
  float fade = smoothstep(0.4, 3.0, t) * exp(-t * 0.018);
  vec3 txt  = vec3(0.16, 0.60, 0.86) * s.r;
  vec3 miss = vec3(1.00, 0.30, 0.18) * s.g;
  return (txt + miss) * bright * fade;
}

vec3 wall(vec3 ro, vec3 rd){
  vec3 col = vec3(0.010, 0.022, 0.042) + vec3(0.008, 0.022, 0.04) * (0.5 + 0.5 * rd.y);
  float b = uWallBright;
  col += wallLayer(ro, rd, 30.0, 0.050, uWallScroll * 0.4 + vec2(0.13, 0.41), 0.50 * b);
  col += wallLayer(ro, rd, 22.0, 0.062, uWallScroll * 0.6 + vec2(0.52, 0.19), 0.55 * b);
  col += wallLayer(ro, rd, 15.5, 0.072, uWallScroll * 0.8 + vec2(0.71, 0.07), 0.62 * b);
  col += wallLayer(ro, rd, 11.0, 0.078, uWallScroll * 1.0 + vec2(0.29, 0.63), 0.70 * b);
  col += wallLayer(ro, rd,  7.5, 0.085, uWallScroll * 1.3 + vec2(0.88, 0.33), 0.62 * b);
  col += wallLayer(ro, rd,  4.6, 0.100, uWallScroll * 1.7 + vec2(0.45, 0.91), 0.50 * b);
  return col;
}

vec3 trace(vec3 ro, vec3 rd0){
  vec3 pos = ro;
  vec3 rd = rd0;
  vec3 bend = vec3(0.0);
  vec3 emit = vec3(0.0);
  float T = 1.0;
  for (int i = 0; i < 5; i++){
    float appear = uSlabC[i].z;
    if (appear < 0.002) continue;
    vec3 c = uSlabA[i].xyz;
    mat3 R = uSlabR[i];
    vec3 n = R[2];
    float denom = dot(rd, n);
    if (abs(denom) < 1e-5) continue;
    float t = dot(c - pos, n) / denom;
    if (t <= 0.0) continue;
    vec3 p = pos + rd * t;
    vec3 rel = p - c;
    vec3 q = vec3(dot(rel, R[0]), dot(rel, R[1]), dot(rel, R[2]));
    vec2 hs = uSlabB[i].xy * appear;
    float rad = min(uSlabB[i].z, min(hs.x, hs.y) * 0.95);
    float d = sdRoundBox(q.xy, hs, rad);
    float glow = uSlabB[i].w;
    if (d > 0.0){
      emit += T * glow * vec3(0.10, 0.28, 0.46) * exp(-d * 6.0) * 0.30;
      continue;
    }
    float e = -d;
    float bw = 0.13 + 0.10 * glow;
    float u = clamp(e / bw, 0.0, 1.0);
    vec2 h = vec2(0.004, 0.0);
    vec2 g = vec2(sdRoundBox(q.xy + h.xy, hs, rad) - sdRoundBox(q.xy - h.xy, hs, rad),
                  sdRoundBox(q.xy + h.yx, hs, rad) - sdRoundBox(q.xy - h.yx, hs, rad));
    g /= (length(g) + 1e-6);
    float edge = 1.0 - u;
    float prof = edge * edge * (3.0 - 2.0 * edge);
    vec3 gw = R[0] * g.x + R[1] * g.y;
    vec3 dRay = gw * prof * uSlabC[i].y;
    rd = normalize(rd + dRay);
    bend += dRay;
    vec3 nf = dot(n, rd0) > 0.0 ? -n : n;
    vec3 Nw = normalize(nf + gw * (0.2 + 1.6 * prof));
    vec3 L = normalize(vec3(-0.55, 0.75, 0.55));
    vec3 H = normalize(L - rd0);
    float spec = pow(max(dot(Nw, H), 0.0), 36.0);
    float fres = pow(1.0 - max(dot(Nw, -rd0), 0.0), 3.0);
    float rim = pow(edge, 3.0);
    emit += T * glow * (vec3(0.55, 0.80, 1.0) * (rim * 0.55 + fres * 0.30) + vec3(1.0) * spec * 0.8 * (0.25 + edge));
    emit += T * glow * vec3(0.010, 0.026, 0.044);
    float ca = uSlabA[i].w;
    if (ca > 0.001){
      vec2 uvc = vec2(q.x / (2.0 * hs.x) + 0.5, 0.5 - q.y / (2.0 * hs.y));
      vec2 cell = vec2(mod(uSlabC[i].x, 3.0), floor(uSlabC[i].x / 3.0));
      vec3 cc = texture(uCovers, (cell + clamp(uvc, 0.002, 0.998)) / vec2(3.0, 2.0)).rgb;
      float a = ca * smoothstep(0.0, 0.02, e);
      emit += T * cc * a;
      T *= (1.0 - a);
    }
    pos = p;
  }
  vec3 col;
  if (dot(bend, bend) < 1e-8){
    col = wall(ro, rd0);
  } else {
    col.r = wall(ro, normalize(rd0 + bend * 0.92)).r;
    col.g = wall(ro, normalize(rd0 + bend)).g;
    col.b = wall(ro, normalize(rd0 + bend * 1.08)).b;
  }
  return emit + T * col;
}

void main(){
  vec2 ndc = (gl_FragCoord.xy / uRes) * 2.0 - 1.0;
  ndc += uShake;
  ndc.x *= uRes.x / uRes.y;
  vec3 rd0 = normalize(uCamRot * vec3(ndc * uTanHalf, -1.0));
  vec3 acc = vec3(0.0);
  for (int k = 0; k < 8; k++){
    if (k >= uSamples) break;
    float f = (float(k) + 0.5) / float(uSamples);
    acc += trace(mix(uCam0, uCam1, f), rd0);
  }
  acc /= float(uSamples);
  float r = length(ndc);
  acc += vec3(0.85, 0.93, 1.0) * uLight * exp(-r * r * 2.4);
  acc = clamp(acc, 0.0, 1.0);
  acc *= 1.0 - 0.5 * smoothstep(0.6, 1.6, r);
  acc = mix(acc, vec3(1.0), clamp(uFlash, 0.0, 1.0));
  float alpha = 1.0;
  if (uHole.z > 0.0) {
    vec2 hp = (gl_FragCoord.xy - vec2(uHole.x, uRes.y - uHole.y)) / uHole.zw;
    alpha = smoothstep(0.985, 1.0, length(hp));
  }
  outColor = vec4(acc * alpha, alpha);
}`;

type GLState = {
  gl: WebGL2RenderingContext;
  prog: WebGLProgram;
  loc: Record<string, WebGLUniformLocation | null>;
  wallTex: WebGLTexture;
  coverTex: WebGLTexture;
};

const NAMES = [
  "uRes", "uTime", "uCam0", "uCam1", "uCamRot", "uTanHalf", "uSamples", "uWall", "uCovers",
  "uSlabA[0]", "uSlabB[0]", "uSlabC[0]", "uSlabR[0]", "uWallScroll", "uWallBright", "uLight", "uFlash", "uShake", "uHole",
];

function draw(st: GLState, f: LensFrame, w: number, h: number) {
  const { gl, loc } = st;
  gl.viewport(0, 0, w, h);
  gl.useProgram(st.prog);
  gl.uniform2f(loc.uRes, w, h);
  gl.uniform1f(loc.uTime, f.time);
  gl.uniform3f(loc.uCam0, ...f.cam0);
  gl.uniform3f(loc.uCam1, ...f.cam1);
  gl.uniformMatrix3fv(loc.uCamRot, false, new Float32Array(rotYXZ(f.yaw, f.pitch, f.roll)));
  gl.uniform1f(loc.uTanHalf, f.tanHalf);
  gl.uniform1i(loc.uSamples, Math.max(1, Math.min(8, f.samples)));
  gl.uniform2f(loc.uWallScroll, ...f.wallScroll);
  gl.uniform1f(loc.uWallBright, f.wallBright);
  gl.uniform1f(loc.uLight, f.light);
  gl.uniform1f(loc.uFlash, f.flash);
  gl.uniform2f(loc.uShake, ...f.shake);
  gl.uniform4f(loc.uHole, ...(f.hole ?? [0, 0, 0, 0]));
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, st.wallTex);
  gl.uniform1i(loc.uWall, 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, st.coverTex);
  gl.uniform1i(loc.uCovers, 1);

  // front-to-back order by distance to the camera
  const cam = f.cam1;
  const sorted = [...f.slabs].sort((a, b) => {
    const da = (a.c[0] - cam[0]) ** 2 + (a.c[1] - cam[1]) ** 2 + (a.c[2] - cam[2]) ** 2;
    const db = (b.c[0] - cam[0]) ** 2 + (b.c[1] - cam[1]) ** 2 + (b.c[2] - cam[2]) ** 2;
    return da - db;
  });
  const A = new Float32Array(20), B = new Float32Array(20), Cc = new Float32Array(20), R = new Float32Array(45);
  for (let i = 0; i < 5; i++) {
    const s = sorted[i];
    if (!s) {
      Cc[i * 4 + 2] = 0;
      continue;
    }
    A.set([...s.c, s.cover], i * 4);
    B.set([s.hs[0], s.hs[1], s.r, s.glow], i * 4);
    Cc.set([s.idx, s.k, s.appear, 0], i * 4);
    const m: Mat3 = rotYXZ(s.yaw, s.pitch, s.roll);
    R.set(m, i * 9);
  }
  gl.uniform4fv(loc["uSlabA[0]"], A);
  gl.uniform4fv(loc["uSlabB[0]"], B);
  gl.uniform4fv(loc["uSlabC[0]"], Cc);
  gl.uniformMatrix3fv(loc["uSlabR[0]"], false, R);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

export const LensCanvas: React.FC<{ frame: LensFrame; width?: number; height?: number }> = ({
  frame,
  width = 1920,
  height = 1080,
}) => {
  const ref = useRef<HTMLCanvasElement>(null);
  const st = useRef<GLState | null>(null);
  const latest = useRef(frame);
  latest.current = frame;
  const [handle] = useState(() => delayRender("lens init"));

  useLayoutEffect(() => {
    let dead = false;
    (async () => {
      const cv = ref.current!;
      const gl = cv.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false })!;
      const prog = compile(gl, FULLSCREEN_VS, FRAG);
      gl.bindVertexArray(fullscreenQuad(gl, prog));
      const loc: Record<string, WebGLUniformLocation | null> = {};
      for (const n of NAMES) loc[n.replace("[0]", "") === n ? n : n] = gl.getUniformLocation(prog, n);
      loc.uSlabA = loc["uSlabA[0]"];
      const wallTex = texFromSource(gl, makeWallCanvas(11), { repeat: true });
      const ims = await Promise.all([1, 2, 3, 4, 5].map((i) => loadImage(staticFile(`covers/v${i}.jpg`))));
      const atlas = document.createElement("canvas");
      atlas.width = 1860; // 3 x 2 cells of 620 x 827
      atlas.height = 1654;
      const actx = atlas.getContext("2d")!;
      actx.fillStyle = "#fff";
      actx.fillRect(0, 0, atlas.width, atlas.height);
      ims.forEach((im, i) => actx.drawImage(im, (i % 3) * 620, Math.floor(i / 3) * 827, 620, 827));
      const coverTex = texFromSource(gl, atlas, {});
      if (dead) return;
      st.current = { gl, prog, loc, wallTex, coverTex };
      draw(st.current, latest.current, width, height);
      continueRender(handle);
    })().catch((e) => {
      console.error(e);
      throw e;
    });
    return () => {
      dead = true;
    };
  }, []);

  useLayoutEffect(() => {
    if (st.current) draw(st.current, frame, width, height);
  });

  return <canvas ref={ref} width={width} height={height} style={{ position: "absolute", left: 0, top: 0, width, height }} />;
};
