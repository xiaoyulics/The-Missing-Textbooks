import React, { useLayoutEffect, useMemo, useRef, useState } from "react";
import { ThreeCanvas } from "@remotion/three";
import { continueRender, delayRender, staticFile, useCurrentFrame } from "remotion";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { BOOKS } from "../data";
import { makeWallCanvas } from "../lib/wall";
import { E, clamp, hash1, lerp, mulberry32, smoothstep, TAU } from "../lib/math";

// ---------------------------------------------------------------- layout
export const BW = 0.75, BH = 1.0;
export const SLOT = 0.98;
export const bookX = (i: number) => (i - (BOOKS.length - 1) / 2) * SLOT; // all the books, centred
export const bookD = (i: number) => 0.13 + (BOOKS[i].pages / 394) * 0.08;
export const LANDS = [45, 60, 75, 90, 105]; // landing frames (on the beat), one per volume
export const BURST = 120; // page storm starts
export const COLLAPSE = 210; // pages blow past the camera

const COLORS = ["#2aa3c4", "#9a6bff", "#e8bd5a", "#e8502f", "#8a93ff"];

// ---------------------------------------------------------------- textures
function loadImg(url: string) {
  return new Promise<HTMLImageElement>((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error("failed " + url));
    im.src = url;
  });
}

type Assets = { covers: THREE.Texture[]; spines: THREE.Texture[]; edge: THREE.Texture; atlas: THREE.Texture; wall: THREE.Texture };

function spineTexture(i: number) {
  const b = BOOKS[i];
  const cv = document.createElement("canvas");
  cv.width = 160;
  cv.height = 1024;
  const c = cv.getContext("2d")!;
  const g = c.createLinearGradient(0, 0, 160, 0);
  g.addColorStop(0, b.spine2);
  g.addColorStop(0.5, b.spine);
  g.addColorStop(1, b.spine2);
  c.fillStyle = g;
  c.fillRect(0, 0, 160, 1024);
  c.save();
  c.translate(80, 512);
  c.rotate(Math.PI / 2);
  c.fillStyle = "rgba(255,255,255,0.94)";
  c.font = "700 54px Palatino, serif";
  c.textAlign = "center";
  c.textBaseline = "middle";
  const maxW = 820;
  let size = 54;
  while (c.measureText(b.title).width > maxW && size > 26) { size -= 2; c.font = `700 ${size}px Palatino, serif`; }
  c.fillText(b.title, 0, 0);
  c.restore();
  c.fillStyle = "rgba(255,255,255,0.85)";
  c.font = "700 60px Palatino, serif";
  c.textAlign = "center";
  c.fillText(b.roman, 80, 960);
  c.fillRect(30, 40, 100, 3);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

async function buildAssets(): Promise<Assets> {
  const covers = await Promise.all(
    BOOKS.map(async (b) => {
      const im = await loadImg(staticFile(b.cover));
      const t = new THREE.Texture(im);
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 8;
      t.needsUpdate = true;
      return t;
    }),
  );
  const spines = BOOKS.map((_, i) => spineTexture(i));
  const ec = document.createElement("canvas");
  ec.width = 64; ec.height = 512;
  const e = ec.getContext("2d")!;
  for (let y = 0; y < 512; y += 4) { e.fillStyle = y % 8 === 0 ? "#f3eee0" : "#d9d1bc"; e.fillRect(0, y, 64, 4); }
  const edge = new THREE.CanvasTexture(ec);
  edge.colorSpace = THREE.SRGBColorSpace;
  edge.wrapS = edge.wrapT = THREE.RepeatWrapping;
  // atlas of real interior pages
  const COLS = 15, ROWS = 12, CW = 256, CH = 341; // 5 volumes x 36 sampled pages = 180 cells
  const ac = document.createElement("canvas");
  ac.width = COLS * CW; ac.height = ROWS * CH;
  const a = ac.getContext("2d")!;
  a.fillStyle = "#fff";
  a.fillRect(0, 0, ac.width, ac.height);
  const urls: string[] = [];
  for (let v = 1; v <= BOOKS.length; v++) for (let k = 0; k < 36; k++) urls.push(staticFile(`pages/v${v}-${String(k).padStart(2, "0")}.jpg`));
  const ims = await Promise.all(urls.map(loadImg));
  ims.forEach((im, i) => a.drawImage(im, (i % COLS) * CW, Math.floor(i / COLS) * CH, CW, CH));
  const atlas = new THREE.CanvasTexture(ac);
  atlas.colorSpace = THREE.SRGBColorSpace;
  atlas.anisotropy = 8;
  atlas.generateMipmaps = true;
  atlas.minFilter = THREE.LinearMipmapLinearFilter;
  const wall = new THREE.CanvasTexture(makeWallCanvas(11));
  wall.wrapS = wall.wrapT = THREE.RepeatWrapping;
  wall.generateMipmaps = true;
  wall.minFilter = THREE.LinearMipmapLinearFilter;
  return { covers, spines, edge, atlas, wall };
}

// ---------------------------------------------------------------- the camera
export function finaleCam(f: number) {
  const K = (pairs: [number, number][], ease: (t: number) => number = E.inOutCubic) => {
    if (f <= pairs[0][0]) return pairs[0][1];
    for (let i = 1; i < pairs.length; i++) if (f <= pairs[i][0]) return lerp(pairs[i - 1][1], pairs[i][1], ease(clamp((f - pairs[i - 1][0]) / (pairs[i][0] - pairs[i - 1][0]))));
    return pairs[pairs.length - 1][1];
  };
  return {
    px: K([[0, 0], [110, 0.1], [135, 0.0], [196, 3.0], [210, 2.4], [240, 0.0], [330, 0.0], [470, -0.5]]),
    py: K([[0, 2.8], [110, 1.2], [135, 1.7], [196, 3.4], [210, 1.4], [240, 1.0], [330, 1.0], [470, 0.9]]),
    pz: K([[0, 7.4], [110, 6.4], [135, 7.6], [196, 9.4], [210, 4.6], [240, 6.1], [330, 5.9], [470, 6.6]]),
    lx: K([[0, 0], [210, 0], [240, 0], [330, 0.0], [470, 1.0]]),
    ly: K([[0, 0.4], [130, 0.7], [196, 1.5], [240, 1.1], [330, 1.1], [470, 0.62]]),
    lz: K([[0, 0], [240, 0], [330, 0], [470, -1.0]]),
  };
}

// ---------------------------------------------------------------- scene pieces
const Backdrop: React.FC<{ a: Assets; f: number }> = ({ a, f }) => {
  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: { uTex: { value: a.wall }, uOff: { value: new THREE.Vector2() }, uGain: { value: 0.6 } },
        vertexShader: `varying vec2 vUv; varying vec3 vW; void main(){ vUv = uv; vec4 w = modelMatrix*vec4(position,1.0); vW = w.xyz; gl_Position = projectionMatrix*viewMatrix*w; }`,
        fragmentShader: `uniform sampler2D uTex; uniform vec2 uOff; uniform float uGain; varying vec2 vUv; varying vec3 vW;
          void main(){ vec4 s = texture2D(uTex, vec2(vW.x, vW.y) * 0.075 + uOff);
            float fade = smoothstep(7.0, 0.0, length(vW.xy - vec2(0.0, 1.0))) * 0.8 + 0.2;
            vec3 col = (vec3(0.05,0.28,0.45)*s.r + vec3(0.6,0.12,0.07)*s.g) * uGain * fade;
            gl_FragColor = vec4(col, 1.0); }`,
      }),
    [a],
  );
  useLayoutEffect(() => {
    mat.uniforms.uOff.value.set(f * 0.00018, f * 0.00009);
    mat.uniforms.uGain.value = 0.55 * smoothstep(0, 30, f);
  });
  return (
    <mesh position={[0, 4, -1.05]} material={mat}>
      <planeGeometry args={[120, 36]} />
    </mesh>
  );
};

const Shelf: React.FC = () => (
  <group>
    <mesh position={[0, -0.09, 0.15]} receiveShadow>
      <boxGeometry args={[120, 0.18, 1.7]} />
      <meshStandardMaterial color="#0d1626" roughness={0.28} metalness={0.15} transparent opacity={0.9} />
    </mesh>
    <mesh position={[0, 4, -1.12]} receiveShadow>
      <planeGeometry args={[120, 36]} />
      <meshStandardMaterial color="#050a14" roughness={0.9} />
    </mesh>
    <mesh position={[0, -0.185, 0.98]}>
      <boxGeometry args={[120, 0.01, 0.02]} />
      <meshBasicMaterial color="#6fd0ff" />
    </mesh>
  </group>
);

const GhostSlot: React.FC<{ x: number; f: number; k: number }> = ({ x, f, k }) => {
  const ref = useRef<THREE.LineSegments>(null);
  const geom = useMemo(() => {
    const w = BW, h = BH, z = 0.12;
    const pts = [
      -w / 2, 0, z, w / 2, 0, z, w / 2, 0, z, w / 2, h, z, w / 2, h, z, -w / 2, h, z, -w / 2, h, z, -w / 2, 0, z,
    ];
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return g;
  }, []);
  const mat = useMemo(() => new THREE.LineDashedMaterial({ color: "#b4e4ff", dashSize: 0.05, gapSize: 0.035, transparent: true, opacity: 0, depthWrite: false }), []);
  useLayoutEffect(() => {
    ref.current?.computeLineDistances();
  }, [geom]);
  useLayoutEffect(() => {
    const appear = smoothstep(30 + k * 5, 52 + k * 5, f);
    const ghostIn = smoothstep(246 + k * 7, 264 + k * 7, f);
    const pulse = 0.55 + 0.45 * Math.sin(f * 0.12 + k * 1.7);
    mat.opacity = (k < 0 ? appear * (1 - smoothstep(LANDS[Math.max(0, Math.min(4, k + 5))] - 2, LANDS[Math.max(0, Math.min(4, k + 5))] + 6, f)) : ghostIn) * (0.35 + 0.5 * pulse) * Math.exp(-Math.abs(k) * 0.12);
  });
  return <lineSegments ref={ref} geometry={geom} material={mat} position={[x, 0, 0]} />;
};

type BookProps = { i: number; a: Assets; f: number };
const BookMesh: React.FC<BookProps> = ({ i, a, f }) => {
  const g = useRef<THREE.Group>(null);
  const d = bookD(i);
  const mats = useMemo(() => {
    const side = new THREE.MeshStandardMaterial({ map: a.edge, roughness: 0.8 });
    const cover = new THREE.MeshStandardMaterial({ map: a.covers[i], roughness: 0.55, metalness: 0.0, color: new THREE.Color(0.86, 0.86, 0.88) });
    const spine = new THREE.MeshStandardMaterial({ map: a.spines[i], roughness: 0.5 });
    const back = new THREE.MeshStandardMaterial({ color: BOOKS[i].spine2, roughness: 0.7 });
    const top = new THREE.MeshStandardMaterial({ map: a.edge, roughness: 0.85 });
    return [side, spine, top, top, cover, back];
  }, [a, i]);
  useLayoutEffect(() => {
    const t = f - LANDS[i];
    const fall0 = LANDS[i] - 26;
    let y = 0;
    let sq = 1;
    if (f < fall0) y = 9;
    else if (t < 0) {
      const u = (f - fall0) / 26;
      y = lerp(9, 0, u * u);
    } else {
      // bounce
      const b = Math.exp(-t * 0.28) * Math.abs(Math.sin(t * 0.55)) * 0.16;
      y = b;
      sq = 1 - 0.045 * Math.exp(-t * 0.5);
    }
    const spin = f < LANDS[i] ? (1 - clamp((f - fall0) / 26)) * 0.5 * (i % 2 ? 1 : -1) : 0;
    g.current!.position.set(bookX(i), y + (BH * sq) / 2, 0);
    g.current!.scale.set(1, sq, 1);
    g.current!.rotation.set(0, spin, 0);
    g.current!.visible = f >= fall0 - 1;
  });
  return (
    <group ref={g}>
      <mesh castShadow receiveShadow material={mats}>
        <boxGeometry args={[BW, BH, d]} />
      </mesh>
    </group>
  );
};

const Dust: React.FC<{ f: number }> = ({ f }) => {
  const N = BOOKS.length * 60;
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    g.setAttribute("aA", new THREE.BufferAttribute(new Float32Array(N), 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 50);
    return g;
  }, [N]);
  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        transparent: true, depthWrite: false,
        vertexShader: `attribute float aA; varying float vA; void main(){ vec4 mv = modelViewMatrix*vec4(position,1.0); gl_Position = projectionMatrix*mv; vA = aA; gl_PointSize = 90.0 / -mv.z; }`,
        fragmentShader: `varying float vA; void main(){ vec2 c = gl_PointCoord-0.5; float a = exp(-dot(c,c)*14.0); gl_FragColor = vec4(vec3(0.62,0.72,0.85), a*vA*0.35); }`,
      }),
    [],
  );
  useLayoutEffect(() => {
    const pos = geom.attributes.position as THREE.BufferAttribute;
    const al = geom.attributes.aA as THREE.BufferAttribute;
    for (let b = 0; b < BOOKS.length; b++) {
      const t = f - LANDS[b];
      for (let k = 0; k < 60; k++) {
        const idx = b * 60 + k;
        if (t < 0 || t > 40) { al.setX(idx, 0); continue; }
        const r = mulberry32(b * 100 + k);
        const a = r() * TAU, sp = 0.025 + r() * 0.05;
        const side = r() < 0.5 ? -1 : 1;
        pos.setXYZ(idx, bookX(b) + Math.cos(a) * sp * t * 3.2 * 0.7 + side * 0.3 * (1 - Math.exp(-t * 0.3)) * r(), 0.04 + Math.abs(Math.sin(a)) * sp * t * 1.2, 0.18 + Math.sin(a) * sp * t * 2.0);
        al.setX(idx, Math.max(0, 1 - t / 40));
      }
    }
    pos.needsUpdate = true;
    al.needsUpdate = true;
  });
  return <points geometry={geom} material={mat} frustumCulled={false} />;
};

// ---------------------------------------------------------------- the page storm
const NPAGES = BOOKS.length * 36;
const Pages: React.FC<{ a: Assets; f: number }> = ({ a, f }) => {
  const ref = useRef<THREE.InstancedMesh>(null);
  const data = useMemo(() => {
    const r = mulberry32(88);
    return Array.from({ length: NPAGES }, (_, i) => ({
      book: Math.floor(i / 36),
      cell: i,
      r0: 0.7 + r() * 0.9,
      R: 1.1 + r() * 3.2,
      th0: r() * TAU,
      w: (0.8 + r() * 1.4) * (r() < 0.5 ? 1 : -1) * 0.045,
      h: 0.6 + r() * 4.4,
      delay: r() * 22,
      tumble: [r() * TAU, r() * TAU, r() * TAU] as [number, number, number],
      tw: [(r() - 0.5) * 0.25, (r() - 0.5) * 0.25, (r() - 0.5) * 0.25] as [number, number, number],
      ph: r() * TAU,
    }));
  }, []);
  const geom = useMemo(() => {
    const g = new THREE.PlaneGeometry(BW * 0.9, BH * 0.9, 10, 10);
    const cell = new Float32Array(NPAGES * 2);
    data.forEach((p, i) => { cell[i * 2] = i % 15; cell[i * 2 + 1] = Math.floor(i / 15); });
    g.setAttribute("aCell", new THREE.InstancedBufferAttribute(cell, 2));
    const ph = new Float32Array(NPAGES);
    data.forEach((p, i) => (ph[i] = p.ph));
    g.setAttribute("aPh", new THREE.InstancedBufferAttribute(ph, 1));
    return g;
  }, [data]);
  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        side: THREE.DoubleSide,
        transparent: true,
        uniforms: { uAtlas: { value: a.atlas }, uTime: { value: 0 }, uFade: { value: 1 } },
        vertexShader: `
          attribute vec2 aCell; attribute float aPh; uniform float uTime;
          varying vec2 vUv; varying vec2 vCell; varying float vShade; varying float vDepth;
          void main(){
            vUv = uv; vCell = aCell;
            vec3 p = position;
            p.z += sin(p.y * 4.0 + uTime * 3.0 + aPh) * 0.07 + sin(p.x * 5.0 + aPh * 2.0 + uTime * 2.0) * 0.03;
            mat4 m = modelMatrix * instanceMatrix;
            vec4 w = m * vec4(p, 1.0);
            vec3 n = normalize(mat3(m) * vec3(0.0, 0.0, 1.0));
            vShade = 0.55 + 0.45 * abs(dot(n, normalize(vec3(-0.4, 0.8, 0.6))));
            vec4 mv = viewMatrix * w;
            vDepth = -mv.z;
            gl_Position = projectionMatrix * mv;
          }`,
        fragmentShader: `
          uniform sampler2D uAtlas; uniform float uFade;
          varying vec2 vUv; varying vec2 vCell; varying float vShade; varying float vDepth;
          void main(){
            vec2 uv = (vCell + vec2(vUv.x, 1.0 - vUv.y)) / vec2(15.0, 12.0);
            uv.y = (vCell.y + (1.0 - vUv.y)) / 12.0;
            vec3 c = texture2D(uAtlas, uv).rgb;
            if (!gl_FrontFacing) c = c * 0.9;
            float edge = smoothstep(0.0, 0.03, min(min(vUv.x, 1.0 - vUv.x), min(vUv.y, 1.0 - vUv.y)));
            float a = uFade * smoothstep(0.6, 2.0, vDepth) * smoothstep(40.0, 24.0, vDepth);
            gl_FragColor = vec4(c * vShade * 0.82, a);
            if (edge < 0.001) discard;
          }`,
      }),
    [a],
  );
  const tmp = useMemo(() => ({ m: new THREE.Matrix4(), q: new THREE.Quaternion(), e: new THREE.Euler(), p: new THREE.Vector3(), s: new THREE.Vector3(1, 1, 1) }), []);
  useLayoutEffect(() => {
    const mesh = ref.current!;
    mat.uniforms.uTime.value = f / 30;
    const show = f >= BURST - 1 && f < COLLAPSE + 40;
    mesh.visible = show;
    if (!show) return;
    data.forEach((p, i) => {
      const t = f - BURST - p.delay;
      const x0 = bookX(p.book), y0 = 0.55;
      let px: number, py: number, pz: number;
      let ex = p.tumble[0] + p.tw[0] * t, ey = p.tumble[1] + p.tw[1] * t, ez = p.tumble[2] + p.tw[2] * t;
      if (t < 0) {
        px = x0; py = y0; pz = 0.0;
        tmp.s.set(0.001, 0.001, 0.001);
      } else {
        const burst = E.outCubic(clamp(t / 34));
        const th = p.th0 + p.w * t;
        const Rr = lerp(p.r0 * 0.6, p.R, burst);
        const cx = 0, cz = 0;
        const gx = x0 + (cx - x0) * burst * 0.8;
        px = gx + Math.cos(th) * Rr * burst;
        pz = cz + Math.sin(th) * Rr * burst * 0.9;
        py = y0 + p.h * burst + Math.sin(t * 0.06 + p.ph) * 0.25;
        // blow-out toward the camera
        const bo = clamp((f - COLLAPSE + p.delay * 0.15) / 22);
        const eo = bo * bo;
        px += (px - 0.0) * 2.2 * eo + 0.0;
        py += 1.0 * eo * (p.h - 1.5);
        pz += 14 * eo;
        const sc = Math.min(1, t / 8) * (1 - 0.0);
        tmp.s.set(sc, sc, sc);
      }
      tmp.p.set(px, py, pz);
      tmp.e.set(ex, ey, ez);
      tmp.q.setFromEuler(tmp.e);
      tmp.m.compose(tmp.p, tmp.q, tmp.s);
      mesh.setMatrixAt(i, tmp.m);
    });
    mesh.instanceMatrix.needsUpdate = true;
  });
  return <instancedMesh ref={ref} args={[geom, mat, NPAGES]} frustumCulled={false} />;
};

// ---------------------------------------------------------------- lights
const Lights: React.FC<{ f: number }> = ({ f }) => {
  const pls = useRef<(THREE.PointLight | null)[]>([]);
  useLayoutEffect(() => {
    pls.current.forEach((l, i) => {
      if (!l) return;
      const t = f - LANDS[i];
      const on = t < 0 ? 0 : 1;
      l.intensity = on * (0.45 + 5.0 * Math.exp(-Math.max(0, t) * 0.12));
    });
  });
  return (
    <>
      <ambientLight intensity={0.3} color="#8fb0d8" />
      <directionalLight position={[-3.5, 6.5, 5]} intensity={1.05} color="#fff4e6" castShadow shadow-mapSize={[2048, 2048]} shadow-camera-left={-7} shadow-camera-right={7} shadow-camera-top={7} shadow-camera-bottom={-4} shadow-camera-near={1} shadow-camera-far={25} shadow-bias={-0.0004} />
      <directionalLight position={[4, 2.5, -2]} intensity={0.55} color="#8fd0ff" />
      {COLORS.map((c, i) => (
        <pointLight key={i} ref={(r) => { pls.current[i] = r; }} position={[bookX(i), 1.3, 1.1]} color={c} intensity={0} distance={5.5} decay={1.6} />
      ))}
    </>
  );
};

const Rig: React.FC<{ f: number }> = ({ f }) => {
  const camera = useThree((s) => s.camera);
  useLayoutEffect(() => {
    const c = finaleCam(f);
    const shake = LANDS.reduce((s, l) => { const d = f - l; return s + (d >= 0 && d < 10 ? Math.exp(-d * 0.35) * (hash1(f * 3.1 + l) - 0.5) * 0.05 : 0); }, 0);
    camera.position.set(c.px + shake, c.py + shake * 0.7, c.pz);
    camera.lookAt(c.lx, c.ly, c.lz);
    (camera as THREE.PerspectiveCamera).fov = 32;
    (camera as THREE.PerspectiveCamera).updateProjectionMatrix();
    camera.updateMatrixWorld();
  });
  return null;
};

const Post: React.FC = () => {
  const { gl, scene, camera, size } = useThree();
  const composer = useMemo(() => {
    const c = new EffectComposer(gl);
    c.addPass(new RenderPass(scene, camera));
    c.addPass(new UnrealBloomPass(new THREE.Vector2(size.width, size.height), 0.32, 0.5, 0.92));
    c.addPass(new OutputPass());
    return c;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gl, scene, camera]);
  useLayoutEffect(() => {
    composer.setPixelRatio(1);
    composer.setSize(size.width, size.height);
  }, [composer, size]);
  useFrame(() => { composer.render(); }, 1);
  return null;
};

const Content: React.FC<{ f: number; assets: Assets }> = ({ f, assets }) => {
  return (
    <>
      <color attach="background" args={["#03070e"]} />
      <fog attach="fog" args={["#03070e", 14, 46]} />
      <Rig f={f} />
      <Lights f={f} />
      <Backdrop a={assets} f={f} />
      <Shelf />
      {BOOKS.map((_, i) => <BookMesh key={i} i={i} a={assets} f={f} />)}
      {[-5, -4, -3, -2, -1].map((k) => <GhostSlot key={"p" + k} x={bookX(k + 5)} f={f} k={k} />)}
      {Array.from({ length: 16 }, (_, k) => k + 1).map((k) => (
        <React.Fragment key={"g" + k}>
          <GhostSlot x={bookX(BOOKS.length - 1) + k * SLOT} f={f} k={k} />
          {k <= 6 && <GhostSlot x={bookX(0) - k * SLOT} f={f} k={k} />}
        </React.Fragment>
      ))}
      <Dust f={f} />
      <Pages a={assets} f={f} />
      <Post />
    </>
  );
};

export const FinaleScene: React.FC = () => {
  const f = useCurrentFrame();
  const [assets, setAssets] = useState<Assets | null>(null);
  const [handle] = useState(() => delayRender("finale assets"));
  useLayoutEffect(() => {
    let dead = false;
    buildAssets().then((a) => {
      if (!dead) setAssets(a);
    }).catch((e) => { console.error(e); });
    return () => { dead = true; };
  }, []);
  // release the hold only after the canvas (which takes its own hold in a layout effect) has mounted
  React.useEffect(() => {
    if (assets) continueRender(handle);
  }, [assets, handle]);
  if (!assets) return null;
  return (
    <ThreeCanvas
      width={1920}
      height={1080}
      dpr={1}
      shadows
      gl={{ antialias: true, alpha: false, preserveDrawingBuffer: true, powerPreference: "high-performance" }}
      camera={{ fov: 32, near: 0.1, far: 200, position: [0, 2.6, 6.8] }}
    >
      <Content f={f} assets={assets} />
    </ThreeCanvas>
  );
};
void smoothstep;


const _pc = new THREE.PerspectiveCamera(32, 1920 / 1080, 0.1, 200);
/** world -> 1920x1080 screen pixels for the finale camera at frame f */
export function finaleProject(x: number, y: number, z: number, f: number): [number, number] {
  const c = finaleCam(f);
  _pc.position.set(c.px, c.py, c.pz);
  _pc.lookAt(c.lx, c.ly, c.lz);
  _pc.updateProjectionMatrix();
  _pc.updateMatrixWorld();
  const v = new THREE.Vector3(x, y, z).project(_pc);
  return [(v.x * 0.5 + 0.5) * 1920, (-v.y * 0.5 + 0.5) * 1080];
}
