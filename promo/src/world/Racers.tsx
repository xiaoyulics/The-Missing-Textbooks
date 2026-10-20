import React, { useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import { MIN_POS, RACERS, racerPos, RACE_F0, STEPS, stepAt, worldX, worldZ } from "./terrain";
import { smoothstep, TAU } from "../lib/math";

const SUB = 6;
const TRAIL_VERT = /* glsl */ `
attribute float aStep;
uniform float uNow; uniform float uSize; uniform float uHeight; uniform float uShow;
varying float vA;
void main(){
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float vis = step(aStep, uNow) * uShow;
  float age = uNow - aStep;
  vA = vis * (0.30 + 0.70 * (1.0 - smoothstep(0.0, 420.0, age)));
  float head = 1.0 - smoothstep(0.0, 30.0, age);
  gl_PointSize = max(1.0, uSize * (0.75 + 0.7 * head) * projectionMatrix[1][1] * uHeight * 0.5 / -mv.z);
}`;
const TRAIL_FRAG = /* glsl */ `
uniform vec3 uColor; varying float vA;
void main(){ vec2 c = gl_PointCoord - 0.5; float a = exp(-dot(c, c) * 4.0 * 3.0); gl_FragColor = vec4(uColor * 1.15, a * vA * 0.95); }`;

const HEAD_VERT = /* glsl */ `
attribute vec3 aColor; uniform float uHeight; uniform float uSize; uniform float uShow;
varying vec3 vC;
void main(){
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = max(1.0, uSize * projectionMatrix[1][1] * uHeight * 0.5 / -mv.z) * uShow;
  vC = aColor;
}`;
const HEAD_FRAG = /* glsl */ `
varying vec3 vC;
void main(){ vec2 c = gl_PointCoord - 0.5; float d = dot(c, c) * 4.0;
  float core = exp(-d * 16.0); float halo = exp(-d * 3.0);
  gl_FragColor = vec4(vC * halo * 0.65 + vec3(1.0) * core * 0.55, max(halo * 0.8, core)); }`;

export const Racers: React.FC<{ f: number }> = ({ f }) => {
  const racers = useMemo(() => RACERS(), []);
  const trails = useMemo(
    () =>
      racers.map((r, i) => {
        const n = STEPS * SUB + 1;
        const pos = new Float32Array(n * 3);
        const stp = new Float32Array(n);
        for (let k = 0; k < n; k++) {
          const p = racerPos(i, k / SUB, 0.045);
          pos.set(p, k * 3);
          stp[k] = k / SUB;
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        g.setAttribute("aStep", new THREE.BufferAttribute(stp, 1));
        const m = new THREE.ShaderMaterial({
          vertexShader: TRAIL_VERT,
          fragmentShader: TRAIL_FRAG,
          transparent: true,
          depthWrite: false,
          blending: THREE.NormalBlending,
          uniforms: {
            uNow: { value: 0 }, uSize: { value: 0.05 }, uHeight: { value: 1080 }, uShow: { value: 0 },
            uColor: { value: new THREE.Color(r.color) },
          },
        });
        return { g, m };
      }),
    [racers],
  );
  const heads = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(racers.length * 3), 3));
    const col = new Float32Array(racers.length * 3);
    racers.forEach((r, i) => {
      const c = new THREE.Color(r.color);
      col.set([c.r, c.g, c.b], i * 3);
    });
    g.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 100);
    const m = new THREE.ShaderMaterial({
      vertexShader: HEAD_VERT, fragmentShader: HEAD_FRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uHeight: { value: 1080 }, uSize: { value: 0.30 }, uShow: { value: 0 } },
    });
    return { g, m };
  }, [racers]);
  // step-shape auras (the unit ball of each optimiser's norm)
  const auras = useMemo(
    () =>
      racers.map((r) => {
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(65 * 3), 3));
        g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 100);
        const m = new THREE.LineBasicMaterial({ color: new THREE.Color(r.color), transparent: true, opacity: 0.0, blending: THREE.AdditiveBlending, depthWrite: false });
        return new THREE.Line(g, m);
      }),
    [racers],
  );
  const beam = useMemo(() => {
    const n = 500;
    const pos = new Float32Array(n * 3), a = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      pos.set([MIN_POS[0], t * 4.2, MIN_POS[2]], i * 3);
      a[i] = t;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("aStep", new THREE.BufferAttribute(a, 1));
    const m = new THREE.ShaderMaterial({
      vertexShader: `attribute float aStep; uniform float uHeight; uniform float uShow; varying float vA;
        void main(){ vec4 mv = modelViewMatrix*vec4(position,1.0); gl_Position = projectionMatrix*mv; vA = (1.0-aStep)*uShow; gl_PointSize = max(1.0, 0.07*projectionMatrix[1][1]*uHeight*0.5/-mv.z); }`,
      fragmentShader: `varying float vA; void main(){ vec2 c = gl_PointCoord-0.5; float a = exp(-dot(c,c)*12.0); gl_FragColor = vec4(vec3(1.0,0.92,0.7), a*vA*0.22); }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uHeight: { value: 1080 }, uShow: { value: 0 } },
    });
    return { g, m };
  }, []);

  useLayoutEffect(() => {
    const s = stepAt(f);
    const show = smoothstep(RACE_F0 - 60, RACE_F0 - 40, f);
    trails.forEach((t) => {
      t.m.uniforms.uNow.value = f < RACE_F0 ? -1 : s;
      t.m.uniforms.uShow.value = f < RACE_F0 ? 0 : 1;
    });
    const pos = heads.g.attributes.position as THREE.BufferAttribute;
    racers.forEach((_, i) => {
      const p = racerPos(i, f < RACE_F0 ? 0 : s, 0.08);
      pos.setXYZ(i, p[0], p[1], p[2]);
      // aura
      const a = auras[i];
      const ap = a.geometry.attributes.position as THREE.BufferAttribute;
      const path = racers[i].path;
      const k = Math.min(path.length - 2, Math.floor(f < RACE_F0 ? 0 : s));
      const dx = worldX(path[k + 1][0]) - worldX(path[k][0]), dz = worldZ(path[k + 1][1]) - worldZ(path[k][1]);
      const ang = Math.atan2(dz, dx);
      for (let j = 0; j <= 64; j++) {
        const t = (j / 64) * TAU;
        let x: number, z: number;
        if (i === 3) {
          // square: the l-infinity ball
          const q = (j / 64) * 4, side = Math.floor(q) % 4, r = q - Math.floor(q), h = 0.27;
          const pts = [[-h + 2 * h * r, -h], [h, -h + 2 * h * r], [h - 2 * h * r, h], [-h, h - 2 * h * r]];
          x = pts[side][0]; z = pts[side][1];
        } else if (i === 2) {
          // adaptive ellipse stretched along the motion
          const ex = Math.cos(t) * 0.44, ez = Math.sin(t) * 0.16;
          x = ex * Math.cos(ang) - ez * Math.sin(ang); z = ex * Math.sin(ang) + ez * Math.cos(ang);
        } else {
          x = Math.cos(t) * 0.27; z = Math.sin(t) * 0.27;
        }
        ap.setXYZ(j, p[0] + x, p[1] - 0.02, p[2] + z);
      }
      ap.needsUpdate = true;
      (a.material as THREE.LineBasicMaterial).opacity = (f < RACE_F0 ? 0 : 0.85) * smoothstep(RACE_F0 - 10, RACE_F0 + 6, f);
    });
    pos.needsUpdate = true;
    heads.m.uniforms.uShow.value = show;
    beam.m.uniforms.uShow.value = smoothstep(540, 580, f);
  });

  return (
    <group>
      {trails.map((t, i) => (
        <points key={i} geometry={t.g} material={t.m} frustumCulled={false} />
      ))}
      <points geometry={heads.g} material={heads.m} frustumCulled={false} />
      {auras.map((a, i) => (
        <primitive key={i} object={a} />
      ))}
      <points geometry={beam.g} material={beam.m} frustumCulled={false} />
    </group>
  );
};
