import React, { useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import { heightUV, U0, U1, V0, V1, worldX, worldZ } from "./terrain";
import { smoothstep } from "../lib/math";

const VERT = /* glsl */ `
varying vec3 vP; varying vec3 vN;
void main(){ vP = position; vN = normal; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const FRAG = /* glsl */ `
varying vec3 vP; varying vec3 vN;
uniform float uFade; uniform float uTime; uniform float uPulse; uniform vec4 uBounds;
void main(){
  float h = vP.y;
  float t = h * 3.2;
  float fw = max(fwidth(t), 1e-4);
  float line = 1.0 - smoothstep(0.0, 1.4 * fw, abs(fract(t - 0.5) - 0.5));
  float tm = t / 5.0;
  float fm = max(fwidth(tm), 1e-4);
  float major = 1.0 - smoothstep(0.0, 1.6 * fm, abs(fract(tm - 0.5) - 0.5));
  float hh = clamp(h / 2.6, 0.0, 1.0);
  vec3 low = vec3(0.05, 0.85, 0.80), high = vec3(1.0, 0.62, 0.22);
  vec3 lineCol = mix(low, high, pow(hh, 0.75));
  float steep = clamp(0.30 / (fw * 7.0), 0.12, 1.0);
  vec3 col = lineCol * (line * 0.50 + major * 0.85) * steep;
  vec2 g = abs(fract(vP.xz * 2.0 - 0.5) - 0.5) / max(fwidth(vP.xz * 2.0), vec2(1e-4));
  float gl = 1.0 - min(min(g.x, g.y), 1.0);
  col += vec3(0.05, 0.16, 0.18) * gl * 0.55;
  col += vec3(0.0, 0.22, 0.26) * smoothstep(0.9, 0.0, h) * 0.4;
  float lam = clamp(dot(normalize(vN), normalize(vec3(-0.4, 0.8, 0.5))), 0.0, 1.0);
  col += mix(vec3(0.006, 0.03, 0.035), vec3(0.04, 0.07, 0.07), lam) * (0.6 + 0.8 * hh);
  float e = min(min(vP.x - uBounds.x, uBounds.y - vP.x), min(vP.z - uBounds.z, uBounds.w - vP.z));
  col *= smoothstep(0.0, 0.9, e);
  gl_FragColor = vec4(col * uFade, 1.0);
}`;

export const TerrainMesh: React.FC<{ f: number }> = ({ f }) => {
  const geom = useMemo(() => {
    const nx = 260, nz = 220;
    const pos = new Float32Array((nx + 1) * (nz + 1) * 3);
    let k = 0;
    for (let iz = 0; iz <= nz; iz++)
      for (let ix = 0; ix <= nx; ix++) {
        const u = U0 + ((U1 - U0) * ix) / nx, v = V0 + ((V1 - V0) * iz) / nz;
        pos[k++] = worldX(u);
        pos[k++] = heightUV(u, v);
        pos[k++] = worldZ(v);
      }
    const idx: number[] = [];
    for (let iz = 0; iz < nz; iz++)
      for (let ix = 0; ix < nx; ix++) {
        const a = iz * (nx + 1) + ix, b = a + 1, c = a + nx + 1, d = c + 1;
        idx.push(a, c, b, b, c, d);
      }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  }, []);
  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        uniforms: { uFade: { value: 0 }, uTime: { value: 0 }, uPulse: { value: 0 }, uBounds: { value: new THREE.Vector4(worldX(U0), worldX(U1), worldZ(V1), worldZ(V0)) } },
        side: THREE.DoubleSide,
      }),
    [],
  );
  useLayoutEffect(() => {
    mat.uniforms.uFade.value = smoothstep(452, 504, f);
    mat.uniforms.uTime.value = f / 30;
  });
  return <mesh geometry={geom} material={mat} frustumCulled={false} visible={f > 440} />;
};
