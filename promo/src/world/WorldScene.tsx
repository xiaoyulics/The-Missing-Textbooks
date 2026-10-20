import React, { useLayoutEffect, useMemo, useRef } from "react";
import { ThreeCanvas } from "@remotion/three";
import { useCurrentFrame } from "remotion";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { buildWorld, N } from "./shapes";
import { applyCam, worldCam } from "./camera";
import { stageAt } from "./timeline";
import { TerrainMesh } from "./TerrainMesh";
import { Racers } from "./Racers";
import { E, clamp, smoothstep } from "../lib/math";

const BG = new THREE.Color("#030f1b");

const VERT = /* glsl */ `
attribute vec3 aPos0; attribute vec3 aPos1; attribute vec3 aPos2; attribute vec3 aPos3;
attribute vec3 aPos4; attribute vec3 aPos5; attribute vec3 aPos6; attribute vec3 aPos7;
attribute vec4 aSigns; attribute vec4 aTone; attribute vec4 aRand; attribute vec3 aColB;
uniform float uA; uniform float uB; uniform float uMix; uniform float uTime;
uniform float uSize; uniform float uTurb; uniform float uReveal; uniform float uHeight;
uniform float uDim; uniform float uSpecPulse;
varying vec3 vColor; varying float vAlpha;

vec3 P(float k){
  if (k < 0.5) return aPos0; if (k < 1.5) return aPos1; if (k < 2.5) return aPos2; if (k < 3.5) return aPos3;
  if (k < 4.5) return aPos4; if (k < 5.5) return aPos5; if (k < 6.5) return aPos6; return aPos7;
}
float sg(float k){ if (k < 1.5) return aSigns.x; if (k < 2.5) return aSigns.y; if (k < 3.5) return aSigns.z; return aSigns.w; }
float tn(float k){ if (k < 1.5) return aTone.x; if (k < 2.5) return aTone.y; if (k < 3.5) return aTone.z; return aTone.w; }
vec3 colorFor(float k, vec3 p){
  if (k < 0.5) return vec3(0.42, 0.78, 1.0) * 0.8;
  if (k < 4.5) {
    vec3 warm = vec3(1.0, 0.58, 0.24), cool = vec3(0.26, 0.70, 1.0);
    return mix(cool, warm, step(0.0, sg(k))) * (0.55 + 0.9 * tn(k));
  }
  if (k < 5.5) return aColB * (1.0 + 0.5 * uSpecPulse);
  if (k < 6.5) return mix(vec3(0.25, 0.85, 1.0), vec3(1.0, 0.50, 0.30), smoothstep(-2.0, 2.0, p.y));
  float h = clamp(p.y / 3.2, 0.0, 1.0);
  return mix(vec3(0.20, 0.90, 0.85), vec3(1.0, 0.70, 0.30), h);
}
float gainFor(float k){ if (k < 0.5) return 0.075; if (k < 4.5) return 0.021; if (k < 5.5) return 0.034; if (k < 6.5) return 0.04; return 0.2; }
float szFor(float k){ if (k < 0.5) return 0.9; if (k < 4.5) return 1.0; if (k < 5.5) return 1.05; if (k < 6.5) return 1.0; return 0.8; }

void main(){
  vec3 pa = P(uA), pb = P(uB);
  float d = aRand.x;
  float m = clamp((uMix * 1.6 - d * 0.6), 0.0, 1.0);
  m = m * m * (3.0 - 2.0 * m);
  vec3 p = mix(pa, pb, m);
  float tb = sin(3.14159265 * m) * uTurb;
  p += tb * vec3(sin(p.y * 1.7 + uTime * 2.0 + aRand.z * 6.283),
                 sin(p.z * 1.3 + uTime * 1.7 + aRand.z * 3.1),
                 sin(p.x * 1.5 + uTime * 2.3 + aRand.z * 5.0));
  // slow breathing for orbital clouds
  float breathe = 1.0 + 0.012 * sin(uTime * 1.3 + aRand.z * 6.283);
  p *= mix(1.0, breathe, step(0.5, uA) * step(uA, 4.5));
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  float sz = mix(szFor(uA), szFor(uB), m) * uSize * (0.55 + aRand.y * 0.9);
  gl_PointSize = max(1.0, sz * projectionMatrix[1][1] * uHeight * 0.5 / -mv.z);
  vColor = mix(colorFor(uA, pa), colorFor(uB, pb), m);
  float vis = 1.0;
  if (uReveal < 1.0) vis = smoothstep(uReveal * 5.2, uReveal * 5.2 - 0.5, length(p.xz));
  vAlpha = vis * uDim * mix(gainFor(uA), gainFor(uB), m) * (0.55 + 0.45 * aRand.y);
}`;
const FRAG = /* glsl */ `
varying vec3 vColor; varying float vAlpha;
void main(){
  vec2 c = gl_PointCoord - 0.5;
  float d = dot(c, c) * 4.0;
  float a = exp(-d * 3.2);
  gl_FragColor = vec4(vColor, a * vAlpha);
}`;

const ParticleField: React.FC<{ f: number }> = ({ f }) => {
  const attrs = useMemo(buildWorld, []);
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    attrs.pos.forEach((p, k) => g.setAttribute(`aPos${k}`, new THREE.BufferAttribute(p, 3)));
    g.setAttribute("aSigns", new THREE.BufferAttribute(attrs.signs, 4));
    g.setAttribute("aTone", new THREE.BufferAttribute(attrs.tone, 4));
    g.setAttribute("aRand", new THREE.BufferAttribute(attrs.rand, 4));
    g.setAttribute("aColB", new THREE.BufferAttribute(attrs.colB, 3));
    g.setAttribute("position", new THREE.BufferAttribute(attrs.pos[0], 3)); // required by three for counts
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 100);
    return g;
  }, [attrs]);
  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uA: { value: 0 }, uB: { value: 0 }, uMix: { value: 0 }, uTime: { value: 0 }, uSize: { value: 0.05 },
          uTurb: { value: 0.7 }, uReveal: { value: 1 }, uHeight: { value: 1080 }, uDim: { value: 1 }, uSpecPulse: { value: 0 },
        },
      }),
    [],
  );
  const st = stageAt(f);
  useLayoutEffect(() => {
    const u = mat.uniforms;
    u.uA.value = st.a;
    u.uB.value = st.b;
    u.uMix.value = st.mix;
    u.uTime.value = f / 30;
    u.uSize.value = st.size;
    u.uTurb.value = st.turb;
    u.uReveal.value = st.reveal;
    u.uDim.value = st.dim;
    u.uSpecPulse.value = st.pulse;
  });
  return <points geometry={geom} material={mat} frustumCulled={false} />;
};

const Rig: React.FC<{ f: number }> = ({ f }) => {
  const camera = useThree((s) => s.camera);
  useLayoutEffect(() => {
    applyCam(camera, worldCam(f));
  });
  return null;
};

const Bloom: React.FC<{ strength?: number; radius?: number; threshold?: number }> = ({ strength = 0.75, radius = 0.6, threshold = 0.12 }) => {
  const { gl, scene, camera, size } = useThree();
  const composer = useMemo(() => {
    const c = new EffectComposer(gl);
    c.addPass(new RenderPass(scene, camera));
    c.addPass(new UnrealBloomPass(new THREE.Vector2(size.width, size.height), strength, radius, threshold));
    return c;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gl, scene, camera]);
  useLayoutEffect(() => {
    composer.setPixelRatio(1);
    composer.setSize(size.width, size.height);
  }, [composer, size]);
  useFrame(() => {
    composer.render();
  }, 1);
  return null;
};

export const WorldScene: React.FC<{ offset?: number }> = ({ offset = 0 }) => {
  const frame = useCurrentFrame() + offset;
  return (
    <ThreeCanvas
      width={1920}
      height={1080}
      dpr={1}
      gl={{ antialias: false, alpha: false, preserveDrawingBuffer: true, powerPreference: "high-performance" }}
      camera={{ fov: 35, near: 0.1, far: 200, position: [0, 0, 12] }}
    >
      <color attach="background" args={[BG.getStyle()]} />
      <Rig f={frame} />
      <TerrainMesh f={frame} />
      <ParticleField f={frame} />
      <Racers f={frame} />
      <Bloom />
    </ThreeCanvas>
  );
};
void E; void clamp; void smoothstep; void useRef;
