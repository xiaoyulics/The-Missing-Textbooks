import * as THREE from "three";
import { E, clamp, lerp } from "../lib/math";
import { RACERS, racerPos, stepAt, START_POS } from "./terrain";

export type Cam = { D: number; az: number; el: number; tx: number; ty: number; tz: number; fov: number };

/** keyframes: [frame, value]; smooth (ease in-out) between neighbours */
export type Keys = [number, number][];
export const sample = (keys: Keys, f: number, ease: (t: number) => number = E.inOutCubic) => {
  if (f <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    if (f <= keys[i][0]) {
      const [f0, v0] = keys[i - 1], [f1, v1] = keys[i];
      return lerp(v0, v1, ease(clamp((f - f0) / (f1 - f0))));
    }
  }
  return keys[keys.length - 1][1];
};

// World time f: 0 = start of volume 2 (global frame 960). Volume 3 starts at f = 480.
const KEY = {
  D: [[0, 8.6], [70, 11.0], [104, 12.2], [184, 12.6], [226, 12.9], [410, 12.9], [480, 13.2]] as Keys,
  az: [[0, 0.15], [66, 0.35], [190, 2.3], [226, 0.0], [410, 0.0], [480, -0.33]] as Keys,
  el: [[0, 0.52], [66, 0.42], [190, 0.22], [226, 0.0], [410, 0.0], [480, 0.76]] as Keys,
  tx: [[0, 0], [410, 0], [480, 0.0]] as Keys,
  ty: [[0, 0], [410, 0], [480, 0.8]] as Keys,
};
// the race (volume 3)
const RK = {
  D: [[480, 13.2], [540, 12.6], [580, 11.2], [660, 10.2], [752, 9.4], [800, 12.2], [900, 12.8], [960, 14.5]] as Keys,
  az: [[480, -0.33], [540, -0.30], [580, -0.26], [660, -0.06], [752, 0.2], [800, 0.0], [900, -0.1], [960, -0.2]] as Keys,
  el: [[480, 0.76], [540, 0.86], [580, 0.88], [660, 0.85], [752, 0.80], [800, 1.2], [900, 1.28], [960, 1.38]] as Keys,
};

export function worldCam(f: number): Cam {
  if (f < 480) {
    return {
      D: sample(KEY.D, f),
      az: sample(KEY.az, f, E.inOutSine),
      el: sample(KEY.el, f, E.inOutSine),
      tx: sample(KEY.tx, f),
      ty: sample(KEY.ty, f),
      tz: 0,
      fov: 35,
    };
  }
  // target: overview -> start area -> centroid of the pack -> the minimum -> overview
  const st = START_POS();
  const s = stepAt(f);
  let cx = 0, cy = 0, cz = 0;
  const n = RACERS().length;
  for (let i = 0; i < n; i++) {
    const p = racerPos(i, s);
    cx += p[0] / n; cy += p[1] / n; cz += p[2] / n;
  }
  const a = E.inOutCubic(clamp((f - 500) / 70)); // overview -> start
  let tx = lerp(0.0, st[0] + 2.0, a * 0.7), ty = lerp(0.8, 0.9, a), tz = lerp(0, st[2], a * 0.4);
  const follow = E.inOutCubic(clamp((f - 584) / 40));
  tx = lerp(tx, cx * 0.85, follow); ty = lerp(ty, cy * 0.5, follow); tz = lerp(tz, cz * 0.85, follow);
  // after the race: rise to a top-down map, shifted so the terrain sits left of the interlude panel
  const toMap = E.inOutCubic(clamp((f - 756) / 44));
  tx = lerp(tx, 3.0, toMap); ty = lerp(ty, 0.2, toMap); tz = lerp(tz, 0.25, toMap);
  return {
    D: sample(RK.D, f),
    az: sample(RK.az, f, E.inOutSine),
    el: sample(RK.el, f, E.inOutSine),
    tx, ty, tz, fov: 35,
  };
}

export function applyCam(camera: THREE.Camera, c: Cam) {
  const pc = camera as THREE.PerspectiveCamera;
  pc.position.set(
    c.tx + c.D * Math.sin(c.az) * Math.cos(c.el),
    c.ty + c.D * Math.sin(c.el),
    c.tz + c.D * Math.cos(c.az) * Math.cos(c.el),
  );
  pc.lookAt(c.tx, c.ty, c.tz);
  pc.fov = c.fov;
  pc.near = 0.1;
  pc.far = 200;
  pc.updateProjectionMatrix();
  pc.updateMatrixWorld();
}

const tmpCam = new THREE.PerspectiveCamera(35, 1920 / 1080, 0.1, 200);
/** project a world point to 1920x1080 screen pixels using the camera at world-time f */
export function project(x: number, y: number, z: number, f: number, cam: Cam = worldCam(f)): [number, number, number] {
  applyCam(tmpCam, cam);
  tmpCam.aspect = 1920 / 1080;
  tmpCam.updateProjectionMatrix();
  const v = new THREE.Vector3(x, y, z).project(tmpCam);
  return [(v.x * 0.5 + 0.5) * 1920, (-v.y * 0.5 + 0.5) * 1080, v.z];
}
