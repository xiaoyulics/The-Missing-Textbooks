import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { LensCanvas, LensFrame, SlabParams } from "../components/LensCanvas";
import { RevealText } from "../components/RevealText";
import { C, F } from "../theme";
import { E, clamp, hash1, rmap, smoothstep } from "../lib/math";
import { BEAT } from "../lib/timing";
import { useZh } from "../lang";

const DUR = 240;
const TAN = Math.tan((38 * Math.PI) / 360); // vertical fov 38°

// camera path: drift, then a long accelerating dive through the gap between the books
const DIVE0 = 138;
const DIVE1 = 226;
const camZ = (f: number) => {
  const u = clamp((f - DIVE0) / (DIVE1 - DIVE0));
  return -7.4 * Math.pow(u, 2.3);
};

// five slabs in a shallow fan; the dive goes through the gap between the third and the fourth
const slabBase = [
  { x: -3.36, z: -7.2, yaw: 0.36, bob: 0.0, idx: 0 },
  { x: -1.68, z: -6.6, yaw: 0.18, bob: 1.3, idx: 1 },
  { x: 0.0, z: -6.3, yaw: 0.0, bob: 2.6, idx: 2 },
  { x: 1.68, z: -6.6, yaw: -0.18, bob: 3.9, idx: 3 },
  { x: 3.36, z: -7.2, yaw: -0.36, bob: 5.2, idx: 4 },
];
const GAP_X = 0.84; // middle of the gap between slab 3 and slab 4
const camX = (f: number) => GAP_X * E.inOutCubic(clamp((f - 112) / 72));
/** frame at which slab k pops in (on the beat) */
const slabStart = (k: number) => 45 + k * BEAT;

export const S0Missing: React.FC = () => {
  const frame = useCurrentFrame();
  const zh = useZh();
  // without the Chinese line underneath, the English line sits a little higher to keep the same visual weight
  const textBottom = zh ? 118 : 150;
  const t = frame / 30;

  const slabs: SlabParams[] = slabBase.map((s, k) => {
    const start = slabStart(k);
    const a = clamp((frame - start) / 14);
    const appear = a <= 0 ? 0 : E.outBack(a, 1.4);
    const glow = smoothstep(start, start + 10, frame);
    // the covers try to exist: a flicker at "They don't."
    const flick = frame > 120 ? hash1(Math.floor(frame / 2) * 3.1 + k * 7.7) : 0;
    // the covers try to exist and cannot (flicker), then settle low, then become real as we fall toward the light
    const cover =
      frame < 118 ? 0 : frame < 150 ? 0.7 * (flick > 0.5 ? 1 : 0.08) : 0.28 + 0.14 * smoothstep(150, 185, frame) + 0.58 * smoothstep(186, 216, frame);
    return {
      c: [s.x, Math.sin(t * 0.9 + s.bob) * 0.06, s.z],
      yaw: s.yaw + Math.sin(t * 0.6 + s.bob) * 0.04,
      pitch: Math.sin(t * 0.5 + s.bob * 1.7) * 0.03,
      roll: 0,
      hs: [0.62, 0.83],
      r: 0.1,
      cover: cover * glow,
      glow,
      idx: s.idx,
      k: 0.9,
      appear,
    };
  });

  const u = clamp((frame - DIVE0) / (DIVE1 - DIVE0));
  const shutter = 0.5 + 2.5 * smoothstep(0.1, 0.9, u);
  const z0 = camZ(frame - shutter);
  const z1 = camZ(frame + shutter);
  const swayX = Math.sin(t * 0.35) * 0.18 * (1 - u);
  const swayY = Math.cos(t * 0.28) * 0.08 * (1 - u);
  const popShake = slabBase.reduce((acc, _s, k) => {
    const d = frame - slabStart(k);
    return acc + (d >= 0 && d < 8 ? Math.exp(-d * 0.6) * (hash1(frame * 3.7 + k) - 0.5) * 0.012 : 0);
  }, 0);
  const lf: LensFrame = {
    time: t,
    cam0: [swayX + camX(frame - shutter), swayY, z0],
    cam1: [swayX + camX(frame + shutter), swayY, z1],
    yaw: Math.sin(t * 0.3) * 0.03 * (1 - u),
    pitch: Math.sin(t * 0.22) * 0.015,
    roll: 0,
    tanHalf: TAN,
    samples: 1 + Math.round(7 * smoothstep(0.12, 0.7, u)),
    slabs,
    wallScroll: [t * 0.0035, t * 0.0018],
    wallBright: smoothstep(0, 40, frame) * (0.8 + 0.5 * smoothstep(0.3, 1, u)),
    light: 1.4 * smoothstep(0.5, 1.0, u),
    flash: smoothstep(0.86, 1.0, u) + 0.0,
    shake: [popShake, popShake * 0.6],
  };

  // text
  const l3In = 146;
  const l3 = clamp((frame - l3In) / 80);
  const l3Scale = 1 + 1.6 * E.inCubic(l3);
  const l3Op = 1 - smoothstep(180, 214, frame);

  return (
    <AbsoluteFill style={{ background: C.night }}>
      <LensCanvas frame={lf} />

      <AbsoluteFill style={{ alignItems: "center", justifyContent: "flex-end", paddingBottom: 118 }}>
        {frame < 130 && (
          <div style={{ position: "absolute", bottom: textBottom, textAlign: "center" }}>
            <RevealText
              text="Some textbooks should exist."
              start={30}
              exit={110}
              style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 74, color: "rgba(244,247,251,0.94)", letterSpacing: "0.005em" }}
            />
            {zh && (
              <>
                <div style={{ height: 14 }} />
                <RevealText
                  text="有些教科书，本该存在。"
                  start={52}
                  chars
                  stagger={2}
                  exit={110}
                  blur={8}
                  rise={10}
                  style={{ fontFamily: F.cn, fontSize: 32, color: "rgba(244,247,251,0.6)", letterSpacing: "0.5em" }}
                />
              </>
            )}
          </div>
        )}
        {frame >= 118 && frame < 190 && (
          <div style={{ position: "absolute", bottom: textBottom, textAlign: "center" }}>
            <RevealText
              text="They don’t."
              start={120}
              stagger={5}
              exit={140}
              style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 74, color: "rgba(244,247,251,0.96)" }}
            />
            {zh && (
              <>
                <div style={{ height: 14 }} />
                <RevealText
                  text="它们并不存在。"
                  start={128}
                  chars
                  stagger={2}
                  exit={140}
                  blur={8}
                  rise={10}
                  style={{ fontFamily: F.cn, fontSize: 32, color: "rgba(244,247,251,0.6)", letterSpacing: "0.5em" }}
                />
              </>
            )}
          </div>
        )}
      </AbsoluteFill>

      {frame >= l3In && frame < 218 && (
        <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", opacity: l3Op }}>
          <div style={{ transform: `scale(${l3Scale})`, textAlign: "center", filter: `blur(${E.inCubic(l3) * 5}px)` }}>
            <RevealText
              text="Not yet."
              start={l3In}
              stagger={6}
              dur={16}
              style={{
                fontFamily: F.serif,
                fontStyle: "italic",
                fontSize: 128,
                color: "#fff",
                textShadow: "0 0 36px rgba(0,12,32,0.95), 0 0 80px rgba(0,12,32,0.8), 0 0 30px rgba(120,200,255,0.5)",
              }}
            />
          </div>
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};
export const S0_DUR = DUR;
