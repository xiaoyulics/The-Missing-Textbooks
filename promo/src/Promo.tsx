import React from "react";
import { AbsoluteFill, Audio, Sequence, staticFile, useCurrentFrame } from "remotion";
import { S0Missing } from "./scenes/S0Missing";
import { S1Title } from "./scenes/S1Title";
import { S2Sea, S2_PRE } from "./scenes/S2Sea";
import { WorldScene } from "./world/WorldScene";
import { S3Atom } from "./scenes/S3Atom";
import { S4Wild } from "./scenes/S4Wild";
import { S5Lab } from "./scenes/S5Lab";
import { S6Finale } from "./scenes/S6Finale";
import { S7Verifiable } from "./scenes/S7Verifiable";
import { Film } from "./components/Film";
import { C } from "./theme";
import { SCENE } from "./lib/timing";
import { Lang, LangContext } from "./lang";

const S2Layer: React.FC = () => {
  const f = useCurrentFrame();
  const u = Math.min(1, Math.max(0, (f - (S2_PRE - 92)) / 92));
  const sc = 1 + 0.55 * Math.pow(1 - u, 2.2);
  return (
    <AbsoluteFill style={{ transform: `scale(${sc})`, transformOrigin: "50% 50%" }}>
      <S2Sea />
    </AbsoluteFill>
  );
};

const PromoInner: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: C.night }}>
      <Sequence from={SCENE.missing.from} durationInFrames={SCENE.missing.dur}>
        <S0Missing />
      </Sequence>
      <Sequence from={SCENE.v1.from - S2_PRE} durationInFrames={SCENE.v1.dur + S2_PRE}>
        <S2Layer />
      </Sequence>
      <Sequence from={SCENE.title.from} durationInFrames={SCENE.title.dur}>
        <S1Title />
      </Sequence>
      <Sequence from={SCENE.v2.from} durationInFrames={SCENE.v2.dur + SCENE.v3.dur}>
        <WorldScene />
      </Sequence>
      <Sequence from={SCENE.v2.from} durationInFrames={SCENE.v2.dur}>
        <S3Atom />
      </Sequence>
      <Sequence from={SCENE.v3.from} durationInFrames={SCENE.v3.dur}>
        <S4Wild />
      </Sequence>
      <Sequence from={SCENE.v4.from} durationInFrames={SCENE.v4.dur}>
        <S5Lab />
      </Sequence>
      <Sequence from={SCENE.v5.from} durationInFrames={SCENE.v5.dur}>
        <S7Verifiable />
      </Sequence>
      <Sequence from={SCENE.finale.from} durationInFrames={SCENE.finale.dur}>
        <S6Finale />
      </Sequence>
      <Film />
      <Audio src={staticFile("audio/score.wav")} />
    </AbsoluteFill>
  );
};

export const Promo: React.FC<{ lang?: Lang }> = ({ lang = "bilingual" }) => (
  <LangContext.Provider value={lang}>
    <PromoInner />
  </LangContext.Provider>
);
