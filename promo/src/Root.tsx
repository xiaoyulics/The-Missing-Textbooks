import React from "react";
import { Composition } from "remotion";
import { Promo } from "./Promo";
import { FPS, TOTAL } from "./lib/timing";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/* the master cut: English with a few Chinese lines (the series' Chinese name and captions) */}
      <Composition id="Promo" component={Promo} defaultProps={{ lang: "bilingual" as const }} durationInFrames={TOTAL} fps={FPS} width={1920} height={1080} />
      {/* English only, for X and other English-language channels */}
      <Composition id="PromoEN" component={Promo} defaultProps={{ lang: "en" as const }} durationInFrames={TOTAL} fps={FPS} width={1920} height={1080} />
    </>
  );
};
