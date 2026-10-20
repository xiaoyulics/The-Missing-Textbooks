import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { FinaleScene, BURST, COLLAPSE, finaleProject, bookX, SLOT, BW, BH } from "../world/Finale";
import { Lockup } from "../components/Lockup";
import { RevealText } from "../components/RevealText";
import { BOOKS, TOTAL_PAGES } from "../data";
import { F } from "../theme";
import { E, clamp, smoothstep } from "../lib/math";
import { useZh } from "../lang";

const fmt = (n: number) => n.toLocaleString("en-US");

export const S6Finale: React.FC = () => {
  const f = useCurrentFrame();
  const zh = useZh();
  const whiteIn = 1 - smoothstep(0, 16, f); // the white-out that ends volume 5 clears
  // pages counter
  const cIn = smoothstep(BURST - 6, BURST + 10, f), cOut = 1 - smoothstep(COLLAPSE - 6, COLLAPSE + 4, f);
  const count = Math.round(TOTAL_PAGES * E.outCubic(clamp((f - BURST) / 80)));
  const flash = Math.exp(-Math.pow((f - 211) / 7, 2)) * 0.9;
  const lockOp = smoothstep(238, 242, f) * (1 - smoothstep(334, 352, f));
  const end = smoothstep(466, 479, f);
  return (
    <AbsoluteFill style={{ background: "#03070e" }}>
      <FinaleScene />
      <AbsoluteFill style={{ opacity: cIn * cOut }}>
        <div style={{ position: "absolute", left: 0, right: 0, bottom: 70, textAlign: "center", color: "#fff", textShadow: "0 0 40px rgba(120,190,255,0.5)" }}>
          <div style={{ fontFamily: F.serif, fontSize: 150, lineHeight: 1, fontVariantNumeric: "tabular-nums", letterSpacing: "0.02em" }}>{fmt(count)}</div>
          <div style={{ fontFamily: F.sans, fontSize: 26, letterSpacing: "0.5em", color: "rgba(244,247,251,0.75)", marginTop: 8, paddingLeft: "0.5em" }}>PAGES · {BOOKS.length} VOLUMES</div>
        </div>
      </AbsoluteFill>
      <Lockup t={f - 240} cx={960} cy={212} scale={0.6} opacity={lockOp} march={f} />
      {zh && (
        <AbsoluteFill style={{ opacity: smoothstep(238, 242, f) }}>
          <div style={{ position: "absolute", left: 0, right: 0, top: 372, textAlign: "center" }}>
            <RevealText text="不存在的教科书" start={266} chars stagger={3} dur={14} blur={8} rise={12} exit={294} exitDur={8}
              style={{ fontFamily: F.cn, fontWeight: 900, fontSize: 50, letterSpacing: "0.5em", color: "#f4f7fb", paddingLeft: "0.5em" }} />
          </div>
        </AbsoluteFill>
      )}
      <AbsoluteFill style={{ alignItems: "center" }}>
        <div style={{ position: "absolute", top: 368, textAlign: "center" }}>
          <RevealText text="Five are written." start={300} stagger={5} exit={344} exitDur={10} style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 64, color: "#fff", textShadow: "0 0 30px rgba(0,10,30,0.9)" }} />
        </div>
      </AbsoluteFill>
      <AbsoluteFill>
        <div style={{ position: "absolute", left: 120, top: 120 }}>
          <RevealText text="The shelf is not full." start={360} stagger={5} dur={16} style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 92, color: "#fff", textShadow: "0 0 40px rgba(0,10,30,0.9)" }} />
          {zh && (
            <>
              <div style={{ height: 14 }} />
              <RevealText text="书架还没满。" start={388} chars stagger={4} dur={14} blur={8} rise={12} style={{ fontFamily: F.cn, fontSize: 40, letterSpacing: "0.5em", color: "rgba(244,247,251,0.72)" }} />
            </>
          )}
        </div>
        <div style={{ position: "absolute", left: 0, right: 0, bottom: 64, textAlign: "center" }}>
          <RevealText text="WRITTEN BY CLAUDE  ·  CONCEIVED AND EDITED BY XIAOYU LI" start={420} stagger={0} dur={22} rise={10} blur={6}
            style={{ fontFamily: F.sans, fontSize: 24, letterSpacing: "0.34em", color: "rgba(244,247,251,0.72)", fontWeight: 600 }} />
        </div>
      </AbsoluteFill>
      {f >= 396 && (() => {
        // the next slot waits, with a cursor in it
        const x5 = bookX(BOOKS.length - 1) + SLOT;
        const [x0, y0] = finaleProject(x5 - BW / 2, BH, 0.12, f);
        const [x1, y1] = finaleProject(x5 + BW / 2, 0, 0.12, f);
        const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
        const op = smoothstep(396, 412, f) * (1 - smoothstep(466, 479, f));
        const blink = Math.floor(f / 15) % 2 === 0 ? 1 : 0.1;
        return (
          <AbsoluteFill style={{ opacity: op }}>
            <div style={{ position: "absolute", left: cx, top: y0 - 40, transform: "translateX(-50%)", fontFamily: F.sans, fontSize: 22, letterSpacing: "0.4em", color: "rgba(190,228,255,0.9)", whiteSpace: "nowrap", paddingLeft: "0.4em" }}>VOL. VI</div>
            <div style={{ position: "absolute", left: cx - 2, top: cy - (y1 - y0) * 0.1, width: 4, height: (y1 - y0) * 0.2, background: "rgba(220,240,255,0.95)", opacity: blink, boxShadow: "0 0 12px rgba(150,210,255,0.9)" }} />
          </AbsoluteFill>
        );
      })()}
      <AbsoluteFill style={{ background: "#fff", opacity: flash, pointerEvents: "none" }} />
      <AbsoluteFill style={{ background: "#fff", opacity: whiteIn, pointerEvents: "none" }} />
      <AbsoluteFill style={{ background: "#000", opacity: end }} />
    </AbsoluteFill>
  );
};
