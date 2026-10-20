# The Missing Textbooks — promo film

A 112 s, 1920×1080, 30 fps promotional film for the series and its five volumes, built with [Remotion](https://www.remotion.dev).
Eight scenes on one 120 BPM beat grid (60 frames per bar); every picture hit has a matching sound in the score.

| bars | time | scene | what happens |
|---|---|---|---|
| 1–4 | 0:00 | **The missing** | five glass slabs refract a wall of notation; the books flicker and cannot exist; "Not yet."; a dive through the gap toward a white light |
| 5–8 | 0:08 | **Title** | 17 000 particles implode into THE / MISSING / TEXTBOOKS; the word *missing* drains into a dashed outline; five dashed books fill with colour; the camera falls into the second **O** |
| 9–16 | 0:16 | **Vol. 1 · Dissolving Problems** | hammer and chisel fail on the nut; the sea rises; the shell dissolves; only the kernel is left |
| 17–24 | 0:32 | **Vol. 2 · The Hydrogen Atoms of Deep Learning** | the kernel becomes a nucleus; Bohr rings → orbitals → the Balmer spectrum → the double-descent curve → a loss landscape |
| 25–32 | 0:48 | **Vol. 3 · Optimizers in the Wild** | SGD, momentum, Adam and Lion race down a banana valley; the ℓ₂ ball and the ℓ∞ ball |
| 33–40 | 1:04 | **Vol. 4 · Deep Learning as an Experimental Science** | ten runs of the same algorithm; verdict stamps that do what they say; here be dragons; OPEN |
| 41–48 | 1:20 | **Vol. 5 · Verifiable Intelligence** | the title page's own codeword, the 2048-bit Hadamard encoding of 1985, lies on the floor as a wall of digits; the verifier reads three bits at a time (the linearity test, run on the real bits), accepts the honest prover, and catches the one that cheats; the three questions every chapter answers |
| 49–56 | 1:36 | **Finale** | five real books drop on a shelf; a storm of real pages (180 sampled from the 1,818); "The shelf is not full." |

## Cuts

| composition | what it is |
|---|---|
| `Promo` | the master: English, with seven Chinese lines (the series' Chinese name twice, five short captions) |
| `PromoEN` | English only, for X and other English-language channels; same timing, same score |

`src/lang.tsx` holds the switch; every Chinese line in `src/scenes/` sits behind `useZh()`, and where its absence leaves a gap the English text moves into it.

## Run it

```bash
npm install
python3 scripts/prepare_assets.py      # covers + sample pages from the five PDFs (needs pdftoppm)
python3 audio/make_score.py            # synthesizes public/audio/score.wav (numpy/scipy)
npm run studio                         # preview (pick Promo or PromoEN)
npx remotion render src/index.ts PromoEN out/The-Missing-Textbooks-promo-EN-1080p.mp4 \
  --codec=h264 --crf=18 --color-space=bt709 --concurrency=5 --timeout=180000 --audio-codec=aac --audio-bitrate=256k
scripts/export_x.sh out/The-Missing-Textbooks-promo-EN-1080p.mp4 out/The-Missing-Textbooks-promo-EN-X-1080p.mp4
```

The first render downloads a headless Chrome (about 170 MB) into `node_modules`. A fresh clone reproduces the frames exactly: three frames rendered in a clean checkout matched the original render pixel for pixel. The scenes use macOS system fonts (Palatino, Songti SC, PingFang SC, Menlo), so on another system the text will be set in whatever the fallback is.

Always render with `--color-space=bt709`. Remotion's default converts RGB to YUV with the BT.601 matrix but leaves the matrix untagged, so most players decode HD video as Rec. 709 and every colour shifts by about ten code values (a vermilion of 217/72/43 comes out near 228/82/38).

## Posting on X

`scripts/export_x.sh` re-encodes a master for X: H.264 High level 4.2, 1080p30, AAC-LC 48 kHz, video capped at 12 Mbps with a one-second buffer (the peak one-second window measures about 20 Mbps), BT.709 tags, faststart. Third-party spec pages give X's limits for standard accounts as 140 s, 512 MB, 1920×1200 and 25 Mbps; the English cut is 112 s and 128 MB. X autoplays muted, and the film reads with the sound off.

The audio is encoded once, from the lossless score, after `audio/limit_truepeak.py`: a look-ahead limiter that keeps the true peak under -2 dBTP, which leaves about -1.3 dBTP after AAC and so headroom for X's own re-encode. It touches 1.7 % of the samples, by at most 1.7 dB, and the file stays at -16.1 LUFS. Re-encoding the master's AAC instead stacked two lossy generations and measured +0.1 dBTP; ffmpeg's `loudnorm` was rejected because in one pass it raised the loudness to -15 LUFS and squashed the loudness range from 6.9 to 4.7 LU. (The score's own ceiling in `make_score.py` limits sample peaks, so the masters sit at -0.9 dBTP; the X file is the one that is safe for re-encoding.)

`out/X-thumbnail.jpg` is the title frame (frame 379 of `PromoEN`, with the book row taken from frame 386 so that all five books have settled).

The title implosion (8.1–9.7 s) is the hardest content in the film: thousands of one-pixel particle trails plus film grain. It softens under any capped encode; raising the cap to a 40 Mbps peak only gains about 3 dB there, so the capped file is kept. The 1080p master keeps the full detail.

Stills for checking one or many frames in a single browser session:

```bash
node scripts/stills.mjs Promo 0,300,900 --scale 0.5 --out out/stills
python3 scripts/sheet.py out/sheet.png 3 640 out/stills/Promo-*.png
```

## Layout

- `src/Promo.tsx` — the timeline; `src/lib/timing.ts` — the beat grid and scene boundaries.
- `src/scenes/` — one file per scene; `src/world/` — the shared 3D world (particles, terrain, racers) used by volumes 2 and 3 and the finale.
- `src/world/codeword.ts` — volume 5's emblem: the Hadamard codeword of 1985 in the order of the book's title page (NumPy `default_rng(1985).permutation(2048)`, the call in `figures/scripts/00-emblem.py`), plus the linearity tests the film runs on it.
- `src/components/LensCanvas.tsx` — the refractive glass shader of the opening.
- `audio/` — the score: `dsp.py` is the synthesis kit, `make_score.py` places every sound at its picture cue.
- `scripts/check_sync.py` — finds where the picture changes and where the sound starts around every impact cue of a finished MP4 and reports the lag in frames.
- Fonts are the macOS system fonts (Palatino, Songti SC, Menlo), chosen to match the books.
