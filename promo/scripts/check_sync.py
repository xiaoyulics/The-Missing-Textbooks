#!/usr/bin/env python3
"""
Does the sound land where the picture changes?

For each impact cue (a time at which both something visible and something audible should happen) this finds
  - the video frame with the largest frame-to-frame change within +-WIN seconds, and
  - the audio onset (spectral flux peak) within +-WIN seconds,
and reports the lag in frames (positive = sound is late). Both are read from the finished MP4, so the check
includes the muxing offset.   usage: scripts/check_sync.py out/v5/The-Missing-Textbooks-5vol-EN-1080p.mp4
"""
import subprocess, sys
import numpy as np
import scipy.signal as sg

FPS = 30
WIN = 0.30
# (seconds, label): impacts that change many pixels at once, so that the picture has an unambiguous frame
CUES = [
    (8.00, "title implosion begins (flash)"), (9.50, "title particles arrive"),
    (18.50, "hammer 1"), (19.50, "hammer 2"), (20.50, "hammer 3 / chisel snaps"),
    (64.00, "volume 4 cut-in"), (69.00, "VANISHED stamp"),
    (70.50, "card 1"), (71.50, "card 2"), (72.50, "card 3"), (73.50, "card 4"), (74.50, "card 5"),
    (78.50, "OPEN stamp"), (80.00, "volume 5 begins"), (89.00, "Caught."),
    (96.00, "finale release"), (97.50, "book 1 lands"), (98.00, "book 2 lands"), (98.50, "book 3 lands"),
    (99.00, "book 4 lands"), (99.50, "book 5 lands"), (103.03, "page storm flash"),
]


def decode_video(path, w=96, h=54):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
    return a


def decode_audio(path, sr=44100):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32), sr


def main(path):
    v = decode_video(path)
    flux_v = np.r_[0.0, np.abs(np.diff(v, axis=0)).mean(axis=(1, 2))]          # change that frame k shows relative to k-1
    x, sr = decode_audio(path)
    hop = sr // 100
    f, t, Z = sg.stft(x, sr, nperseg=1024, noverlap=1024 - hop, boundary=None)
    mag = np.log1p(40 * np.abs(Z))
    flux_a = np.r_[0.0, np.maximum(0, np.diff(mag, axis=1)).sum(axis=0)]
    ta = t - (1024 / 2) / sr + 512 / sr * 0.0                                     # centre of each analysis window
    print(f"video: {len(v)} frames ({len(v)/FPS:.2f} s)   audio: {len(x)/sr:.2f} s")
    lags = []
    print(f"{'cue':>7s}  {'what':32s} {'video frame':>11s} {'audio onset':>11s} {'lag (frames)':>12s}")
    for cue, name in CUES:
        if cue + WIN > len(v) / FPS:
            continue
        k0, k1 = int((cue - WIN) * FPS), int((cue + WIN) * FPS) + 1
        kv = k0 + int(np.argmax(flux_v[k0:k1]))
        m = (ta >= cue - WIN) & (ta <= cue + WIN)
        # the onset is where the audio flux first reaches 60% of its peak in the window (the attack), not the peak itself
        seg = flux_a[m]; tseg = ta[m]
        pk = seg.max()
        first = np.argmax(seg >= 0.6 * pk)
        t_on = tseg[first]
        lag = (t_on - kv / FPS) * FPS
        lags.append(lag)
        flag = "" if abs(lag) <= 1.5 else "   <-- check"
        print(f"{cue:7.2f}  {name:32s} {kv:11d} {t_on*FPS:11.1f} {lag:12.1f}{flag}")
    lags = np.array(lags)
    print(f"\nmedian lag {np.median(lags):+.1f} frames, mean |lag| {np.abs(lags).mean():.1f}, within +-1.5 frames: {(np.abs(lags) <= 1.5).sum()} of {len(lags)}")


if __name__ == "__main__":
    main(sys.argv[1])
