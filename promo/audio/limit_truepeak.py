#!/usr/bin/env python3
"""
A small look-ahead true-peak limiter for the score.

The sample peak of a signal can sit well below its true peak (the peak of the reconstructed waveform between samples),
and an AAC encode overshoots a little more. This finds the true peaks on a 4x oversampled copy, builds one smooth gain
curve that keeps every one of them under the ceiling, and applies that curve to the original samples. Everything below
the ceiling is left exactly as it was.

usage: audio/limit_truepeak.py in.wav out.wav [ceiling_dBTP = -2.0]
"""
import sys, wave
import numpy as np
import scipy.signal as sg
from scipy.ndimage import minimum_filter1d, uniform_filter1d

OS = 4


def read(path):
    w = wave.open(path)
    sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).reshape(-1, ch).astype(np.float64) / 32768.0
    return x, sr


def write(path, x, sr):
    out = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(out.tobytes())


def true_peak_db(x):
    os_ = np.stack([sg.resample_poly(x[:, c], OS, 1) for c in range(x.shape[1])], 1)
    return 20 * np.log10(np.abs(os_).max() + 1e-12)


def limit(x, sr, ceiling_db):
    ceiling = 10 ** (ceiling_db / 20.0)
    os_ = np.stack([sg.resample_poly(x[:, c], OS, 1) for c in range(x.shape[1])], 1)
    env = np.abs(os_).max(axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))          # gain that would put this oversampled sample at the ceiling
    # fold the oversampled requirement back to the original rate (the smallest gain among the OS sub-samples)
    n = len(x)
    need = need[: n * OS].reshape(n, OS).min(axis=1)
    look = int(0.0015 * sr)                                          # 1.5 ms look-ahead / smoothing window
    g = minimum_filter1d(need, size=2 * look + 1, mode="nearest")    # hold the lowest gain across the window ...
    g = uniform_filter1d(g, size=2 * look + 1, mode="nearest")       # ... then smooth, which can never exceed the held minimum
    # a slow release so that the gain does not flutter between close peaks
    rel = np.exp(-1.0 / (0.040 * sr))
    out = np.empty_like(g)
    cur = 1.0
    for i in range(n):
        cur = g[i] if g[i] < cur else g[i] + (cur - g[i]) * rel
        out[i] = cur
    out = np.minimum(out, 1.0)
    return x * out[:, None], out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ceiling = float(sys.argv[3]) if len(sys.argv) > 3 else -2.0
    x, sr = read(src)
    before = true_peak_db(x)
    y, g = limit(x, sr, ceiling)
    # the smoothing is conservative, so check and top up once if a peak still pokes through
    after = true_peak_db(y)
    if after > ceiling + 0.05:
        y, g2 = limit(y, sr, ceiling)
        after = true_peak_db(y)
    write(dst, y, sr)
    print(f"true peak {before:+.2f} dBTP -> {after:+.2f} dBTP (ceiling {ceiling:+.1f});"
          f" gain reduced on {100.0 * (g < 0.999).mean():.2f}% of samples, by at most {-20 * np.log10(g.min()):.2f} dB")


if __name__ == "__main__":
    main()
