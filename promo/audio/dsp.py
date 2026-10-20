"""Small synthesis kit for the promo score (numpy only, deterministic)."""
import numpy as np
import scipy.signal as sg

SR = 44100
RNG = np.random.default_rng(2026)


def tt(dur):
    return np.arange(int(round(dur * SR))) / SR


def db(x):
    return 10 ** (x / 20.0)


def fade(y, a=0.003, r=0.01):
    n = len(y)
    na, nr = int(a * SR), int(r * SR)
    if na > 0:
        y[:na] *= np.linspace(0, 1, na)
    if nr > 0:
        y[-nr:] *= np.linspace(1, 0, nr)
    return y


def adsr(n, a, d, s, r):
    e = np.ones(n)
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na = min(na, n)
    e[:na] = np.linspace(0, 1, na) if na else 1
    nd = min(nd, n - na)
    if nd > 0:
        e[na:na + nd] = np.linspace(1, s, nd)
        e[na + nd:] = s
    nr = min(nr, n)
    if nr > 0:
        e[-nr:] *= np.linspace(1, 0, nr) * 1.0
    return e


def lowpass(x, fc, order=2):
    fc = min(fc, SR * 0.45)
    sos = sg.butter(order, fc / (SR / 2), "low", output="sos")
    return sg.sosfilt(sos, x)


def highpass(x, fc, order=2):
    sos = sg.butter(order, fc / (SR / 2), "high", output="sos")
    return sg.sosfilt(sos, x)


def bandpass(x, lo, hi, order=2):
    hi = min(hi, SR * 0.45)
    sos = sg.butter(order, [lo / (SR / 2), hi / (SR / 2)], "band", output="sos")
    return sg.sosfilt(sos, x)


def noise(dur):
    return RNG.standard_normal(int(round(dur * SR)))


def sine(f, dur, ph=0.0):
    return np.sin(2 * np.pi * f * tt(dur) + ph)


def glide(f0, f1, dur, curve="exp"):
    t = tt(dur)
    u = t / max(dur, 1e-9)
    if curve == "exp":
        f = f0 * (f1 / f0) ** u
    else:
        f = f0 + (f1 - f0) * u
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def saw(f, dur, detune_cents=0.0):
    t = tt(dur)
    f2 = f * 2 ** (detune_cents / 1200.0)
    return 2 * ((f2 * t) % 1.0) - 1


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)


NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def nm(name):
    """'D4' / 'Bb3' / 'F#5' -> frequency"""
    p = NOTE[name[0]]
    i = 1
    if name[i] in "#b":
        p += 1 if name[i] == "#" else -1
        i += 1
    o = int(name[i:])
    return midi(12 * (o + 1) + p)


# ------------------------------------------------------------------ instruments
def kick(dur=0.5, f0=165, f1=46, tau=0.04, decay=9.0, click=0.18):
    t = tt(dur)
    f = f1 + (f0 - f1) * np.exp(-t / tau)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * decay)
    y += click * highpass(noise(dur), 1500) * np.exp(-t * 350)
    return fade(np.tanh(y * 1.3), 0.001, 0.02)


def sub(f, dur, att=0.02, rel=0.15):
    t = tt(dur)
    y = np.sin(2 * np.pi * f * t) + 0.18 * np.sin(2 * np.pi * 2 * f * t)
    return y * adsr(len(t), att, 0.1, 0.9, rel)


def bell(f, dur=2.5, bright=1.0):
    t = tt(dur)
    parts = [(1.0, 1.0, 2.2), (2.76, 0.55, 3.4), (5.40, 0.30, 5.0), (8.93, 0.16 * bright, 8.0), (13.3, 0.08 * bright, 12.0)]
    y = sum(a * np.sin(2 * np.pi * f * p * t) * np.exp(-t * d) for p, a, d in parts)
    return fade(y * 0.5, 0.0008, 0.05)


def pluck(f, dur=0.6, idx=2.2, decay=4.5):
    t = tt(dur)
    I = idx * np.exp(-t * 7)
    y = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * t))
    y += 0.25 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 9)
    return fade(y * np.exp(-t * decay), 0.001, 0.02)


def epiano(f, dur=2.5):
    t = tt(dur)
    I = 1.6 * np.exp(-t * 3.0)
    y = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * t * 1.0))
    y += 0.35 * np.sin(2 * np.pi * f * 2 * t + 0.8 * I * np.sin(2 * np.pi * f * 14 * t)) * np.exp(-t * 6)
    y += 0.1 * np.sin(2 * np.pi * f * 3 * t) * np.exp(-t * 5)
    return fade(y * np.exp(-t * 1.5), 0.002, 0.2)


def pad(f, dur, cutoff=1500, att=1.0, rel=1.4, bright=1.0):
    t = tt(dur)
    y = (saw(f, dur, -7) + saw(f, dur, 0) + saw(f, dur, 8)) / 3.0
    y += 0.5 * np.sin(2 * np.pi * f * 0.5 * t)
    y = lowpass(y, cutoff * bright, 2)
    # slow movement
    y *= 1.0 + 0.06 * np.sin(2 * np.pi * 0.23 * t + f)
    return y * adsr(len(t), att, 0.0, 1.0, rel)


def chord_pad(freqs, dur, **kw):
    out = np.zeros(int(round(dur * SR)))
    for k, f in enumerate(freqs):
        out += pad(f, dur, **kw) / len(freqs) ** 0.5
    return out


def hat(dur=0.08, open_=False, bright=1.0):
    d = 0.35 if open_ else dur
    t = tt(d)
    y = highpass(noise(d), 7000 * bright, 3) * np.exp(-t * (14 if open_ else 55))
    return fade(y, 0.0005, 0.01)


def clap(dur=0.28):
    t = tt(dur)
    y = np.zeros(len(t))
    for off in (0.0, 0.011, 0.023):
        n = bandpass(noise(dur), 1100, 3600, 2) * np.exp(-np.maximum(t - off, 0) * 26)
        n[t < off] = 0
        y += n
    return fade(y * 0.7, 0.0005, 0.05)


def snare(dur=0.3):
    t = tt(dur)
    y = bandpass(noise(dur), 900, 7000, 2) * np.exp(-t * 18)
    y += 0.6 * np.sin(2 * np.pi * np.cumsum(190 + 70 * np.exp(-t * 30)) / SR) * np.exp(-t * 20)
    return fade(y, 0.0005, 0.04)


def tom(f=90, dur=0.5):
    t = tt(dur)
    fr = f * (1 + 0.6 * np.exp(-t * 28))
    y = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 8)
    y += 0.12 * highpass(noise(dur), 900) * np.exp(-t * 200)
    return fade(y, 0.0008, 0.03)


def thud(f=95, dur=0.45, paper=0.45):
    """a stamp / book landing: low body plus a papery slap"""
    t = tt(dur)
    fr = f * (1 + 0.8 * np.exp(-t * 35))
    y = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 11)
    s = bandpass(noise(dur), 700, 5200, 2) * np.exp(-t * 55) * paper
    return fade(y + s, 0.0006, 0.03)


def clank(dur=0.9, base=1900.0):
    t = tt(dur)
    parts = [(1.0, 1.0, 9), (1.47, 0.7, 11), (2.09, 0.5, 14), (2.56, 0.35, 18), (3.31, 0.22, 24)]
    y = sum(a * np.sin(2 * np.pi * base * p * t + 0.3 * p) * np.exp(-t * d) for p, a, d in parts)
    y += 0.6 * highpass(noise(dur), 2500) * np.exp(-t * 90)
    return fade(y * 0.4, 0.0004, 0.05)


def boom(dur=3.0, f0=70, f1=30, noise_amt=0.7):
    t = tt(dur)
    fr = f1 + (f0 - f1) * np.exp(-t * 6)
    y = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 1.3)
    nz = lowpass(noise(dur), 900, 2) * np.exp(-t * 2.6) * noise_amt
    return fade(np.tanh((y + nz) * 1.2), 0.002, 0.2)


def crash(dur=2.2):
    t = tt(dur)
    y = highpass(noise(dur), 1800, 2) * np.exp(-t * 2.2)
    return fade(y * 0.7, 0.002, 0.2)


def stft_sweep(dur, f0, f1, bw_oct=1.0, amp=None, curve="exp", seed=None):
    """noise passed through a moving band (smooth, no clicks)"""
    n = int(round(dur * SR))
    x = (np.random.default_rng(seed) if seed is not None else RNG).standard_normal(n)
    nper = 2048
    f, tm, Z = sg.stft(x, SR, nperseg=nper, noverlap=nper * 3 // 4)
    u = np.clip(tm / max(dur, 1e-9), 0, 1)
    fc = f0 * (f1 / f0) ** u if curve == "exp" else f0 + (f1 - f0) * u
    lf = np.log2(np.maximum(f, 1.0))[:, None]
    mask = np.exp(-0.5 * ((lf - np.log2(fc)[None, :]) / (bw_oct / 2.355)) ** 2)
    _, y = sg.istft(Z * mask, SR, nperseg=nper, noverlap=nper * 3 // 4)
    y = y[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    y = y / (np.max(np.abs(y)) + 1e-9)
    if amp is not None:
        y = y * amp(np.arange(n) / SR)
    return y


def riser(dur, f0=300, f1=9000, peak=1.0, power=2.0, tone=True, seed=None):
    n = int(round(dur * SR))
    amp = lambda t: (t / dur) ** power * peak
    y = stft_sweep(dur, f0, f1, bw_oct=2.2, amp=amp, seed=seed)
    if tone:
        t = tt(dur)
        u = t / dur
        fr = 180 * (6.0 ** u)
        trem = 0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(4 + 16 * u ** 2) / SR)
        tn = np.sin(2 * np.pi * np.cumsum(fr) / SR) * (0.35 + 0.65 * trem) * amp(t) * 0.35
        y = y[:n] + tn[:n]
    return y


def reverse(y):
    return y[::-1].copy()


def whoosh(dur=0.7, f0=500, f1=5000, peak=1.0, shape=2.0, seed=None):
    amp = lambda t: np.sin(np.pi * np.clip(t / dur, 0, 1)) ** shape * peak
    return stft_sweep(dur, f0, f1, bw_oct=1.6, amp=amp, seed=seed)


def reverb_ir(rt60=3.4, damp=5200, seed=5):
    rng = np.random.default_rng(seed)
    n = int(rt60 * 1.2 * SR)
    t = np.arange(n) / SR
    out = []
    for ch in range(2):
        x = rng.standard_normal(n)
        x = lowpass(x, damp, 2) * np.exp(-6.9 * t / rt60)
        x2 = lowpass(rng.standard_normal(n), 1200, 2) * np.exp(-6.9 * t / (rt60 * 1.35)) * 0.6
        ir = x + x2
        ir[: int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))
        out.append(ir / np.sqrt(np.sum(ir ** 2)))
    return np.stack(out, 1)


def convolve_rev(stereo, ir):
    L = sg.fftconvolve(stereo[:, 0], ir[:, 0])[: len(stereo)]
    R = sg.fftconvolve(stereo[:, 1], ir[:, 1])[: len(stereo)]
    return np.stack([L, R], 1)


class Bus:
    def __init__(self, dur):
        self.n = int(round(dur * SR))
        self.y = np.zeros((self.n, 2))

    def add(self, sig, t0, gain=1.0, pan=0.0):
        i0 = int(round(t0 * SR))
        if i0 >= self.n or i0 + len(sig) <= 0:
            return
        a = max(0, -i0)
        b = min(len(sig), self.n - i0)
        seg = sig[a:b]
        th = (pan + 1) * np.pi / 4
        self.y[i0 + a:i0 + b, 0] += seg * gain * np.cos(th)
        self.y[i0 + a:i0 + b, 1] += seg * gain * np.sin(th)

    def add_st(self, l, r, t0, gain=1.0):
        i0 = int(round(t0 * SR))
        if i0 >= self.n:
            return
        b = min(len(l), self.n - i0)
        self.y[i0:i0 + b, 0] += l[:b] * gain
        self.y[i0:i0 + b, 1] += r[:b] * gain

    def gate(self, t0, t1, ramp=0.02):
        """silence the bus between t0 and t1 (a 'breath' before a drop)"""
        i0, i1 = int(t0 * SR), int(t1 * SR)
        g = np.ones(self.n)
        r = int(ramp * SR)
        g[i0:i1] = 0
        g[max(0, i0 - r):i0] = np.linspace(1, 0, min(r, i0))
        g[i1:i1 + r] = np.linspace(0, 1, len(g[i1:i1 + r]))
        self.y *= g[:, None]


def sidechain(bus, kick_times, depth=0.6, rel=0.28, atk=0.004):
    g = np.ones(bus.n)
    for tk in kick_times:
        i0 = int(tk * SR)
        L = int((rel + atk) * SR)
        if i0 >= bus.n:
            continue
        e = np.ones(L)
        na = int(atk * SR)
        e[:na] = np.linspace(1, 1 - depth, na)
        e[na:] = (1 - depth) + depth * (1 - np.exp(-np.linspace(0, 5, L - na)))
        b = min(L, bus.n - i0)
        g[i0:i0 + b] = np.minimum(g[i0:i0 + b], e[:b])
    bus.y *= g[:, None]


# ------------------------------------------------------------------ digital voices for volume 5
def blip(f, dur=0.08, glide_to=None):
    """a short sine data-blip with a hint of its octave; optional downward/upward chirp"""
    t = tt(dur)
    if glide_to is None:
        ph = 2 * np.pi * f * t
    else:
        ph = 2 * np.pi * np.cumsum(f * (glide_to / f) ** (t / dur)) / SR
    y = np.sin(ph) * np.exp(-t * 36) + 0.25 * np.sin(2 * ph) * np.exp(-t * 62)
    return fade(y, 0.0004, 0.012)


def tick(dur=0.022, bright=1.0):
    """a dry click, like a counter flipping"""
    t = tt(dur)
    return highpass(noise(dur), 3500 * bright) * np.exp(-t * 170)


def siren(f0, f1, dur, cutoff=2800):
    """a falling saw alarm, distorted and decaying"""
    t = tt(dur)
    fr = f0 * (f1 / f0) ** (t / dur)
    y = 2 * ((np.cumsum(fr) / SR) % 1.0) - 1
    y = lowpass(y, cutoff, 2) * np.exp(-t * 2.4)
    return fade(np.tanh(y * 1.9), 0.002, 0.06)


def sonar(f=880.0, dur=1.3):
    """a round ping with a very small downward pitch bend"""
    t = tt(dur)
    y = np.sin(2 * np.pi * np.cumsum(f * (1 + 0.014 * np.exp(-t * 22))) / SR) * np.exp(-t * 3.6)
    return fade(y, 0.001, 0.06)


def swell_saw(freqs, dur, c0=350.0, c1=9000.0, att=1.6, rel=0.08, detunes=(-12, -5, 0, 6, 13)):
    """a stack of detuned saws whose filter opens through the note"""
    t = tt(dur)
    raw = np.zeros(len(t))
    for f in freqs:
        for d in detunes:
            raw += saw(f, dur, d)
    raw /= len(freqs) * len(detunes) ** 0.5
    cuts = np.geomspace(c0, c1, 5)
    layers = [lowpass(raw, c, 2) for c in cuts]
    pos = np.clip(t / dur, 0, 1) * (len(cuts) - 1)
    y = np.zeros(len(t))
    for i, L in enumerate(layers):
        w = np.clip(1 - np.abs(pos - i), 0, 1)
        y += L * w
    return y * adsr(len(t), att, 0.0, 1.0, rel)
