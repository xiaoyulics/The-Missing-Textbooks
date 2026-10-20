"""
Score for "The Missing Textbooks" promo.  120 BPM, D minor, 112 s.
Every picture hit has a time here; the same times drive the frames in src/ (frame = seconds * 30).
Usage: python3 audio/make_score.py  ->  public/audio/score.wav
"""
import sys, json, wave
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from dsp import *  # noqa

DUR = 112.0
BEAT = 0.5
F = lambda frame: frame / 30.0  # frame -> seconds

drums = Bus(DUR); bass = Bus(DUR); padb = Bus(DUR); arpb = Bus(DUR)
bells = Bus(DUR); fxb = Bus(DUR); hitb = Bus(DUR)
kicks = []  # for sidechain


def add_kick(t, g=0.9):
    drums.add(kick(), t, g, 0.0)
    kicks.append(t)


def four_floor(t0, t1, g=0.85):
    t = t0
    while t < t1 - 1e-6:
        add_kick(t, g)
        t += BEAT


def hats8(t0, t1, g=0.12, off=False):
    t = t0
    while t < t1 - 1e-6:
        drums.add(hat(0.07, False), t + (BEAT / 2 if off else 0), g * (0.7 if int(round((t - t0) / (BEAT / 2))) % 2 else 1.0), 0.25)
        t += BEAT / 2 if not off else BEAT


def hats16(t0, t1, g=0.09):
    t = t0
    i = 0
    while t < t1 - 1e-6:
        drums.add(hat(0.05), t, g * (1.0 if i % 4 == 0 else 0.55 if i % 2 == 0 else 0.35), -0.2 + 0.4 * (i % 2))
        t += BEAT / 4
        i += 1


def backbeat(t0, t1, g=0.55, use="clap"):
    t = t0 + BEAT  # beats 2 and 4
    while t < t1 - 1e-6:
        drums.add(clap() if use == "clap" else snare(), t, g, 0.0)
        t += BEAT * 2


def pad_chord(freqs, t0, dur, g=0.2, cutoff=1500, att=1.0, rel=1.4, bright=1.0):
    l = chord_pad(freqs, dur, cutoff=cutoff, att=att, rel=rel, bright=bright)
    r = chord_pad([f * 1.0035 for f in freqs], dur, cutoff=cutoff, att=att, rel=rel, bright=bright)
    padb.add_st(l, r, t0, g)


def arp(notes, t0, t1, step=BEAT / 2, g=0.1, pat="updown", oct_up=False, pan_w=0.5):
    seq = list(notes)
    if pat == "updown":
        seq = seq + seq[-2:0:-1]
    t = t0
    i = 0
    while t < t1 - 1e-6:
        f = seq[i % len(seq)] * (2 if oct_up else 1)
        arpb.add(pluck(f, 0.7), t, g * (1.0 if i % 4 == 0 else 0.75), pan_w * np.sin(i * 0.9))
        t += step
        i += 1


def bass_line(root, t0, t1, g=0.33, pattern="eighths"):
    t = t0
    i = 0
    while t < t1 - 1e-6:
        if pattern == "eighths":
            if i % 2 == 0 or i % 8 == 7:
                d = BEAT * 0.45
                bass.add(sub(root * (2 if (i % 8 == 7) else 1), d, 0.005, 0.08), t, g, 0.0)
        else:
            bass.add(sub(root, BEAT * 1.9, 0.02, 0.2), t, g, 0.0)
        t += BEAT / 2 if pattern == "eighths" else BEAT * 2
        i += 1


D2, A2, E3, F3, A3, C3, G3, Bb2, F2, C2, G2 = nm("D2"), nm("A2"), nm("E3"), nm("F3"), nm("A3"), nm("C3"), nm("G3"), nm("Bb2"), nm("F2"), nm("C2"), nm("G2")
Dm9 = [nm("D3"), nm("F3"), nm("A3"), nm("E4")]
Bbmaj7 = [nm("Bb2"), nm("D3"), nm("F3"), nm("A3")]
Fmaj7 = [nm("F2"), nm("A2"), nm("C3"), nm("E3")]
Cadd9 = [nm("C3"), nm("E3"), nm("G3"), nm("D4")]
Gm9 = [nm("G2"), nm("Bb2"), nm("D3"), nm("F3"), nm("A3")]
Dm = [nm("D3"), nm("F3"), nm("A3"), nm("D4")]
Bb = [nm("Bb2"), nm("D3"), nm("F3"), nm("Bb3")]
Fc = [nm("F2"), nm("A2"), nm("C3"), nm("F3")]
Cc = [nm("C3"), nm("E3"), nm("G3"), nm("C4")]
Dsus2 = [nm("D2"), nm("A2"), nm("E3"), nm("A3")]
arp_Dm = [nm("D4"), nm("F4"), nm("A4"), nm("D5")]
arp_Bb = [nm("Bb3"), nm("D4"), nm("F4"), nm("Bb4")]
arp_F = [nm("F4"), nm("A4"), nm("C5"), nm("F5")]
arp_C = [nm("C4"), nm("E4"), nm("G4"), nm("C5")]
arp_Dm9 = [nm("D4"), nm("F4"), nm("A4"), nm("E5")]

cues = {}

# =============================================================== S0  (0 - 8 s): the missing
# drone
t = tt(8.0)
drone = (np.sin(2 * np.pi * D2 * t) * 0.6 + lowpass(saw(D2, 8.0), 320) * 0.5 + np.sin(2 * np.pi * A2 * t) * 0.18)
drone *= np.minimum(1, t / 1.4) ** 1.5
padb.add(drone, 0.0, 0.34, 0.0)
bells.add(bell(nm('A5'), 2.5), 1.0, 0.12, 0.0)
# data hiss
hiss = highpass(noise(8.0), 5000) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)) * np.minimum(1, t / 1.5)
fxb.add(hiss, 0.0, 0.012, 0.0)
# glass slabs materialise
for k, (f, tk) in enumerate(zip(["D6", "F6", "A6", "C7", "E7"], [1.5, 2.0, 2.5, 3.0, 3.5])):
    bells.add(bell(nm(f), 3.0), tk, 0.24, -0.6 + 0.3 * k)
    fxb.add(whoosh(0.5, 900, 4200, 0.25, 1.5, seed=10 + k), tk - 0.3, 0.12, -0.6 + 0.3 * k)
# "They don't." - a dissonant cluster and glitch
for f in (nm("D2"), nm("Eb2") if False else midi(39), midi(42)):
    fxb.add(sine(f, 1.6) * adsr(int(1.6 * SR), 0.5, 0, 1, 0.8), 4.0, 0.07, 0)
g = np.random.default_rng(3)
for i in range(18):
    tg = 4.0 + g.random() * 1.0
    fxb.add(highpass(noise(0.025), 1500) * np.exp(-tt(0.025) * 80), tg, 0.14, g.random() * 2 - 1)
# the dive: riser, then a breath of silence before the drop
fxb.add(riser(3.1, 260, 9000, 0.55, 2.2, seed=21), 4.6, 1.0, 0.0)
fxb.add(riser(3.1, 500, 12000, 0.3, 3.0, tone=False, seed=22), 4.6, 1.0, 0.0)

# =============================================================== S1  (8 - 16 s): title
# 8.0 drop: boom + implosion
hitb.add(boom(3.6, 74, 29), 8.0, 0.95, 0.0)
fxb.add(crash(1.6), 8.0, 0.35, 0.0)
fxb.add(riser(1.45, 400, 14000, 0.55, 1.7, tone=False, seed=32), 8.05, 1.0, 0.0)
# 9.5 arrival: metallic hit
hitb.add(boom(2.6, 82, 33), 9.5, 0.8, 0.0)
hitb.add(crash(1.2), 9.5, 0.3, 0.0)
for f, p in (("D5", -0.3), ("A5", 0.3), ("D6", 0.0)):
    bells.add(bell(nm(f), 3.2), 9.5, 0.2, p)
add_kick(9.5, 1.0)
four_floor(10.0, 15.5, 0.8)
# the drain at 9.77
_g = glide(1400, 90, 0.95)
fxb.add(_g * np.exp(-tt(0.95)[:len(_g)] * 1.8), 9.77, 0.12, 0.0)
fxb.add(stft_sweep(0.95, 7000, 300, 1.4, amp=lambda x: np.exp(-x * 2.2), seed=33), 9.77, 0.45, 0.0)
# harmony
pad_chord(Dm9, 9.5, 4.5, 0.2, cutoff=1700)
pad_chord(Bbmaj7, 12.0, 2.5, 0.2, cutoff=1700)
pad_chord(Fmaj7, 14.0, 2.2, 0.2, cutoff=1700)
for r, a, b in ((D2, 9.5, 12.0), (Bb2 / 2, 12.0, 14.0), (F2 / 2 * 1.0, 14.0, 16.0)):
    bass.add(sub(r, b - a - 0.05), a, 0.28, 0.0)
arp(arp_Dm, 10.0, 12.0, g=0.1)
arp(arp_Bb, 12.0, 14.0, g=0.1)
arp(arp_F, 14.0, 15.5, g=0.1)
# five books fill in on the beats (local frames 75, 90, 105, 120, 135): a Dm9 arpeggio, one note per volume
for k, f in enumerate(["D5", "F5", "A5", "C6", "E6"]):
    bells.add(bell(nm(f), 2.6), 10.5 + 0.5 * k, 0.2, -0.6 + 0.3 * k)
    hitb.add(thud(110, 0.3, 0.2), 10.5 + 0.5 * k, 0.25, 0)
# the dive into the O (13.0 - 16.0)
fxb.add(riser(3.0, 220, 11000, 0.8, 2.6, seed=41), 13.0, 1.0, 0.0)
_g = glide(180, 1800, 3.0)
fxb.add(_g * (tt(3.0)[:len(_g)] / 3.0) ** 2.5, 13.0, 0.10, 0.0)
cues["s1_end"] = 16.0

# =============================================================== S2  (16 - 32 s): dissolving problems
# breath: a soft chord as the sea appears inside the O
pad_chord(Dm9, 15.9, 4.3, 0.17, cutoff=1300, att=0.6, rel=1.6)
pad_chord(Bbmaj7, 20.0, 4.3, 0.17, cutoff=1300, att=1.2, rel=1.6)
pad_chord(Fmaj7, 24.0, 4.3, 0.17, cutoff=1300, att=1.2, rel=1.6)
pad_chord(Cadd9, 28.0, 4.3, 0.17, cutoff=1100, att=1.2, rel=1.8)
for r, a in ((D2, 16.0), (Bb2 / 2, 20.0), (F2 / 2, 24.0), (C2, 28.0)):
    bass.add(sub(r, 3.9), a, 0.26, 0.0)
# ocean bed
t = tt(16.0)
ocean = lowpass(noise(16.0), 1300) * (0.55 + 0.45 * np.sin(2 * np.pi * 0.11 * t + 1.0))
rise = np.clip((t - 5.5) / 5.7, 0, 1) ** 2 * 0.9 + 0.12
fxb.add(ocean * rise, 16.0, 0.18, 0.0)
# sparse bells over the calm opening
for tk, f in ((16.5, "A5"), (17.5, "F5"), (18.0, "D5")):
    bells.add(bell(nm(f), 3.0), tk, 0.10, 0.2)
# hammer, hammer, snap  (18.5, 19.5, 20.5)
for k, tk in enumerate((18.5, 19.5, 20.5)):
    fxb.add(whoosh(0.16, 700, 3600, 0.5, 1.2, seed=50 + k), tk - 0.15, 0.25, 0.0)
    hitb.add(clank(1.0, 1800 + 120 * k), tk, 0.5 + 0.12 * k, 0.1)
    hitb.add(thud(90 - 6 * k, 0.4, 0.3), tk, 0.55, 0.0)
# 20.5: the chisel snaps: a glassy break and two small pieces landing
for i, f in enumerate(["F7", "D7", "A6", "F6", "D6"]):
    bells.add(bell(nm(f), 1.4, 1.4), 20.5 + 0.03 * i, 0.12, -0.4 + 0.2 * i)
hitb.add(crash(0.7), 20.5, 0.3, 0.0)
for tk, p in ((21.0, -0.4), (21.2, 0.4)):
    hitb.add(clank(0.5, 2600), tk, 0.14, p)
# sea rises 21.5 - 27.2, shell dissolves 22.7 - 25.3
fxb.add(stft_sweep(5.8, 250, 2200, 1.8, amp=lambda x: (x / 5.8) ** 1.6, seed=60), 21.5, 0.55, 0.0)
fxb.add(sine(44, 6.5) * adsr(int(6.5 * SR), 3.0, 0, 1, 1.5), 21.5, 0.26, 0.0)
g = np.random.default_rng(8)
for i in range(70):
    u = g.random()
    tk = 22.6 + 3.0 * u
    dens = np.sin(np.pi * u)
    if g.random() < dens:
        bells.add(bell(nm(["D6", "F6", "A6", "C7", "E6", "G6"][g.integers(0, 6)]), 0.9, 1.3) * 0.5, tk, 0.07, g.random() * 1.6 - 0.8)
# the kernel appears: a sustained shimmer
for f, gn in (("D5", 0.10), ("A5", 0.07), ("D6", 0.05)):
    s = sine(nm(f), 7.5) * adsr(int(7.5 * SR), 3.0, 0, 1, 2.0)
    bells.add(s, 24.8, gn, 0.0)
bass.add(sine(D2 / 2, 8.0) * adsr(int(8.0 * SR), 3.0, 0, 1, 2.0), 24.5, 0.3, 0.0)
# soft heartbeat while we sink
for k in range(8):
    drums.add(lowpass(kick(0.5, 110, 42, 0.05, 8.0), 220), 28.0 + k * 0.5 * 1.0, 0.45, 0.0)
    kicks.append(28.0 + k * 0.5)
fxb.add(reverse(whoosh(2.0, 300, 3000, 0.7, 1.0, seed=66)), 30.0, 0.25, 0.0)

# =============================================================== S3  (32 - 48 s): the hydrogen atoms
hitb.add(boom(2.4, 62, 31, 0.3), 32.0, 0.55, 0.0)
for k, f in enumerate(["D5", "A5", "D6", "F6", "A6"]):
    bells.add(bell(nm(f), 3.0), 32.1 + 0.34 * k, 0.11, -0.5 + 0.25 * k)
pad_chord(Dm9, 32.0, 4.2, 0.2, cutoff=2000, att=0.8, bright=1.1)
pad_chord([nm("Bb2"), nm("D3"), nm("F3"), nm("A3"), nm("E4")], 36.0, 4.2, 0.2, cutoff=2000, att=0.8)
pad_chord(Fmaj7 + [nm("A3")], 40.0, 4.2, 0.2, cutoff=2000, att=0.8)
pad_chord(Gm9, 44.0, 4.2, 0.2, cutoff=2000, att=0.8)
for r, a in ((D2, 32.0), (Bb2 / 2, 36.0), (F2 / 2, 40.0), (G2 / 2, 44.0)):
    bass.add(sub(r, 3.9), a, 0.3, 0.0)
# orbitals: whoosh up, soft thump on arrival (34.2, 35.53, 36.87)
four_floor(34.0, 47.5, 0.55)
for k, tk in enumerate((34.2, 35.53, 36.87)):
    fxb.add(whoosh(0.9, 400, 5000, 0.6, 1.2, seed=70 + k), tk - 0.1, 0.3, 0.0)
    hitb.add(thud(70, 0.5, 0.0), tk + 0.9, 0.3, 0.0)
arp([nm("D5"), nm("F5"), nm("A5"), nm("E5")], 34.2, 36.0, step=BEAT / 4, g=0.07, pat="up", pan_w=0.7)
arp([nm("Bb4"), nm("D5"), nm("F5"), nm("A5")], 36.0, 38.0, step=BEAT / 4, g=0.07, pat="up", pan_w=0.7)
arp([nm("A4"), nm("C5"), nm("F5"), nm("E5")], 40.0, 42.0, step=BEAT / 4, g=0.07, pat="up", pan_w=0.7)
# spectrum (38.2 - 39.7) and its four lines
fxb.add(riser(1.6, 500, 11000, 0.65, 1.6, seed=80), 38.1, 1.0, 0.0)
hitb.add(crash(1.0), 39.7, 0.2, 0.0)
for k, (tk, f) in enumerate(zip((39.87, 40.13, 40.40, 40.67), ["D5", "F5", "A5", "C6"])):
    bells.add(bell(nm(f), 3.0), tk, 0.2, -0.5 + 0.33 * k)
arp([nm("D5"), nm("A5"), nm("F5"), nm("C6")], 41.0, 42.0, step=BEAT / 4, g=0.06, pat="up", pan_w=0.7)
# the curve (42.1 - 43.5)
fxb.add(reverse(whoosh(1.4, 600, 7000, 0.9, 1.0, seed=82)), 42.1, 0.5, 0.0)
bells.add(bell(nm("D6"), 4.0), 43.4, 0.18, 0.0)
# simulation dots: a sparkle run (44.0 - 45.1)
g = np.random.default_rng(12)
penta = ["D5", "F5", "G5", "A5", "C6", "D6", "F6", "A6"]
for i in range(34):
    tk = 44.0 + i * 0.033
    bells.add(bell(nm(penta[min(7, int(i / 34 * 8) + g.integers(-1, 2) % 2)]), 0.6, 1.3) * 0.6, tk, 0.055, g.random() * 1.2 - 0.6)
# terrain 45.7 - 47.8: falling
_g = glide(900, 70, 2.1)
fxb.add(_g * np.exp(-tt(2.1)[:len(_g)] * 0.8), 45.7, 0.14, 0.0)
fxb.add(stft_sweep(2.1, 6000, 250, 1.6, amp=lambda x: np.sin(np.pi * np.clip(x / 2.1, 0, 1)) ** 1.5, seed=84), 45.7, 0.5, 0.0)
fxb.add(riser(0.9, 400, 9000, 0.6, 2.0, tone=False, seed=85), 47.1, 1.0, 0.0)

# =============================================================== S4  (48 - 64 s): optimizers in the wild
hitb.add(boom(2.6, 80, 31), 48.0, 0.85, 0.0)
hitb.add(crash(1.4), 48.0, 0.3, 0.0)
pad_chord([nm("D3"), nm("F3"), nm("A3"), nm("D4"), nm("F4")], 48.0, 4.2, 0.22, cutoff=2200, att=0.4)
pad_chord([nm("Bb2"), nm("D3"), nm("F3"), nm("Bb3"), nm("D4")], 52.0, 4.2, 0.22, cutoff=2200, att=0.4)
pad_chord([nm("F2"), nm("A2"), nm("C3"), nm("F3"), nm("A3")], 56.0, 4.2, 0.22, cutoff=2200, att=0.4)
pad_chord([nm("C3"), nm("E3"), nm("G3"), nm("C4"), nm("E4")], 60.0, 4.2, 0.22, cutoff=2200, att=0.4)
# legend ticks
for k in range(4):
    bells.add(bell(nm(["D6", "F6", "A6", "C7"][k]), 0.8, 1.4) * 0.5, 50.3 + 0.3 * k, 0.08, -0.5 + 0.33 * k)
# pickup fill into the race (starts 51.33)
for i in range(10):
    drums.add(snare(0.2), 50.4 + i * 0.1, 0.12 + 0.05 * i, 0.0)
hitb.add(chord_pad([nm("D3"), nm("A3"), nm("F4"), nm("A4")], 0.6, cutoff=3500, att=0.01, rel=0.3), 51.33, 0.5, 0.0)
hitb.add(crash(0.9), 51.33, 0.18, 0.0)
# the beat: 51.5 - 57.0 full, 57.0 - 57.5 break, 57.5 - 62 filtered, 62 - 64 build
four_floor(51.5, 57.0, 0.9)
backbeat(51.5, 57.0, 0.55, "snare")
hats8(51.5, 57.0, 0.12)
hats16(51.5, 57.0, 0.04)
bass_line(D2 / 2 * 2, 51.5, 52.0, 0.3)
bass_line(Bb2, 52.0, 56.0, 0.3)
bass_line(F2, 56.0, 57.0, 0.3)
arp(arp_Dm, 51.5, 52.0, g=0.085)
arp(arp_Bb, 52.0, 56.0, g=0.085)
arp(arp_F, 56.0, 57.0, g=0.085)
fxb.add(riser(0.5, 800, 8000, 0.5, 1.4, tone=False, seed=90), 57.0, 1.0, 0.0)
# interlude: the norm panel (57.73 - 62.7)
four_floor(57.5, 62.0, 0.5)
hats8(57.5, 62.0, 0.07, off=True)
arp(arp_C, 57.5, 60.0, g=0.07, step=BEAT)
arp(arp_Dm9, 60.0, 62.0, g=0.07, step=BEAT)
fxb.add(whoosh(0.7, 900, 5200, 0.5, 1.2, seed=91), 57.55, 0.3, 0.0)
bass_line(C2, 57.5, 60.0, 0.24, "long")
bass_line(D2, 60.0, 62.0, 0.24, "long")
# build to the smash cut at 64.0
fxb.add(riser(2.0, 300, 13000, 0.9, 2.4, seed=92), 62.0, 1.0, 0.0)
for i in range(16):
    drums.add(snare(0.2), 62.0 + i * 0.125, 0.10 + 0.04 * i, 0.0)
cues["s4_end"] = 64.0

# =============================================================== S5  (64 - 80 s): the verdicts
hitb.add(boom(1.8, 90, 36, 0.9), 64.0, 0.9, 0.0)
hitb.add(crash(1.0), 64.0, 0.45, 0.0)
hitb.add(thud(120, 0.4, 0.8), 64.0, 0.6, 0.0)
pad_chord([nm("D2"), nm("A2"), nm("D3")], 64.0, 8.0, 0.14, cutoff=900, att=0.3)
# lab beat
for t0 in np.arange(66.0, 77.0, 2.0):
    add_kick(t0, 0.6)
    add_kick(t0 + 1.0, 0.5)
    drums.add(clap(), t0 + 0.5, 0.35, 0.0)
    drums.add(clap(), t0 + 1.5, 0.35, 0.0)
    for k in range(4):
        drums.add(hat(0.04, False, 1.3), t0 + k * 0.5 + 0.25, 0.10, 0.3)
for r, a in ((D2, 66.0), (D2, 68.0), (Bb2, 70.0), (Bb2, 72.0), (F2, 74.0), (C2, 76.0)):
    bass.add(sub(r, 1.9, 0.01, 0.1), a, 0.26, 0.0)
# curves draw (66.5 - 67.4), "ten runs" (68.3), VANISHED (69.0)
for _t, _a, _b, _p in ((66.5, 300, 900, -0.3), (66.9, 220, 700, 0.3)):
    _g = glide(_a, _b, 0.6, "lin")
    fxb.add(_g * np.hanning(len(_g)), _t, 0.07, _p)
fxb.add(whoosh(0.8, 500, 6000, 0.5, 1.2, seed=100), 68.2, 0.3, 0.0)
hitb.add(thud(70, 0.8, 1.0), 69.0, 0.9, 0.0)
hitb.add(boom(1.2, 80, 38, 0.5), 69.0, 0.35, 0.0)
fxb.add(stft_sweep(1.4, 8000, 300, 1.8, amp=lambda x: np.exp(-x * 2.0), seed=101), 69.05, 0.4, 0.0)
# five cards: 70.5, 71.5, 72.5, 73.5, 74.5
for k, tk in enumerate((70.5, 71.5, 72.5, 73.5, 74.5)):
    fxb.add(whoosh(0.3, 600, 4200, 0.5, 1.2, seed=110 + k), tk - 0.42, 0.22, -0.6 if k % 2 == 0 else 0.6)
    hitb.add(thud(86 + 6 * k, 0.5, 0.65), tk, 0.85, 0.0)
fxb.add(reverse(whoosh(0.5, 300, 3000, 0.5, 1.0, seed=120)), 70.6, 0.2, 0.0)   # REVERSED: the flip
_g = glide(200, 900, 0.7, "lin")
_m = np.sin(2 * np.pi * np.cumsum(7 + 12 * np.linspace(0, 1, len(_g))) / SR) ** 2
fxb.add(_g * _m, 71.55, 0.1, 0.0)  # tangle
_g = glide(700, 300, 0.5)
fxb.add(_g * np.hanning(len(_g)), 72.55, 0.1, 0.0)  # artifact smooths
for j in range(3):
    bells.add(bell(nm("A5") * (1 - 0.06 * j), 0.5, 1.3) * 0.5, 73.7 + 0.1 * j, 0.1 / (1 + j), 0.4 * (j - 1))  # unreplicated echoes
_g = glide(2200, 600, 0.25)
fxb.add(_g * np.hanning(len(_g)), 74.6, 0.1, 0.0)  # strike-through
# the storm: twelve stamps in 1.4 s, hats roll, riser
storm = [330, 334, 337, 341, 345, 349, 352, 356, 360, 364, 367, 371]
for i, fr in enumerate(storm):
    tk = 64.0 + F(fr)   # the frames are local to the act, which starts at 64.0 s
    hitb.add(thud(80 + 7 * i, 0.4, 0.7), tk, 0.7, -0.5 + (i % 5) * 0.25)
fxb.add(riser(1.5, 600, 12000, 0.7, 1.6, tone=True, seed=130), 74.9, 1.0, 0.0)
hats16(75.0, 77.2, 0.10)
# cards fall (76.87), calm plot (77.2), OPEN (78.5: local frame 435)
fxb.add(whoosh(0.7, 3500, 250, 0.6, 1.0, seed=140), 76.8, 0.3, 0.0)
for f, p in (("D4", -0.3), ("A4", 0.0), ("F5", 0.3)):
    bells.add(bell(nm(f), 2.0), 77.2, 0.07, p)
hitb.add(boom(3.2, 84, 31), 78.5, 1.0, 0.0)
hitb.add(thud(70, 0.8, 1.0), 78.5, 0.9, 0.0)
hitb.add(crash(1.8), 78.5, 0.5, 0.0)
hitb.add(chord_pad([nm("D3"), nm("A3"), nm("F4"), nm("A4")], 1.2, cutoff=4000, att=0.005, rel=0.9), 78.5, 0.45, 0.0)
# the vermilion flood (79.07 - 79.93) carries into volume 5
fxb.add(riser(1.5, 200, 9000, 0.9, 2.2, seed=150), 78.5, 1.0, 0.0)
cues["s5_end"] = 80.0

# =============================================================== S7  (80 - 96 s): verifiable intelligence
# All times come from the act's local frames in src/scenes/S7Verifiable.tsx (the act starts at 80.0 s = frame 0).
T7 = lambda fr: 80.0 + fr / 30.0
Adom = [nm("A2"), nm("C#3"), nm("E3"), nm("G3"), nm("Bb3")]          # A7(b9): the tension of a lie
penta_hi = [nm(x) for x in ("D5", "F5", "G5", "A5", "C6", "D6", "F6", "G6", "A6", "C7")]

# 80.0: the vermilion drains; one deep boom, then a hush while 2048 digits decode
hitb.add(boom(3.4, 78, 30, 0.5), 80.0, 0.85, 0.0)
hitb.add(crash(1.6), 80.0, 0.28, 0.0)
pad_chord(Dm9 + [nm("A4")], 79.95, 4.3, 0.2, cutoff=1500, att=0.6, rel=1.2)
bass.add(sub(D2, 3.9), 80.0, 0.3, 0.0)
fxb.add(highpass(noise(4.0), 6000) * (0.4 + 0.6 * np.sin(2 * np.pi * 0.5 * tt(4.0)) ** 2), 80.0, 0.02, 0.0)
# the decode wave runs outward from the middle of the wall (frames 6 - 44)
_g = glide(160, 1400, 1.3)
fxb.add(_g * np.hanning(len(_g)), T7(6), 0.05, 0.0)
g = np.random.default_rng(80)
for i in range(64):
    u = i / 63
    bells.add(blip(penta_hi[g.integers(0, len(penta_hi))], 0.08), T7(6) + (T7(44) - T7(6)) * u ** 0.85, 0.07 + 0.05 * u, g.random() * 1.6 - 0.8)
# the question, as text: "What can a weak verifier learn from a strong prover that may lie?"
bells.add(bell(nm("D6"), 3.0), T7(74), 0.12, -0.2)
bells.add(bell(nm("A6"), 3.0), T7(96), 0.10, 0.2)
fxb.add(reverse(whoosh(1.2, 400, 5000, 0.6, 1.0, seed=181)), 82.8, 0.25, 0.0)

# 84.0: "Read three bits." - the kick arrives; the three rings of the title page's test pop on the beats
pad_chord(Bbmaj7 + [nm("E4")], 84.0, 4.0, 0.22, cutoff=2400, att=0.4)
bass.add(sub(Bb2 / 2, 3.9), 84.0, 0.3, 0.0)
four_floor(84.0, 88.0, 0.62)
hats8(84.0, 88.0, 0.07, off=True)
for k, (fr, f) in enumerate(((135, "D5"), (150, "F5"), (165, "A5"))):
    bells.add(bell(nm(f), 2.4), T7(fr), 0.22, -0.3 + 0.3 * k)
    hitb.add(tick(), T7(fr), 0.25, 0.0)
# 86.0 (frame 180): the test accepts, 1 + 1 = 0
for f, gn in (("D5", 0.16), ("A5", 0.15), ("D6", 0.14), ("F6", 0.10), ("C7", 0.07)):
    bells.add(bell(nm(f), 3.0), T7(180), gn, 0.0)
hitb.add(thud(84, 0.5, 0.2), T7(180), 0.4, 0.0)
# four quick tests (frames 195, 201, 207, 213), then twenty in a flurry (219 - 239)
for k, fr in enumerate((195, 201, 207, 213)):
    bells.add(bell(nm(["D6", "F6", "A6", "C7"][k]), 1.2, 1.2), T7(fr), 0.11, -0.4 + 0.27 * k)
    hitb.add(tick(), T7(fr), 0.18, 0.0)
arp([nm("Bb4"), nm("D5"), nm("F5"), nm("A5"), nm("C6"), nm("D6")], T7(195), T7(240), step=BEAT / 4, g=0.07, pat="up", pan_w=0.7)
for i in range(20):
    u = i / 19
    tk = T7(219) + (T7(239) - T7(219)) * u ** 0.9
    bells.add(blip(penta_hi[min(9, int(u * 9.99))], 0.06), tk, 0.09, -0.6 + 1.2 * u)
fxb.add(riser(0.9, 600, 12000, 0.7, 1.8, tone=False, seed=182), 87.1, 1.0, 0.0)

# 88.0 (frame 240): "Now the prover cheats." - the kick drops out and the harmony turns to A7(b9)
pad_chord(Adom, 88.0, 2.3, 0.24, cutoff=1900, att=0.15, rel=0.6, bright=1.1)
bass.add(sub(A2 / 2, 2.2, 0.03, 0.2), 88.0, 0.34, 0.0)
hitb.add(thud(58, 0.8, 0.0), 88.0, 0.5, 0.0)
fxb.add(sine(nm("D3") * 0.5, 2.0) * adsr(int(2.0 * SR), 0.6, 0, 1, 0.5), 88.0, 0.07, 0.0)
# the lie flips bits (frames 244 - 252): glitch crackle with falling blips
g = np.random.default_rng(88)
for i in range(11):
    tk = T7(244) + g.random() * (T7(252) - T7(244))
    hitb.add(tick(0.03, 0.7), tk, 0.22, g.random() * 2 - 1)
    bells.add(blip(float(g.choice(penta_hi)) * 0.8, 0.09, glide_to=float(g.choice(penta_hi)) * 0.35), tk, 0.06, g.random() * 1.6 - 0.8)
# five tests that still pass, a little out of tune, and sooner each time
for j, fr in enumerate((252, 256, 260, 264, 267)):
    bells.add(blip(1318.5 * (1 - 0.012 * j), 0.09), T7(fr), 0.12, -0.3 + 0.15 * j)
    hitb.add(tick(), T7(fr), 0.12, 0.0)
fxb.add(riser(0.9, 300, 6000, 0.5, 2.6, tone=False, seed=183), 88.1, 1.0, 0.0)

# 89.0 (frame 270): "Caught." - the first rejected test
hitb.add(boom(2.6, 92, 32), T7(270), 1.0, 0.0)
hitb.add(thud(64, 0.9, 0.8), T7(270), 0.8, 0.0)
hitb.add(crash(1.8), T7(270), 0.5, 0.0)
hitb.add(clank(1.2, 1750), T7(270), 0.45, 0.1)
hitb.add(siren(1900, 130, 1.1), T7(270), 0.40, 0.0)
for f, p in (("D6", -0.3), ("Ab6" if False else "Ab5", 0.3), ("Eb6" if False else "G#6", 0.0)):
    bells.add(bell(nm(f), 2.0, 1.2), T7(270), 0.10, p)   # a cluster that does not resolve
for j, (fr, f) in enumerate(((270, "D6"), (275, "A5"), (280, "F5"))):                              # three expanding rings
    bells.add(sonar(nm(f), 1.4), T7(fr), 0.15, -0.5 + 0.5 * j)
fxb.add(stft_sweep(1.4, 9000, 250, 1.6, amp=lambda x: np.exp(-x * 2.2), seed=184), T7(270), 0.35, 0.0)
# the verifier keeps reading, at a steady dry tempo, while the red washes in (frames 276 - 315)
g = np.random.default_rng(89)
for i in range(8):
    bells.add(blip(float(g.choice(penta_hi)), 0.07), T7(276) + i * (5 / 30) * 1.0, 0.07, g.random() * 1.2 - 0.6)
    hitb.add(tick(), T7(276) + i * (5 / 30), 0.10, 0.0)

# 90.0: the answer - the harmony resolves to Dm9 and the three questions arrive (frames 315, 330, 345)
pad_chord(Dm9 + [nm("A4"), nm("F4")], 90.0, 4.3, 0.26, cutoff=3000, att=0.25, rel=1.0, bright=1.15)
bass.add(sub(D2, 3.9), 90.0, 0.36, 0.0)
four_floor(90.0, 94.0, 0.7)
hats8(90.0, 94.0, 0.09)
for k, (fr, f, th) in enumerate(((315, "F5", 80), (330, "A5", 74), (345, "D6", 68))):
    bells.add(epiano(nm(f), 3.0), T7(fr), 0.16, -0.4 + 0.4 * k)
    bells.add(bell(nm(f) * 2, 2.4), T7(fr), 0.05, 0.3)
    hitb.add(thud(th, 0.5, 0.35), T7(fr), 0.6, -0.4 + 0.4 * k)
arp(arp_Dm9, 91.0, 94.0, g=0.07, pat="updown")
bells.add(bell(nm("E6"), 3.0), T7(396), 0.12, 0.0)                    # "Fifteen protocols. The same three questions."

# 93.0 - 94.0: the build; 94.0 (frame 420): the surge. Every digit turns white, outward from the middle
fxb.add(riser(1.5, 400, 13000, 0.9, 2.2, seed=185), 92.5, 1.0, 0.0)
for i in range(16):
    drums.add(snare(0.2), 93.0 + (i / 16) ** 1.35 * 1.0, 0.10 + 0.04 * i, 0.0)
hitb.add(crash(2.6), 94.0, 0.55, 0.0)
hitb.add(boom(1.6, 80, 34, 0.3), 94.0, 0.5, 0.0)
pad_chord(Bb, 94.0, 2.0, 0.2, cutoff=3200, att=0.1, rel=0.2, bright=1.2)
bass.add(sub(Bb2 / 2, 1.9, 0.03, 0.1), 94.0, 0.36, 0.0)
four_floor(94.0, 95.9, 1.0)
hats8(94.0, 95.9, 0.14)
arp([nm("Bb4"), nm("D5"), nm("F5"), nm("A5"), nm("D6"), nm("F6")], 94.0, 95.9, step=BEAT / 4, g=0.075, pat="up", pan_w=0.8)
padb.add(swell_saw([nm("D3"), nm("A3"), nm("D4"), nm("F4")], 2.0, 400, 10000, att=1.4, rel=0.05), 94.0, 0.16, 0.0)
fxb.add(riser(1.9, 800, 15000, 1.0, 2.0, seed=186), 94.0, 1.0, 0.0)

# =============================================================== S6  (96 - 112 s): the shelf is not full
O = 96.0
hitb.add(boom(3.0, 70, 28, 0.35), O, 0.8, 0.0)               # the release after the white-out
pad_chord([nm("D2"), nm("A2"), nm("D3"), nm("F3"), nm("A3")], O - 0.1, 4.4, 0.24, cutoff=1500, att=0.5, rel=1.0)
bass.add(sub(D2 / 2, 4.0, 0.05, 0.3), O, 0.36, 0.0)
# five books land on the beats (finale frames 45, 60, 75, 90, 105): a descending Dm arpeggio to the root
for k, (tk, f) in enumerate(zip([O + 1.5 + 0.5 * j for j in range(5)], (95, 82, 73, 65, 58))):
    hitb.add(thud(f, 0.6, 0.7), tk, 0.95, -0.5 + 0.25 * k)
    hitb.add(bandpass(noise(0.2), 300, 1200) * np.exp(-tt(0.2) * 25), tk, 0.35, -0.5 + 0.25 * k)
    fxb.add(highpass(noise(0.5), 1500) * np.exp(-tt(0.5) * 9) * 0.5, tk + 0.02, 0.05, 0.0)
    bells.add(bell(nm(["A4", "F4", "D4", "A3", "D3"][k]), 3.0), tk, 0.09, -0.5 + 0.25 * k)
pad_chord(Bb, O + 4.0, 4.3, 0.24, cutoff=2400, att=0.5)
pad_chord(Dm9 + [nm("A4")], O + 8.0, 4.3, 0.26, cutoff=3200, att=0.3, bright=1.2)
bass.add(sub(Bb2 / 2, 3.9), O + 4.0, 0.34, 0.0)
bass.add(sub(D2, 3.9), O + 8.0, 0.38, 0.0)
# the page storm (frames 120 - 210)
four_floor(O + 4.0, O + 7.0, 0.9)
hats8(O + 4.0, O + 7.0, 0.12)
fxb.add(whoosh(1.0, 400, 6500, 0.7, 1.1, seed=160), O + 3.8, 0.35, 0.0)
arp([nm("D4"), nm("F4"), nm("A4"), nm("D5"), nm("F5"), nm("A5")], O + 4.0, O + 7.0, step=BEAT / 4, g=0.07, pat="up", pan_w=0.8)
for i in range(16):
    drums.add(snare(0.2), O + 5.5 + i * 0.09375, 0.09 + 0.045 * i, 0.0)
for i in range(60):   # counter ticks: dense mechanical clicks
    u = i / 59
    hitb.add(highpass(noise(0.015), 3000) * np.exp(-tt(0.015) * 150), O + 4.15 + 2.6 * (u ** 0.8), 0.12, 0.0)
fxb.add(riser(1.0, 500, 14000, 0.9, 2.0, seed=161), O + 6.0, 1.0, 0.0)
hitb.add(boom(3.0, 88, 33), O + 7.03, 1.0, 0.0)
hitb.add(crash(2.4), O + 7.03, 0.6, 0.0)
# lockup (frame 240): the chord and the little theme (D A F E D)
for tk, f in ((O + 8.0, "D5"), (O + 8.5, "A4"), (O + 9.0, "F5"), (O + 9.5, "E5"), (O + 10.0, "D5")):
    bells.add(epiano(nm(f), 3.0), tk, 0.2, 0.0)
    bells.add(bell(nm(f) * 2, 2.5), tk, 0.05, 0.2)
# "Five are written.": the five notes of a Dm9 chord, one per volume
for k, f in enumerate(["D5", "F5", "A5", "C6", "E6"]):
    bells.add(epiano(nm(f), 2.5), O + 10.0 + 0.2 * k, 0.13, -0.4 + 0.2 * k)
four_floor(O + 8.0, O + 12.0, 0.75)
hats8(O + 8.0, O + 12.0, 0.10)
arp(arp_Dm9, O + 8.0, O + 12.0, g=0.075, pat="updown")
# calm: "The shelf is not full." an open (suspended) chord
pad_chord([nm("Bb2"), nm("D3"), nm("F3"), nm("A3")], O + 12.0, 2.2, 0.2, cutoff=1500, att=0.8, rel=1.2)
pad_chord(Dsus2, O + 14.0, 2.4, 0.22, cutoff=1300, att=1.0, rel=2.0)
bass.add(sub(Bb2 / 2, 2.0, 0.05, 0.3), O + 12.0, 0.3, 0.0)
bass.add(sub(D2 / 2, 2.2, 0.05, 0.3), O + 14.0, 0.3, 0.0)
for tk, f in ((O + 12.5, "A4"), (O + 13.5, "C5"), (O + 14.5, "E5")):
    bells.add(epiano(nm(f), 3.0), tk, 0.11, 0.1)
fxb.add(whoosh(0.9, 2500, 600, 0.3, 1.0, seed=170), O + 11.7, 0.15, 0.0)


# =============================================================== mix
# side-chain the pads/bass to the kicks a little
sidechain(padb, kicks, 0.35, 0.28)
sidechain(bass, kicks, 0.30, 0.2)
ir = reverb_ir(3.6)
ir_short = reverb_ir(0.9, damp=3500, seed=9)
dry = np.zeros((int(DUR * SR), 2))
send = np.zeros_like(dry)
send_s = np.zeros_like(dry)
# 0-8 s: S0 contains a breath of silence at 7.72-7.98 before the drop
for name, b, g_dry, g_send in (
    ("drums", drums, 1.0, 0.04), ("bass", bass, 1.0, 0.0), ("pad", padb, 0.8, 0.30),
    ("arp", arpb, 0.9, 0.35), ("bells", bells, 0.8, 0.42), ("fx", fxb, 0.85, 0.30), ("hits", hitb, 1.0, 0.16)):
    if name in ("pad", "arp", "bells", "fx"):
        b2 = Bus(DUR); b2.y = b.y.copy()
    # the breath before the title drop
    gi0, gi1 = int(7.74 * SR), int(8.0 * SR)
    ramp = int(0.02 * SR)
    gg = np.ones(b.n); gg[gi0:gi1] = 0; gg[gi0 - ramp:gi0] = np.linspace(1, 0, ramp)
    if name != "hits" and name != "fx":
        pass
    dry += b.y * g_dry * (gg[:, None] if name in ("pad", "arp", "bells", "drums", "bass") else 1.0)
    send += b.y * g_send * (gg[:, None] if name in ("pad", "arp", "bells") else 1.0)
    send_s += b.y * 0.06 * (1.0 if name in ("drums", "hits") else 0.0)
wet = convolve_rev(send, ir) * 0.9 + convolve_rev(send_s, ir_short) * 0.6
mix = dry + wet
# underwater: low-pass the mix from 26.7 s to 32.0 s, then release as the atom opens
sos = sg.butter(2, 1100 / (SR / 2), "low", output="sos")
lp = np.stack([sg.sosfilt(sos, mix[:, 0]), sg.sosfilt(sos, mix[:, 1])], 1)
tm = np.arange(len(mix)) / SR
w = np.clip((tm - 26.7) / 2.2, 0, 1) * (1 - np.clip((tm - 31.9) / 0.6, 0, 1))
mix = mix * (1 - w[:, None]) + lp * w[:, None]
# master
mix = highpass_stereo = np.stack([highpass(mix[:, 0], 28, 2), highpass(mix[:, 1], 28, 2)], 1)
# the climax of volume 5 is cut a tenth of a second early so that the release at 96.0 lands from silence
gate = np.ones(len(tm))
gate[(tm >= 95.9) & (tm < 96.0)] = 0.0
gate[(tm >= 95.88) & (tm < 95.9)] = np.linspace(1, 0, ((tm >= 95.88) & (tm < 95.9)).sum())
mix *= gate[:, None]
# final fade
tail = np.clip((DUR - tm) / 1.2, 0, 1)
head = np.clip(tm / 0.4, 0, 1)
mix *= (tail * head)[:, None]
rms = np.sqrt(np.mean(mix ** 2))
target = db(-17.5)
mix *= target / rms
# soft limiter + hard ceiling at -1 dBFS
mix = np.tanh(mix * 1.15) / np.tanh(1.15)
peak = np.max(np.abs(mix))
mix *= db(-1.0) / peak
out = (mix * 32767).astype(np.int16)
path = sys.argv[1] if len(sys.argv) > 1 else __file__.rsplit("/", 2)[0] + "/public/audio/score.wav"
import os
os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)   # a fresh clone has no public/audio yet
with wave.open(path, "wb") as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes(out.tobytes())
print("wrote", path, "peak", float(peak), "rms dB", 20 * np.log10(rms))
