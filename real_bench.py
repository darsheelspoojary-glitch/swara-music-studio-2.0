"""Harder synthetic songs: slides between notes (jaru), oscillating gamakas, chords, noise."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import engine as E

SCALES = {  # name -> (scale, aroha-free) used as ground truth
    "Hamsadhwani": (0,2,4,7,11), "Mohanam": (0,2,4,7,9), "Hindolam": (0,3,5,8,10),
    "Shuddha Saveri": (0,2,5,7,9), "Madhyamavati": (0,2,5,7,10), "Abhogi": (0,2,3,5,9),
    "Yaman": (0,2,4,6,7,9,11), "Kafi": (0,2,3,5,7,9,10), "Bhairav": (0,1,4,5,7,8,11),
    "Shankarabharanam": (0,2,4,5,7,9,11), "Kharaharapriya": (0,2,3,5,7,9,10), "Todi": (0,1,3,5,7,8,10),
}

def song(scale, sa_hz, rng, dur=50, glide=0.12, gamaka=0.35, pad=0.08, noise=0.03, drone=0.10):
    deg = sorted({d + o for d in scale for o in (-12, 0, 12) if -7 <= d + o <= 12})
    notes, t, pos = [], 0.4, deg.index(0)
    while t < dur - 3:
        for k in range(int(rng.integers(6, 10))):
            last = k == 8
            pos = int(np.clip(pos + rng.choice([-2, -1, -1, 1, 1, 2]), 0, len(deg) - 1))
            d = rng.uniform(0.3, 0.9)
            s = deg[pos]
            if rng.random() < gamaka and d > 0.45 and pos > 0:             # kampita: wobble onto the swara below
                lo = deg[pos - 1]
                notes += [(s, t, t + .16), (lo, t + .16, t + .22), (s, t + .22, t + .38),
                          (lo, t + .38, t + .44), (s, t + .44, t + d)]
            else:
                notes.append((s, t, t + d))
            t += d
        t += rng.uniform(0.4, 0.8)
    y = E.render_melody(notes, sa_hz, t + .5, sr=16000, vibrato=True, drone=drone, glide=glide)
    tt = np.arange(len(y)) / 16000
    chord = sum(np.sin(2*np.pi*sa_hz*r*h*tt)/h for r in (1.0, 1.26, 1.5) for h in (1, 2, 3))
    y = y + pad * chord / np.abs(chord).max() + rng.normal(0, noise, len(y))
    return (y / np.abs(y).max() * .9).astype(np.float32), notes

if __name__ == "__main__":
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 36
    only = sys.argv[3] if len(sys.argv) > 3 else None
    names = [k for k in SCALES if not only or k == only]
    ok = tot = 0; misses = []
    for i in range(n):
        name = names[i % len(names)]
        sa = 110 * 2 ** (rng.uniform(0, 12) / 12 + rng.integers(0, 2))
        y, _ = song(SCALES[name], sa, rng)
        try:
            r = E.interpret(E.extract_f0(y), 0, "Both")
        except ValueError as e:
            misses.append((name, "ValueError")); tot += 1; continue
        tp = int(round(12 * np.log2(sa / 440) + 69)) % 12
        good = r.sa_pc == tp and r.raga.notes == tuple(SCALES[name]); ok += good; tot += 1
        if not good:
            misses.append((name, f"true Sa {E.PITCH_NAMES[tp]} -> {r.raga.name[:22]} Sa {E.PITCH_NAMES[r.sa_pc]}"))
    print(f"scale+tonic correct {ok}/{tot}")
    for m in misses: print("  miss", m)
