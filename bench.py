"""Random-melody benchmark: synthetic songs with vibrato, drone, chords and noise."""
import sys, time, numpy as np
sys.path.insert(0, "..")
import engine as E

CASES = {  # name -> (scale)
    "Yaman": (0,2,4,6,7,9,11), "Bhoopali": (0,2,4,7,9), "Kafi": (0,2,3,5,7,9,10),
    "Hanumatodi (Todi) (Mela 8)": (0,1,3,5,7,8,10), "Malkauns": (0,3,5,8,10),
    "Bhairavi": (0,1,3,5,7,8,10), "Dheerasankarabharanam (Shankarabharanam) (Mela 29)": (0,2,4,5,7,9,11),
    "Kharaharapriya (Mela 22)": (0,2,3,5,7,9,10), "Mohanam": (0,2,4,7,9), "Hindolam": (0,3,5,8,10),
    "Bhairav": (0,1,4,5,7,8,11), "Charukesi (Mela 26)": (0,2,4,5,7,8,10),
}

def make(scale, sa_hz, rng, dur=45, accomp=True, drone=0.12):
    deg = sorted(set(scale) | {s + 12 for s in scale} | {s - 12 for s in scale if s - 12 >= -7})
    deg = [d for d in deg if -7 <= d <= 12]
    notes, t, pos = [], 0.4, deg.index(0)
    while t < dur - 3:
        n = rng.integers(5, 9)
        for k in range(n):
            if k == n - 1:  # end phrase on Sa / Pa mostly
                tgt = [d for d in deg if d in (0, 7, 12)]
                pos = deg.index(min(tgt, key=lambda d: abs(d - deg[pos])))
            else:
                pos = int(np.clip(pos + rng.choice([-2, -1, -1, 1, 1, 2]), 0, len(deg) - 1))
            d = rng.uniform(0.3, 0.8) * (2.2 if k == n - 1 else 1)
            notes.append((deg[pos], t, t + d)); t += d
        t += rng.uniform(0.4, 0.8)
    y = E.render_melody(notes, sa_hz, t + 0.5, sr=16000, vibrato=True, drone=drone if accomp else 0)
    if accomp:  # chord pad + noise at lower level
        tt = np.arange(len(y)) / 16000
        pad = sum(np.sin(2*np.pi*sa_hz*r*h*tt)/h for r in (1.0, 1.26, 1.5) for h in (1, 2, 3))
        y = y + 0.06 * pad / np.max(np.abs(pad)) + rng.normal(0, 0.02, len(y))
    return (y / np.max(np.abs(y)) * 0.9).astype(np.float32), notes

def frame_acc(pred_notes, true_notes, sa_shift=0):
    hit = tot = 0
    for semi, t0, t1 in true_notes:
        for tm in np.arange(t0 + 0.1, t1 - 0.1, 0.05):
            tot += 1
            for ps, p0, p1 in pred_notes:
                if p0 <= tm < p1:
                    hit += (ps == semi); break
    return hit / max(tot, 1)

ACC, DRONE = True, 0.12
if __name__ == "__main__":
    import os; DRONE = float(os.environ.get("DRONE", 0.12)); ACC = os.environ.get("ACC", "1") == "1"
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    ok_scale = ok_name = 0; accs = []; fails = []
    for i in range(n):
        name = list(CASES)[i % len(CASES)]
        scale = CASES[name]
        sa_hz = 110 * 2 ** (rng.uniform(0, 12) / 12 * 1.0 + rng.integers(0, 2))   # 110-440 Hz, any cents
        y, notes = make(scale, sa_hz, rng, accomp=ACC, drone=DRONE)
        f0 = E.extract_f0(y)
        try:
            r = E.interpret(f0, 0, "Both")
        except ValueError as e:
            fails.append((name, str(e)[:40])); continue
        true_pc = int(round(12 * np.log2(sa_hz / 440) + 69)) % 12
        right_scale = (r.sa_pc == true_pc) and (r.raga.notes == tuple(scale))
        ok_scale += right_scale
        ok_name += (r.raga.name == name) or (name in r.same_scale)
        acc = frame_acc(r.notes, notes) if right_scale else 0
        accs.append(acc)
        if not right_scale:
            fails.append((name, f"Sa {E.PITCH_NAMES[true_pc]} -> {r.raga.name} {E.PITCH_NAMES[r.sa_pc]}"))
    print(f"tonic+scale correct: {ok_scale}/{n}   named raga (or same scale twin): {ok_name}/{n}   "
          f"mean note accuracy (when scale right): {np.mean([a for a in accs if a]):.2f}")
    for f in fails: print("  miss:", f)
