"""Run:  python tests/test_engine.py      (no pytest needed)"""
import os, shutil, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import engine as E

def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        check.failed += 1
check.failed = 0

# --- database ---
M = {int(r.name.split("Mela ")[1].rstrip(")")): r for r in E.CARNATIC}
check("72 melakartas, all distinct", len({r.notes for r in E.CARNATIC}) == 72)
check("Mela 29 = Shankarabharanam", M[29].notes == (0, 2, 4, 5, 7, 9, 11))
check("Mela 65 = Kalyani", M[65].notes == (0, 2, 4, 6, 7, 9, 11))
check("Mela 15 = Mayamalavagowla", M[15].notes == (0, 1, 4, 5, 7, 8, 11))
check("Mela 22 = Kharaharapriya", M[22].notes == (0, 2, 3, 5, 7, 9, 10))
check("Mela 8 = Todi scale", M[8].notes == (0, 1, 3, 5, 7, 8, 10))

# --- demo through the whole pipeline (decode -> track -> analyse) ---
wav = E.demo_song()
y = E.decode_audio(wav, ".wav")
tr = E.extract_f0(y)
r = E.interpret(tr, 0, "Both")
check("demo: Sa = A#", E.PITCH_NAMES[r.sa_pc] == "A#")
check("demo: Yaman/Kalyani scale", r.raga.notes == (0, 2, 4, 6, 7, 9, 11))
check("demo: Hindustani-only finds Yaman", E.interpret(tr, 0, "Hindustani").raga.name == "Yaman")
check("demo: Carnatic-only finds Kalyani", "Kalyani" in E.interpret(tr, 0, "Carnatic").raga.name)
txt = E.build_text(r)
check("demo: notes + lines + text", r.n_notes > 50 and r.lines and "Raga" in txt and "Sa" in txt)
check("demo: Sa near 233 Hz", abs(r.sa_hz - 233.08) < 4)
check("demo: no lone octave glitches", all(abs(a[0] - b[0]) < 20 for a, b in zip(r.notes, r.notes[1:])))
check("demo: confidence not Low", r.confidence in ("High", "Medium"))

# --- whole-song notation: lines, styles, repeats ---
check("demo: many lines, none empty", len(r.lines) >= 6 and all(len(l.notes) > 0 for l in r.lines))
check("demo: song repeats are detected", sum(1 for l in r.lines if l.repeat_of) >= 3)
check("demo: tags are letters", all(l.tag[:1].isalpha() for l in r.lines))
for st in E.STYLES:
    t_ = E.build_text(r, st)
    check(f"style {st!r} renders", len(t_.splitlines()) > len(r.lines) and E.swara_name(7, st) in t_)
check("collapse repeats shortens text", len(E.build_text(r, collapse=True)) < len(E.build_text(r)))
check("notes_table has every note", len(E.notes_table(r)) == sum(len(l.notes) for l in r.lines))
check("lines respect the max-notes limit", max(len(l.notes) for l in r.lines) <= 24)

# --- forced settings ---
r2 = E.interpret(tr, 0, "Both", sa_pc=5)
check("forced Sa is honoured", r2.sa_pc == 5)
r3 = E.interpret(tr, 0, "Both", raga_name=E.RAGA_BY_NAME["Bilawal"].name)
check("forced raga is honoured", r3.raga.name == "Bilawal")
check("transpose table rows == notes in raga", len(E.transpose_table(r.raga, r.sa_pc, 3)) == len(r.raga.notes))
bad = E.transpose_table(r.raga, r.sa_pc, 3, "S R G M")
check("transpose table honours note style", bad[0]["Swara"] == "S")
check("snap=False and dashes=False run", E.interpret(tr, 0, "Both", snap=False, dashes=False).n_notes > 0)
check("extreme min_dur runs", E.interpret(tr, 0, "Both", min_dur=0.35, phrase_gap=1.5).n_notes > 5)
check("section offset shows in timestamps", E.interpret(tr, 125, "Both").lines[0].stamp == "02:05")

# --- friendly errors, never crashes ---
def raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True
check("empty file -> ValueError", raises(lambda: E.decode_audio(b"", ".mp3")))
check("garbage bytes -> ValueError", raises(lambda: E.decode_audio(os.urandom(5000), ".mp3")))
sil = E.to_wav_bytes(np.zeros(16000 * 5, np.float32), 16000)
check("silence -> ValueError", raises(lambda: E.decode_audio(sil, ".wav")))
short = E.to_wav_bytes(np.random.randn(8000).astype(np.float32) * .1, 16000)
check("1-second clip -> ValueError", raises(lambda: E.decode_audio(short, ".wav")))
noise = np.random.default_rng(0).normal(0, .3, 16000 * 20).astype(np.float32)
check("pure noise -> ValueError", raises(lambda: E.interpret(E.extract_f0(noise), 0, "Both")))
check("tiny f0 array -> ValueError", raises(lambda: E.interpret(np.full(50, 200.0), 0, "Both")))
check("weird suffix is safe", E.decode_audio(wav, "../../etc/passwd").shape[0] > 0)

# --- other containers (video etc.) when ffmpeg can make them ---
exe = shutil.which("ffmpeg")
if exe:
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "a.wav"); open(src, "wb").write(wav)
        for ext, extra in [("mp4", ["-f", "lavfi", "-i", "color=c=blue:s=64x64:d=70", "-c:v", "mpeg4", "-c:a", "aac", "-shortest"]),
                           ("mkv", ["-f", "lavfi", "-i", "color=c=red:s=64x64:d=70", "-c:v", "mpeg4", "-c:a", "pcm_s16le", "-shortest"]),
                           ("flac", []), ("ogg", ["-c:a", "libvorbis"]), ("m4a", ["-c:a", "aac"]), ("mp3", [])]:
            out = os.path.join(d, "o." + ext)
            cmd = [exe, "-y", "-v", "error", "-i", src] + extra + [out]
            if subprocess.run(cmd, capture_output=True).returncode != 0:
                print("SKIP", ext, "(ffmpeg build cannot make it)"); continue
            yy = E.decode_audio(open(out, "rb").read(), "." + ext)
            rr = E.interpret(E.extract_f0(yy), 0, "Both")
            check(f".{ext} decodes and gives A# / Kalyani scale", E.PITCH_NAMES[rr.sa_pc] == "A#" and rr.raga.notes == (0, 2, 4, 6, 7, 9, 11))

# --- synth for the Listen tab ---
audio = E.render_melody(r.notes[:40], r.sa_hz, r.notes[39][2] + 0.3, drone=0.1, timbre="harmonium")
wb = E.to_wav_bytes(audio, 22050)
check("render_melody -> valid wav", wb[:4] == b"RIFF" and len(audio) > 22050)
back = E.decode_audio(wb, ".wav")
rb = E.interpret(E.extract_f0(back), 0, "Both")
check("re-listening to our own transcription gives the same Sa", rb.sa_pc == r.sa_pc)

# --- a FULL-LENGTH song (4+ minutes) gives the whole song, fast ---
import time
sys.path.insert(0, os.path.dirname(__file__))
from real_bench import song, SCALES
rng = np.random.default_rng(3)
big, _ = song(SCALES["Hamsadhwani"], 196.0, rng, dur=260)
t0 = time.time(); tr_big = E.extract_f0(big); r_big = E.interpret(tr_big, 0, "Both"); dt = time.time() - t0
check(f"4+ min song analysed in {dt:.1f}s (< 20 s)", dt < 20)
check(f"full song notated: {len(r_big.lines)} lines, {r_big.n_notes} notes", len(r_big.lines) > 25 and r_big.n_notes > 300)
check("full song: last line is near the end", r_big.lines[-1].t1 > 230)
check("Hamsadhwani is not called Mohanam", r_big.raga.notes == (0, 2, 4, 7, 11))
ham = 0
for sd in range(3):
    rg = np.random.default_rng(100 + sd); y3, _ = song(SCALES["Hamsadhwani"], 150 + 40 * sd, rg, dur=40)
    ham += E.interpret(E.extract_f0(y3), 0, "Carnatic").raga.name == "Hamsadhwani"
check("Carnatic mode names Hamsadhwani (3 songs, different keys)", ham == 3)

print("\nALL PASSED" if not check.failed else f"\n{check.failed} FAILED")
sys.exit(1 if check.failed else 0)
