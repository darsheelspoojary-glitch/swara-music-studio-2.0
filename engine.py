"""Raga / sargam analysis engine.

Pure numpy + scipy (librosa is only needed for the optional pYIN engine), no Streamlit imports,
so everything here can be unit-tested on its own.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d

SR = 16000                       # analysis sample rate
HOP = 320                        # 20 ms per pitch frame
DT = HOP / SR
N_FFT = 2048
MAX_SECONDS = 900                # decode at most 15 minutes of a file
F_MIN = 90.0                     # lowest melody pitch searched (Hz)
BINS_PER_OCT = 60                # 20-cent resolution
N_BINS = 216                     # 90 Hz ... ~1.1 kHz
LOCAL_GATE = 0.30                # frames this much quieter than the loudest nearby moment = 'no melody'

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
STYLES = {   # how the 12 swaras (semitones above Sa) are written
    "Sa Re Ga Ma": ["Sa", "Re♭", "Re", "Ga♭", "Ga", "Ma", "Ma♯", "Pa", "Dha♭", "Dha", "Ni♭", "Ni"],
    "Sa Ri Ga Ma": ["Sa", "Ri♭", "Ri", "Ga♭", "Ga", "Ma", "Ma♯", "Pa", "Dha♭", "Dha", "Ni♭", "Ni"],
    "S R G M": ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"],
    "सा रे ग म": ["सा", "रे♭", "रे", "ग♭", "ग", "म", "म♯", "प", "ध♭", "ध", "नि♭", "नि"],
}
DEFAULT_STYLE = "Sa Re Ga Ma"
SWARA = STYLES[DEFAULT_STYLE]
# harmonium "scale" (kattai) names for each key, counted from the lowest C
KATTAI = ["1 White", "1 Black", "2 White", "2 Black", "3 White", "4 White",
          "4 Black", "5 White", "5 Black", "6 White", "6 Black", "7 White"]


# =========================================================================== #
# Raga database  (notes = semitones above Sa)
# =========================================================================== #
@dataclass(frozen=True)
class Raga:
    name: str
    system: str
    notes: tuple
    vadi: int | None = None
    samvadi: int | None = None
    aroha: tuple | None = None       # ascending swaras if different from `notes`
    avaroha: tuple | None = None     # descending swaras if different from `notes`

    @property
    def up(self) -> tuple:
        return self.aroha or self.notes

    @property
    def down(self) -> tuple:
        return self.avaroha or self.notes


_MELA = (
    "Kanakangi Ratnangi Ganamurti Vanaspati Manavati Tanarupi Senavati Hanumatodi Dhenuka "
    "Natakapriya Kokilapriya Rupavati Gayakapriya Vakulabharanam Mayamalavagowla Chakravakam "
    "Suryakantam Hatakambari Jhankaradhwani Natabhairavi Keeravani Kharaharapriya Gaurimanohari "
    "Varunapriya Mararanjani Charukesi Sarasangi Harikambhoji Dheerasankarabharanam Naganandini "
    "Yagapriya Ragavardhini Gangeyabhushani Vagadheeswari Shulini Chalanata Salagam Jalarnavam "
    "Jhalavarali Navaneetam Pavani Raghupriya Gavambodhi Bhavapriya Shubhapantuvarali "
    "Shadvidamargini Suvarnangi Divyamani Dhavalambari Namanarayani Kamavardhini Ramapriya "
    "Gamanashrama Vishwambhari Shamalangi Shanmukhapriya Simhendramadhyamam Hemavati Dharmavati "
    "Neetimati Kantamani Rishabhapriya Latangi Vachaspati Mechakalyani Chitrambari Sucharitra "
    "Jyotiswarupini Dhatuvardhani Nasikabhushani Kosalam Rasikapriya"
).split()
assert len(_MELA) == 72

_ALIAS = {
    8: "Hanumatodi (Todi)",
    29: "Dheerasankarabharanam (Shankarabharanam)",
    51: "Kamavardhini (Pantuvarali)",
    65: "Mechakalyani (Kalyani)",
}
_RG = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]        # (Ri, Ga) in semitones
_DN = [(8, 9), (8, 10), (8, 11), (9, 10), (9, 11), (10, 11)]  # (Dha, Ni) in semitones

CARNATIC = []
for _i in range(72):
    _ma = 5 if _i < 36 else 6
    _ri, _ga = _RG[(_i % 36) // 6]
    _dha, _ni = _DN[_i % 6]
    _label = _ALIAS.get(_i + 1, _MELA[_i])
    CARNATIC.append(Raga(f"{_label} (Mela {_i + 1})", "Carnatic", (0, _ri, _ga, _ma, 7, _dha, _ni)))

_FULL = (0, 2, 4, 5, 7, 9, 11)
# name, notes, aroha (None = same as notes), avaroha (None = same as notes)
_JANYA = [
    ("Mohanam", (0, 2, 4, 7, 9), None, None),
    ("Hamsadhwani", (0, 2, 4, 7, 11), None, None),
    ("Shuddha Saveri", (0, 2, 5, 7, 9), None, None),
    ("Madhyamavati", (0, 2, 5, 7, 10), None, None),
    ("Hindolam", (0, 3, 5, 8, 10), None, None),
    ("Shuddha Dhanyasi", (0, 3, 5, 7, 10), None, None),
    ("Abhogi", (0, 2, 3, 5, 9), None, None),
    ("Sriranjani", (0, 2, 3, 5, 9, 10), None, None),
    ("Revati", (0, 1, 5, 7, 10), None, None),
    ("Saveri", (0, 1, 4, 5, 7, 8, 10), (0, 1, 5, 7, 8), None),
    ("Bilahari", _FULL, (0, 2, 4, 7, 9), None),
    ("Arabhi", _FULL, (0, 2, 5, 7, 9), None),
    ("Begada", _FULL, None, None),
    ("Kambhoji", (0, 2, 4, 5, 7, 9, 10), (0, 2, 4, 5, 7, 9), None),
    ("Bhoopalam", (0, 1, 4, 7, 8), None, None),
    ("Malahari", (0, 1, 5, 7, 8), None, None),
    ("Gambhira Nattai", (0, 4, 5, 7, 11), None, None),
    ("Valaji", (0, 2, 4, 7, 9), None, None),
]
JANYA = [Raga(n, "Carnatic", s, aroha=a, avaroha=d) for n, s, a, d in _JANYA]

# name, notes, vadi, samvadi
_HINDUSTANI = [
    ("Yaman", (0, 2, 4, 6, 7, 9, 11), 4, 11),
    ("Bilawal", (0, 2, 4, 5, 7, 9, 11), 9, 4),
    ("Khamaj", (0, 2, 4, 5, 7, 9, 10, 11), 4, 11),
    ("Desh", (0, 2, 4, 5, 7, 9, 10, 11), 2, 7),
    ("Kafi", (0, 2, 3, 5, 7, 9, 10), 7, 0),
    ("Bageshree", (0, 2, 3, 5, 7, 9, 10), 5, 0),
    ("Asavari", (0, 2, 3, 5, 7, 8, 10), 8, 3),
    ("Darbari Kanada", (0, 2, 3, 5, 7, 8, 10), 2, 7),
    ("Bhairavi", (0, 1, 3, 5, 7, 8, 10), 5, 0),
    ("Bhairav", (0, 1, 4, 5, 7, 8, 11), 8, 1),
    ("Ahir Bhairav", (0, 1, 4, 5, 7, 9, 10), 5, 0),
    ("Todi", (0, 1, 3, 6, 7, 8, 11), 8, 3),
    ("Purvi", (0, 1, 4, 6, 7, 8, 11), 4, 11),
    ("Marwa", (0, 1, 4, 6, 9, 11), 1, 9),
    ("Bhoopali", (0, 2, 4, 7, 9), 4, 9),
    ("Malkauns", (0, 3, 5, 8, 10), 5, 0),
    ("Durga", (0, 2, 5, 7, 9), 5, 0),
    ("Megh", (0, 2, 5, 7, 10), 0, 7),
    ("Hindol", (0, 2, 4, 6, 9, 11), 9, 2),
    ("Bihag", (0, 2, 4, 5, 6, 7, 9, 11), 4, 11),
    ("Hamsadhwani (Hindustani)", (0, 2, 4, 7, 11), None, None),
    ("Pahadi", (0, 2, 4, 7, 9), None, None),
]
HINDUSTANI = [Raga(n, "Hindustani", s, v, sv) for n, s, v, sv in _HINDUSTANI]

ALL_RAGAS = CARNATIC + JANYA + HINDUSTANI
RAGA_BY_NAME = {r.name: r for r in ALL_RAGAS}


def _template(r: Raga) -> np.ndarray:
    t = np.zeros(12)
    t[list(r.notes)] = 1.0
    t[0] *= 1.3                      # Sa is the anchor
    if 7 in r.notes:
        t[7] *= 1.15                 # Pa is the second anchor
    if r.vadi is not None:
        t[r.vadi] *= 1.25
    if r.samvadi is not None:
        t[r.samvadi] *= 1.1
    return t / np.linalg.norm(t)


EPS_FLOOR = 0.06      # probability mass reserved for stray / out-of-raga notes


def _logq(r: "Raga") -> np.ndarray:
    """log P(swara | raga): emphasised swaras (Sa, Pa, vadi) get more mass, strays get a small floor."""
    w = np.zeros(12)
    w[list(r.notes)] = 1.0
    w[0] *= 1.5
    if 7 in r.notes:
        w[7] *= 1.2
    if r.vadi is not None:
        w[r.vadi] *= 1.25
    if r.samvadi is not None:
        w[r.samvadi] *= 1.1
    q = w / w.sum() * (1 - EPS_FLOOR) + EPS_FLOOR / 12
    return np.log(q)


def _logp(notes: tuple, eps: float = 0.03) -> np.ndarray:
    """log P(landing swara | raga) for a stepwise move: uniform over the allowed swaras."""
    n = len(notes)
    p = np.full(12, eps / max(12 - n, 1))
    p[list(notes)] = (1 - eps) / n
    return np.log(p)


TEMPLATES = {r.name: _template(r) for r in ALL_RAGAS}
LOGQ = {r.name: _logq(r) for r in ALL_RAGAS}
LOGP_UP = {r.name: _logp(r.up) for r in ALL_RAGAS}
LOGP_DOWN = {r.name: _logp(r.down) for r in ALL_RAGAS}


def ragas_for(tradition: str) -> list[Raga]:
    return [r for r in ALL_RAGAS if tradition == "Both" or r.system == tradition]


# =========================================================================== #
# 1. Decode ANY audio / video file
# =========================================================================== #
def _ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:                              # bundled binary shipped by the imageio-ffmpeg package
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _decode_fallback(path: str, max_seconds: float) -> np.ndarray:
    """Used only when ffmpeg is unavailable (plain audio formats)."""
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=True, frames=int(MAX_SECONDS * 48000))
        y = y.mean(axis=1)
    except Exception:
        try:
            import librosa
            y, sr = librosa.load(path, sr=None, mono=True, duration=max_seconds)
        except Exception as exc:
            raise ValueError("Could not read this file. Try mp3, wav, m4a, flac, ogg or mp4.") from exc
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g)
    return y.astype(np.float32)[: int(max_seconds * SR)]


def decode_audio(data: bytes, suffix: str = "", max_seconds: float = MAX_SECONDS) -> np.ndarray:
    """Any audio/video bytes -> mono float32 samples at 16 kHz, peak-normalised."""
    if not data:
        raise ValueError("The file is empty.")
    suffix = suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,6}", suffix or "") else ".bin"
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "input" + suffix)
        with open(src, "wb") as fh:
            fh.write(data)
        y, why = None, ""
        exe = _ffmpeg()
        if exe:
            cmd = [exe, "-nostdin", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(SR),
                   "-t", str(max_seconds), "-f", "f32le", "pipe:1"]
            try:
                p = subprocess.run(cmd, capture_output=True, timeout=300)
                buf = p.stdout[: len(p.stdout) // 4 * 4]
                if p.returncode == 0 and len(buf) >= 4:
                    y = np.frombuffer(buf, dtype="<f4").astype(np.float32)
                else:
                    why = p.stderr.decode(errors="ignore").strip()[-200:]
            except subprocess.TimeoutExpired:
                why = "decoding timed out"
        if y is None:
            if exe and why:
                raise ValueError("This file has no readable audio track. " + why)
            y = _decode_fallback(src, max_seconds)

    y = np.nan_to_num(y.astype(np.float32))
    if len(y) < 2 * SR:
        raise ValueError("The audio is shorter than 2 seconds - please use a longer clip.")
    peak = float(np.max(np.abs(y)))
    if peak < 1e-4:
        raise ValueError("The audio is silent.")
    return (y / peak * 0.9).astype(np.float32)


# =========================================================================== #
# 2. Melody tracking
#    (a) full-mix engine: harmonic-salience + Viterbi  (numpy only, fast)
#    (b) solo engine: pYIN (librosa, slower, very precise on clean single voices)
# =========================================================================== #
_BIN_HZ = SR / N_FFT
_F0S = F_MIN * 2.0 ** (np.arange(N_BINS) / BINS_PER_OCT)
_NH = 8
_HW = 0.8 ** np.arange(_NH)
_FWEIGHT = np.clip(0.55 + 0.45 * np.log2(_F0S / F_MIN), 0.55, 1.0).astype(np.float32)  # tame bass
_HARM = []
for _h in range(1, _NH + 1):
    _p = _h * _F0S / _BIN_HZ
    _i0 = np.floor(_p).astype(int)
    _HARM.append((np.clip(_i0, 0, N_FFT // 2 - 1), (_p - _i0).astype(np.float32),
                  ((_i0 + 1) <= N_FFT // 2).astype(np.float32)))


def compute_salience(y: np.ndarray):
    """Harmonic-summation pitch salience, shape (N_BINS, frames), plus per-frame RMS."""
    T = len(y) // HOP
    if T < 20:
        raise ValueError("The audio is too short to analyse.")
    half = N_FFT // 2
    ypad = np.concatenate([np.zeros(half, np.float32), y.astype(np.float32),
                           np.zeros(N_FFT + HOP, np.float32)])
    win = np.hanning(N_FFT).astype(np.float32)
    sal = np.zeros((N_BINS, T), np.float32)
    rms = np.zeros(T, np.float32)
    CH = 1200                                           # 24 s per chunk keeps memory small
    for c0 in range(0, T, CH):
        c1 = min(T, c0 + CH)
        seg = ypad[c0 * HOP: c1 * HOP + N_FFT]
        fr = sliding_window_view(seg, N_FFT)[::HOP][: c1 - c0]
        rms[c0:c1] = np.sqrt((fr ** 2).mean(axis=1))
        spec = np.abs(np.fft.rfft(fr * win, axis=1)).astype(np.float32).T
        peaks = np.sqrt(np.maximum(spec - uniform_filter1d(spec, 41, axis=0, mode="nearest"), 0))
        s = np.zeros((N_BINS, c1 - c0), np.float32)
        for h, (i0, a, okm) in enumerate(_HARM):
            s += (_HW[h] * okm)[:, None] * ((1 - a)[:, None] * peaks[i0] + a[:, None] * peaks[i0 + 1])
        sal[:, c0:c1] = s * _FWEIGHT[:, None]
    return sal, rms


def _viterbi(obs: np.ndarray, unv: float = 0.25, lam: float = 0.012, cap: float = 0.5,
             mu: float = 0.35) -> np.ndarray:
    """Best pitch path through the salience map. State N_BINS = 'no pitch'. Vectorised per frame."""
    K, T = obs.shape
    idx = np.arange(K, dtype=np.float64)
    ii = np.arange(K)
    bp_v = np.full((T, K), K, dtype=np.int16)
    bp_u = np.full(T, K, dtype=np.int16)
    v, u = obs[:, 0].copy(), float(unv)
    for t in range(1, T):
        w = v + lam * idx                                  # best predecessor j <= i
        rm = np.maximum.accumulate(w)
        af = np.maximum.accumulate(np.where(w >= rm, ii, 0))
        f = rm - lam * idx
        w2 = v - lam * idx                                 # best predecessor j >= i
        rm2 = np.maximum.accumulate(w2[::-1])[::-1]
        ab = np.minimum.accumulate(np.where(w2 >= rm2, ii, K)[::-1])[::-1]
        g = rm2 + lam * idx
        use_f = f >= g
        best = np.where(use_f, f, g)
        src = np.where(use_f, af, ab)
        jg = int(np.argmax(v))                             # long jumps cost at most `cap`
        jump = (v[jg] - cap) > best
        best = np.where(jump, v[jg] - cap, best)
        src = np.where(jump, jg, src)
        from_u = (u - mu) > best                           # entering a voiced phrase
        best = np.where(from_u, u - mu, best)
        src = np.where(from_u, K, src)
        vmax = v[jg] - mu                                  # leaving to silence
        if u >= vmax:
            new_u, bp_u[t] = u + unv, K
        else:
            new_u, bp_u[t] = vmax + unv, jg
        v = best + obs[:, t]
        u = new_u
        bp_v[t] = src
        m = max(float(v.max()), u)                         # keep numbers small
        v -= m
        u -= m
    state = K if u > v.max() else int(np.argmax(v))
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = state
        state = int(bp_u[t]) if state == K else int(bp_v[t][state])
    return path


@dataclass
class Track:
    f0: np.ndarray                     # Hz per 20 ms frame, NaN = no melody
    bg: np.ndarray | None = None       # average pitch energy per search bin (drone / bass / harmony)
    quality: float | None = None       # how "pitched" the audio is (noise ~2, clear melody 7+)
    bg_scale: float = 1.0              # trust in `bg` (1 = measured during pauses, lower = measured over the melody)


def track_melody(y: np.ndarray) -> Track:
    """Full-mix melody tracker."""
    sal, rms = compute_salience(y)
    T = sal.shape[1]
    active = rms > max(10 ** (-45 / 20) * np.percentile(rms, 95), 1e-5)
    if not active.any():
        return Track(np.full(T, np.nan))
    ref = np.percentile(sal.max(axis=0)[active], 95) + 1e-9
    obs = np.clip(sal / ref, 0, 1.5).astype(np.float64)
    obs[:, ~active] = 0.0
    local = maximum_filter1d(rms, size=100, mode="nearest")  # a sudden drop in loudness = the singer paused;
    paused = active & (rms < LOCAL_GATE * local)              # what remains (drone, soft chords) is not melody
    obs[:, paused] = 0.0
    path = _viterbi(obs)
    f0 = np.full(T, np.nan)
    t_idx = np.flatnonzero(path < N_BINS)
    if len(t_idx):
        k = path[t_idx]
        c, l, r = obs[k, t_idx], obs[np.maximum(k - 1, 0), t_idx], obs[np.minimum(k + 1, N_BINS - 1), t_idx]
        den = l - 2 * c + r
        delta = np.where(den < -1e-9, 0.5 * (l - r) / np.where(den < -1e-9, den, 1), 0.0)
        f0[t_idx] = F_MIN * 2.0 ** ((k + np.clip(delta, -0.5, 0.5)) / BINS_PER_OCT)
    contrast = sal.max(axis=0)[active] / (sal.mean(axis=0)[active] + 1e-9)
    # tonic evidence: accompaniment heard while the melody pauses (drone / bass / chords sit on Sa and Pa)
    src_frames = paused if paused.sum() >= 40 else active
    return Track(f0, sal[:, src_frames].mean(axis=1), float(np.median(contrast)),
                 1.0 if paused.sum() >= 40 else 0.3)


def track_melody_pyin(y: np.ndarray) -> Track:
    """Solo voice / instrument engine (needs librosa)."""
    import librosa
    f0, voiced, prob = librosa.pyin(y, fmin=70.0, fmax=1100.0, sr=SR, frame_length=2048, hop_length=HOP)
    return Track(np.where(voiced & (prob > 0.5), f0, np.nan))


def extract_f0(y: np.ndarray, method: str = "salience") -> Track:
    if method == "pyin":
        try:
            return track_melody_pyin(y)
        except ImportError:
            pass                                           # librosa missing -> use the built-in engine
    return track_melody(y)


# =========================================================================== #
# 3. Tonic + raga + notes (whole song, line by line)
# =========================================================================== #
@dataclass
class NoteOut:
    pc: int            # swara, semitones above Sa (0..11)
    octave: int        # -1 lower, 0 middle, 1 upper ...
    dashes: int        # extra beats the note is held
    t0: float          # seconds from the start of the analysed section
    t1: float

    @property
    def semi(self) -> int:
        return self.pc + 12 * self.octave


@dataclass
class Line:
    index: int                      # 1-based line number
    t0: float
    t1: float
    stamp: str                      # mm:ss in the original file
    notes: list
    tag: str = "A"                  # melody tag: lines with the same tune share a letter
    repeat_of: int | None = None    # earlier line this one repeats
    similarity: float = 0.0


@dataclass
class Analysis:
    start: float
    duration: float
    sa_pc: int
    sa_hz: float
    cents: float
    raga: Raga
    same_scale: list
    table: list
    alternatives: list      # [(Sa pitch class, raga name, closeness %)]
    profile: np.ndarray     # share of each swara (index 0 = Sa)
    lines: list
    notes: list             # [(semitone_rel_Sa, t0, t1)]
    n_notes: int
    confidence: str
    margin: float
    contour: tuple
    warnings: list


def swara_name(pc: int, style: str = DEFAULT_STYLE) -> str:
    return STYLES.get(style, STYLES[DEFAULT_STYLE])[int(pc) % 12]


def note_text(pc: int, octave: int = 0, style: str = DEFAULT_STYLE) -> str:
    mark = "'" * octave if octave > 0 else "," * (-octave)
    return swara_name(pc, style) + mark


def label(semi: int, octave: bool = True, style: str = DEFAULT_STYLE) -> str:
    o, pc = divmod(int(semi), 12)
    return note_text(pc, o if octave else 0, style)


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _tuning(midi: np.ndarray) -> float:
    """How far the recording sits from A=440 tuning, in semitones (circular mean of steady frames)."""
    if len(midi) >= 9:                                  # ignore slides / gamakas: use steady stretches only
        win = sliding_window_view(midi, 5)
        steady = np.zeros(len(midi), bool)
        steady[2:-2] = np.ptp(win, axis=1) < 0.35
        if steady.sum() >= 50:
            midi = midi[steady]
    return float(np.angle(np.mean(np.exp(2j * np.pi * (midi - np.round(midi))))) / (2 * np.pi))


W_END, W_BG, W_TR = 0.4, 4.5, 0.1      # weights of the extra tonic / ascent-descent cues


def rank(profile, pool, tonic=None, ends=None, asc=None, desc=None, bg=None, n_ends=0, final=None, bg_scale=1.0):
    """Score every (Sa, raga) pair (expected log-likelihood of the melody under the raga); best first."""
    out = []
    for t in ([tonic] if tonic is not None else range(12)):
        p = np.roll(profile, -t)
        bonus = 0.0
        if ends is not None:                                                 # phrases land on Sa (or Pa)
            e = np.roll(ends, -t)
            bonus += W_END * min(1.0, n_ends / 10) * (e[0] + 0.5 * e[7])
        if bg is not None:
            bonus += W_BG * bg_scale * np.roll(bg, -t)[0]                    # drone / bass sits on Sa
        if final is not None:                                                # songs usually finish on Sa
            bonus += 0.3 if (final - t) % 12 == 0 else 0.1 if (final - t) % 12 == 7 else 0.0
        a_rel = np.roll(asc, -t) if asc is not None else np.zeros(12)
        d_rel = np.roll(desc, -t) if desc is not None else np.zeros(12)
        n_moves = a_rel.sum() + d_rel.sum()
        for r in pool:
            score = float(p @ LOGQ[r.name]) + bonus
            if n_moves >= 20:                                                # ascent / descent fit
                score += W_TR * float(a_rel @ LOGP_UP[r.name] + d_rel @ LOGP_DOWN[r.name]) / n_moves
            out.append((score, t, r))
    out.sort(key=lambda x: -x[0])
    return out


def _snap_delta(pc: int, allowed) -> int:
    best = min(allowed, key=lambda a: (min((pc - a) % 12, (a - pc) % 12), ((a - pc + 6) % 12) - 6 > 0))
    return ((best - pc + 6) % 12) - 6


def _split_long(group: list, limit: int) -> list:
    """Cut a very long unbroken phrase at its biggest internal pause."""
    out = []
    while len(group) > limit:
        k = max(range(limit // 2, limit + 1), key=lambda i: group[i][1] - group[i - 1][2])
        out.append(group[:k])
        group = group[k:]
    out.append(group)
    return out


def _letter(k: int) -> str:
    return chr(65 + k % 26) + (str(k // 26) if k >= 26 else "")


def tag_repeats(lines: list, threshold: float = 0.8) -> None:
    """Mark lines that repeat an earlier line (built-in difflib) and give each tune a letter."""
    from difflib import SequenceMatcher
    seqs = [[n.semi for n in ln.notes] for ln in lines]
    originals: list[int] = []
    sm = SequenceMatcher(None, autojunk=False)
    for i, ln in enumerate(lines):
        best, bj = 0.0, None
        if len(seqs[i]) >= 4:
            sm.set_seq2(seqs[i])
            for j in originals:
                sm.set_seq1(seqs[j])
                if sm.real_quick_ratio() >= threshold and sm.quick_ratio() >= threshold:
                    r = sm.ratio()
                    if r > best:
                        best, bj = r, j
        if bj is not None and best >= threshold:
            ln.repeat_of, ln.similarity, ln.tag = lines[bj].index, best, lines[bj].tag
        else:
            ln.tag = _letter(len(originals))
            originals.append(i)


def interpret(f0, start=0.0, tradition="Both", sa_pc=None, raga_name=None, min_dur=0.12,
              phrase_gap=0.45, snap=True, dashes=True, max_line_notes=24) -> Analysis:
    bg_bins = getattr(f0, "bg", None)
    bg_scale = getattr(f0, "bg_scale", 1.0)
    quality = getattr(f0, "quality", None)
    if quality is not None and quality < 4.0:
        raise ValueError("No clear melody found - this audio sounds like noise or speech. "
                         "Try a section with singing or an instrument playing the tune.")
    f0 = np.asarray(getattr(f0, "f0", f0), dtype=float)
    ok = np.isfinite(f0) & (f0 > 20)
    if ok.sum() < 100:
        raise ValueError("Not enough clear melody found (need 2+ seconds of pitched sound). "
                         "Try another section, or a cleaner recording.")
    midi = np.full(f0.shape, np.nan)
    midi[ok] = 69 + 12 * np.log2(f0[ok] / 440.0)
    tuning = _tuning(midi[ok])
    midi -= tuning

    # ---- notes (absolute semitones; Sa not known yet) ----
    absm = np.round(midi)
    idx = np.flatnonzero(ok)
    segs = []
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        vals = median_filter(absm[run], size=5, mode="nearest") if len(run) >= 5 else absm[run]
        for part in np.split(np.arange(len(run)), np.flatnonzero(np.diff(vals) != 0) + 1):
            segs.append([int(vals[part[0]]), run[part[0]] * DT, (run[part[-1]] + 1) * DT])
    segs = [s for s in segs if s[2] - s[1] >= min_dur]
    for i in range(1, len(segs) - 1):          # a lone note an octave off its neighbours = tracker glitch
        a, b, c = segs[i - 1][0], segs[i][0], segs[i + 1][0]
        if abs(a - c) <= 5 and abs(b - a) >= 9 and abs(b - c) >= 9:
            segs[i][0] = b + 12 * int(round(((a + c) / 2 - b) / 12))
    if len(segs) < 5:
        raise ValueError("Couldn't find stable notes. Try a smaller 'shortest note' value or another section.")
    prof_abs = np.zeros(12)                    # swara histogram from held notes only
    for s in segs:
        prof_abs[s[0] % 12] += s[2] - s[1]
    prof_abs /= prof_abs.sum()

    ends, n_ends = np.zeros(12), 0             # where do phrases finish?
    asc, desc = np.zeros(12), np.zeros(12)     # where do stepwise moves land?
    for i, s in enumerate(segs):
        last = i + 1 == len(segs)
        if last or segs[i + 1][1] - s[2] > phrase_gap:
            ends[s[0] % 12] += s[2] - s[1]
            n_ends += 1
        if not last:
            b = segs[i + 1]
            if b[1] - s[2] <= phrase_gap and 1 <= abs(b[0] - s[0]) <= 4:
                (asc if b[0] > s[0] else desc)[b[0] % 12] += 1
    ends = ends / ends.sum() if ends.sum() > 0 and len(segs) >= 8 else None
    bg = None
    if bg_bins is not None and np.sum(bg_bins) > 0:
        bin_pc = np.round(69 + 12 * np.log2(_F0S / 440.0) - tuning).astype(int) % 12
        bg = np.bincount(bin_pc, weights=np.asarray(bg_bins, float), minlength=12)
        bg = bg / bg.sum()

    # ---- tonic + raga ----
    pool = ragas_for(tradition)
    if raga_name:
        pool = [r for r in pool if r.name == raga_name] or pool
    final = segs[-1][0] % 12 if len(segs) >= 20 else None
    ranked = rank(prof_abs, pool, sa_pc, ends, asc, desc, bg, n_ends, final, bg_scale)
    score, t, raga = ranked[0]

    def pct(sc: float) -> int:
        return int(round(100 * np.exp((sc - score) / 0.5)))

    table, seen = [], set()
    for sc, tt, r in ranked:
        if r.name in seen:
            continue
        seen.add(r.name)
        table.append({"Raga": r.name, "System": r.system, "Sa": PITCH_NAMES[tt], "Closeness %": pct(sc)})
        if len(table) == 8:
            break
    alts, seen_alt = [], {(t, raga.notes)}
    for sc, tt, r in ranked[1:]:
        if (tt, r.notes) not in seen_alt:
            seen_alt.add((tt, r.notes))
            alts.append((tt, r.name, pct(sc)))
        if len(alts) == 3:
            break
    same = [r.name for r in ALL_RAGAS if r.notes == raga.notes and r.name != raga.name]
    gap = next((score - sc for sc, tt, r in ranked[1:] if tt != t or r.notes != raga.notes), 0.5)
    confidence = "High" if gap >= 0.20 else "Medium" if gap >= 0.08 else "Low"
    if len(segs) < 30 and confidence == "High":
        confidence = "Medium"

    # ---- Sa frequency and octave reference ----
    med = float(np.median(midi[ok]))
    sa_mid = t + 12 * np.floor((med - t) / 12)
    sa_hz = 440.0 * 2 ** ((sa_mid + tuning - 69) / 12)

    # ---- notate relative to Sa ----
    for s in segs:
        s[0] = int(s[0] - sa_mid)
    if snap:
        allowed = set(raga.notes)
        for s in segs:
            if s[0] % 12 not in allowed:
                s[0] += _snap_delta(s[0] % 12, allowed)
    merged = []
    for s in segs:
        if merged and merged[-1][0] == s[0] and s[1] - merged[-1][2] <= 0.15:
            merged[-1][2] = s[2]
        else:
            merged.append(s)

    unit = float(np.clip(np.median([m[2] - m[1] for m in merged]), 0.15, 0.8))
    groups, cur, last_end = [], [], 0.0
    for m in merged:
        if cur and m[1] - last_end > phrase_gap:
            groups.append(cur)
            cur = []
        cur.append(m)
        last_end = m[2]
    if cur:
        groups.append(cur)
    lines = []
    for g in (x for grp in groups for x in _split_long(grp, max(8, int(max_line_notes)))):
        notes = []
        for semi, t0, t1 in g:
            o, pc = divmod(semi, 12)
            n_dash = int(np.clip(int((t1 - t0) / unit + 0.5), 1, 4)) - 1 if dashes else 0
            notes.append(NoteOut(pc, o, n_dash, t0, t1))
        lines.append(Line(len(lines) + 1, g[0][1], g[-1][2], mmss(start + g[0][1]), notes))
    tag_repeats(lines)

    dur = len(f0) * DT
    warns = []
    if len(merged) < 25 or (dur > 40 and len(merged) / dur * 60 < 12):
        warns.append("Very few notes were found for this length. The melody may be buried under instruments - "
                     "try a vocal-only version, the 'Solo' engine, or a different section.")
    if confidence == "Low":
        warns.append("Several readings fit almost equally well. Check 'Other readings' and set the Sa if you know it.")
    if abs(tuning) > 0.35:
        warns.append(f"This recording is tuned about {abs(tuning) * 100:.0f} cents "
                     f"{'sharp' if tuning > 0 else 'flat'} of standard pitch - the Sa shown is the nearest key.")

    keep = np.flatnonzero(ok)[::2]
    contour = ((keep * DT).astype(np.float32), (midi[keep] - sa_mid).astype(np.float32))
    return Analysis(float(start), dur, t, sa_hz, 100 * tuning, raga, same, table, alts,
                    np.roll(prof_abs, -t), lines, [tuple(m) for m in merged], len(merged),
                    confidence, float(gap), contour, warns)


def scale_text(raga: Raga, style: str = DEFAULT_STYLE) -> tuple[str, str]:
    up = "  ".join(label(n, False, style) for n in sorted(raga.up)) + "  " + label(12, True, style)
    down = label(12, True, style) + "  " + "  ".join(label(n, False, style) for n in sorted(raga.down, reverse=True))
    return up, down


def build_text(res: Analysis, style: str = DEFAULT_STYLE, collapse: bool = False, wrap: int = 16) -> str:
    """The whole song as plain text, one lyric line per row."""
    up, down = scale_text(res.raga, style)
    head = [f"Raga    : {res.raga.name}",
            f"Sa      : {PITCH_NAMES[res.sa_pc]}  (about {res.sa_hz:.1f} Hz)   Harmonium scale: {KATTAI[res.sa_pc]}",
            f"Aroha   : {up}", f"Avaroha : {down}",
            "Legend  : ♭ komal, ♯ tivra, ' upper octave, , lower octave, – hold the note", ""]
    width = len(str(len(res.lines)))
    body = []
    for ln in res.lines:
        prefix = f"{ln.index:>{width}}. [{ln.stamp}] {ln.tag}  "
        if collapse and ln.repeat_of:
            body.append(prefix + f"↻ same as line {ln.repeat_of}")
            continue
        toks = [note_text(n.pc, n.octave, style) + " –" * n.dashes for n in ln.notes]
        chunks = [toks[i:i + wrap] for i in range(0, len(toks), wrap)]
        body.append(prefix + "  ".join(chunks[0]))
        for c in chunks[1:]:
            body.append(" " * len(prefix) + "  ".join(c))
    return "\n".join(head + body)


def transpose_table(raga: Raga, original_sa: int, target_sa: int, style: str = DEFAULT_STYLE) -> list:
    return [{"Swara": swara_name(n, style),
             f"Original (Sa = {PITCH_NAMES[original_sa]})": PITCH_NAMES[(original_sa + n) % 12],
             f"Your key (Sa = {PITCH_NAMES[target_sa]})": PITCH_NAMES[(target_sa + n) % 12]}
            for n in sorted(raga.notes)]


def notes_table(res: Analysis, style: str = DEFAULT_STYLE) -> list:
    """Every note with its time, for the CSV download."""
    rows = []
    for ln in res.lines:
        for n in ln.notes:
            rows.append({"line": ln.index, "start_s": round(res.start + n.t0, 2), "end_s": round(res.start + n.t1, 2),
                         "swara": note_text(n.pc, n.octave, style), "tune": ln.tag})
    return rows


# =========================================================================== #
# 4. Hear the result: synthesise the notes (and a built-in demo song)
# =========================================================================== #
def to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def render_melody(notes, sa_hz: float, total: float, sr: int = 22050, vibrato: bool = False,
                  drone: float = 0.0, glide: float = 0.03, timbre: str = "vocal") -> np.ndarray:
    """notes = [(semitone_rel_Sa, t0, t1)] -> audio. timbre 'harmonium' = two slightly detuned reeds."""
    n = int(total * sr) + 1
    fvec = np.full(n, float(sa_hz))
    amp = np.zeros(n)
    for semi, t0, t1 in notes:
        i0, i1 = max(int(t0 * sr), 0), min(int(t1 * sr), n)
        if i1 > i0:
            fvec[i0:i1] = sa_hz * 2 ** (semi / 12)
            amp[i0:i1] = 1.0
    hold = np.maximum.accumulate(np.where(amp > 0, np.arange(n), 0))   # carry pitch through rests
    fvec = uniform_filter1d(fvec[hold], max(1, int(glide * sr)))
    tt = np.arange(n) / sr
    if vibrato:
        fvec = fvec * 2 ** ((25 / 1200) * np.sin(2 * np.pi * 5.5 * tt))
    phase = 2 * np.pi * np.cumsum(fvec) / sr
    if timbre == "harmonium":
        y = sum((np.sin(h * phase) + np.sin(h * phase * 1.003)) / h ** 0.9 for h in range(1, 9))
    else:
        y = sum(np.sin(h * phase) / h ** 1.2 for h in range(1, 7))
    y = y * uniform_filter1d(amp, max(1, int(0.025 * sr)))
    if drone > 0:
        for f, g in ((sa_hz * 0.75, 1.0), (sa_hz * 0.5, 1.2), (sa_hz, 0.8)):
            y = y + drone * g * sum(np.sin(2 * np.pi * f * h * tt) / h ** 1.5 for h in range(1, 5)) * (
                0.8 + 0.2 * np.sin(2 * np.pi * 0.35 * tt))
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.85).astype(np.float32)


def render_notes(notes, sa_hz: float, speed: float = 1.0, drone: float = 0.0, timbre: str = "harmonium",
                 sr: int = 22050, max_seconds: float = 150.0) -> bytes:
    """WAV bytes of the transcription, at practice speed (1.0 = original, 0.5 = half speed)."""
    speed = float(np.clip(speed, 0.25, 2.0))
    scaled = []
    for semi, t0, t1 in notes:
        if t0 / speed >= max_seconds:
            break
        scaled.append((semi, t0 / speed, min(t1 / speed, max_seconds)))
    if not scaled:
        raise ValueError("No notes to play.")
    shift = scaled[0][1] - 0.2                       # start playing right away
    scaled = [(s, a - shift, b - shift) for s, a, b in scaled]
    y = render_melody(scaled, sa_hz, scaled[-1][2] + 0.4, sr, drone=drone, glide=0.02, timbre=timbre)
    return to_wav_bytes(y, sr)


_DEMO = [  # (semitone from Sa, beats)  - Raga Yaman
    [(-1, .5), (2, .5), (4, .5), (6, .5), (7, 1.2), (4, .5), (2, .5), (0, 1.4)],
    [(4, .4), (6, .4), (7, .4), (9, .4), (11, .6), (12, 1.3), (11, .4), (9, .4), (7, .4), (6, .4), (4, 1.2)],
    [(7, .4), (6, .4), (4, .4), (2, .4), (0, 1.4), (-1, .5), (2, .5), (4, .5), (6, .5), (7, .8), (12, 1.4)],
    [(12, .4), (11, .4), (9, .4), (7, .4), (6, .4), (4, .4), (2, .4), (0, 1.8)],
]


def demo_song(sa_hz: float = 233.08, sr: int = 22050) -> bytes:
    """A short Raga Yaman melody with vibrato and drone, as WAV bytes (used by the 'Try demo' button)."""
    notes, t = [], 0.5
    for _ in range(3):
        for phrase in _DEMO:
            for semi, beats in phrase:
                notes.append((semi, t, t + beats * 0.8))
                t += beats * 0.8
            t += 0.7
    return to_wav_bytes(render_melody(notes, sa_hz, t + 0.5, sr, vibrato=True, drone=0.12), sr)
