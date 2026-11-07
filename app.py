"""Raga & Sargam Notes - Streamlit app (run:  streamlit run app.py)."""
import hashlib
import html
import importlib.util
import json
import os
import time

import pandas as pd
import streamlit as st

import engine as E
import ui

st.set_page_config(page_title="Raga & Sargam Notes", page_icon="🎶", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
st.markdown(ui.hero(), unsafe_allow_html=True)

ss = st.session_state
HAS_PYIN = importlib.util.find_spec("librosa") is not None


# --------------------------------------------------------------------------- #
# cached heavy work (re-running with other display settings is instant)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=2)
def load_samples(key: str, _data: bytes, suffix: str):
    return E.decode_audio(_data, suffix)


@st.cache_data(show_spinner=False, max_entries=8)
def get_track(key: str, a: int, b: int, method: str, _y):
    t0 = time.time()
    track = E.extract_f0(_y[a * E.SR: b * E.SR], method)
    return track, time.time() - t0


@st.cache_data(show_spinner=False, max_entries=1)
def demo_bytes():
    return E.demo_song()


@st.cache_data(show_spinner=False, max_entries=6)
def synth(notes: tuple, sa_hz: float, speed: float, drone: float, timbre: str):
    return E.render_notes(list(notes), sa_hz, speed, drone, timbre)


def with_loader(msg, fn, *args):
    slot = st.empty()
    slot.markdown(ui.loader(msg), unsafe_allow_html=True)
    try:
        return fn(*args)
    finally:
        slot.empty()


def use_reading(sa_name: str, raga_name: str):          # button callback: try another Sa / raga reading
    ss["sa_choice"] = sa_name
    ss["raga_choice"] = raga_name


def stop_demo():
    ss["demo"] = False


def start_demo():
    ss["demo"] = True


# --------------------------------------------------------------------------- #
# 1. choose a source: upload / record / demo
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 2], gap="large")
with left:
    up = st.file_uploader("Drop any audio or video file here", key="uploader", on_change=stop_demo,
                          help="mp3, m4a, wav, flac, ogg, aac, opus, wma, mp4, mkv, mov, webm, avi … almost anything with sound.")
    st.caption("🎵 mp3 · m4a · wav · flac · ogg · aac · opus · wma   🎬 mp4 · mkv · mov · webm · avi · 3gp · and more")
with right:
    st.button("▶  Try the demo song", on_click=start_demo, help="A short Raga Yaman melody so you can see how it works.")
    rec = None
    if hasattr(st, "audio_input"):
        rec = st.audio_input("🎤 Or hum / play a tune", on_change=stop_demo)

if ss.get("demo"):
    data, name, suffix = demo_bytes(), "demo_yaman.wav", ".wav"
elif up is not None:
    data, name = up.getvalue(), up.name
    suffix = os.path.splitext(name)[1].lower()
elif rec is not None:
    data, name, suffix = rec.getvalue(), "recording.wav", ".wav"
else:
    st.markdown(ui.steps(), unsafe_allow_html=True)
    st.markdown('<div class="tipbox">💡 <b>For the best notes:</b> use the clearest version of the song you have — '
                "a vocal-only or karaoke-free recording gives the most accurate result. "
                "Everything is processed in this session and nothing is stored.</div>", unsafe_allow_html=True)
    st.stop()

# --------------------------------------------------------------------------- #
# 2. decode (cached) and analysis settings
# --------------------------------------------------------------------------- #
key = hashlib.md5(data).hexdigest()
try:
    y = with_loader("Reading your file…", load_samples, key, data, suffix)
except ValueError as e:
    st.error(f"😕 {e}")
    st.stop()
except Exception as e:                                   # never show a raw traceback to the user
    st.error("Something unexpected happened while reading this file. Please try another file.")
    st.caption(f"Technical detail: {type(e).__name__}")
    st.stop()

total = len(y) / E.SR
total_i = max(int(total), 2)
st.markdown(ui.file_strip(name, len(data) / 1e6, total), unsafe_allow_html=True)

with st.expander("⚙️ Fine-tune (optional)", expanded=False):
    c1, c2, c3 = st.columns(3)
    tradition = c1.radio("Raga system", ["Both", "Carnatic", "Hindustani"], key="tradition")
    sa_choice = c2.selectbox("Sa (tonic)", ["Auto"] + E.PITCH_NAMES, key="sa_choice",
                             help="If you know the singer's Sa, set it here. This is the most reliable way to improve the raga.")
    opts = ["Auto"] + [r.name for r in E.ragas_for(tradition)]
    if ss.get("raga_choice") not in opts:
        ss["raga_choice"] = "Auto"
    raga_choice = c3.selectbox("Raga", opts, key="raga_choice")
    engines = ["Full song mix (fast)"] + (["Solo voice / instrument (precise, slower)"] if HAS_PYIN else [])
    engine_choice = st.radio("Melody engine", engines, horizontal=True, key="engine")
    section = st.slider("Section to analyse (seconds)", 0, total_i, (0, total_i), key=f"section_{key}")
    d1, d2 = st.columns(2)
    min_note = d1.slider("Shortest note (seconds)", 0.06, 0.40, 0.12, 0.01,
                         help="Lower = more (faster) notes, higher = cleaner.")
    line_gap = d2.slider("New line after a pause of (seconds)", 0.20, 1.50, 0.45, 0.05,
                         help="Lyric lines are separated by the singer's breaths. Lower = shorter lines.")
    e1, e2 = st.columns(2)
    snap = e1.checkbox("Clean notes using the detected raga", True)
    dashes = e2.checkbox("Mark long notes with –", True)

a, b = int(section[0]), int(section[1])
method = "pyin" if engine_choice.startswith("Solo") else "salience"
if method == "pyin" and b - a > 150:
    b = a + 150
    st.caption("Solo engine analyses up to 150 seconds at a time - using the first 150 seconds of your section.")
if b - a < 3:
    st.warning("Please choose a section of at least 3 seconds.")
    st.stop()

# --------------------------------------------------------------------------- #
# 3. analyse
# --------------------------------------------------------------------------- #
try:
    track, took = with_loader("Listening to the melody…", get_track, key, a, b, method, y)
    res = E.interpret(track, float(a), tradition,
                      sa_pc=None if sa_choice == "Auto" else E.PITCH_NAMES.index(sa_choice),
                      raga_name=None if raga_choice == "Auto" else raga_choice,
                      min_dur=min_note, phrase_gap=line_gap, snap=snap, dashes=dashes)
except ValueError as e:
    st.error(f"😕 {e}")
    st.stop()
except Exception as e:
    st.error("Something unexpected happened while analysing. Please try another section or file.")
    with st.expander("Technical details"):
        st.code(repr(e))
    st.stop()

n_lines, n_unique = len(res.lines), len({ln.tag for ln in res.lines})
conf_color = {"High": ("#5ee08a", "#3dd6a3"), "Medium": ("#ffb86b", "#ff9f43"), "Low": ("#ff6b6b", "#f4845f")}[res.confidence]

# --------------------------------------------------------------------------- #
# 4. result cards
# --------------------------------------------------------------------------- #
twins = ("Same swaras: " + ", ".join(res.same_scale[:3])) if res.same_scale else res.raga.system
cards = st.columns(4)
cards[0].markdown(ui.stat_card("Raga", res.raga.name, twins, 0, small=len(res.raga.name) > 18), unsafe_allow_html=True)
cards[1].markdown(ui.stat_card("Sa (pitch)", E.PITCH_NAMES[res.sa_pc],
                               f"≈ {res.sa_hz:.1f} Hz · harmonium {E.KATTAI[res.sa_pc]}", 80, c1="#4cc9f0", c2="#8b7bff"),
                  unsafe_allow_html=True)
cards[2].markdown(ui.stat_card("Confidence", ui.badge(res.confidence), "of this raga + Sa reading", 160,
                               c1=conf_color[0], c2=conf_color[1], raw_value=True), unsafe_allow_html=True)
cards[3].markdown(ui.stat_card("Whole song", f"{n_lines} lines", f"{res.n_notes} notes · {n_unique} tunes · ⚡ {took:.1f}s", 240,
                               c1="#c3a6ff", c2="#ff5e8a"), unsafe_allow_html=True)

for w in res.warnings[:2]:
    st.warning(w)

if res.alternatives:
    st.markdown("**Doesn't sound right?** Try another reading:")
    cols = st.columns(len(res.alternatives))
    for i, (pc, rname, pct) in enumerate(res.alternatives):
        cols[i].button(f"Sa = {E.PITCH_NAMES[pc]} · {rname.split(' (')[0]} ({pct}%)", key=f"alt{i}",
                       on_click=use_reading, args=(E.PITCH_NAMES[pc], rname))

# --------------------------------------------------------------------------- #
# 5. tabs
# --------------------------------------------------------------------------- #
t_notes, t_harm, t_prac, t_raga, t_graph = st.tabs(
    ["🎼 Song notes", "🎹 Harmonium", "🔊 Practice", "🧭 Raga", "📈 Pitch graph"])

# ---- Song notes: the whole song, line by line ---- #
with t_notes:
    n1, n2, n3 = st.columns([3, 2, 2])
    style = n1.radio("Write notes as", list(E.STYLES), horizontal=True, key="style")
    collapse = n2.checkbox("Shorten repeated lines", False, help="Show '↻ same tune as line 3' instead of repeating notes.")
    view = n3.radio("View", ["Colour", "Text"], horizontal=True, key="view")
    st.markdown(ui.legend(), unsafe_allow_html=True)
    if view == "Colour":
        st.markdown(ui.notes_html(res.lines, style, collapse), unsafe_allow_html=True)
    else:
        st.code(E.build_text(res, style, collapse), language=None)
    text = E.build_text(res, style, collapse)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Notes (.txt)", text, file_name="sargam_notes.txt", mime="text/plain")
    d2.download_button("⬇️ Printable sheet (.html)", ui.sheet_html(res, style, collapse, name.rsplit(".", 1)[0]),
                       file_name="sargam_sheet.html", mime="text/html",
                       help="Open it in any browser; use Print → Save as PDF.")
    d3.download_button("⬇️ Every note with time (.csv)",
                       pd.DataFrame(E.notes_table(res, style)).to_csv(index=False).encode("utf-8-sig"),
                       file_name="sargam_notes.csv", mime="text/csv")

# ---- Harmonium keyboard ---- #
with t_harm:
    st.markdown('<div class="tipbox">🎹 Pick the key your harmonium Sa is on. The highlighted keys are the swaras of '
                f"<b>{html.escape(res.raga.name)}</b>. The notes you read in <i>Song notes</i> stay exactly the same in any scale.</div>",
                unsafe_allow_html=True)
    names = [f"{E.KATTAI[i]}  ({E.PITCH_NAMES[i]})" for i in range(12)]
    pick = st.selectbox("Sa on harmonium key", names, index=res.sa_pc, key=f"hsa_{key}_{res.sa_pc}")
    target = names.index(pick)
    st.markdown(ui.keyboard_html(target, res.raga.notes, ss.get("style", E.DEFAULT_STYLE)), unsafe_allow_html=True)
    shift = (target - res.sa_pc + 6) % 12 - 6
    st.write("Same key as the original recording." if shift == 0 else
             f"Compared with the original, play **{abs(shift)} semitone(s) {'higher' if shift > 0 else 'lower'}**.")
    st.dataframe(pd.DataFrame(E.transpose_table(res.raga, res.sa_pc, target, ss.get("style", E.DEFAULT_STYLE))),
                 hide_index=True)

# ---- Practice: hear it slowly, line by line ---- #
with t_prac:
    st.caption("Listen to the detected notes on a harmonium sound, slow them down and practise one line at a time.")
    p1, p2, p3 = st.columns([3, 2, 2])
    choices = ["Whole song (first 2½ min)"] + [f"Line {ln.index} · {ln.stamp} · tune {ln.tag}" for ln in res.lines]
    which = p1.selectbox("Play", choices, key=f"play_{key}")
    speed = p2.select_slider("Speed", options=[0.5, 0.75, 1.0, 1.25], value=1.0, format_func=lambda v: f"{v}×")
    sound = p3.radio("Sound", ["Harmonium", "Soft voice"], horizontal=True, key="sound")
    drone = st.checkbox("Add tanpura-style drone", False)
    idx = choices.index(which)
    if idx == 0:
        play = [n for n in res.notes]
    else:
        ln = res.lines[idx - 1]
        play = [(n.semi, n.t0, n.t1) for n in ln.notes]
        st.markdown(ui.notes_html([ln], ss.get("style", E.DEFAULT_STYLE)), unsafe_allow_html=True)
    try:
        wav = synth(tuple(play), float(res.sa_hz), float(speed), 0.08 if drone else 0.0,
                    "harmonium" if sound == "Harmonium" else "vocal")
        st.audio(wav, format="audio/wav")
    except ValueError as e:
        st.info(str(e))
    st.markdown(f"**Original** (this section starts at {E.mmss(a)})")
    (st.video if suffix.lstrip(".") in ("mp4", "mkv", "mov", "webm", "avi", "3gp", "m4v") else st.audio)(data)

# ---- Raga details ---- #
with t_raga:
    stl = ss.get("style", E.DEFAULT_STYLE)
    st.markdown(f"### {html.escape(res.raga.name)}")
    st.markdown("**Aroha** (going up)")
    st.markdown(ui.scale_html(res.raga.up, stl), unsafe_allow_html=True)
    st.markdown("**Avaroha** (coming down)")
    st.markdown(ui.scale_html(res.raga.down, stl, descending=True), unsafe_allow_html=True)
    if res.same_scale:
        st.info("Ragas with exactly the same swaras: " + ", ".join(res.same_scale) +
                ". They can't be told apart from the notes alone - the phrases (pakad) decide.")
    st.markdown("**Closest ragas**")
    st.dataframe(pd.DataFrame(res.table), hide_index=True)
    with st.expander("How is the raga found?"):
        st.write("The song's melody is tracked, steady notes are counted (slides and quick gamaka wobbles are ignored), "
                 "and every raga is tried in all 12 possible Sa positions. Each (raga, Sa) pair is scored by how well the "
                 "swaras you hear fit it, how phrases start/end, the sound of the background drone or bass, and the "
                 "ascending/descending pattern. Closeness % shows how near each other candidate is to the winner.")

# ---- Pitch graph ---- #
with t_graph:
    dur_view = 30
    v0 = st.slider("Show from (seconds)", int(a), max(int(a), int(b) - dur_view), int(a), key=f"view_{key}") \
        if b - a > dur_view else int(a)
    try:
        import altair as alt
        ct, cs = res.contour
        pts = pd.DataFrame({"t": ct + a, "y": cs})
        pts = pts[(pts.t >= v0) & (pts.t < v0 + dur_view) & (pts.y > -14) & (pts.y < 26)]
        rects = pd.DataFrame([{"t0": a + n.t0, "t1": a + n.t1, "y0": n.semi - 0.38, "y1": n.semi + 0.38,
                               "pc": n.pc, "swara": E.note_text(n.pc, n.octave, ss.get("style", E.DEFAULT_STYLE))}
                              for ln in res.lines for n in ln.notes
                              if a + n.t1 >= v0 and a + n.t0 < v0 + dur_view])
        ticks = [o * 12 + n for o in (-1, 0, 1, 2) for n in res.raga.notes if -13 <= o * 12 + n <= 25]
        names_js = json.dumps(E.STYLES[ss.get("style", E.DEFAULT_STYLE)], ensure_ascii=False)
        yenc = alt.Y("y:Q", title=None, scale=alt.Scale(domain=[-13, 25]),
                     axis=alt.Axis(values=ticks, grid=True, gridOpacity=0.15,
                                   labelExpr=f"{names_js}[((datum.value % 12) + 12) % 12]"))
        dots = alt.Chart(pts).mark_circle(size=14, opacity=0.5, color="#9aa4c2").encode(
            x=alt.X("t:Q", title="seconds", scale=alt.Scale(domain=[v0, v0 + dur_view])), y=yenc)
        layers = dots
        if len(rects):
            bars = alt.Chart(rects).mark_rect(cornerRadius=3, opacity=0.9).encode(
                x="t0:Q", x2="t1:Q", y="y0:Q", y2="y1:Q",
                color=alt.Color("pc:O", scale=alt.Scale(domain=list(range(12)), range=ui.SWARA_COLORS), legend=None),
                tooltip=["swara", "t0", "t1"])
            layers = bars + dots
        st.altair_chart(layers.properties(height=380, width="container"))
        st.caption("Grey dots = the pitch we heard · coloured bars = the notes written down.")
    except Exception:
        st.line_chart(pd.DataFrame({"pitch": res.contour[1]}))

st.caption("Notes are an automatic estimate – always check them by ear. Nothing you upload is stored.")
