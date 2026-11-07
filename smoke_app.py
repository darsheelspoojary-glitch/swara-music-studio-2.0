"""Runs app.py end-to-end against a tiny fake `streamlit` (no Streamlit install needed).
Checks: no crashes, friendly errors, HTML safe for Markdown.   python tests/smoke_app.py"""
import os, re, runpy, subprocess, sys, tempfile, types
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import engine as E

class Stop(Exception): pass
class Upload:
    def __init__(self, name, data): self.name, self._d = name, data
    def getvalue(self): return self._d

def make_fake(ss, upload=None):
    log = {"html": [], "errors": [], "warnings": [], "calls": []}
    st = types.ModuleType("streamlit")
    st.session_state = ss
    class Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __getattr__(self, n): return getattr(st, n)
    def widget(default_fn):
        def f(label, *a, key=None, **k):
            v = ss.get(key) if key and key in ss else default_fn(label, *a, **k)
            if key: ss[key] = v
            return v
        return f
    def _opts(o): return list(o)
    st.radio = widget(lambda l, o, index=0, **k: _opts(o)[index])
    def selectbox(label, options, index=0, key=None, **k):
        o = _opts(options); v = ss.get(key) if key in ss else o[index]
        v = v if v in o else o[index]
        if key: ss[key] = v
        return v
    st.selectbox = selectbox
    def slider(label, mn=None, mx=None, value=None, step=None, key=None, **k):
        v = ss[key] if key and key in ss else value
        if key: ss[key] = v
        return v
    st.slider = slider
    st.select_slider = lambda l, options=None, value=None, **k: value
    st.checkbox = lambda l, value=False, key=None, **k: ss.get(key, value) if key else value
    def button(label, on_click=None, args=(), key=None, **k):
        log["calls"].append(("button", label)); return False
    st.button = button
    st.file_uploader = lambda *a, **k: upload
    st.audio_input = lambda *a, **k: None
    st.columns = lambda spec, **k: [Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda labels: [Ctx() for _ in labels]
    st.expander = lambda *a, **k: Ctx()
    st.empty = lambda: Ctx()
    st.empty_markdown = None
    Ctx.markdown = lambda self, s, unsafe_allow_html=False: st.markdown(s, unsafe_allow_html)
    Ctx.empty = lambda self: None
    def markdown(s, unsafe_allow_html=False):
        if unsafe_allow_html: log["html"].append(s)
    st.markdown = markdown
    st.error = lambda m, *a, **k: log["errors"].append(str(m))
    st.warning = lambda m, *a, **k: log["warnings"].append(str(m))
    for n in ("caption", "info", "write", "code", "dataframe", "line_chart", "audio", "video", "set_page_config", "subheader"):
        setattr(st, n, lambda *a, **k: log["calls"].append("x"))
    st.download_button = lambda *a, **k: log["calls"].append(("download", a[0])) or False
    def stop(): raise Stop()
    st.stop = stop
    def cache_data(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k: return a[0]
        return lambda f: f
    st.cache_data = cache_data
    return st, log

def run(ss, upload=None):
    st, log = make_fake(ss, upload)
    sys.modules["streamlit"] = st
    for m in ("engine", "ui"): sys.modules.pop(m, None)
    sys.path.insert(0, ROOT)
    try:
        runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
        log["finished"] = True
    except Stop:
        log["finished"] = False
    return log

fails = 0
def check(name, cond):
    global fails
    print(("PASS" if cond else "FAIL"), name); fails += (not cond)

def html_ok(log):
    bad = [h[:60] for h in log["html"] if "\n\n" in h or re.search(r"\n {4,}", h)]
    return not bad

# 1. nothing chosen -> landing page, no crash
lg = run({}); check("landing page renders", not lg["errors"] and html_ok(lg) and any("step" in h for h in lg["html"]))
# 2. demo song, default settings
lg = run({"demo": True})
check("demo: app runs to the end", lg["finished"] and not lg["errors"])
check("demo: result cards + notes + keyboard shown", any("Yaman" in h for h in lg["html"]) and any('class="kb"' in h for h in lg["html"]) and any("notes-wrap" in h for h in lg["html"]))
check("demo: HTML safe for Markdown", html_ok(lg))
check("demo: 3 download buttons", sum(1 for c in lg["calls"] if isinstance(c, tuple) and c[0] == "download") == 3)
# 3. other display settings
for extra in ({"style": "S R G M", "view": "Text"}, {"style": "सा रे ग म"}, {"tradition": "Carnatic"}, {"tradition": "Hindustani", "sa_choice": "C"},
              {"engine": "Full song mix (fast)", "sound": "Soft voice"}):
    lg = run({"demo": True, **extra}); check(f"demo with {extra}", lg["finished"] and not lg["errors"] and html_ok(lg))
# 4. real uploads in several containers
wav = E.demo_song()
with tempfile.TemporaryDirectory() as d:
    open(f"{d}/a.wav", "wb").write(wav)
    for ext, extra in [("mp3", []), ("m4a", ["-c:a", "aac"]), ("ogg", ["-c:a", "libvorbis"]),
                       ("mp4", ["-f", "lavfi", "-i", "color=c=blue:s=64x64:d=70", "-c:v", "mpeg4", "-c:a", "aac", "-shortest"])]:
        out = f"{d}/o.{ext}"
        if subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", f"{d}/a.wav"] + extra + [out], capture_output=True).returncode: continue
        lg = run({}, Upload("My Song." + ext, open(out, "rb").read()))
        check(f"upload .{ext} -> full result", lg["finished"] and not lg["errors"] and any("Yaman" in h or "Kalyani" in h for h in lg["html"]))
# 5. bad inputs give friendly errors, never a crash
lg = run({}, Upload("notes.pdf", os.urandom(4000))); check("garbage file -> friendly error", lg["errors"] and not lg["finished"])
lg = run({}, Upload("empty.mp3", b"")); check("empty file -> friendly error", lg["errors"] and not lg["finished"])
import numpy as np
lg = run({}, Upload("silence.wav", E.to_wav_bytes(np.zeros(16000 * 6, np.float32), 16000))); check("silent file -> friendly error", lg["errors"])
lg = run({}, Upload("noise.wav", E.to_wav_bytes((np.random.randn(16000 * 15) * .2).astype(np.float32), 16000)))
check("noise file -> friendly error", lg["errors"] and not lg["finished"])
print("\nALL PASSED" if not fails else f"\n{fails} FAILED"); sys.exit(1 if fails else 0)
