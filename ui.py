"""Look & feel: CSS, animations and small HTML builders. Pure strings, no Streamlit import.

Every HTML string is returned on ONE line with no blank lines or indentation, because Markdown
would otherwise break the HTML into code blocks.
"""
from __future__ import annotations

import html
import re

from engine import DEFAULT_STYLE, KATTAI, PITCH_NAMES, mmss, note_text, scale_text, swara_name

# one colour per swara (index = semitone above Sa); komal notes are deeper shades of their hue
SWARA_COLORS = ["#ffd166", "#ff8fab", "#ff5d8f", "#c3a6ff", "#8b7bff", "#4cc9f0",
                "#3dd6a3", "#5ee08a", "#ffb86b", "#ff9f43", "#f4845f", "#ff6b6b"]


def _min(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


CSS = "<style>" + _min("""
:root{--accent:#ff8a3d;--line:rgba(255,255,255,.09);--card:rgba(255,255,255,.045)}
.stApp{background:
  radial-gradient(1100px 520px at 8% -8%,rgba(255,138,61,.16),transparent 60%),
  radial-gradient(900px 520px at 104% 2%,rgba(139,123,255,.18),transparent 58%),
  radial-gradient(800px 600px at 50% 120%,rgba(46,230,197,.08),transparent 60%),#0b0f1a}
.block-container{max-width:1100px;padding-top:1.1rem}
footer{visibility:hidden}

@keyframes flow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes eq{0%,100%{transform:scaleY(.18)}50%{transform:scaleY(1)}}
@keyframes pop{from{opacity:0;transform:translateY(8px) scale(.95)}to{opacity:1;transform:none}}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
@keyframes glow{0%,100%{box-shadow:0 0 0 0 rgba(255,138,61,.0)}50%{box-shadow:0 0 28px 2px rgba(255,138,61,.25)}}

.hero{position:relative;overflow:hidden;border-radius:24px;padding:28px 30px 24px;margin:0 0 18px;
  border:1px solid var(--line);background:linear-gradient(120deg,rgba(255,138,61,.22),rgba(255,94,138,.17),
  rgba(139,123,255,.22),rgba(46,230,197,.14));background-size:300% 300%;animation:flow 14s ease infinite}
.hero h1{margin:0;font-size:2.15rem;line-height:1.15;font-weight:800;letter-spacing:-.5px;
  background:linear-gradient(90deg,#ffd166,#ff8a3d,#ff5e8a,#b794f6);-webkit-background-clip:text;
  background-clip:text;color:transparent}
.hero p{margin:.5rem 0 0;color:#cdd3e6;font-size:1.02rem;max-width:620px}
.hero .eq{position:absolute;right:26px;top:22px;height:70px}
.eq{display:flex;gap:5px;align-items:flex-end}
.eq i{display:block;width:7px;height:100%;border-radius:4px;background:linear-gradient(180deg,#ffd166,#ff5e8a);
  transform-origin:bottom;animation:eq 1.15s ease-in-out infinite;animation-delay:calc(var(--i)*-.14s)}

.loader{display:flex;align-items:center;gap:18px;padding:18px 22px;margin:10px 0;border-radius:18px;
  border:1px solid var(--line);background:linear-gradient(90deg,rgba(255,255,255,.03),rgba(255,255,255,.09),
  rgba(255,255,255,.03));background-size:200% 100%;animation:shimmer 2.2s linear infinite}
.loader .eq{height:44px}.loader b{color:#fff}.loader span{color:#aab3d0;font-size:.88rem;display:block}

.stat{position:relative;overflow:hidden;height:100%;border-radius:18px;padding:15px 17px;background:var(--card);
  border:1px solid var(--line);backdrop-filter:blur(6px);animation:pop .55s both;animation-delay:var(--d,0ms)}
.stat:before{content:"";position:absolute;left:0;right:0;top:0;height:3px;
  background:linear-gradient(90deg,var(--c1,#ff8a3d),var(--c2,#ff5e8a))}
.stat .k{font-size:.72rem;letter-spacing:.13em;text-transform:uppercase;color:#9aa4c2}
.stat .v{font-size:1.6rem;font-weight:800;color:#fff;line-height:1.2;margin-top:5px;word-break:break-word}
.stat .v.sm{font-size:1.08rem}.stat .s{font-size:.82rem;color:#aab3d0;margin-top:5px}

.badge{display:inline-block;padding:3px 12px;border-radius:999px;font-weight:800;font-size:.95rem}
.badge.high{background:rgba(94,224,138,.18);color:#6df0a0;border:1px solid rgba(94,224,138,.5)}
.badge.medium{background:rgba(255,184,107,.16);color:#ffc680;border:1px solid rgba(255,184,107,.5)}
.badge.low{background:rgba(255,107,107,.16);color:#ff8f8f;border:1px solid rgba(255,107,107,.5)}

.notes-wrap{max-height:560px;overflow-y:auto;padding:10px 16px;border-radius:18px;
  background:rgba(8,11,20,.72);border:1px solid var(--line)}
.nrow{display:flex;gap:12px;align-items:center;padding:7px 0;border-bottom:1px dashed rgba(255,255,255,.07);
  animation:pop .45s both;animation-delay:var(--rd,0ms)}
.nrow:last-child{border-bottom:none}
.ts{flex:0 0 auto;font:600 .78rem ui-monospace,Menlo,Consolas,monospace;color:#8f9ac0;
  background:rgba(255,255,255,.06);border-radius:8px;padding:3px 8px}
.nl{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.sw{display:inline-flex;align-items:center;justify-content:center;min-width:44px;padding:5px 11px;
  border-radius:999px;background:var(--c);color:#12162a;font-weight:800;font-size:.98rem;
  box-shadow:0 3px 12px -4px var(--c);transition:transform .15s}
.sw:hover{transform:translateY(-2px) scale(1.08)}
.sw sup,.sw sub{font-size:.62em;margin-left:1px;line-height:0}
.hold{color:#7f8ab0;font-weight:800;padding:0 2px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.82rem;color:#aab3d0;margin:.1rem 0 .7rem}
.scale{display:flex;flex-wrap:wrap;gap:8px;margin:.3rem 0 .9rem}
.scale .sw{font-size:1.05rem;padding:7px 15px}

.step{height:100%;padding:18px;border-radius:18px;background:var(--card);border:1px solid var(--line);
  animation:pop .6s both;animation-delay:var(--d,0ms);transition:transform .2s,border-color .2s}
.step:hover{transform:translateY(-4px);border-color:rgba(255,138,61,.6)}
.step .n{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-weight:800;color:#12162a;background:linear-gradient(135deg,#ffd166,#ff5e8a);margin-bottom:10px}
.step b{color:#fff;font-size:1.02rem}.step p{color:#aab3d0;font-size:.9rem;margin:.35rem 0 0}
.strip{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;padding:10px 16px;border-radius:14px;
  background:rgba(255,255,255,.04);border:1px solid var(--line);margin:8px 0 6px;color:#cdd3e6;font-size:.92rem}
.strip b{color:#fff}

[data-testid="stFileUploaderDropzone"]{border:2px dashed rgba(255,159,67,.65);border-radius:18px;
  background:rgba(255,159,67,.06);transition:all .25s}
[data-testid="stFileUploaderDropzone"]:hover{background:rgba(255,159,67,.12);transform:translateY(-1px);
  animation:glow 1.6s ease-in-out infinite}
.stButton>button,.stDownloadButton>button{border-radius:12px;border:1px solid rgba(255,159,67,.55);
  transition:all .2s}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-2px);
  box-shadow:0 8px 20px -6px rgba(255,159,67,.45);border-color:#ff9f43}
.stTabs [data-baseweb="tab-list"]{gap:.35rem}
.stTabs [data-baseweb="tab"]{border-radius:12px 12px 0 0;padding:.55rem 1rem}
div[data-testid="stExpander"]{border-radius:14px;border:1px solid var(--line);background:rgba(255,255,255,.025)}

@keyframes pulse{0%,100%{box-shadow:inset 0 0 0 2px #ffd166,0 0 10px rgba(255,209,102,.35)}50%{box-shadow:inset 0 0 0 2px #ffd166,0 0 22px rgba(255,209,102,.85)}}
.tag{flex:0 0 auto;width:26px;height:26px;border-radius:8px;display:flex;align-items:center;justify-content:center;
  font-weight:800;font-size:.8rem;color:#12162a;background:var(--c)}
.ln{flex:0 0 auto;min-width:26px;text-align:right;color:#6f7aa0;font-weight:700;font-size:.85rem}
.rep{color:#9fe3b6;font-size:.86rem;background:rgba(94,224,138,.1);border:1px solid rgba(94,224,138,.35);
  border-radius:999px;padding:3px 12px}
.nrow.isrep .nl{opacity:.92}
.kb-scroll{overflow-x:auto;padding:6px 2px 14px;-webkit-overflow-scrolling:touch}
.kb{position:relative;display:flex;min-width:780px;height:200px;border-radius:0 0 12px 12px;
  padding:0;background:#0d1020;border:1px solid var(--line)}
.wk{flex:1;position:relative;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:2px;
  padding-bottom:7px;background:linear-gradient(#f7f4ec,#e9e4d6);border:1px solid #1a1e33;
  border-radius:0 0 9px 9px;color:#12162a;transition:transform .15s}
.wk.hl{background:linear-gradient(#ffffff,var(--c))}
.wk .l{font-weight:800;font-size:.84rem}.wk .kn{font-size:.62rem;color:#7b7f94}
.bk{position:absolute;top:0;height:62%;width:2.9%;z-index:2;background:linear-gradient(#262b45,#0b0e1c);
  border-radius:0 0 7px 7px;display:flex;align-items:flex-end;justify-content:center;padding-bottom:6px;
  color:#fff;font-weight:800;font-size:.68rem;border:1px solid #05070f}
.bk.hl{background:linear-gradient(#ffffff55,var(--c));color:#12162a}
.wk.sa,.bk.sa{animation:pulse 2.2s ease-in-out infinite}
.kb sup,.kb sub{font-size:.65em;line-height:0}
.tipbox{padding:12px 16px;border-radius:14px;background:rgba(76,201,240,.07);border:1px solid rgba(76,201,240,.3);
  color:#cfe9f5;font-size:.92rem;margin:.4rem 0 .8rem}
@media (max-width:700px){.nrow{flex-wrap:wrap}.sw{min-width:38px;padding:4px 9px;font-size:.9rem}}
@media (max-width:700px){.hero .eq{display:none}.hero h1{font-size:1.65rem}.stat .v{font-size:1.3rem}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
""") + "</style>"


def _eq(n: int = 10) -> str:
    return '<div class="eq">' + "".join(f'<i style="--i:{i}"></i>' for i in range(n)) + "</div>"


def hero() -> str:
    return ('<div class="hero">' + _eq() + "<h1>🎶 Raga &amp; Sargam Notes</h1>"
            "<p>Drop in any song or video. Get the raga, the Sa (pitch), a transposition guide and the "
            "notes in <b>Sa Ri Ga Ma Pa</b> – ready to play.</p></div>")


def loader(msg: str) -> str:
    return ('<div class="loader">' + _eq(8) + f"<div><b>{html.escape(msg)}</b>"
            "<span>Hang on, this only takes a moment…</span></div></div>")


def stat_card(label: str, value: str, sub: str = "", delay: int = 0, small: bool = False,
              c1: str = "#ff8a3d", c2: str = "#ff5e8a", raw_value: bool = False) -> str:
    v = value if raw_value else html.escape(value)
    cls = "v sm" if small else "v"
    s = f'<div class="s">{html.escape(sub)}</div>' if sub else ""
    return (f'<div class="stat" style="--d:{delay}ms;--c1:{c1};--c2:{c2}">'
            f'<div class="k">{html.escape(label)}</div><div class="{cls}">{v}</div>{s}</div>')


def badge(level: str) -> str:
    return f'<span class="badge {level.lower()}">{html.escape(level)}</span>'


TAG_COLORS = ["#ffd166", "#4cc9f0", "#ff8fab", "#5ee08a", "#c3a6ff", "#ffb86b", "#3dd6a3", "#ff6b6b"]


def tag_color(tag: str) -> str:
    return TAG_COLORS[(ord(tag[0]) - 65) % len(TAG_COLORS)]


def _chip(pc: int, octave: int = 0, style: str = DEFAULT_STYLE) -> str:
    mark = ""
    if octave > 0:
        mark = "<sup>" + "•" * octave + "</sup>"
    elif octave < 0:
        mark = "<sub>" + "•" * (-octave) + "</sub>"
    return f'<span class="sw" style="--c:{SWARA_COLORS[pc % 12]}">{html.escape(swara_name(pc, style))}{mark}</span>'


def notes_html(lines, style: str = DEFAULT_STYLE, collapse: bool = False, max_lines: int = 500) -> str:
    out = ['<div class="notes-wrap">']
    for i, ln in enumerate(lines[:max_lines]):
        head = (f'<span class="ln">{ln.index}</span><span class="ts">{ln.stamp}</span>'
                f'<span class="tag" style="--c:{tag_color(ln.tag)}">{html.escape(ln.tag)}</span>')
        if collapse and ln.repeat_of:
            body = f'<span class="rep">↻ same tune as line {ln.repeat_of}</span>'
        else:
            body = "".join(_chip(n.pc, n.octave, style) + '<span class="hold">–</span>' * n.dashes
                           for n in ln.notes)
            if ln.repeat_of:
                body += f'<span class="rep">↻ like line {ln.repeat_of}</span>'
        out.append(f'<div class="nrow{" isrep" if ln.repeat_of else ""}" style="--rd:{min(i, 24) * 30}ms">'
                   f'{head}<div class="nl">{body}</div></div>')
    if len(lines) > max_lines:
        out.append(f'<div class="nrow"><span class="ts">…</span><div class="nl">'
                   f"{len(lines) - max_lines} more lines are in the download</div></div>")
    out.append("</div>")
    return "".join(out)


def legend() -> str:
    items = ["♭ komal (flat)", "♯ tivra (sharp)", "• above = upper octave", "• below = lower octave",
             "– hold the note", "letter = tune (same letter = same melody)"]
    return '<div class="legend">' + "".join(f"<span>{html.escape(i)}</span>" for i in items) + "</div>"


def scale_html(pcs, style: str = DEFAULT_STYLE, descending: bool = False) -> str:
    seq = sorted(pcs, reverse=descending)
    chips = [_chip(p, 0, style) for p in seq]
    chips = [_chip(0, 1, style)] + chips if descending else chips + [_chip(0, 1, style)]
    return '<div class="scale">' + "".join(chips) + "</div>"


def keyboard_html(sa_pc: int, raga_notes, style: str = DEFAULT_STYLE, octaves: int = 3) -> str:
    """A 3-octave harmonium keyboard with the raga's swaras lit up for the chosen Sa."""
    black = {1, 3, 6, 8, 10}
    n_keys = 12 * octaves
    whites = [i for i in range(n_keys) if i % 12 not in black]
    sa_index = 12 + sa_pc                                   # Sa in the middle octave
    notes = set(raga_notes)

    def info(i):
        rel = i - sa_index
        pc, octv = rel % 12, rel // 12
        lit = pc in notes
        lab = ""
        if lit:
            mark = "<sup>" + "•" * octv + "</sup>" if octv > 0 else "<sub>" + "•" * (-octv) + "</sub>" if octv < 0 else ""
            lab = html.escape(swara_name(pc, style)) + mark
        return lit, pc, lab

    out = ['<div class="kb-scroll"><div class="kb">']
    for i in whites:
        lit, pc, lab = info(i)
        cls = "wk" + (" hl" if lit else "") + (" sa" if lit and pc == 0 else "")
        c = f' style="--c:{SWARA_COLORS[pc]}"' if lit else ""
        out.append(f'<div class="{cls}"{c}><span class="l">{lab}</span><span class="kn">{PITCH_NAMES[i % 12]}</span></div>')
    w = len(whites)
    for i in range(n_keys):
        if i % 12 in black:
            left = (sum(1 for k in whites if k < i)) / w * 100 - 1.45
            lit, pc, lab = info(i)
            cls = "bk" + (" hl" if lit else "") + (" sa" if lit and pc == 0 else "")
            c = f";--c:{SWARA_COLORS[pc]}" if lit else ""
            out.append(f'<div class="{cls}" style="left:{left:.3f}%{c}">{lab}</div>')
    out.append("</div></div>")
    return "".join(out)


def sheet_html(res, style: str = DEFAULT_STYLE, collapse: bool = False, title: str = "Sargam notes") -> str:
    """A self-contained, printable page (opens on any device; 'Print' -> 'Save as PDF')."""
    up, down = scale_text(res.raga, style)
    rows = []
    for ln in res.lines:
        if collapse and ln.repeat_of:
            body = f'<i class="rep">same tune as line {ln.repeat_of}</i>'
        else:
            body = ""
            for n in ln.notes:
                mark = "<sup>" + "•" * n.octave + "</sup>" if n.octave > 0 else (
                    "<sub>" + "•" * (-n.octave) + "</sub>" if n.octave < 0 else "")
                body += (f'<span class="n" style="--c:{SWARA_COLORS[n.pc]}">{html.escape(swara_name(n.pc, style))}{mark}</span>'
                         + '<span class="h">–</span>' * n.dashes)
        rows.append(f'<div class="r"><b class="no">{ln.index}</b><span class="t">{ln.stamp}</span>'
                    f'<span class="g">{html.escape(ln.tag)}</span><div class="s">{body}</div></div>')
    css = ("body{font-family:system-ui,'Noto Sans','Noto Sans Devanagari',sans-serif;margin:0;padding:22px;color:#14172b;"
           "background:#fff;max-width:900px;margin:auto}h1{margin:0 0 4px;font-size:1.5rem}"
           ".m{color:#4a5070;margin:2px 0}.box{border:1px solid #d9dcea;border-radius:12px;padding:12px 16px;margin:14px 0;"
           "background:#f8f9fd}.r{display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-bottom:1px dashed #d9dcea;"
           "page-break-inside:avoid}.no{min-width:28px;text-align:right;color:#6b7194}.t{font:12px monospace;color:#6b7194;"
           "padding-top:4px}.g{background:#14172b;color:#fff;border-radius:6px;padding:1px 7px;font-size:12px;font-weight:700;"
           "margin-top:2px}.s{flex:1;line-height:2.1}.n{display:inline-block;margin:0 3px;padding:0 7px;font-weight:700;"
           "font-size:18px;border-bottom:4px solid var(--c)}.h{color:#8a90b0;font-weight:700}.rep{color:#2f8f55}"
           "sup,sub{font-size:.6em;line-height:0}.foot{color:#8a90b0;font-size:12px;margin-top:18px}"
           "@media print{body{padding:8px}.box{background:#fff}}")
    return ("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{css}</style></head><body>"
            f"<h1>{html.escape(title)}</h1><p class='m'>Raga: <b>{html.escape(res.raga.name)}</b> &nbsp;|&nbsp; "
            f"Sa: <b>{PITCH_NAMES[res.sa_pc]}</b> (harmonium scale {KATTAI[res.sa_pc]})</p>"
            f"<div class='box'><div class='m'>Aroha: <b>{html.escape(up)}</b></div>"
            f"<div class='m'>Avaroha: <b>{html.escape(down)}</b></div>"
            "<div class='m'>♭ komal · ♯ tivra · dot above = upper octave · dot below = lower octave · – hold the note</div></div>"
            + "".join(rows) + "<p class='foot'>Made with Raga &amp; Sargam Notes. Notes are an automatic estimate - "
            "check them by ear.</p></body></html>")


def steps() -> str:
    data = [("Upload", "Any audio or video file – mp3, m4a, wav, flac, mp4, mkv, mov… or record a tune."),
            ("We listen", "The melody is tracked, the Sa (pitch) is found and the raga is matched."),
            ("You play", "Get the whole song line by line in sargam, see the harmonium keys, and practise slowly.")]
    cards = "".join(f'<div class="step" style="--d:{i * 120}ms"><div class="n">{i + 1}</div><b>{t}</b><p>{d}</p></div>'
                    for i, (t, d) in enumerate(data))
    return f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px">{cards}</div>'


def file_strip(name: str, size_mb: float, duration: float) -> str:
    return (f'<div class="strip">🎧 <b>{html.escape(name)}</b><span>{size_mb:.1f} MB</span>'
            f"<span>⏱ {mmss(duration)}</span></div>")
