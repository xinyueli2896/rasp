#!/usr/bin/env python3
"""Generate the rule-adapter architecture figures.

Emits three SVGs describing the CP-transformer + yin-yang cross-attention
adapter, contrasting the two ways rule_hidden is produced:

    A  explicit rule input   --paired_chord_seq   (ChordSeqRuleModel)
    B  self-derived rule     --bidirectional      (ar_to_rule encoder)

Each figure is written twice:
    figures/<name>.svg          standalone, light palette, for papers/slides
    figures/<name>.body.svg     shapes only, for inlining into a themed page

Run:  python3 figures/make_arch_figures.py
"""

import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── palette (standalone files only; the inline copies use CSS tokens) ────────
LIGHT = {
    'ink': '#1B1F27', 'muted': '#616A78', 'rule': '#D7DCE4',
    'panel': '#FFFFFF', 'panel2': '#F2F4F8',
    'frozen_fill': '#EDEFF3', 'frozen_stroke': '#C4CBD6',
    'flow': '#2B5FD9', 'flow_fill': '#E7EDFC',
    'aux': '#A96A12', 'aux_fill': '#FBF1E0',
}

CSS = """
.pnl   {{ fill: {panel}; stroke: {rule}; stroke-width: 1; }}
.bx    {{ fill: {panel2}; stroke: {rule}; stroke-width: 1; }}
.bxf   {{ fill: {frozen_fill}; stroke: {frozen_stroke}; stroke-width: 1; }}
.bxa   {{ fill: {flow_fill}; stroke: {flow}; stroke-width: 1.2; }}
.bxx   {{ fill: {aux_fill}; stroke: {aux}; stroke-width: 1.2; }}
.bxd   {{ fill: none; stroke: {muted}; stroke-width: 1; stroke-dasharray: 4 4; }}
.slot  {{ fill: {panel}; stroke: {flow}; stroke-width: .8; }}
.tick  {{ fill: {frozen_fill}; stroke: {frozen_stroke}; stroke-width: .7; }}
.ttl   {{ font: 600 12px 'IBM Plex Sans','Helvetica Neue',sans-serif; fill: {ink};
          letter-spacing: .09em; }}
.sub   {{ font: 400 10.5px 'IBM Plex Sans','Helvetica Neue',sans-serif; fill: {muted}; }}
.lbl   {{ font: 500 11px 'IBM Plex Sans','Helvetica Neue',sans-serif; fill: {ink}; }}
.lblf  {{ font: 500 11px 'IBM Plex Sans','Helvetica Neue',sans-serif; fill: {flow}; }}
.lblx  {{ font: 600 10px 'IBM Plex Sans','Helvetica Neue',sans-serif; fill: {aux}; }}
.mono  {{ font: 400 9.5px 'IBM Plex Mono',ui-monospace,monospace; fill: {muted}; }}
.monof {{ font: 400 9.5px 'IBM Plex Mono',ui-monospace,monospace; fill: {flow}; }}
.tag   {{ font: 400 9px 'IBM Plex Sans','Helvetica Neue',sans-serif; fill: {muted};
          letter-spacing: .06em; }}
.arw   {{ stroke: {ink}; stroke-width: 1.4; fill: none; }}
.arwf  {{ stroke: {flow}; stroke-width: 1.5; fill: none; }}
.arwx  {{ stroke: {aux}; stroke-width: 1.2; fill: none; stroke-dasharray: 5 4; }}
.hair  {{ stroke: {flow}; stroke-width: .7; fill: none; opacity: .55; }}
"""


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def rect(x, y, w, h, cls, rx=4):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'


def txt(x, y, s, cls='lbl', anchor='start'):
    a = f' text-anchor="{anchor}"' if anchor != 'start' else ''
    return f'<text x="{x}" y="{y}"{a} class="{cls}">{esc(s)}</text>'


def line(x1, y1, x2, y2, cls='arw', marker=None):
    m = f' marker-end="url(#{marker})"' if marker else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"{m}/>'


def poly(pts, cls='arw', marker=None):
    m = f' marker-end="url(#{marker})"' if marker else ''
    p = ' '.join(f'{a},{b}' for a, b in pts)
    return f'<polyline points="{p}" class="{cls}"{m}/>'


def box2(x, y, w, h, main, sub, cls='bx'):
    """Two-line box: label above, mono detail below."""
    return (rect(x, y, w, h, cls)
            + txt(x + 11, y + 15, main, 'lbl')
            + txt(x + 11, y + 28, sub, 'mono'))


def box1(x, y, w, h, main, tag, cls='bx', main_cls='lbl'):
    """Single-line box with a right-aligned tag."""
    out = rect(x, y, w, h, cls) + txt(x + 11, y + h / 2 + 4, main, main_cls)
    if tag:
        out += txt(x + w - 11, y + h / 2 + 4, tag, 'tag', anchor='end')
    return out


def markers():
    d = []
    for name, col in (('ink', '{ink}'), ('flow', '{flow}'), ('aux', '{aux}')):
        for suffix, rot in (('', 0), ('-l', 180), ('-d', 90)):
            d.append(
                f'<marker id="ar-{name}{suffix}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{col}"/></marker>')
            break   # orient="auto-start-reverse" handles direction; one per colour
    return ''.join(d)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — the two architectures
# ═══════════════════════════════════════════════════════════════════════════

CHORDS = [('I', 'G'), ('IV', 'C'), ('V', 'D'), ('I', 'G')] * 2

BASE_X, BASE_W = 250, 200
RULE_X, RULE_W = 22, 190

LAYERS = [(196, '1'), (272, '2'), (366, '12')]   # unit top y, layer label


def base_column():
    """The frozen CP transformer, identical in both panels."""
    s = []
    s.append(box2(BASE_X, 92, BASE_W, 34, 'CP tokens', '(B, 64, S)  4 bars x 16 subbeats'))
    s.append(line(350, 126, 350, 142, 'arw', 'ar-ink'))
    s.append(box2(BASE_X, 144, BASE_W, 34, 'local encoder', '(B, 64, 768)'))
    s.append(line(350, 178, 350, 194, 'arw', 'ar-ink'))

    for i, (top, name) in enumerate(LAYERS):
        s.append(box1(BASE_X, top, BASE_W, 30, f'RoFormer layer {name}', 'frozen', 'bxf'))
        s.append(box1(BASE_X, top + 34, BASE_W, 30,
                      f'cross-attn adapter {name}', 'trainable', 'bxa', 'lblf'))
        if i < len(LAYERS) - 1 and top != 272:
            s.append(line(350, top + 64, 350, top + 74, 'arw', 'ar-ink'))
    s.append(line(350, 260, 350, 270, 'arw', 'ar-ink'))
    # continuation between layer 2 and layer 12
    s.append(f'<line x1="350" y1="336" x2="350" y2="344" class="arw" '
             f'stroke-dasharray="2 3"/>')
    s.append(f'<line x1="350" y1="360" x2="350" y2="366" class="arw" '
             f'stroke-dasharray="2 3" marker-end="url(#ar-ink)"/>')
    s.append(txt(350, 354, 'x 12 layers  ·  n_skip = 1', 'tag', anchor='middle'))

    s.append(line(350, 430, 350, 446, 'arw', 'ar-ink'))
    s.append(box2(BASE_X, 448, BASE_W, 34, 'local decoder', 'note tuples per subbeat'))
    s.append(line(350, 482, 350, 498, 'arw', 'ar-ink'))
    s.append(box2(BASE_X, 500, BASE_W, 34, 'generated CP tokens', '(B, 64, S)'))
    return ''.join(s)


def panel_a():
    s = [rect(0.5, 28, 470, 612, 'pnl', 8)]
    s.append(txt(20, 50, 'A  ·  EXPLICIT RULE INPUT', 'ttl'))
    s.append(txt(20, 68, '--paired_chord_seq — the progression is handed in', 'sub'))
    s.append(base_column())

    # rule source
    s.append(box2(RULE_X, 92, RULE_W, 34, 'chord_seq', '(B, 8)   7 0 2 7 7 0 2 7'))
    s.append(line(117, 126, 117, 142, 'arwf', 'ar-flow'))
    s.append(box2(RULE_X, 144, RULE_W, 34, 'ChordSeqRuleModel', 'analytic · 0 parameters'))
    s.append(line(117, 178, 117, 194, 'arwf', 'ar-flow'))

    s.append(rect(RULE_X, 196, RULE_W, 234, 'bxa'))
    s.append(txt(34, 212, 'rule_hidden  (B, 8, 16)', 'lblf'))
    s.append(txt(34, 224, '12-d triad chromagram · 4-d phase', 'mono'))
    for c, (deg, root) in enumerate(CHORDS):
        y = 230 + c * 24
        s.append(rect(34, y, 166, 20, 'slot', 3))
        s.append(txt(42, y + 14, str(c), 'mono'))
        s.append(txt(66, y + 14, deg, 'lbl'))
        s.append(txt(192, y + 14, root, 'monof', anchor='end'))

    for y in (245, 321, 415):
        s.append(line(212, y, 246, y, 'arwf', 'ar-flow'))
    s.append(txt(229, 316, 'K,V', 'monof', anchor='middle'))
    s.append(txt(22, 448, 'computed once · shared by all 12 adapters', 'tag'))

    s.append(rect(RULE_X, 520, RULE_W, 56, 'bx'))
    s.append(txt(34, 540, 'At inference', 'lbl'))
    s.append(txt(34, 556, 'chord_seq must still be given.', 'sub'))
    s.append(txt(34, 570, 'The model is told the answer.', 'sub'))
    return ''.join(s)


def panel_b():
    s = [rect(0.5, 28, 470, 612, 'pnl', 8)]
    s.append(txt(20, 50, 'B  ·  SELF-DERIVED RULE', 'ttl'))
    s.append(txt(20, 68, '--bidirectional — no rule input exists', 'sub'))
    s.append(base_column())

    s.append(rect(RULE_X, 92, RULE_W, 34, 'bxd'))
    s.append(txt(117, 114, 'no rule input', 'sub', anchor='middle'))

    for i, (top, name) in enumerate(LAYERS):
        s.append(rect(60, top + 4, 152, 52, 'bxa'))
        s.append(txt(70, top + 24, 'ar_to_rule', 'lblf'))
        s.append(txt(70, top + 38, 'Linear(768 → 16)', 'mono'))
        # h out of the frozen layer, left into the encoder
        s.append(line(248, top + 15, 216, top + 15, 'arwf', 'ar-flow'))
        # proxy back right into the adapter
        s.append(line(214, top + 49, 246, top + 49, 'arwf', 'ar-flow'))
        if i == 0:
            s.append(txt(231, top + 10, 'h', 'monof', anchor='middle'))
            s.append(txt(231, top + 44, 'K,V', 'monof', anchor='middle'))
        # dashed rail to the auxiliary loss
        s.append(line(60, top + 44, 41, top + 44, 'arwx'))

    s.append(line(41, 240, 41, 448, 'arwx', 'ar-aux'))
    s.append(txt(60, 444, 'one projection per layer · 12 separate Linears', 'tag'))

    s.append(rect(RULE_X, 452, RULE_W, 54, 'bxx'))
    s.append(txt(34, 470, 'auxiliary loss · train only', 'lblx'))
    s.append(txt(34, 486, 'BCE(proxy[:12], chromagram)', 'mono'))
    s.append(txt(34, 500, '+ CE(proxy[12:], phase)', 'mono'))

    s.append(rect(RULE_X, 520, RULE_W, 56, 'bx'))
    s.append(txt(34, 540, 'At inference', 'lbl'))
    s.append(txt(34, 556, 'nothing is supplied.', 'sub'))
    s.append(txt(34, 570, 'The key is inferred from the prompt.', 'sub'))
    return ''.join(s)


def fig1():
    return ('<g>' + panel_a() + '</g>'
            + '<g transform="translate(520,0)">' + panel_b() + '</g>')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — inside one adapter block
# ═══════════════════════════════════════════════════════════════════════════

def fig2():
    s = []
    s.append(box2(30, 52, 210, 40, 'music  h[t]', '(768)  from RoFormer layer'))
    s.append(box2(30, 176, 210, 40, 'rule_hidden[c]', '(16)  chromagram | phase'))

    s.append(line(240, 72, 286, 72, 'arw', 'ar-ink'))
    s.append(box1(288, 52, 150, 40, 'q_proj', '768→256', 'bxa', 'lblf'))
    s.append(poly([(240, 196), (264, 196), (264, 150), (286, 150)], 'arwf', 'ar-flow'))
    s.append(box1(288, 130, 150, 40, 'k_proj', '16→256', 'bxa', 'lblf'))
    s.append(poly([(240, 196), (264, 196), (264, 216), (286, 216)], 'arwf', 'ar-flow'))
    s.append(box1(288, 196, 150, 40, 'v_proj', '16→256', 'bxa', 'lblf'))

    s.append(rect(490, 92, 220, 104, 'bx'))
    s.append(txt(600, 122, 'softmax(Q Kᵀ / √32)', 'lbl', anchor='middle'))
    s.append(txt(600, 142, '8 heads × 32 dims', 'mono', anchor='middle'))
    s.append(txt(600, 172, 'attn @ V', 'lbl', anchor='middle'))
    s.append(line(438, 72, 486, 116, 'arw', 'ar-ink'))
    s.append(txt(452, 96, 'Q', 'mono'))
    s.append(line(438, 150, 486, 144, 'arwf', 'ar-flow'))
    s.append(txt(452, 138, 'K', 'monof'))
    s.append(line(438, 216, 486, 176, 'arwf', 'ar-flow'))
    s.append(txt(452, 200, 'V', 'monof'))

    s.append(line(710, 144, 756, 144, 'arw', 'ar-ink'))
    s.append(box1(758, 124, 150, 40, 'out_proj', '256→768', 'bxa', 'lblf'))
    s.append(line(833, 164, 833, 194, 'arw', 'ar-ink'))
    s.append(f'<circle cx="833" cy="212" r="18" class="bxa"/>')
    s.append(txt(833, 217, '× g', 'lblf', anchor='middle'))
    s.append(txt(858, 216, 'learned gate', 'tag'))
    s.append(line(833, 230, 833, 262, 'arw', 'ar-ink'))
    s.append(rect(610, 264, 350, 40, 'bx'))
    s.append(txt(626, 289, 'h[t]  ←  h[t]  +  g · Δ        residual, every layer', 'lbl'))

    s.append(rect(30, 250, 540, 58, 'bxd'))
    s.append(txt(46, 270, '--positional_qk', 'lblf'))
    s.append(txt(46, 286, 'Q and K are replaced by sinusoidal PE × 20 — alignment is', 'sub'))
    s.append(txt(46, 300, 'hardcoded, not learned. --qk_content_residual adds the', 'sub'))
    s.append(txt(46, 314, 'zero-init projections back on top.', 'sub'))
    return ''.join(s)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — query/key alignment
# ═══════════════════════════════════════════════════════════════════════════

X0, TW, STEP = 60, 12.0, 13.75


def tick_row(y, h=18):
    return ''.join(rect(X0 + i * STEP, y, TW, h, 'tick', 2) for i in range(64))


def tick_cx(i):
    return X0 + i * STEP + TW / 2


def fig3():
    s = []
    # ---- A: 64 queries → 8 keys
    s.append(txt(0, 20, 'A  ·  chord-seq mode — 64 subbeat queries, 8 chord keys '
                        '(key_stride = 8)', 'ttl'))
    for c, (deg, root) in enumerate(CHORDS):
        x = X0 + c * 8 * STEP
        s.append(rect(x, 34, 8 * STEP - 4, 20, 'slot', 3))
        s.append(txt(x + (8 * STEP - 4) / 2, 48, f'{deg}  {root}', 'lblf', anchor='middle'))
    s.append(tick_row(62))
    for c in range(8):
        kx = X0 + (c * 8 + 3.5) * STEP + TW / 2
        for i in range(c * 8, c * 8 + 8):
            s.append(line(tick_cx(i), 80, kx, 112, 'hair'))
        s.append(f'<path d="M {kx} 112 l 6 6 l -6 6 l -6 -6 z" class="bxa"/>')
    s.append(txt(X0, 146, 'key PE sits at the slot centre  c·8 + 3.5  — every subbeat '
                          'in a slot reads its own chord', 'mono'))

    # ---- B: 64 → 64
    s.append(txt(0, 186, 'B  ·  self-derived mode — 64 queries, 64 keys '
                         '(key_stride = 1, causal)', 'ttl'))
    s.append(tick_row(200))
    for i in range(64):
        s.append(line(tick_cx(i), 218, tick_cx(i), 240, 'hair'))
        s.append(f'<circle cx="{tick_cx(i)}" cy="246" r="4" class="bxa"/>')
    s.append(txt(X0, 274, 'proxy = ar_to_rule(h[t]) — one 16-d vector per subbeat, '
                          'no chord labels anywhere', 'mono'))
    return ''.join(s)


# ═══════════════════════════════════════════════════════════════════════════

FIGS = [
    ('adapter_architecture', 1000, 660, fig1,
     'Two ways the rule signal reaches the frozen transformer'),
    ('adapter_block', 1000, 330, fig2,
     'Inside one cross-attention adapter block'),
    ('adapter_alignment', 1000, 290, fig3,
     'How 64 music positions align to the rule keys'),
]


def write_all():
    for name, w, h, fn, desc in FIGS:
        body = markers().format(**LIGHT) + fn()
        # standalone, light palette
        standalone = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{esc(desc)}">'
            f'<style>{CSS.format(**LIGHT)}</style>'
            f'<rect width="{w}" height="{h}" fill="{LIGHT["panel"]}"/>'
            f'{body}</svg>')
        with open(os.path.join(OUT, f'{name}.svg'), 'w') as f:
            f.write(standalone)
        # body only, for inlining into a themed page
        with open(os.path.join(OUT, f'{name}.body.svg'), 'w') as f:
            f.write(body)
        print(f'  {name}.svg  ({w}x{h})')


if __name__ == '__main__':
    print('Writing figures to', OUT)
    write_all()
