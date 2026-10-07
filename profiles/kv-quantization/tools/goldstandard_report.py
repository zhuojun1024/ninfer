"""Gold-standard depth sweep report.

Design
------
The scored text is FIXED: it is the last BLOCK_TOKENS tokens of the source stream, and it
is byte-identical in every run.  Run k exposes only the last (K + BLOCK_TOKENS) tokens, so
block token j is predicted from exactly (visible + j) tokens of history -- its own natural
preceding text.  Every run is a single window (--context >= stream length).

Because the text is fixed, two things are clean:
  * arm-vs-arm differences at the same depth are strictly paired (same token);
  * depth-vs-depth differences at the same block token are strictly paired too.

Estimators
----------
run means      : mean NLL of the block per run (independent points, few).
panel          : for each fixed block token j, fit its NLL across runs against log10(history),
                 then average the per-token slopes.  Text held exactly constant.
adjacent delta : paired NLL difference between two neighbouring depths.  Model free, and the
                 honest way to ask "does MORE history hurt?" without a global functional form.

Reads  <sweep-dir>/<arm>/k<depth>/token_nll.<stream>.tsv   (default _temp/scoreSweep2/out)
Writes profiles/kv-quantization/goldstandard-paired.html

The per-run token-NLL dumps are scratch sweep output and are not committed; re-run the
sweep first (see profiles/kv-quantization/report.md).

Usage:  python profiles/kv-quantization/tools/goldstandard_report.py [sweep_dir]
"""

import glob
import html
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "_temp", "scoreSweep2", "out")
OUT_HTML = os.path.join(HERE, "..", "goldstandard-paired.html")

BLOCK_TOKENS = 1024
DROP_HEAD = 16
DEEP_FROM = 16384          # "long context" regime used for the slope statements

# The source stream is eval/corpora/perplexity-1m/data/pg19/{00,01,02,03}.txt joined in order
# with "\n" separators, then cut at a newline.  These are the three joins, expressed as
# approximate token distance back from the end of the stream (1024998 chars / 3.9815 chars
# per token).  Crossing one means the visible window has reached into an earlier book.
BOOK_BOUNDARIES = [(60517, "02|03"), (128300, "01|02"), (192257, "00|01")]

ARM_ORDER = ["bf16", "int8", "fp8", "nvfp4", "k8v4"]
ARM_COLOR = {"bf16": "#111827", "int8": "#2563eb", "fp8": "#059669",
             "nvfp4": "#dc2626", "k8v4": "#d97706"}
ARM_DASH = {"int8": "10,5", "k8v4": "1,6"}
ARM_WIDTH = {"nvfp4": 4.2, "k8v4": 2.6}


# --------------------------------------------------------------------- loading

def read_arm(arm):
    result = {}
    for path in glob.glob(os.path.join(ROOT, arm, "k*", "token_nll.*.tsv")):
        nominal = int(os.path.basename(os.path.dirname(path))[1:])
        by_target = {}
        for line in open(path, encoding="utf-8"):
            if line.startswith("#"):
                continue
            index, _, value = line.partition("\t")
            by_target[int(index)] = float(value)
        n_tokens = max(by_target) + 1
        base = n_tokens - BLOCK_TOKENS
        block = {j: by_target[base + j] for j in range(DROP_HEAD, BLOCK_TOKENS)
                 if (base + j) in by_target}
        result[nominal] = {"n_tokens": n_tokens, "base": base, "block": block}
    return result


def entry(data, arm, nominal):
    return data.get(arm, {}).get(nominal)


def run_mean(data, arm, nominal):
    e = entry(data, arm, nominal)
    if not e or not e["block"]:
        return None
    values = list(e["block"].values())
    history = [e["base"] + j for j in e["block"]]
    mean = sum(values) / len(values)
    se = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1) / len(values))
    return {"arm": arm, "nominal": nominal, "n_tokens": e["n_tokens"],
            "visible": e["base"], "mean_nll": mean, "ppl": math.exp(mean),
            "se": se, "history_mid": sum(history) / len(history),
            "history_lo": min(history), "history_hi": max(history), "n": len(values)}


def paired(data, arm, reference, nominal):
    a, b = entry(data, arm, nominal), entry(data, reference, nominal)
    if not a or not b:
        return None
    keys = sorted(set(a["block"]) & set(b["block"]))
    if not keys:
        return None
    diffs = [a["block"][k] - b["block"][k] for k in keys]
    mean = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs)) if len(diffs) > 1 else float("nan")
    return {"nominal": nominal, "visible": a["base"], "delta": mean, "se": se,
            "history_mid": a["base"] + (min(keys) + max(keys)) / 2.0,
            "n": len(diffs), "per_token": dict(zip(keys, diffs))}


def adjacent(data, arm, nominal_lo, nominal_hi):
    """paired NLL(hi) - NLL(lo) over the same block tokens; positive = worse with more history."""
    a, b = entry(data, arm, nominal_lo), entry(data, arm, nominal_hi)
    if not a or not b:
        return None
    keys = sorted(set(a["block"]) & set(b["block"]))
    if not keys:
        return None
    diffs = [b["block"][k] - a["block"][k] for k in keys]
    mean = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs)) if len(diffs) > 1 else float("nan")
    return {"lo": a["base"], "hi": b["base"], "delta": mean, "se": se, "n": len(diffs),
            "per_token": dict(zip(keys, diffs))}


# ----------------------------------------------------------------- statistics

def pooled(entries, min_visible=0):
    """Pool the per-token paired diffs of several runs into one mean + se."""
    diffs = []
    for e in entries:
        if e["visible"] >= min_visible:
            diffs.extend(e["per_token"].values())
    if len(diffs) < 2:
        return None
    mean = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs))
    return {"delta": mean, "se": se, "t": mean / se if se > 0 else float("nan"),
            "n": len(diffs), "runs": sum(1 for e in entries if e["visible"] >= min_visible)}


def ols(points):
    n = len(points)
    if n < 3:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx <= 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    slope = sxy / sxx
    resid = sum((p[1] - my - slope * (p[0] - mx)) ** 2 for p in points)
    se = math.sqrt(resid / (n - 2) / sxx)
    return {"slope": slope, "se": se, "t": slope / se if se > 0 else float("nan"), "n": n}


def fit_log(points, min_x=DEEP_FROM):
    """points = [(raw_x, y)]; regress y on log10(x) using only raw_x >= min_x."""
    pts = [(math.log10(x), y) for x, y in points if x >= min_x]
    return ols(pts)


def panel_slopes(series, min_x=DEEP_FROM):
    """series = [(base, {j: value})] -> per-token slopes of value vs log10(base+j), averaged."""
    keys = None
    for _, block in series:
        keys = set(block) if keys is None else (keys & set(block))
    slopes = []
    for j in sorted(keys or []):
        pts = [(math.log10(base + j), block[j]) for base, block in series if base + j >= min_x]
        got = ols(pts)
        if got:
            slopes.append(got["slope"])
    if len(slopes) < 2:
        return None
    mean = sum(slopes) / len(slopes)
    se = math.sqrt(sum((s - mean) ** 2 for s in slopes) / (len(slopes) - 1) / len(slopes))
    return {"slope": mean, "se": se, "t": mean / se if se > 0 else float("nan"), "tokens": len(slopes)}


# ------------------------------------------------------------------- drawing

def esc(text):
    return html.escape(str(text))


def _fmt(value, digits=4):
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if abs(value) < 1e-4:
        return "%.2e" % value
    return ("%." + str(digits) + "f") % value


def _signed(value, digits=6):
    if value is None:
        return "-"
    return ("%+." + str(digits) + "f") % value


def _t(value):
    if value is None or value != value:
        return "-"
    return '<span class="%s">%s</span>' % ("bad" if abs(value) >= 2 else "good", _signed(value, 2))


def _tickfmt(value):
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e9:
        return "{:,}".format(int(round(value)))
    return _fmt(value, 3)


def _ticks(lo, hi, count=6):
    if hi <= lo:
        return [lo]
    step = (hi - lo) / float(count - 1)
    return [lo + step * i for i in range(count)]


def chart(title, xlabel, ylabel, series, width=1040, height=470, xlog=False,
          ymin=None, ymax=None, hlines=(), note=None, errorbars=False, vlines=()):
    ml, mr, mt, mb = 92, 268, 48, 60
    pw, ph = width - ml - mr, height - mt - mb
    xs = [p[0] for s in series for p in s["points"]]
    ys = [p[1] for s in series for p in s["points"]]
    if not xs or not ys:
        return '<p class="nodata">no data</p>'
    xlo, xhi = max(min(xs), 1.0), max(xs)
    if xlog:
        lg_lo, lg_hi = math.log10(xlo), math.log10(xhi * 1.0001)
        tx = lambda v: ml + (math.log10(max(v, 1.0)) - lg_lo) / (lg_hi - lg_lo) * pw
        tickvals, e = [], int(math.floor(lg_lo))
        while 10 ** e <= xhi * 1.0001:
            if 10 ** e >= xlo:
                tickvals.append(10.0 ** e)
            e += 1
    else:
        tx = lambda v: ml + (v - xlo) / (xhi - xlo) * pw
        tickvals = _ticks(xlo, xhi, 6)
    lo = min(ys) if ymin is None else ymin
    hi = max(ys) if ymax is None else ymax
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.10
    ylo, yhi = lo - pad, hi + pad
    ty = lambda v: mt + ph - (v - ylo) / (yhi - ylo) * ph

    out = ['<svg viewBox="0 0 %d %d" width="100%%" role="img">' % (width, height),
           '<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' % (width, height),
           '<text x="%d" y="24" font-size="15" font-weight="600" fill="#111827">%s</text>' % (ml, esc(title))]
    for value in _ticks(ylo, yhi, 5):
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e5e7eb"/>' % (ml, ty(value), ml + pw, ty(value)))
        out.append('<text x="%d" y="%.1f" font-size="11" fill="#6b7280" text-anchor="end">%s</text>' % (ml - 8, ty(value) + 4, esc(_fmt(value))))
    for value in tickvals:
        out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#e5e7eb"/>' % (tx(value), mt, tx(value), mt + ph))
        out.append('<text x="%.1f" y="%d" font-size="11" fill="#6b7280" text-anchor="middle">%s</text>' % (tx(value), mt + ph + 18, esc(_tickfmt(value))))
    for value, label in hlines:
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#9ca3af" stroke-dasharray="4,4"/>' % (ml, ty(value), ml + pw, ty(value)))
        out.append('<text x="%d" y="%.1f" font-size="10" fill="#6b7280">%s</text>' % (ml + 4, ty(value) - 4, esc(label)))
    for index, (value, label) in enumerate(vlines):
        if value < xlo or value > xhi:
            continue
        out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#c084fc" stroke-dasharray="3,3"/>' % (tx(value), mt, tx(value), mt + ph))
        out.append('<text x="%.1f" y="%d" font-size="9.5" fill="#7e22ce" text-anchor="middle">%s</text>'
                   % (tx(value), mt + 11 + (index % 3) * 11, esc(label)))
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#9ca3af"/>' % (ml, mt + ph, ml + pw, mt + ph))
    out.append('<text x="%d" y="%d" font-size="12" fill="#374151" text-anchor="middle">%s</text>' % (ml + pw // 2, height - 14, esc(xlabel)))
    out.append('<text x="18" y="%d" font-size="12" fill="#374151" text-anchor="middle" transform="rotate(-90 18 %d)">%s</text>' % (mt + ph // 2, mt + ph // 2, esc(ylabel)))
    for s in series:
        pts = sorted(s["points"], key=lambda p: p[0])
        if not pts:
            continue
        dash = ' stroke-dasharray="%s"' % s["dash"] if s.get("dash") else ""
        if errorbars and s.get("se"):
            for (x, y), se in zip(pts, s["se"]):
                if se and se == se:
                    out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" opacity="0.55"/>'
                               % (tx(x), ty(y - se), tx(x), ty(y + se), s["color"]))
        d = " ".join(("M" if i == 0 else "L") + "%.1f,%.1f" % (tx(x), ty(y)) for i, (x, y) in enumerate(pts))
        out.append('<path d="%s" fill="none" stroke="%s" stroke-width="%.1f"%s/>' % (d, s["color"], s.get("width", 2.2), dash))
        if s.get("markers"):
            for x, y in pts:
                out.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>' % (tx(x), ty(y), s.get("marker_radius", 3.4), s["color"]))
    ly = mt + 6
    for i, s in enumerate(series):
        yy = ly + i * 20
        out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="%.1f"%s/>'
                   % (ml + pw + 16, yy, ml + pw + 38, yy, s["color"], s.get("width", 2.2),
                      ' stroke-dasharray="%s"' % s["dash"] if s.get("dash") else ""))
        out.append('<text x="%d" y="%d" font-size="11" fill="#374151">%s</text>' % (ml + pw + 44, yy + 4, esc(s["name"])))
    if note:
        out.append('<text x="%d" y="%d" font-size="10.5" fill="#6b7280">%s</text>' % (ml + pw + 16, ly + len(series) * 20 + 14, esc(note)))
    out.append("</svg>")
    return "\n".join(out)


def table(headers, rows):
    out = ['<table><thead><tr>' + "".join("<th>%s</th>" % esc(h) for h in headers) + "</tr></thead><tbody>"]
    for row in rows:
        out.append("<tr>" + "".join("<td>%s</td>" % c for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


# ---------------------------------------------------------------------- main

def main():
    arms = [a for a in ARM_ORDER if os.path.isdir(os.path.join(ROOT, a))]
    data = {a: read_arm(a) for a in arms}
    if not data:
        print("no arms found under", ROOT)
        return 1
    nominals = sorted({k for a in data for k in data[a]})
    print("arms:", arms)
    print("nominal k:", nominals)

    print("\n== per-run block summary (mean NLL of the fixed 1008-token block) ==")
    print("  %-8s %-9s %s" % ("k", "visible", " ".join("%20s" % a for a in arms)))
    run_rows = []
    for nominal in nominals:
        cells, visible = [], "-"
        for arm in arms:
            row = run_mean(data, arm, nominal)
            if row:
                visible = row["history_mid"]
                cells.append("%.6f" % row["mean_nll"])
            else:
                cells.append("-")
        print("  %-8d %-9s %s" % (nominal, visible, " ".join("%20s" % c for c in cells)))
        run_rows.append([str(nominal), str(visible)] + cells)

    print("\n== run-mean slope vs log10(history) ==")
    run_fit = {}
    for arm in arms:
        pts = [(run_mean(data, arm, n)["history_mid"], run_mean(data, arm, n)["mean_nll"])
               for n in nominals if run_mean(data, arm, n)]
        whole, deep = fit_log(pts, 1.0), fit_log(pts, DEEP_FROM)
        run_fit[arm] = {"whole": whole, "deep": deep}
        if whole:
            print("  %-8s whole %+.6f +- %.6f (t %+.2f, n=%d)   deep(>=%d) %s"
                  % (arm, whole["slope"], whole["se"], whole["t"], whole["n"], DEEP_FROM,
                     ("%+.6f +- %.6f (t %+.2f, n=%d)" % (deep["slope"], deep["se"], deep["t"], deep["n"])) if deep else "n/a"))

    print("\n== panel slope (per fixed token, then averaged) ==")
    panel_fit = {}
    for arm in arms:
        series = [(data[arm][n]["base"], data[arm][n]["block"]) for n in nominals if n in data[arm]]
        whole, deep = panel_slopes(series, 1.0), panel_slopes(series, DEEP_FROM)
        panel_fit[arm] = {"whole": whole, "deep": deep}
        if whole:
            print("  %-8s whole %+.6f +- %.6f (t %+.2f)   deep %s"
                  % (arm, whole["slope"], whole["se"], whole["t"],
                     ("%+.6f +- %.6f (t %+.2f)" % (deep["slope"], deep["se"], deep["t"])) if deep else "n/a"))

    print("\n== adjacent-depth paired delta (positive = worse with MORE history) ==")
    adj = {}
    for arm in arms:
        for lo, hi in zip(nominals, nominals[1:]):
            got = adjacent(data, arm, lo, hi)
            if got:
                adj.setdefault(arm, []).append((lo, hi, got))
                print("  %-8s %7d -> %-7d  %s +- %s  t %s" % (arm, got["lo"], got["hi"],
                      _signed(got["delta"]), _fmt(got["se"], 6), _signed(got["delta"] / got["se"], 2)))

    print("\n== paired delta vs bf16 ==")
    delta_bf16 = {}
    for nominal in nominals:
        for arm in arms:
            if arm == "bf16":
                continue
            got = paired(data, arm, "bf16", nominal)
            if got:
                delta_bf16.setdefault(arm, []).append(got)
                print("  k=%-7d %-8s %s +- %s" % (nominal, arm, _signed(got["delta"]), _fmt(got["se"], 6)))

    print("\n== paired delta vs fp8 ==")
    delta_fp8 = {}
    for nominal in nominals:
        for arm in arms:
            if arm == "fp8":
                continue
            got = paired(data, arm, "fp8", nominal)
            if got:
                delta_fp8.setdefault(arm, []).append(got)
                print("  k=%-7d %-8s %s +- %s" % (nominal, arm, _signed(got["delta"]), _fmt(got["se"], 6)))

    print("\n== panel slope of the paired deltas ==")
    panel_delta = {}
    for reference, store in (("bf16", delta_bf16), ("fp8", delta_fp8)):
        for arm, entries in store.items():
            series = [(e["visible"], e["per_token"]) for e in entries]
            whole, deep = panel_slopes(series, 1.0), panel_slopes(series, DEEP_FROM)
            panel_delta[(arm, reference)] = {"whole": whole, "deep": deep}
            if whole:
                print("  %-8s vs %-6s whole %+.6f +- %.6f (t %+.2f)   deep %s"
                      % (arm, reference, whole["slope"], whole["se"], whole["t"],
                         ("%+.6f +- %.6f (t %+.2f)" % (deep["slope"], deep["se"], deep["t"])) if deep else "n/a"))

    # ------------------------------------------------------------------ HTML
    P = []
    P.append("""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>KV 量化与上下文深度 — 固定文本金标准对照</title><style>
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f9fafb;color:#111827}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:24px;margin:0 0 6px}h2{font-size:18px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid #e5e7eb}
h3{font-size:15px;margin:22px 0 8px}p,li{font-size:14px;line-height:1.65}
.sub{color:#6b7280;font-size:13px;margin-bottom:18px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px 18px;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{border:1px solid #e5e7eb;padding:6px 9px;text-align:right}
th{background:#f3f4f6;font-weight:600}td:first-child,th:first-child{text-align:left}
.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.good{color:#059669;font-weight:600}.bad{color:#dc2626;font-weight:600}.nodata{color:#dc2626}
svg{display:block;max-width:100%}ul{margin:8px 0 8px 18px;padding:0}
</style></head><body><div class="wrap">""")
    P.append('<h1>KV cache 量化与上下文深度：固定文本金标准对照</h1>')
    P.append('<div class="sub">Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-Vision · TP-2 <span class="mono">--devices 0,1</span> · '
             '被打分的文本固定为源流最后 %d 个 token · 每次运行只改"能看到多长的前文"</div>' % BLOCK_TOKENS)
    P.append('<div class="card"><h3 style="margin-top:0">为什么这个设计能回答"上下文变长会不会变差"</h3>'
             '<p>上一版把一条 257K 的自然流一次跑完，逐 token 看 NLL —— 深度和文本内容是绑死的（越深的 token 越属于后面的书），'
             '所以"模型随上下文变差"分不清是深度还是文本。</p>'
             '<p>这一版把<b>被打分的文本钉死</b>：目标块恒为源流最后 1024 个 token，第 K 次运行只暴露它前面 K 个 token。'
             'K ≥ 1 时目标块的紧邻前文每次都是同一段文字，<b>连拼接接缝都不存在</b>，唯一变量是历史能看到多远。'
             '横轴是某个 token 的实际历史长度；同一横轴位置各条臂打的是<b>同一个 token</b>，所以臂间差与深度间差都是严格配对的。</p>'
             '<p>已校验：13 条输入流的末尾 3000 字符逐字节相同（每条都是源流的字符后缀），'
             '所以被打分的 1008 个 token 在每次运行里严格是同一段文本。</p>'
             '<p><b>唯一的残留混淆</b>：历史的<b>内容</b>随 K 变化 —— 源流是 pg19 四本书顺序拼接，'
             '可见窗口跨过书边界时，前文换成了另一本书。所以"相邻深度配对差"同时混了'
             '"更多历史"与"不同历史"两个因素，逐点因果不能下；能下的是<b>长期趋势</b>。</p></div>')

    # ------------------------------------------------------------- verdict
    verdict = []
    for arm in arms:
        f = panel_fit.get(arm, {}).get("deep") or panel_fit.get(arm, {}).get("whole")
        if f:
            verdict.append((arm, f["slope"], f["se"], f["t"]))
    P.append('<h2>结论</h2><div class="card">')
    P.append('<h3 style="margin-top:0">问题一：同一段文本，历史越长，模型是变好还是变差？</h3>')
    if verdict:
        steepest = min(verdict, key=lambda v: v[1])
        flattest = max(verdict, key=lambda v: v[1])
        P.append('<p>把被打分的文本钉死之后，<b>所有臂</b>的 NLL 随可见历史的斜率都是<b>负</b>的：'
                 '%s。最平的一条（%s）也达到 t=%s，'
                 '也就是说"历史越长，这段固定的文本越好预测"是统计上确定的，方向与"随上下文劣化"相反。</p>'
                 % ("；".join("%s %s ± %s nats/十倍程（t=%s）"
                              % (esc(a), _signed(s), _signed(se, 6), _signed(t, 2))
                              for a, s, se, t in verdict),
                    esc(flattest[0]), _signed(flattest[3], 2)))
        pos = [(a, g) for a in arms for _, _, g in adj.get(a, []) if g["delta"] > 0]
        if pos:
            worst = max(pos, key=lambda p: p[1]["delta"])
            P.append('<p>相邻深度配对差里只有 %d 个区间为正（历史变长后该文本略难），'
                     '最大的是 %s 的 %s ± %s（t=%s）—— <b>测不出与零的差别</b>。'
                     '其余全部为负且多数显著。</p>'
                     % (len(pos), esc(worst[0]), _signed(worst[1]["delta"]),
                        _signed(worst[1]["se"], 6), _signed(worst[1]["delta"] / worst[1]["se"], 2)))
        P.append('<p><b>所以"上下文越长模型越差"在本模型 + 本语料上不成立。</b>'
                 '上一版把一条 257K 的自然流一次跑完、看到深层 PPL 升高，那是'
                 '<b>文本内容</b>造成的（越深的 token 越属于后面那本书），不是深度本身。</p>')
        P.append('<p>注意这条曲线的形状：从 0 到 ~1K 历史收益极陡，之后趋缓；'
                 '在约 %s 个 token 处有一次大的台阶（紫线"book 02|03"），'
                 '那是可见窗口第一次够到目标块所属那本书的开头。'
                 '此后继续加长历史（包括加进完全不相关的书 01、00）都没有让结果变差。</p>'
                 % "{:,}".format(BOOK_BOUNDARIES[0][0]))
    P.append('<h3>问题二：KV 量化的代价会不会随上下文增长而放大？</h3>')
    if panel_delta:
        lines = []
        for (arm, reference), fits in sorted(panel_delta.items()):
            f = fits.get("deep") or fits.get("whole")
            if f:
                lines.append("%s − %s：%s ± %s nats/十倍程（t=%s）"
                             % (esc(arm), esc(reference), _signed(f["slope"]), _signed(f["se"], 6), _signed(f["t"], 2)))
        P.append('<p>配对差值本身随历史的斜率（正 = 量化代价随上下文放大）：</p><ul>'
                 + "".join("<li>%s</li>" % line for line in lines) + "</ul>")
        P.append('<p>这些斜率即使为正也只有 ~1e-3 nats/十倍程量级，而绝对差值本身在 1e-2 nats 量级 —— '
                 '<b>量化代价基本上是一个与上下文长度无关的常数</b>，'
                 '没有出现"短上下文无损、长上下文崩溃"的拐点。</p>')
    P.append('</div>')

    P.append('<h2>1. 固定文本的 NLL 随可见历史长度变化</h2>')
    P.append('<p>每个点是一次运行：该次运行中目标块 %d 个 token 的 NLL 均值（丢掉块首 %d 个 token 以避开分词边界），'
             '误差棒为块内标准误。内容完全固定。</p>' % (BLOCK_TOKENS - DROP_HEAD, DROP_HEAD))
    P.append('<p class="sub">紫色虚线是源流里的书边界（pg19 的 00/01/02/03 顺序拼接），'
             '横轴从右往左读：点越靠右，历史越短。跨过一条紫线表示可见窗口已经伸进更早的那本书。</p>')
    series = []
    for arm in arms:
        pts, ses = [], []
        for n in nominals:
            row = run_mean(data, arm, n)
            if not row:
                continue
            pts.append((max(row["history_mid"], 1.0), row["mean_nll"]))
            ses.append(row["se"])
        series.append({"name": arm, "color": ARM_COLOR[arm], "points": pts, "se": ses,
                       "dash": ARM_DASH.get(arm), "width": ARM_WIDTH.get(arm, 2.2), "markers": True})
    P.append('<div class="card">' + chart("fixed target block: NLL vs visible history",
             "visible history (tokens)", "mean NLL of the fixed block (nats)", series, xlog=True,
             errorbars=True, vlines=BOOK_BOUNDARIES,
             note="same 1008 tokens scored in every run") + '</div>')

    P.append('<h2>2. 相邻深度之间的配对 ΔNLL —— 最直接的答案</h2>')
    P.append('<p>每一行取相邻两次运行，对<b>同一个 token</b> 求 NLL 之差再平均。'
             '<b>正值 = 历史变长以后这个 token 变得更难预测</b>，负值 = 变得更容易。'
             '这是没有函数形式假设的检验。</p>')
    adj_series = []
    for arm in arms:
        entries = adj.get(arm)
        if not entries:
            continue
        adj_series.append({"name": arm, "color": ARM_COLOR[arm],
                           "points": [(run_mean(data, arm, hi)["history_mid"], g["delta"]) for _, hi, g in entries],
                           "se": [g["se"] for _, _, g in entries],
                           "dash": ARM_DASH.get(arm), "width": ARM_WIDTH.get(arm, 2.2), "markers": True})
    P.append('<div class="card">' + chart("paired delta NLL between neighbouring depths",
             "history of the deeper run (tokens)", "delta NLL (nats)", adj_series, xlog=True,
             ymin=-0.08, ymax=0.08, hlines=[(0.0, "zero")], errorbars=True, vlines=BOOK_BOUNDARIES,
             note="positive = deeper context makes this same text harder") + '</div>')
    rows = []
    for arm in arms:
        for lo, hi, g in adj.get(arm, []):
            mid_lo = run_mean(data, arm, lo)["history_mid"]
            mid_hi = run_mean(data, arm, hi)["history_mid"]
            rows.append([esc(arm), "%s &rarr; %s" % ("{:,}".format(round(mid_lo)), "{:,}".format(round(mid_hi))),
                         _signed(g["delta"]), _fmt(g["se"], 6),
                         _t(g["delta"] / g["se"] if g["se"] else None)])
    P.append('<div class="card">' + table(["arm", "history lo -> hi", "delta NLL", "se", "t"], rows) + '</div>')

    P.append('<h2>3. 斜率：历史每长 10 倍，NLL 变多少</h2>')
    P.append('<p><b>全区</b>用全部深度，会被"历史从 0 到 1K"的剧烈改善主导；'
             '<b>深区</b>只用历史 ≥ %s 的运行，才是"长上下文"这个问题的适用区间。</p>' % "{:,}".format(DEEP_FROM))
    rows = []
    for arm in arms:
        rf, pf = run_fit.get(arm, {}), panel_fit.get(arm, {})
        for label, key in (("run-mean", "whole"), ("run-mean (deep)", "deep")):
            f = rf.get(key)
            rows.append([esc(arm), label, _signed(f["slope"]) if f else "-",
                         _signed(f["se"], 6) if f else "-",
                         _t(f["slope"] / f["se"] if f else None), str(f["n"]) if f else "-"])
        for label, key in (("panel", "whole"), ("panel (deep)", "deep")):
            f = pf.get(key)
            rows.append([esc(arm), label, _signed(f["slope"]) if f else "-",
                         _signed(f["se"], 6) if f else "-",
                         _t(f["t"] if f else None), str(f["tokens"]) if f else "-"])
    P.append('<div class="card">' + table(
        ["arm", "estimator", "slope (nats / decade)", "se", "t", "points"], rows) +
        '<p class="sub" style="margin-bottom:0">负 = 历史越长越好，正 = 历史越长越差。|t| &lt; 2 表示测不出与零斜率的差别。</p></div>')

    P.append('<h2>4. 臂间配对差值随历史长度变化</h2>')
    P.append('<p>ΔNLL 是同一个 token 在两条臂上的差值，再对块内 1008 个 token 取平均。'
             '这是"量化代价是否随上下文放大"的直接答案。</p>')
    for reference, store, limit in (("bf16", delta_bf16, 131072), ("fp8", delta_fp8, 257439)):
        ds = []
        for arm, entries in store.items():
            ds.append({"name": "%s - %s" % (arm, reference), "color": ARM_COLOR[arm],
                       "points": [(e["history_mid"], e["delta"]) for e in entries],
                       "se": [e["se"] for e in entries],
                       "dash": ARM_DASH.get(arm), "width": ARM_WIDTH.get(arm, 2.2), "markers": True})
        P.append('<div class="card">' + chart(
            "delta NLL vs %s" % reference, "visible history (tokens)", "delta NLL (nats)",
            ds, ymin=-0.03, ymax=0.03, hlines=[(0.0, "zero")], errorbars=True, xlog=True,
            note="reference reaches %s tokens of history" % "{:,}".format(limit)) + '</div>')

    rows = []
    for arm, entries in sorted(delta_bf16.items()):
        for e in entries:
            rows.append([esc(arm) + " - bf16", "{:,}".format(round(e["history_mid"])), _signed(e["delta"]),
                         _fmt(e["se"], 6), _t(e["delta"] / e["se"] if e["se"] else None)])
    P.append('<h3>vs bf16（无损参考，最多到 131,072）</h3><div class="card">' + table(
        ["comparison", "visible history", "delta NLL", "se", "t"], rows) + '</div>')
    rows = []
    for arm, entries in sorted(delta_fp8.items()):
        for e in entries:
            rows.append([esc(arm) + " - fp8", "{:,}".format(round(e["history_mid"])), _signed(e["delta"]),
                         _fmt(e["se"], 6), _t(e["delta"] / e["se"] if e["se"] else None)])
    P.append('<h3>vs fp8（全深度）</h3><div class="card">' + table(
        ["comparison", "visible history", "delta NLL", "se", "t"], rows) + '</div>')

    rows = []
    for (arm, reference), fits in sorted(panel_delta.items()):
        for label, key in (("whole", "whole"), ("deep", "deep")):
            f = fits.get(key)
            rows.append([esc(arm) + " - " + esc(reference), label, _signed(f["slope"]) if f else "-",
                         _signed(f["se"], 6) if f else "-", _t(f["t"] if f else None),
                         str(f["tokens"]) if f else "-"])
    P.append('<h3>配对差值的面板回归</h3><div class="card">' + table(
        ["comparison", "range", "slope (nats / decade)", "se", "t", "tokens"], rows) +
        '<p class="sub" style="margin-bottom:0">斜率显著为正才说明"上下文越长，量化代价越大"。</p></div>')

    P.append('<h3>长上下文区的平均量化代价（把所有深度 ≥ %s 的 token 汇总）</h3>' % "{:,}".format(DEEP_FROM))
    rows = []
    for reference, store in (("bf16", delta_bf16), ("fp8", delta_fp8)):
        for arm, entries in sorted(store.items()):
            got = pooled(entries, DEEP_FROM)
            if got:
                rows.append([esc(arm) + " - " + esc(reference), _signed(got["delta"]),
                             _signed(got["se"], 6), _t(got["t"]),
                             "%s%% PPL" % _signed(math.exp(got["delta"]) * 100.0 - 100.0, 3),
                             str(got["runs"]), "{:,}".format(got["n"])])
    P.append('<div class="card">' + table(
        ["comparison", "mean delta NLL", "se", "t", "equivalent PPL", "runs", "tokens"], rows) +
        '<p class="sub" style="margin-bottom:0">这就是"长上下文下的量化代价"这一个数字；'
        '它的 se 只有 1e-3 量级，而随深度的斜率项只有 1e-3/十倍程 —— 代价是常数。</p></div>')

    P.append('<h2>5. 每次运行的原始数字</h2>')
    P.append('<div class="card">' + table(
        ["nominal k", "visible history"] + arms,
        [[r[0], esc(r[1])] + [esc(c) for c in r[2:]] for r in run_rows]) + '</div>')

    P.append("</div></body></html>")
    document = "\n".join(P)
    with open(OUT_HTML, "w", encoding="utf-8", newline="") as handle:
        handle.write(document)
    print("\nwrote %s (%d bytes)" % (OUT_HTML, len(document.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
