"""Final HTML report for the long-context retrieval probes.

Two probes, both using the prefix-chain trick (each (arm, subset) is one chain
of strictly-extending requests, so the whole chain costs about one full prefill
of the longest length):

  soft  8 needles, needles at the very front, english + chinese
        -> profiles/kv-quantization/data/niah_chain_<arm>.json
  hard  32 needles + a 39-char random access token to copy verbatim, english
        -> profiles/kv-quantization/data/niah_chain_hard_<arm>.json

Usage:
  python profiles/kv-quantization/tools/niah_probe_report.py
"""

from __future__ import annotations

import html
import json
import math
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "niah-probe.html"

ARMS = ("int8", "nvfp4")
ARM_COLOR = {"int8": "#2f6fd0", "nvfp4": "#d0402f"}
ARM_LABEL = {"int8": "int8-g64", "nvfp4": "nvfp4-g16"}
SUBSETS = (
    ("english", "英文 haystack (PaulGraham_Essays)"),
    ("chinese", "中文 haystack (Journey_to_the_West)"),
)
ACCESS = "XK4Q9M2B7T1V8W3N6R5P2H9D4C8L1G7F"
ACCESS_NORM = re.sub(r"[^A-Za-z0-9]", "", ACCESS).upper()


def esc(value) -> str:
    return html.escape(str(value))


def norm_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def access_score(prediction: str) -> tuple[int, int, str]:
    """Best character-level match between the answer's token line and the target."""
    best = ""
    for line in prediction.splitlines():
        if "=" in line:
            continue
        cand = norm_token(line)
        if len(cand) > len(best):
            best = cand
    n = len(ACCESS_NORM)
    if not best:
        return 0, n, ""
    if len(best) <= n:
        hits = sum(1 for a, b in zip(best, ACCESS_NORM) if a == b)
        return hits, n, best
    best_hits, best_window = -1, best[:n]
    for i in range(len(best) - n + 1):
        window = best[i:i + n]
        hits = sum(1 for a, b in zip(window, ACCESS_NORM) if a == b)
        if hits > best_hits:
            best_hits, best_window = hits, window
    return best_hits, n, best_window


def is_complete(rec: dict) -> bool:
    """The answer ran to the end instead of being cut by the context budget."""
    if rec.get("hits", -1) < 0:
        return False
    if rec.get("truncated") is True or rec.get("finish_reason") == "length":
        return False
    stripped = rec["prediction"].strip()
    if not stripped:
        return False
    last = stripped.splitlines()[-1]
    return bool(re.search(r"=\s*\d{5}\s*$", last)) or len(norm_token(last)) >= len(ACCESS_NORM)


def load(prefix: str) -> dict:
    data: dict = {}
    for arm in ARMS:
        path = DATA / f"{prefix}{arm}.json"
        if not path.exists():
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            if rec.get("depth_mode") != "front" or rec.get("hits", -1) < 0:
                continue
            data.setdefault((rec["subset"], arm), []).append(rec)
    for key in data:
        data[key].sort(key=lambda r: r["prompt_tokens"])
    return data


def chart(series, title, xlabel, ylabel, width=1000, height=340, ymin=None, ymax=None,
          yticks=None, note=None, fmt="{:g}", hlines=()):
    ml, mr, mt, mb = 82, 214, 40, 52
    pw, ph = width - ml - mr, height - mt - mb
    xs = [p[0] for s in series for p in s["points"]]
    ys = [p[1] for s in series for p in s["points"]]
    if not xs:
        return ""
    xlo, xhi = math.log10(min(xs)), math.log10(max(xs))
    if xhi - xlo < 1e-9:
        xhi = xlo + 1
    ylo = min(ys) if ymin is None else ymin
    yhi = max(ys) if ymax is None else ymax
    if yhi - ylo < 1e-9:
        yhi = ylo + 1
    pad = (yhi - ylo) * 0.08
    ylo = (ylo - pad) if ymin is None else ylo
    yhi = yhi + pad

    def px(x):
        return ml + (math.log10(x) - xlo) / (xhi - xlo) * pw

    def py(y):
        return mt + ph - (y - ylo) / (yhi - ylo) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="Segoe UI,Helvetica,Arial,sans-serif">',
           f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{ml}" y="24" font-size="15" font-weight="600" fill="#1a1a1a">{esc(title)}</text>']
    for tv in (yticks or [ylo + (yhi - ylo) * i / 5 for i in range(6)]):
        y = py(tv)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#e8e8e8" stroke-width="1"/>')
        out.append(f'<text x="{ml-10}" y="{y+4:.1f}" font-size="11" fill="#666" text-anchor="end">{esc(fmt.format(tv))}</text>')
    for hv in hlines:
        y = py(hv)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#bbb" '
                   f'stroke-width="1" stroke-dasharray="4,4"/>')
    xv_ticks = []
    for decade in range(int(math.floor(xlo)), int(math.ceil(xhi)) + 1):
        for mult in (1, 2, 5):
            v = mult * (10 ** decade)
            if xlo - 1e-9 <= math.log10(v) <= xhi + 1e-9:
                xv_ticks.append(v)
    for xv in sorted(set(xv_ticks)):
        x = px(xv)
        out.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" stroke="#f0f0f0" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{mt+ph+20}" font-size="11" fill="#666" text-anchor="middle">{int(xv):,}</text>')
    out.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#999" stroke-width="1.2"/>')
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#999" stroke-width="1.2"/>')
    out.append(f'<text x="{ml+pw/2:.0f}" y="{height-12}" font-size="12" fill="#444" text-anchor="middle">{esc(xlabel)}</text>')
    out.append(f'<text x="18" y="{mt+ph/2:.0f}" font-size="12" fill="#444" text-anchor="middle" '
               f'transform="rotate(-90 18 {mt+ph/2:.0f})">{esc(ylabel)}</text>')
    for s in series:
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in s["points"])
        out.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" stroke-width="2.4"/>')
        for x, y in s["points"]:
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.4" fill="{s["color"]}"/>')
    ly = mt + 6
    for s in series:
        out.append(f'<line x1="{ml+pw+14}" y1="{ly}" x2="{ml+pw+44}" y2="{ly}" stroke="{s["color"]}" stroke-width="2.4"/>')
        out.append(f'<text x="{ml+pw+50}" y="{ly+4}" font-size="12" fill="#333">{esc(s["name"])}</text>')
        ly += 20
    if note:
        for chunk in note.split("\n"):
            out.append(f'<text x="{ml+pw+14}" y="{ly+8}" font-size="10.5" fill="#888">{esc(chunk)}</text>')
            ly += 14
    out.append("</svg>")
    return "".join(out)


def table(headers, rows):
    out = ['<table><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead><tbody>"]
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def soft_section(data: dict) -> tuple[str, list]:
    body, summary = [], []
    for subset, label in SUBSETS:
        series_acc, series_lat, rows = [], [], []
        by_len: dict = {}
        for arm in ARMS:
            recs = data.get((subset, arm), [])
            if not recs:
                continue
            series_acc.append({"name": ARM_LABEL[arm], "color": ARM_COLOR[arm],
                               "points": [(r["prompt_tokens"], r["hits"]) for r in recs]})
            series_lat.append({"name": ARM_LABEL[arm], "color": ARM_COLOR[arm],
                               "points": [(r["prompt_tokens"], r["latency_s"]) for r in recs]})
            for r in recs:
                by_len.setdefault(r["prompt_tokens"], {})[arm] = r
            summary.append((label, ARM_LABEL[arm],
                            sum(r["hits"] for r in recs), sum(r["total"] for r in recs)))
        for tokens in sorted(by_len):
            entry = by_len[tokens]
            cells = [f'<span class="num">{tokens:,}</span>']
            for arm in ARMS:
                r = entry.get(arm)
                cells.append(f'<span class="num">{r["hits"]}/{r["total"]}</span>' if r else "&ndash;")
            for arm in ARMS:
                r = entry.get(arm)
                cells.append(f'<span class="num">{r["latency_s"]:.0f}s</span>' if r else "&ndash;")
            cells.append('<span class="num">0</span>' if len(entry) == 2 else "")
            rows.append(cells)
        body.append(f"<h2>{esc(label)}</h2>")
        body.append('<div class="card">' + chart(
            series_acc, f"检索命中数 vs 可见上下文长度 — {label}", "prompt tokens (log)",
            "命中的 needle 数 (共 8)", ymin=0, ymax=8,
            yticks=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            note="needle 位于文档最前，提问在最后") + "</div>")
        body.append('<div class="card">' + chart(
            series_lat, f"单次请求耗时 — {label}", "prompt tokens (log)", "latency (s)",
            ymin=0, fmt="{:,.0f}", note="上升 = 前缀复用生效（只付增量）") + "</div>")
        body.append(table(["prompt tokens", "int8-g64", "nvfp4-g16", "int8 耗时",
                           "nvfp4 耗时", "nvfp4 &minus; int8"], rows))
    return "".join(body), summary


def hard_section(data: dict) -> tuple[str, list, list]:
    body, summary, bad = [], [], []
    subset = "english"
    series_acc, series_tok, series_lat, rows = [], [], [], []
    by_len: dict = {}
    for arm in ARMS:
        recs = [r for r in data.get((subset, arm), []) if is_complete(r)]
        cut = [r for r in data.get((subset, arm), []) if not is_complete(r)]
        bad.extend((arm, r) for r in cut)
        if not recs:
            continue
        series_acc.append({"name": ARM_LABEL[arm], "color": ARM_COLOR[arm],
                           "points": [(r["prompt_tokens"], r["hits"]) for r in recs]})
        series_lat.append({"name": ARM_LABEL[arm], "color": ARM_COLOR[arm],
                           "points": [(r["prompt_tokens"], r["latency_s"]) for r in recs]})
        tok_points = []
        for r in recs:
            hits, total, _ = access_score(r["prediction"])
            tok_points.append((r["prompt_tokens"], 100.0 * hits / total))
        series_tok.append({"name": ARM_LABEL[arm], "color": ARM_COLOR[arm], "points": tok_points})
        for r in recs:
            by_len.setdefault(r["prompt_tokens"], {})[arm] = r
        summary.append((ARM_LABEL[arm], sum(r["hits"] for r in recs), sum(r["total"] for r in recs),
                        sum(access_score(r["prediction"])[0] for r in recs),
                        len(recs) * len(ACCESS_NORM)))

    for tokens in sorted(by_len):
        entry = by_len[tokens]
        cells = [f'<span class="num">{tokens:,}</span>']
        for arm in ARMS:
            r = entry.get(arm)
            cells.append(f'<span class="num">{r["hits"]}/{r["total"]}</span>' if r else "&ndash;")
        for arm in ARMS:
            r = entry.get(arm)
            if r:
                hits, total, _ = access_score(r["prediction"])
                cls = "pos" if hits == total else "neg"
                cells.append(f'<span class="num {cls}">{hits}/{total}</span>')
            else:
                cells.append("&ndash;")
        for arm in ARMS:
            r = entry.get(arm)
            cells.append(f'<span class="num">{r["latency_s"]:.0f}s</span>' if r else "&ndash;")
        rows.append(cells)

    body.append(f"<h2>难点探针：32 条 needle + 39 字符随机 token（英文 haystack）</h2>")
    body.append('<div class="card">' + chart(
        series_acc, "32 条 needle 的命中数 vs 可见上下文长度", "prompt tokens (log)",
        "命中的 needle 数 (共 32)", ymin=0, ymax=32,
        yticks=[0, 8, 16, 24, 32], note="needle 位于文档最前") + "</div>")
    body.append('<div class="card">' + chart(
        series_tok, "39 字符 access token 的逐字符复现率", "prompt tokens (log)",
        "字符正确率 (%)", ymin=0, ymax=100, yticks=[0, 25, 50, 75, 100], fmt="{:,.0f}",
        hlines=(100,), note="逐字符比对，归一化后取最优对齐窗口") + "</div>")
    body.append('<div class="card">' + chart(
        series_lat, "单次请求耗时（难点探针）", "prompt tokens (log)", "latency (s)",
        ymin=0, fmt="{:,.0f}", note="上升 = 前缀复用生效（只付增量）") + "</div>")
    body.append(table(["prompt tokens", "int8 命中", "nvfp4 命中", "int8 token 字符",
                       "nvfp4 token 字符", "int8 耗时", "nvfp4 耗时"], rows))
    if bad:
        items = "；".join(
            f"{ARM_LABEL[a]} @ {r['prompt_tokens']:,} tok 只输出 {r['hits']}/{r['total']}"
            for a, r in bad)
        body.append(f'<div class="note"><b>被上下文预算截断、已从曲线剔除的点</b>：{esc(items)}。'
                    f'完整答案需要 352 个输出 token，而 <code>max_context</code>=262,144 是'
                    f'「prompt + 输出」的总预算，所以 prompt 一旦超过约 261,790 就放不下答案，'
                    f'模型会在数字中间被硬切（实测 <code>Lagos = 3</code>）。这是测量伪影，不是检索失败。</div>')
    return "".join(body), summary, bad


def main() -> int:
    soft = load("niah_chain_")
    hard = load("niah_chain_hard_")
    soft_body, soft_summary = soft_section(soft)
    hard_body, hard_summary, _ = hard_section(hard)

    doc = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>KV 量化：长上下文检索精度随可见长度变化</title>
<style>
 body {{ font-family: "Segoe UI",Helvetica,Arial,sans-serif; margin:0; padding:28px 34px 60px;
        background:#fafafa; color:#1a1a1a; }}
 h1 {{ font-size:25px; margin:0 0 6px; }}
 h2 {{ font-size:18px; margin:34px 0 10px; }}
 .sub {{ color:#666; font-size:13.5px; margin-bottom:20px; }}
 .card {{ background:#fff; border:1px solid #e3e3e3; border-radius:9px; padding:12px 14px;
         margin:14px 0; box-shadow:0 1px 2px rgba(0,0,0,.04); overflow-x:auto; }}
 .note {{ background:#fff; border:1px solid #e3e3e3; border-radius:9px; padding:16px 20px; margin:16px 0; }}
 table {{ border-collapse:collapse; width:100%; font-size:12.5px; background:#fff;
         border:1px solid #e3e3e3; border-radius:9px; overflow:hidden; margin:14px 0; }}
 th {{ background:#f4f4f6; text-align:right; padding:7px 10px; font-weight:600; color:#444;
      border-bottom:1px solid #e3e3e3; }}
 th:first-child, td:first-child {{ text-align:left; }}
 td {{ padding:6px 10px; text-align:right; border-bottom:1px solid #f2f2f2; }}
 tr:last-child td {{ border-bottom:none; }}
 .num {{ font-variant-numeric: tabular-nums; }}
 .pos {{ color:#1a7f37; font-weight:600; }}
 .neg {{ color:#b3261e; font-weight:600; }}
 code {{ background:#f2f2f4; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
</style></head><body>
<h1>KV 量化：长上下文检索精度随可见上下文长度变化</h1>
<div class="sub">模型 Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-Vision · TP-2 (<code>--devices 0,1</code>)
 · 贪心解码 · 每条 (臂, 语料) 是一条 11 点的<b>严格前缀链</b>，整条链只付一次最长 prefill</div>

<div class="note">
<b>设计</b>：请求体 = <code>[needle 备忘录] + haystack[0:K] + [提问]</code>。K 从 16,384 递增到约 262,000，
每个请求都是上一个的严格前缀延长，因此 ninfer-serve 的前缀复用把 11 次 prefill 合并成一次最长 prefill
（实测：精确重发 0.9s vs 冷启动 127.6s，即 139×；服务端日志也报 <code>cache 228,352 (91.4%)</code>）。
needle 全部钉在文档最前面，所以"needle 到提问的距离"恰好等于 K —— 这是 stock NIAH 网格里最难、最敏感的
0% 深度档。
<br><br>
<b>为什么这比逐深度网格便宜一个数量级</b>：深度网格里每个样本的 needle 位置不同 ⇒ haystack 的 token 序列
不同 ⇒ 完全没有前缀复用，88 个样本每个都要付一次完整的 128K/260K prefill（约 5.5 小时）。前缀链把成本压到
一次最长 prefill（约 25 分钟），代价是丢掉"深度"这一维。
<br><br>
<b>注意</b>：needle 与提问都是英文，中文档只是换 haystack —— 与 stock NIAH 适配器的做法一致。
</div>

<div class="note">
<b>汇总</b><br>
{'<br>'.join(f'8-needle 探针 · {s} · {a}: {h}/{t}' for s, a, h, t in soft_summary)}<br>
{'<br>'.join(f'32-needle 探针 · {a}: 城市码 {h}/{t} · access token 字符 {th}/{tt}' for a, h, t, th, tt in hard_summary)}
</div>

<h1 style="font-size:20px;margin-top:34px">探针 1：8 条 needle</h1>
{soft_body}

<h1 style="font-size:20px;margin-top:34px">探针 2：32 条 needle + 逐字复现</h1>
{hard_body}
</body></html>
"""
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    for s, a, h, t in soft_summary:
        print(f"  soft {s} / {a}: {h}/{t}")
    for a, h, t, th, tt in hard_summary:
        print(f"  hard {a}: codes {h}/{t}  token chars {th}/{tt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
