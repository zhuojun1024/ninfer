#!/usr/bin/env python3
"""A/B a speculative backend's acceptance between two artifacts on the serving route.

The acceptance metric is the one docs/performance/methodology.md defines: accepted draft tokens
over drafted tokens, where a request's drafted count is the live verified extent and a request's
accepted count is the committed accepted prefix. This runner replays a fixed fixture and seed plan
against one persistent ninfer-serve process and records the complete speculative counters from
every request_done event, including accepted_per_position. The compare subcommand then reports the
two aggregations the methodology keeps apart - the ratio of summed tokens, and the mean of the
per-request ratios - plus the per-position decomposition and the paired per-request difference with
a confidence interval.

Use it to attribute a measured acceptance difference to a configuration change. Hold everything
except the changed dimension equal: draft width, proposal head, sampling profile, concurrency,
KV dtype, context ceiling, build, and the rendered prompt.

  python3 tools/bench/run_serve_spec_ab.py run \
      --artifact baseline.ninfer --label baseline \
      --spec dflash2 --draft-tokens 7 --lm-head-draft \
      --fixture scenario_code_cuda --fixture scenario_story_zh_scifi --seeds 5 \
      --pad-chars 140000 --prefix-reuse --outdir profiles/bench/spec-ab-baseline

  python3 tools/bench/run_serve_spec_ab.py run \
      --artifact candidate.ninfer --label candidate ... --outdir profiles/bench/spec-ab-candidate

  python3 tools/bench/run_serve_spec_ab.py compare \
      --a profiles/bench/spec-ab-baseline/summary.json \
      --b profiles/bench/spec-ab-candidate/summary.json --label-a baseline --label-b candidate

Outputs per run directory: summary.json (records plus server facts), requests.jsonl (the raw
server request log), server.out.log and server.err.log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "examples" / "cli" / "manifest.json"

SCENARIO_FIXTURES = (
    "scenario_code_cuda",
    "scenario_code_python",
    "scenario_code_typescript",
    "scenario_story_zh_scifi",
    "scenario_story_en_mystery",
    "scenario_story_zh_dialogue",
    "scenario_translation_zh_en",
    "scenario_translation_en_zh",
    "scenario_translation_markdown",
    "scenario_structured_jsonl",
    "scenario_structured_csv",
    "scenario_structured_sql",
)
CATEGORY_PREFIXES = (
    ("scenario_code", "code"),
    ("scenario_story", "story"),
    ("scenario_translation", "translation"),
    ("scenario_structured", "structured"),
)
# Document text prepended by --pad-chars, so a deep-context prompt stays real text.
PAD_FIXTURE = "long_niah_64k"
# The published campaign's seeds; --seeds N takes the first N.
SEEDS = (
    7632647173703958409,
    7968175640111700217,
    912910298659544128,
    9060622443728853932,
    4939353812939007330,
)
STOCHASTIC_PROFILE = (
    "--temperature", "0.6",
    "--top-p", "0.95",
    "--top-k", "20",
    "--min-p", "0",
    "--presence-penalty", "1.0",
    "--frequency-penalty", "0",
)
STARTUP_TIMEOUT_SECONDS = 1800.0
REQUEST_TIMEOUT_SECONDS = 24.0 * 60.0 * 60.0
LOG_EVENT_TIMEOUT_SECONDS = 30.0
# Two-tailed 95% t quantiles; the table covers the sample counts these campaigns use.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
       16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 22: 2.074, 24: 2.064,
       26: 2.056, 28: 2.048, 30: 2.042, 40: 2.021, 60: 2.000}


class CampaignError(RuntimeError):
    pass


@dataclass(frozen=True)
class Fixture:
    name: str
    messages: list
    thinking: bool
    max_new: int


def category_of(name: str) -> str:
    for prefix, category in CATEGORY_PREFIXES:
        if name.startswith(prefix):
            return category
    return "other"


def t95(degrees: int) -> float:
    if degrees in T95:
        return T95[degrees]
    for key in sorted(T95):
        if degrees <= key:
            return T95[key]
    return 1.96


def load_fixtures(names) -> dict:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"failed to read {MANIFEST_PATH}: {error}") from error
    cases = {case["name"]: case for case in manifest["cases"]}
    fixtures = {}
    for name in (*names, PAD_FIXTURE):
        if name not in cases:
            raise CampaignError(f"unknown fixture {name!r}")
        case = cases[name]
        path = MANIFEST_PATH.parent / case["messages"]
        try:
            messages = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CampaignError(f"failed to read {path}: {error}") from error
        fixtures[name] = Fixture(name, messages, bool(case["thinking"]), int(case["max_new"]))
    return fixtures


def build_plan(fixture_names, seeds) -> list:
    plan = []
    for name in fixture_names:
        for seed in seeds:
            plan.append((name, seed))
    return plan


def pad_text(fixtures: dict, chars: int) -> str:
    if chars <= 0:
        return ""
    document = next(m["content"] for m in fixtures[PAD_FIXTURE].messages if m["role"] == "user")
    if chars > len(document):
        raise CampaignError(f"--pad-chars {chars} exceeds the {len(document)}-character {PAD_FIXTURE} document")
    return document[:chars]


def request_payload(model_id: str, fixture: Fixture, seed: int, max_tokens: int, pad: str) -> dict:
    messages = json.loads(json.dumps(fixture.messages))
    if pad:
        for message in messages:
            if message.get("role") == "user":
                message["content"] = pad + "\n\n" + message["content"]
                break
    return {
        "model": model_id,
        "messages": messages,
        "max_completion_tokens": min(max_tokens, fixture.max_new),
        "seed": seed,
        "stream": False,
        "enable_thinking": fixture.thinking,
    }


def speculative_metrics(event: dict) -> dict:
    speculative = event.get("speculative") or {}
    return {
        "backend": speculative.get("backend"),
        "draft_window": speculative.get("draft_window"),
        "rounds": int(speculative.get("rounds", 0)),
        "drafted_tokens": int(speculative.get("drafted_tokens", 0)),
        "accepted_tokens": int(speculative.get("accepted_tokens", 0)),
        "fallback_steps": int(speculative.get("fallback_steps", 0)),
        "accepted_per_position": list(speculative.get("accepted_per_position", [])),
    }


def build_record(fixture: str, seed: int, status: int, payload: dict, wall_seconds: float,
                 event: dict, keep_text: bool) -> dict:
    result = event.get("result") or {}
    request = event.get("request") or {}
    message = {}
    finish_reason = None
    if status == 200:
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")
    text = (message.get("content") or "") or ""
    reasoning = (message.get("reasoning_content") or "") or ""
    record = {
        "fixture": fixture,
        "category": category_of(fixture),
        "seed": seed,
        "http": status,
        "wall_seconds": wall_seconds,
        "prompt_tokens": int(result.get("prompt_tokens", -1)),
        "completion_tokens": int(result.get("completion_tokens", -1)),
        "finish_reason": finish_reason,
        "event_seed": (request.get("sampling") or {}).get("seed"),
        "event_requested_output_tokens": request.get("requested_output_tokens"),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_chars": len(text),
        "reasoning_chars": len(reasoning),
    }
    record.update(speculative_metrics(event))
    if keep_text:
        record["text"] = text
    return record


def acceptance(record: dict) -> float:
    return record["accepted_tokens"] / record["drafted_tokens"] if record["drafted_tokens"] else float("nan")


def summarize(records) -> dict:
    accepted = sum(record["accepted_tokens"] for record in records)
    drafted = sum(record["drafted_tokens"] for record in records)
    rounds = sum(record["rounds"] for record in records)
    per_request = [acceptance(record) for record in records if record["drafted_tokens"]]
    width = max((len(record["accepted_per_position"]) for record in records), default=0)
    per_position = [
        sum(record["accepted_per_position"][index] for record in records
            if len(record["accepted_per_position"]) > index)
        for index in range(width)
    ]
    conditional = []
    for index in range(width):
        previous = rounds if index == 0 else per_position[index - 1]
        conditional.append(per_position[index] / previous if previous else float("nan"))
    return {
        "requests": len(records),
        "accepted_tokens": accepted,
        "drafted_tokens": drafted,
        "rounds": rounds,
        "fallback_steps": sum(record["fallback_steps"] for record in records),
        "completion_tokens": sum(record["completion_tokens"] for record in records),
        # Ratio of summed tokens (corpus/wave aggregation).
        "pooled_acceptance": accepted / drafted if drafted else float("nan"),
        # Mean of per-request ratios (phase aggregation).
        "per_request_mean": statistics.fmean(per_request) if per_request else float("nan"),
        "per_request_sd": statistics.stdev(per_request) if len(per_request) > 1 else float("nan"),
        "tokens_per_round": 1.0 + accepted / rounds if rounds else float("nan"),
        "first_position_rate": per_position[0] / rounds if rounds and per_position else float("nan"),
        "accepted_per_position": per_position,
        "conditional_curve": conditional,
    }


def paired_deltas(a_records, b_records, key) -> list:
    """Per-request b-minus-a differences over the keys both runs share."""

    index = {key(record): record for record in a_records}
    deltas = []
    for record in b_records:
        match = index.get(key(record))
        if match is None:
            continue
        if not match["drafted_tokens"] or not record["drafted_tokens"]:
            continue
        deltas.append(acceptance(record) - acceptance(match))
    return deltas


def pair_key(record: dict):
    return (record["fixture"], record["seed"])


def format_percent(value: float) -> str:
    return "n/a" if value != value else f"{value * 100:.2f}%"


def compare_report(a: dict, b: dict, label_a: str, label_b: str) -> str:
    records_a = a["records"]
    records_b = b["records"]
    stats_a = summarize(records_a)
    stats_b = summarize(records_b)
    shared = {pair_key(record) for record in records_a} & {pair_key(record) for record in records_b}
    lines = [
        f"# speculative acceptance A/B   baseline={label_a}  treatment={label_b}",
        f"# sampling={a.get('sampling')} spec={a.get('spec')} draft_tokens={a.get('draft_tokens')} "
        f"kv_dtype={a.get('kv_dtype')} max_context={a.get('max_context')} max_tokens={a.get('max_tokens')}",
        f"# paired requests={len(shared)}  only_baseline={len(records_a) - len(shared)}  "
        f"only_treatment={len(records_b) - len(shared)}",
        "",
        "## Aggregate",
        "pooled acceptance is the ratio of summed tokens; per-request mean is the mean of the "
        "per-request ratios. Do not interchange them.",
        f"{'metric':<26}{label_a:>14}{label_b:>14}{'delta':>12}",
    ]
    rows = (
        ("pooled acceptance", format_percent(stats_a["pooled_acceptance"]),
         format_percent(stats_b["pooled_acceptance"]),
         format_percent(stats_b["pooled_acceptance"] - stats_a["pooled_acceptance"])),
        ("per-request mean", format_percent(stats_a["per_request_mean"]),
         format_percent(stats_b["per_request_mean"]),
         format_percent(stats_b["per_request_mean"] - stats_a["per_request_mean"])),
        ("per-request sd", format_percent(stats_a["per_request_sd"]),
         format_percent(stats_b["per_request_sd"]), ""),
        ("first-position rate", format_percent(stats_a["first_position_rate"]),
         format_percent(stats_b["first_position_rate"]),
         format_percent(stats_b["first_position_rate"] - stats_a["first_position_rate"])),
        ("tokens/round", f"{stats_a['tokens_per_round']:.3f}", f"{stats_b['tokens_per_round']:.3f}",
         f"{stats_b['tokens_per_round'] - stats_a['tokens_per_round']:+.3f}"),
        ("rounds", str(stats_a["rounds"]), str(stats_b["rounds"]), ""),
        ("drafted tokens", str(stats_a["drafted_tokens"]), str(stats_b["drafted_tokens"]), ""),
        ("accepted tokens", str(stats_a["accepted_tokens"]), str(stats_b["accepted_tokens"]), ""),
        ("fallback steps", str(stats_a["fallback_steps"]), str(stats_b["fallback_steps"]), ""),
        ("completion tokens", str(stats_a["completion_tokens"]), str(stats_b["completion_tokens"]), ""),
    )
    for name, left, right, delta in rows:
        lines.append(f"{name:<26}{left:>14}{right:>14}{delta:>12}")
    lines += ["", "## Per-position"]
    width = max(len(stats_a["accepted_per_position"]), len(stats_b["accepted_per_position"]))
    lines.append(f"{'position':<10}{'A_j baseline':>14}{'A_j treatment':>14}{'marginal A':>12}"
                 f"{'marginal B':>12}{'cond A':>10}{'cond B':>10}{'delta':>10}")
    for index in range(width):
        left = stats_a["accepted_per_position"][index] if index < len(stats_a["accepted_per_position"]) else 0
        right = stats_b["accepted_per_position"][index] if index < len(stats_b["accepted_per_position"]) else 0
        marginal_a = left / stats_a["rounds"] if stats_a["rounds"] else float("nan")
        marginal_b = right / stats_b["rounds"] if stats_b["rounds"] else float("nan")
        cond_a = stats_a["conditional_curve"][index] if index < len(stats_a["conditional_curve"]) else float("nan")
        cond_b = stats_b["conditional_curve"][index] if index < len(stats_b["conditional_curve"]) else float("nan")
        lines.append(f"{index + 1:<10}{left:>14}{right:>14}{format_percent(marginal_a):>12}"
                     f"{format_percent(marginal_b):>12}{format_percent(cond_a):>10}"
                     f"{format_percent(cond_b):>10}{format_percent(marginal_b - marginal_a):>10}")
    deltas = paired_deltas(records_a, records_b, pair_key)
    lines += ["", "## Paired per-request delta (treatment - baseline)"]
    if deltas:
        count = len(deltas)
        mean = statistics.fmean(deltas)
        sd = statistics.stdev(deltas) if count > 1 else 0.0
        se = sd / count ** 0.5 if count else float("nan")
        critical = t95(count - 1)
        lines.append(f"paired n={count}  mean={format_percent(mean)}  sd={format_percent(sd)}  "
                     f"se={format_percent(se)}  95% CI [{format_percent(mean - critical * se)}, "
                     f"{format_percent(mean + critical * se)}]")
        categories = {}
        for record in records_b:
            match = next((item for item in records_a if pair_key(item) == pair_key(record)), None)
            if match is None or not match["drafted_tokens"] or not record["drafted_tokens"]:
                continue
            categories.setdefault(record["category"], []).append(acceptance(record) - acceptance(match))
        lines.append(f"{'category':<16}{'n':>4}{'mean delta':>14}{'sd':>10}")
        for category in sorted(categories):
            values = categories[category]
            sd = statistics.stdev(values) if len(values) > 1 else float("nan")
            lines.append(f"{category:<16}{len(values):>4}{format_percent(statistics.fmean(values)):>14}"
                         f"{format_percent(sd):>10}")
    identical = sum(1 for record in records_b
                    if next((item for item in records_a if pair_key(item) == pair_key(record)),
                            {}).get("text_sha256") == record["text_sha256"])
    lines += ["", f"## Response identity: {identical}/{len(shared)} paired requests hashed the same text",
              "Identical text means both runs emitted the same stream (expected under greedy with an",
              "identical target). A quantized draft changes the accepted length per round, which",
              "perturbs the target's replay state, so partial restores need not match."]
    return "\n".join(lines)


def http_get(url: str, timeout: float) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status
    except Exception:
        return 0


def http_post(url: str, body: dict, timeout: float):
    data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8")), time.monotonic() - started
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {"raw": raw.decode("utf-8", "replace")}
        return error.code, payload, time.monotonic() - started
    except Exception as error:  # transport failures are recorded, not raised
        return 0, {"error": repr(error)}, time.monotonic() - started


def startup_tail(log_path: pathlib.Path, lines: int = 4) -> str:
    """Last few server stderr lines, so a startup failure names its own cause."""

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    tail = [line.strip() for line in text.splitlines() if line.strip()][-lines:]
    return ("; ".join(tail)) if tail else ""


def read_events(log_path: pathlib.Path) -> list:
    if not log_path.exists():
        return []
    events = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def wait_for_request_done(log_path: pathlib.Path, count: int, timeout: float) -> list:
    deadline = time.monotonic() + timeout
    while True:
        done = [event for event in read_events(log_path) if event.get("event") == "request_done"]
        if len(done) >= count or time.monotonic() > deadline:
            return done
        time.sleep(0.25)


def server_command(args, artifact: pathlib.Path, log_path: pathlib.Path) -> list:
    command = [
        str(args.serve), str(artifact),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--model-id", args.label,
        "--max-context", str(args.max_context),
        "--kv-dtype", args.kv_dtype,
        "--log-level", "info",
        "--log-stats-interval-ms", "0",
        "--max-concurrency", "1",
        "--prefill-chunk", "1024",
        "--request-log-jsonl", str(log_path),
    ]
    if args.devices:
        command += ["--devices", args.devices]
    if not args.prefix_reuse:
        command.append("--no-prefix-reuse")
    if args.spec != "none":
        command += ["--spec", args.spec, "--draft-tokens", str(args.draft_tokens)]
        if args.lm_head_draft:
            command.append("--lm-head-draft")
    if args.sampling == "greedy":
        command.append("--greedy")
    else:
        command += list(STOCHASTIC_PROFILE)
    return command


def run(args) -> int:
    artifact = args.artifact.resolve()
    if not artifact.is_file():
        raise CampaignError(f"artifact not found: {artifact}")
    if not args.serve.is_file():
        raise CampaignError(f"ninfer-serve not found: {args.serve}")
    fixture_names = args.fixture or list(SCENARIO_FIXTURES)
    if args.seeds < 1 or args.seeds > len(SEEDS):
        raise CampaignError(f"--seeds must be in 1..{len(SEEDS)}")
    fixtures = load_fixtures(fixture_names)
    plan = build_plan(fixture_names, SEEDS[: args.seeds])
    if args.limit:
        plan = plan[: args.limit]
    pad = pad_text(fixtures, args.pad_chars)

    args.outdir.mkdir(parents=True, exist_ok=True)
    log_path = args.outdir / "requests.jsonl"
    if log_path.exists():
        log_path.unlink()
    command = server_command(args, artifact, log_path)
    (args.outdir / "server_argv.json").write_text(json.dumps(command, indent=1), encoding="utf-8")
    print(f"[{args.label}] {len(plan)} requests, sampling={args.sampling}, spec={args.spec} "
          f"K={args.draft_tokens}, pad={args.pad_chars} chars", flush=True)

    environment = dict(os.environ)
    stdout = (args.outdir / "server.out.log").open("wb")
    stderr = (args.outdir / "server.err.log").open("wb")
    process = subprocess.Popen(command, cwd=str(REPO_ROOT), env=environment, stdout=stdout, stderr=stderr)
    print(f"[{args.label}] server pid={process.pid}", flush=True)
    records = []
    server_start = None
    try:
        base = f"http://127.0.0.1:{args.port}"
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                tail = startup_tail(args.outdir / "server.err.log")
                raise CampaignError(
                    f"ninfer-serve exited during startup with status {process.returncode}"
                    + (f": {tail}" if tail else ""))
            if http_get(base + "/health", 3) == 200:
                break
            time.sleep(1.0)
        else:
            raise CampaignError(f"timed out waiting for ninfer-serve at {base}")
        starts = [event for event in read_events(log_path) if event.get("event") == "server_start"]
        server_start = starts[-1] if starts else None
        model_id = ((server_start or {}).get("server") or {}).get("public_model_id") or args.label
        print(f"[{args.label}] healthy, model_id={model_id}", flush=True)

        for index, (name, seed) in enumerate(plan):
            fixture = fixtures[name]
            body = request_payload(model_id, fixture, seed, args.max_tokens, pad)
            status, payload, wall = http_post(base + "/v1/chat/completions", body, REQUEST_TIMEOUT_SECONDS)
            done = wait_for_request_done(log_path, index + 1, LOG_EVENT_TIMEOUT_SECONDS)
            event = done[index] if len(done) > index else {}
            record = build_record(name, seed, status, payload, wall, event, args.keep_text)
            event_seed = record["event_seed"]
            # Seeds exceed 2^53, so the log's JSON number round-trips through float64.
            if event_seed is not None and float(event_seed) != float(seed):
                raise CampaignError(f"request_done seed {event_seed} does not match planned {seed}")
            if record["event_requested_output_tokens"] not in (None, body["max_completion_tokens"]):
                raise CampaignError(
                    f"request_done requested {record['event_requested_output_tokens']} tokens, "
                    f"planned {body['max_completion_tokens']}")
            records.append(record)
            print(f"[{args.label}] {index + 1}/{len(plan)} {name} seed={seed} http={status} "
                  f"prompt={record['prompt_tokens']} out={record['completion_tokens']} "
                  f"rounds={record['rounds']} acc={format_percent(acceptance(record))} "
                  f"({record['accepted_tokens']}/{record['drafted_tokens']}) wall={wall:.1f}s", flush=True)
            if status != 200:
                raise CampaignError(f"request failed: {json.dumps(payload)[:400]}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        stdout.close()
        stderr.close()

    summary = {
        "label": args.label,
        "artifact": str(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "spec": args.spec,
        "draft_tokens": args.draft_tokens,
        "lm_head_draft": args.lm_head_draft,
        "sampling": args.sampling,
        "kv_dtype": args.kv_dtype,
        "max_context": args.max_context,
        "max_tokens": args.max_tokens,
        "pad_chars": args.pad_chars,
        "prefix_reuse": args.prefix_reuse,
        "devices": args.devices,
        "command": command,
        "server_start": {
            "artifact": (server_start or {}).get("artifact"),
            "engine": (server_start or {}).get("engine"),
            "environment": (server_start or {}).get("environment"),
        },
        "records": records,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    stats = summarize(records)
    print(f"[{args.label}] DONE requests={len(records)} pooled={format_percent(stats['pooled_acceptance'])} "
          f"({stats['accepted_tokens']}/{stats['drafted_tokens']}) tokens/round={stats['tokens_per_round']:.3f}",
          flush=True)
    return 0


def compare(args) -> int:
    baseline = json.loads(args.a.read_text(encoding="utf-8"))
    treatment = json.loads(args.b.read_text(encoding="utf-8"))
    report = compare_report(baseline, treatment, args.label_a, args.label_b)
    print(report)
    if args.json:
        payload = {
            "baseline": summarize(baseline["records"]),
            "treatment": summarize(treatment["records"]),
            "paired_delta": paired_deltas(baseline["records"], treatment["records"], pair_key),
        }
        args.json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    runner = subparsers.add_parser("run", help="replay the plan against one artifact")
    runner.add_argument("--artifact", type=pathlib.Path, required=True)
    runner.add_argument("--outdir", type=pathlib.Path, required=True)
    runner.add_argument("--label", default="candidate", help="server model id and report label")
    runner.add_argument("--serve", type=pathlib.Path,
                        default=REPO_ROOT / "build" / "apps" / "ninfer-serve")
    runner.add_argument("--devices", default=None, help="TP-2 pair such as 0,1; omitted is single-device")
    runner.add_argument("--spec", choices=("none", "mtp", "dflash", "dflash2"), default="dflash2")
    runner.add_argument("--draft-tokens", type=int, default=7)
    runner.add_argument("--lm-head-draft", action="store_true")
    runner.add_argument("--sampling", choices=("stochastic", "greedy"), default="stochastic")
    runner.add_argument("--fixture", action="append", default=[],
                        help="scenario fixture; repeatable, default is every scenario")
    runner.add_argument("--seeds", type=int, default=5, help=f"first N of the campaign's {len(SEEDS)} seeds")
    runner.add_argument("--max-tokens", type=int, default=512)
    runner.add_argument("--max-context", type=int, default=65536)
    runner.add_argument("--kv-dtype", default="fp8")
    runner.add_argument("--pad-chars", type=int, default=0,
                        help=f"prepend this many characters of the {PAD_FIXTURE} document")
    runner.add_argument("--prefix-reuse", action="store_true")
    runner.add_argument("--keep-text", action="store_true", help="store response text in summary.json")
    runner.add_argument("--limit", type=int, default=0, help="run only the first N planned requests")
    runner.add_argument("--port", type=int, default=8123)
    runner.set_defaults(handler=run)

    comparer = subparsers.add_parser("compare", help="paired report over two summaries")
    comparer.add_argument("--a", type=pathlib.Path, required=True, help="baseline summary.json")
    comparer.add_argument("--b", type=pathlib.Path, required=True, help="treatment summary.json")
    comparer.add_argument("--label-a", default="baseline")
    comparer.add_argument("--label-b", default="treatment")
    comparer.add_argument("--json", type=pathlib.Path, default=None)
    comparer.set_defaults(handler=compare)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return args.handler(args)
    except CampaignError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
