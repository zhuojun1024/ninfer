from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bench.run_serve_spec_ab import (
    CampaignError,
    Fixture,
    SEEDS,
    main,
    build_plan,
    build_record,
    category_of,
    compare_report,
    load_fixtures,
    pad_text,
    paired_deltas,
    pair_key,
    parse_args,
    request_payload,
    speculative_metrics,
    summarize,
)


def _record(fixture: str, seed: int, accepted: int, drafted: int, rounds: int,
            per_position: list, text: str = "x") -> dict:
    return {
        "fixture": fixture,
        "category": category_of(fixture),
        "seed": seed,
        "accepted_tokens": accepted,
        "drafted_tokens": drafted,
        "rounds": rounds,
        "fallback_steps": 0,
        "completion_tokens": 10,
        "accepted_per_position": per_position,
        "text_sha256": text,
    }


def test_pooled_and_per_request_aggregations_stay_distinct() -> None:
    records = [
        _record("scenario_code_cuda", SEEDS[0], 50, 100, 20, [20, 10, 5]),
        _record("scenario_story_zh_scifi", SEEDS[0], 8, 10, 3, [3, 3, 2]),
    ]
    stats = summarize(records)
    # Ratio of summed tokens: 58/110.
    assert stats["pooled_acceptance"] == pytest.approx(58 / 110)
    # Mean of per-request ratios: (0.5 + 0.8) / 2.
    assert stats["per_request_mean"] == pytest.approx(0.65)
    assert stats["drafted_tokens"] == 110 and stats["accepted_tokens"] == 58
    assert stats["tokens_per_round"] == pytest.approx(1 + 58 / 23)
    # A_0 / rounds, then A_j / A_{j-1}.
    assert stats["first_position_rate"] == pytest.approx(23 / 23)
    assert stats["accepted_per_position"] == [23, 13, 7]
    assert stats["conditional_curve"][1] == pytest.approx(13 / 23)
    assert stats["conditional_curve"][2] == pytest.approx(7 / 13)


def test_category_of_maps_every_scenario_family() -> None:
    assert category_of("scenario_code_python") == "code"
    assert category_of("scenario_story_en_mystery") == "story"
    assert category_of("scenario_translation_markdown") == "translation"
    assert category_of("scenario_structured_sql") == "structured"
    assert category_of("long_niah_64k") == "other"


def test_build_plan_pairs_every_fixture_with_every_seed() -> None:
    plan = build_plan(("a", "b"), SEEDS[:2])
    assert plan == [("a", SEEDS[0]), ("a", SEEDS[1]), ("b", SEEDS[0]), ("b", SEEDS[1])]


def test_pad_text_prepends_the_document_and_rejects_overrun() -> None:
    fixtures = load_fixtures(("scenario_code_cuda",))
    pad = pad_text(fixtures, 64)
    assert len(pad) == 64
    assert pad_text(fixtures, 0) == ""
    with pytest.raises(CampaignError):
        pad_text(fixtures, 10 ** 7)


def test_request_payload_pads_only_the_first_user_message() -> None:
    fixture = Fixture("scenario_code_cuda", [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "ask"},
    ], thinking=False, max_new=4096)
    body = request_payload("model", fixture, 7, 512, "PAD")
    assert body["messages"][0]["content"] == "sys"
    assert body["messages"][1]["content"] == "PAD\n\nask"
    assert body["max_completion_tokens"] == 512 and body["seed"] == 7
    assert body["enable_thinking"] is False and body["stream"] is False
    # A fixture budget below the requested maximum wins.
    small = Fixture("f", [{"role": "user", "content": "ask"}], False, 128)
    assert request_payload("model", small, 7, 512, "")["max_completion_tokens"] == 128


def test_speculative_metrics_reads_the_request_done_counters() -> None:
    metrics = speculative_metrics({
        "speculative": {
            "backend": "dflash2", "draft_window": 7, "rounds": 5, "drafted_tokens": 35,
            "accepted_tokens": 21, "fallback_steps": 1, "accepted_per_position": [5, 4, 3],
        }
    })
    assert metrics == {
        "backend": "dflash2", "draft_window": 7, "rounds": 5, "drafted_tokens": 35,
        "accepted_tokens": 21, "fallback_steps": 1, "accepted_per_position": [5, 4, 3],
    }
    # A request that never reached a speculative round still produces a complete record.
    empty = speculative_metrics({})
    assert empty["rounds"] == 0 and empty["accepted_per_position"] == [] and empty["backend"] is None


def test_build_record_merges_response_and_server_counters() -> None:
    event = {
        "request": {"requested_output_tokens": 8, "sampling": {"seed": 7}},
        "result": {"prompt_tokens": 11, "completion_tokens": 5},
        "speculative": {"backend": "mtp", "rounds": 2, "drafted_tokens": 6, "accepted_tokens": 3},
    }
    payload = {"choices": [{"finish_reason": "length", "message": {"content": "hello", "reasoning_content": "r"}}]}
    record = build_record("scenario_code_cuda", 7, 200, payload, 1.5, event, keep_text=True)
    assert record["http"] == 200 and record["finish_reason"] == "length"
    assert record["prompt_tokens"] == 11 and record["completion_tokens"] == 5
    assert record["accepted_tokens"] == 3 and record["drafted_tokens"] == 6
    assert record["text_chars"] == 5 and record["reasoning_chars"] == 1
    assert record["text"] == "hello" and record["category"] == "code"
    assert len(record["text_sha256"]) == 64


def test_paired_deltas_use_only_shared_fixture_seed_keys() -> None:
    baseline = [_record("scenario_code_cuda", 1, 50, 100, 20, []),
                _record("scenario_code_cuda", 2, 10, 100, 20, [])]
    treatment = [_record("scenario_code_cuda", 1, 60, 100, 20, []),
                 _record("scenario_story_zh_scifi", 2, 90, 100, 20, [])]
    deltas = paired_deltas(baseline, treatment, pair_key)
    assert deltas == [pytest.approx(0.10)]
    report = compare_report({"records": baseline, "sampling": "greedy", "spec": "dflash2",
                             "draft_tokens": 7, "kv_dtype": "fp8", "max_context": 65536,
                             "max_tokens": 512},
                            {"records": treatment},
                            "baseline", "treatment")
    assert "paired requests=1" in report
    assert "pooled acceptance is the ratio of summed tokens" in report
    assert "Response identity: 1/1" in report


def test_cli_defaults_and_run_validation(tmp_path: Path) -> None:
    args = parse_args(["run", "--artifact", "m.ninfer", "--outdir", "out"])
    assert args.spec == "dflash2" and args.draft_tokens == 7 and args.seeds == 5
    assert args.fixture == [] and args.devices is None and args.prefix_reuse is False
    # compare does not accept the runner's options.
    with pytest.raises(SystemExit):
        parse_args(["compare", "--a", "a.json", "--b", "b.json", "--seeds", "1"])

    artifact = tmp_path / "m.ninfer"
    artifact.touch()
    serve = tmp_path / "ninfer-serve"
    serve.touch()
    common = ["run", "--artifact", str(artifact), "--outdir", str(tmp_path / "out"),
              "--serve", str(serve)]
    assert main([*common, "--seeds", "0"]) == 1
    assert main([*common, "--fixture", "not_a_fixture"]) == 1
    assert not (tmp_path / "out" / "summary.json").exists()
