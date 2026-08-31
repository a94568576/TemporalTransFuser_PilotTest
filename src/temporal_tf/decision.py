"""Pre-registered exploratory verdict for a finalized real-data pilot.

This module deliberately consumes only ``study_results.json``.  It does not
import the selection/finalization engine, follow ``study_selection`` paths, or
inspect test-open markers.  The fixed thresholds below are pilot-continuation
criteria, not model-selection knobs and never a paper or closed-loop GO.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DECISION_SCHEMA_VERSION = 2
VERDICT_SCOPE = "exploratory_offline_adapter_pilot"
AUTHORITATIVE_PRIMARY_METHOD = "past_bev"
REQUIRED_METHODS = (
    "baseline",
    "current_only_matched",
    "current_bev",
    "past_bev",
    "shuffled_past_bev",
)
OPTIONAL_METHODS = ("trajectory_only", "combined")
MINIMUM_SEED_COUNT = 3
MINIMUM_ADE_RELATIVE_IMPROVEMENT = 0.03
MAXIMUM_SMOOTHNESS_RELATIVE_WORSENING = 0.05
EXPECTED_BOOTSTRAP_CONFIDENCE = 0.95
EXPECTED_CHECKPOINT_SELECTION_METRIC = {
    "split": "val",
    "metric": "ade",
    "aggregation": "equal_route_macro",
}
FORBIDDEN_OUTPUT_NAMES = {
    "results.json",
    "selection_manifest.json",
    "study_manifest.json",
    "study_results.json",
    "study_test_opened.marker.json",
    "test_opened.marker.json",
}


class DecisionInputError(ValueError):
    """Raised when a study result cannot support a trustworthy decision."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecisionInputError(f"{path} must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DecisionInputError(f"{path} must be an array")
    return value


def _field(mapping: Mapping[str, Any], name: str, path: str) -> Any:
    if name not in mapping:
        raise DecisionInputError(f"missing {path}.{name}")
    return mapping[name]


def _number(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionInputError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DecisionInputError(f"{path} must be finite")
    if nonnegative and result < 0.0:
        raise DecisionInputError(f"{path} must be non-negative")
    return result


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DecisionInputError(f"{path} must be a positive integer")
    return int(value)


def _test_split(seed_result: Mapping[str, Any], method: str, path: str) -> Mapping[str, Any]:
    if method == "baseline":
        baseline = _mapping(_field(seed_result, "baseline", path), f"{path}.baseline")
        return _mapping(
            _field(baseline, "test", f"{path}.baseline"), f"{path}.baseline.test"
        )
    variants = _mapping(_field(seed_result, "variants", path), f"{path}.variants")
    variant = _mapping(
        _field(variants, method, f"{path}.variants"), f"{path}.variants.{method}"
    )
    splits = _mapping(
        _field(variant, "splits", f"{path}.variants.{method}"),
        f"{path}.variants.{method}.splits",
    )
    return _mapping(
        _field(splits, "test", f"{path}.variants.{method}.splits"),
        f"{path}.variants.{method}.splits.test",
    )


def _extract_metrics(split: Mapping[str, Any], path: str) -> dict[str, float | int]:
    route_macro = _mapping(
        _field(split, "route_macro", path), f"{path}.route_macro"
    )
    return {
        "ade": _number(
            _field(route_macro, "ade", f"{path}.route_macro"),
            f"{path}.route_macro.ade",
            nonnegative=True,
        ),
        "fde": _number(
            _field(route_macro, "fde", f"{path}.route_macro"),
            f"{path}.route_macro.fde",
            nonnegative=True,
        ),
        "smoothness": _number(
            _field(route_macro, "smoothness", f"{path}.route_macro"),
            f"{path}.route_macro.smoothness",
            nonnegative=True,
        ),
        "route_count": _positive_integer(
            _field(split, "route_count", path), f"{path}.route_count"
        ),
    }


def _extract_bootstrap_upper(
    split: Mapping[str, Any], path: str, *, expected_route_count: int
) -> float:
    paired = _mapping(
        _field(split, "paired_vs_baseline", path), f"{path}.paired_vs_baseline"
    )
    if paired.get("delta_definition") != "adapter_minus_baseline":
        raise DecisionInputError(
            f"{path}.paired_vs_baseline.delta_definition must be adapter_minus_baseline"
        )
    paired_route_count = _positive_integer(
        _field(paired, "route_count", f"{path}.paired_vs_baseline"),
        f"{path}.paired_vs_baseline.route_count",
    )
    if paired_route_count != expected_route_count:
        raise DecisionInputError(
            f"{path}.paired_vs_baseline.route_count differs from the test route count"
        )
    intervals = _mapping(
        _field(paired, "route_bootstrap_ci", f"{path}.paired_vs_baseline"),
        f"{path}.paired_vs_baseline.route_bootstrap_ci",
    )
    ade_interval = _mapping(
        _field(intervals, "ade", f"{path}.paired_vs_baseline.route_bootstrap_ci"),
        f"{path}.paired_vs_baseline.route_bootstrap_ci.ade",
    )
    if ade_interval.get("cluster_unit") != "route":
        raise DecisionInputError(
            f"{path}.paired_vs_baseline.route_bootstrap_ci.ade.cluster_unit must be route"
        )
    confidence = _number(
        _field(
            ade_interval,
            "confidence_level",
            f"{path}.paired_vs_baseline.route_bootstrap_ci.ade",
        ),
        f"{path}.paired_vs_baseline.route_bootstrap_ci.ade.confidence_level",
    )
    if not math.isclose(confidence, EXPECTED_BOOTSTRAP_CONFIDENCE, abs_tol=1e-12):
        raise DecisionInputError(
            f"{path}.paired_vs_baseline route bootstrap must use a 95% confidence interval"
        )
    return _number(
        _field(
            ade_interval, "upper", f"{path}.paired_vs_baseline.route_bootstrap_ci.ade"
        ),
        f"{path}.paired_vs_baseline.route_bootstrap_ci.ade.upper",
    )


def _gate(
    *,
    passed: bool,
    requirement: str,
    observed: Mapping[str, Any],
    category: str = "core",
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "category": category,
        "requirement": requirement,
        "observed": dict(observed),
    }


def _validate_study_header(
    study_results: Mapping[str, Any],
) -> tuple[list[str], Mapping[str, Any]]:
    if study_results.get("schema_version") != 1:
        raise DecisionInputError("study_results.schema_version must be 1")
    if study_results.get("status") != "completed":
        raise DecisionInputError("study_results.status must be completed")
    if study_results.get("evaluation_mode") != "final_multiseed":
        raise DecisionInputError(
            "study_results.evaluation_mode must be final_multiseed; selection results are forbidden"
        )
    if study_results.get("single_test_open_event") is not True:
        raise DecisionInputError("study_results.single_test_open_event must be true")
    if study_results.get("primary_metric") != "ade":
        raise DecisionInputError("study_results.primary_metric must be ade")
    study_id = study_results.get("study_id")
    if not isinstance(study_id, str) or not study_id.strip():
        raise DecisionInputError("study_results.study_id must be a non-empty string")
    cache_hash = study_results.get("cache_index_sha256")
    if (
        not isinstance(cache_hash, str)
        or len(cache_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in cache_hash)
    ):
        raise DecisionInputError(
            "study_results.cache_index_sha256 must be a 64-character hexadecimal digest"
        )

    declared_raw = _sequence(
        _field(study_results, "seeds", "study_results"), "study_results.seeds"
    )
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in declared_raw):
        raise DecisionInputError("study_results.seeds must contain integer seed values")
    declared = [str(int(seed)) for seed in declared_raw]
    if len(declared) < MINIMUM_SEED_COUNT:
        raise DecisionInputError(
            f"study_results must contain at least {MINIMUM_SEED_COUNT} seeds"
        )
    if any(not seed.strip() for seed in declared) or len(set(declared)) != len(declared):
        raise DecisionInputError("study_results.seeds must be unique, non-empty identifiers")

    seed_results = _mapping(
        _field(study_results, "seed_results", "study_results"),
        "study_results.seed_results",
    )
    normalized_seed_results = {str(key): value for key, value in seed_results.items()}
    if len(normalized_seed_results) != len(seed_results):
        raise DecisionInputError(
            "study_results.seed_results contains duplicate normalized seed keys"
        )
    if set(normalized_seed_results) != set(declared):
        raise DecisionInputError(
            "study_results.seed_results keys must exactly match study_results.seeds"
        )

    variants = _sequence(
        _field(study_results, "variants", "study_results"), "study_results.variants"
    )
    if any(not isinstance(variant, str) or not variant.strip() for variant in variants):
        raise DecisionInputError(
            "study_results.variants must contain non-empty method names"
        )
    variant_names = list(variants)
    if len(set(variant_names)) != len(variant_names):
        raise DecisionInputError("study_results.variants must be unique")
    missing = [
        method
        for method in REQUIRED_METHODS
        if method != "baseline" and method not in variant_names
    ]
    if missing:
        raise DecisionInputError(f"study_results.variants is missing required methods: {missing}")
    return declared, normalized_seed_results


def evaluate_go_no_go(study_results: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and evaluate one finalized real-data multi-seed study.

    Means are recomputed from child final-test route-macro metrics.  The
    possibly redundant ``method_aggregates`` field is intentionally ignored.
    """

    study_results = _mapping(study_results, "study_results")
    if study_results.get("checkpoint_selection_metric") != EXPECTED_CHECKPOINT_SELECTION_METRIC:
        raise DecisionInputError(
            "study checkpoints must be selected by validation equal-route macro ADE"
        )
    seeds, seed_results = _validate_study_header(study_results)
    methods = list(REQUIRED_METHODS)
    for optional in OPTIONAL_METHODS:
        present = []
        for seed in seeds:
            child = _mapping(seed_results[seed], f"study_results.seed_results.{seed}")
            variants = _mapping(
                _field(child, "variants", f"study_results.seed_results.{seed}"),
                f"study_results.seed_results.{seed}.variants",
            )
            present.append(optional in variants)
        if any(present) and not all(present):
            raise DecisionInputError(
                f"optional method {optional} must be present for every seed or none"
            )
        if all(present):
            methods.append(optional)

    per_seed: dict[str, Any] = {}
    route_count_reference: int | None = None
    for seed in seeds:
        seed_path = f"study_results.seed_results.{seed}"
        child = _mapping(seed_results[seed], seed_path)
        if child.get("status") != "completed" or child.get("evaluation_mode") != "final":
            raise DecisionInputError(f"{seed_path} must be a completed final result")
        if child.get("test_opened") is not True:
            raise DecisionInputError(f"{seed_path}.test_opened must be true")
        if child.get("evidence_level") != "offline_cache_evaluation":
            raise DecisionInputError(
                f"{seed_path} is not real offline-cache evidence; synthetic smoke is forbidden"
            )
        source = _mapping(
            _field(child, "cache_source", seed_path), f"{seed_path}.cache_source"
        )
        if not str(source.get("kind", "")).strip() or source.get("kind") == "synthetic":
            raise DecisionInputError(f"{seed_path}.cache_source must identify a real cache")

        window_counts = _mapping(
            _field(child, "dataset_windows", seed_path), f"{seed_path}.dataset_windows"
        )
        test_windows = _positive_integer(
            _field(window_counts, "test", f"{seed_path}.dataset_windows"),
            f"{seed_path}.dataset_windows.test",
        )
        metrics: dict[str, dict[str, float | int]] = {}
        splits: dict[str, Mapping[str, Any]] = {}
        for method in methods:
            method_path = (
                f"{seed_path}.baseline.test"
                if method == "baseline"
                else f"{seed_path}.variants.{method}.splits.test"
            )
            split = _test_split(child, method, seed_path)
            splits[method] = split
            metrics[method] = _extract_metrics(split, method_path)

        route_counts = {int(metric["route_count"]) for metric in metrics.values()}
        if len(route_counts) != 1:
            raise DecisionInputError(f"{seed_path} methods have inconsistent test route counts")
        route_count = route_counts.pop()
        if test_windows < route_count:
            raise DecisionInputError(f"{seed_path} has fewer test windows than routes")
        if route_count_reference is None:
            route_count_reference = route_count
        elif route_count != route_count_reference:
            raise DecisionInputError("test route counts differ across seeds")

        bootstrap_upper = _extract_bootstrap_upper(
            splits["past_bev"],
            f"{seed_path}.variants.past_bev.splits.test",
            expected_route_count=route_count,
        )
        past_ade = float(metrics["past_bev"]["ade"])
        matched_ade = float(metrics["current_only_matched"]["ade"])
        per_seed[seed] = {
            "route_count": route_count,
            "test_windows": test_windows,
            "metrics": metrics,
            "past_vs_current_only_matched_ade_delta": past_ade - matched_ade,
            "past_vs_baseline_bootstrap_ade_ci_upper": bootstrap_upper,
        }

    method_means: dict[str, dict[str, float]] = {}
    for method in methods:
        method_means[method] = {
            metric: statistics.fmean(
                float(per_seed[seed]["metrics"][method][metric]) for seed in seeds
            )
            for metric in ("ade", "fde", "smoothness")
        }

    baseline = method_means["baseline"]
    matched = method_means["current_only_matched"]
    current_bev = method_means["current_bev"]
    past = method_means["past_bev"]
    shuffled = method_means["shuffled_past_bev"]
    if matched["ade"] > 0.0:
        relative_ade_improvement = (matched["ade"] - past["ade"]) / matched["ade"]
        relative_gate_passed = (
            relative_ade_improvement >= MINIMUM_ADE_RELATIVE_IMPROVEMENT
            or math.isclose(
                relative_ade_improvement,
                MINIMUM_ADE_RELATIVE_IMPROVEMENT,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
    else:
        relative_ade_improvement = None
        relative_gate_passed = False

    seed_direction = {
        seed: per_seed[seed]["past_vs_current_only_matched_ade_delta"] < 0.0
        for seed in seeds
    }
    bootstrap_by_seed = {
        seed: per_seed[seed]["past_vs_baseline_bootstrap_ade_ci_upper"] <= 0.0
        for seed in seeds
    }
    smoothness_limit = baseline["smoothness"] * (
        1.0 + MAXIMUM_SMOOTHNESS_RELATIVE_WORSENING
    )
    gates = {
        "ade_3pct_vs_current_only_matched": _gate(
            passed=relative_gate_passed,
            requirement=(
                "mean route-macro past_bev ADE is at least 3% lower than "
                "current_only_matched"
            ),
            observed={
                "past_bev": past["ade"],
                "current_only_matched": matched["ade"],
                "relative_improvement": relative_ade_improvement,
                "minimum_relative_improvement": MINIMUM_ADE_RELATIVE_IMPROVEMENT,
            },
        ),
        "ade_vs_baseline": _gate(
            passed=past["ade"] < baseline["ade"],
            requirement="mean route-macro past_bev ADE is strictly lower than baseline",
            observed={"past_bev": past["ade"], "baseline": baseline["ade"]},
        ),
        "ade_vs_current_bev": _gate(
            passed=past["ade"] < current_bev["ade"],
            requirement="mean route-macro past_bev ADE is strictly lower than current_bev",
            observed={"past_bev": past["ade"], "current_bev": current_bev["ade"]},
            category="temporal_specific",
        ),
        "ade_vs_shuffled_past_bev": _gate(
            passed=past["ade"] < shuffled["ade"],
            requirement="mean route-macro past_bev ADE is strictly lower than shuffled_past_bev",
            observed={"past_bev": past["ade"], "shuffled_past_bev": shuffled["ade"]},
            category="temporal_specific",
        ),
        "fde_vs_current_only_matched": _gate(
            passed=past["fde"] < matched["fde"],
            requirement="mean route-macro past_bev FDE is strictly lower than current_only_matched",
            observed={"past_bev": past["fde"], "current_only_matched": matched["fde"]},
        ),
        "fde_vs_baseline": _gate(
            passed=past["fde"] < baseline["fde"],
            requirement="mean route-macro past_bev FDE is strictly lower than baseline",
            observed={"past_bev": past["fde"], "baseline": baseline["fde"]},
        ),
        "ade_direction_every_seed": _gate(
            passed=all(seed_direction.values()),
            requirement="past_bev ADE is lower than current_only_matched in every seed",
            observed={
                "passed_by_seed": seed_direction,
                "delta_by_seed": {
                    seed: per_seed[seed]["past_vs_current_only_matched_ade_delta"]
                    for seed in seeds
                },
            },
        ),
        "bootstrap_ci_every_seed": _gate(
            passed=all(bootstrap_by_seed.values()),
            requirement=(
                "each seed's paired-vs-baseline route-bootstrap ADE 95% CI upper "
                "bound is <= 0"
            ),
            observed={
                "passed_by_seed": bootstrap_by_seed,
                "upper_by_seed": {
                    seed: per_seed[seed]["past_vs_baseline_bootstrap_ade_ci_upper"]
                    for seed in seeds
                },
            },
        ),
        "smoothness_vs_baseline": _gate(
            passed=past["smoothness"] <= smoothness_limit,
            requirement=(
                "mean route-macro past_bev smoothness is no more than 5% worse "
                "than baseline"
            ),
            observed={
                "past_bev": past["smoothness"],
                "baseline": baseline["smoothness"],
                "maximum_allowed": smoothness_limit,
                "maximum_relative_worsening": MAXIMUM_SMOOTHNESS_RELATIVE_WORSENING,
            },
        ),
    }

    temporal_gate_names = ("ade_vs_current_bev", "ade_vs_shuffled_past_bev")
    core_gate_names = tuple(name for name in gates if name not in temporal_gate_names)
    all_hard_gates_pass = all(gate["passed"] for gate in gates.values())
    core_bev_gates_pass = all(gates[name]["passed"] for name in core_gate_names)
    temporal_specific_gates_pass = all(gates[name]["passed"] for name in temporal_gate_names)
    if all_hard_gates_pass:
        status = "go"
        reason = (
            "All fixed exploratory pilot gates passed; this only authorizes the "
            "next alignment experiment."
        )
    elif core_bev_gates_pass and not temporal_specific_gates_pass:
        status = "ambiguous"
        reason = (
            "BEV correction gates passed, but current/shuffled controls do not establish a "
            "temporal-specific benefit."
        )
    else:
        status = "no_go"
        reason = "At least one authoritative exploratory pilot core gate failed."

    seed_count = len(seeds)
    route_count = int(route_count_reference or 0)
    limitations = [
        (
            "Regardless of status, this is an exploratory pilot verdict and never a "
            "paper GO or a closed-loop driving GO."
        ),
        (
            "This is offline cached-path evidence only; it does not establish closed-loop "
            "CARLA driving improvement."
        ),
        (
            "Training seeds repeat the same test routes and are not additional independent "
            "route samples."
        ),
    ]
    if seed_count == MINIMUM_SEED_COUNT:
        limitations.append(
            "The study uses exactly the minimum three seeds; seed-to-seed uncertainty "
            "is only coarsely characterized."
        )
    if route_count == 1:
        limitations.append(
            "Only one independent test route is present; the route bootstrap is "
            "degenerate and cannot quantify route-population uncertainty."
        )
    elif route_count == 3:
        limitations.append(
            "Exactly three independent test routes are present. Repeating them across "
            "training seeds does not enlarge the route sample, so this result cannot "
            "support a paper-level efficacy or generalization claim."
        )
    elif route_count < 10:
        limitations.append(
            f"Only {route_count} independent test routes are present; route-bootstrap "
            "interval precision is limited."
        )

    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "verdict_scope": VERDICT_SCOPE,
        "paper_go": False,
        "closed_loop_go": False,
        "authoritative_primary_method": AUTHORITATIVE_PRIMARY_METHOD,
        "combined_role": "diagnostic_only",
        "criteria": {
            "minimum_seed_count": MINIMUM_SEED_COUNT,
            "minimum_ade_relative_improvement": MINIMUM_ADE_RELATIVE_IMPROVEMENT,
            "maximum_smoothness_relative_worsening": MAXIMUM_SMOOTHNESS_RELATIVE_WORSENING,
            "bootstrap_confidence_level": EXPECTED_BOOTSTRAP_CONFIDENCE,
            "primary_aggregation": "equal-weight route macro, then arithmetic mean across seeds",
            "checkpoint_selection_metric": dict(EXPECTED_CHECKPOINT_SELECTION_METRIC),
            "verdict_semantics": "exploratory continuation gate only",
            "combined_affects_verdict": False,
        },
        "study": {
            "study_id": study_results.get("study_id"),
            "seed_count": seed_count,
            "seeds": seeds,
            "test_route_count_per_seed": route_count,
            "test_windows_by_seed": {
                seed: int(per_seed[seed]["test_windows"]) for seed in seeds
            },
            "cache_index_sha256": study_results.get("cache_index_sha256"),
        },
        "method_means": method_means,
        "gates": gates,
        "summary": {
            "all_hard_gates_pass": all_hard_gates_pass,
            "core_bev_gates_pass": core_bev_gates_pass,
            "temporal_specific_gates_pass": temporal_specific_gates_pass,
            "failed_gates": [name for name, gate in gates.items() if not gate["passed"]],
        },
        "per_seed": per_seed,
        "limitations": limitations,
    }


def _format_number(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{100.0 * number:.2f}%" if percent else f"{number:.4f}"


def render_decision_markdown(decision: Mapping[str, Any]) -> str:
    """Render a concise audit trail from :func:`evaluate_go_no_go` output."""

    status = str(decision["status"])
    study = _mapping(decision["study"], "decision.study")
    lines = [
        "# Exploratory real-pilot verdict",
        "",
        (
            f"**Exploratory pilot decision: `{status}` (not a paper or closed-loop GO).** "
            f"{decision['reason']}"
        ),
        "",
        (
            "The authoritative gate uses `past_bev`; `combined` is diagnostic only "
            "and cannot change this verdict."
        ),
        "",
        (
            f"Study `{study.get('study_id')}`: {study['seed_count']} seeds, "
            f"{study['test_route_count_per_seed']} test routes per seed."
        ),
        "",
        "## Fixed authoritative exploratory gates",
        "",
        "| Gate | Result | Observed |",
        "|---|---:|---|",
    ]
    gates = _mapping(decision["gates"], "decision.gates")
    for name, raw_gate in gates.items():
        gate = _mapping(raw_gate, f"decision.gates.{name}")
        observed = _mapping(gate["observed"], f"decision.gates.{name}.observed")
        if name == "ade_3pct_vs_current_only_matched":
            detail = (
                "relative improvement "
                f"{_format_number(observed['relative_improvement'], percent=True)}"
            )
        elif name in ("ade_direction_every_seed", "bootstrap_ci_every_seed"):
            passed_by_seed = _mapping(
                observed["passed_by_seed"], f"decision.gates.{name}.observed.passed_by_seed"
            )
            detail = ", ".join(
                f"seed {seed}: {'pass' if passed else 'fail'}"
                for seed, passed in passed_by_seed.items()
            )
        elif "past_bev" in observed and "maximum_allowed" in observed:
            detail = (
                f"past {_format_number(observed['past_bev'])}; limit "
                f"{_format_number(observed['maximum_allowed'])}"
            )
        elif "past_bev" in observed:
            comparator = next(key for key in observed if key != "past_bev")
            detail = (
                f"past {_format_number(observed['past_bev'])}; {comparator} "
                f"{_format_number(observed[comparator])}"
            )
        else:
            detail = "see JSON"
        lines.append(f"| `{name}` | {'PASS' if gate['passed'] else 'FAIL'} | {detail} |")

    lines.extend(
        [
            "",
            "## Route-macro means across seeds",
            "",
            "| Method | ADE | FDE | Smoothness |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, raw_metrics in _mapping(
        decision["method_means"], "decision.method_means"
    ).items():
        metrics = _mapping(raw_metrics, f"decision.method_means.{method}")
        lines.append(
            f"| {method} | {_format_number(metrics['ade'])} | "
            f"{_format_number(metrics['fde'])} | {_format_number(metrics['smoothness'])} |"
        )

    lines.extend(["", "## Limitations", ""])
    for limitation in _sequence(decision["limitations"], "decision.limitations"):
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def load_and_evaluate(study_results_path: str | Path) -> tuple[dict[str, Any], str]:
    """Read exactly one study-results file and return a decision plus SHA-256."""

    path = Path(study_results_path)
    if not path.is_file():
        raise DecisionInputError(f"study results file does not exist: {path}")
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise DecisionInputError(f"cannot read valid JSON from {path}: {error}") from error
    decision = evaluate_go_no_go(parsed)
    decision["source"] = {
        "study_results": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return decision, hashlib.sha256(raw).hexdigest()


def write_decision_outputs(
    decision: Mapping[str, Any],
    *,
    json_output: str | Path,
    markdown_output: str | Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write only the two requested report files, never experiment markers."""

    json_path = Path(json_output)
    markdown_path = Path(markdown_output)
    if json_path.resolve() == markdown_path.resolve():
        raise ValueError("JSON and Markdown outputs must be different files")
    source = decision.get("source")
    source_path: Path | None = None
    if isinstance(source, Mapping) and isinstance(source.get("study_results"), str):
        source_path = Path(source["study_results"]).resolve()
    for path in (json_path, markdown_path):
        normalized_name = path.name.lower()
        if normalized_name in FORBIDDEN_OUTPUT_NAMES or normalized_name.endswith(
            ".marker.json"
        ):
            raise ValueError(
                f"refusing to write an experiment-state or test marker path: {path}"
            )
        if source_path is not None and path.resolve() == source_path:
            raise ValueError(f"refusing to overwrite the source study results: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing decision output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    payloads = (
        (json_path, json.dumps(decision, indent=2, ensure_ascii=False) + "\n"),
        (markdown_path, render_decision_markdown(decision)),
    )
    for path, payload in payloads:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    return json_path, markdown_path
