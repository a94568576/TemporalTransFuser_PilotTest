"""Human-readable rendering of a pilot result JSON."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _number(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:+.4f}" if signed else f"{numeric:.4f}"


def _percentage(value: Any) -> str:
    if value is None:
        return "—"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "—"
    return f"{100.0 * numeric:.1f}%"


def _preferred_metrics(split_result: dict[str, Any]) -> dict[str, Any]:
    # New results use route-macro as the primary summary. Keep accepting the
    # legacy sample-micro schema so historical smoke artifacts still render.
    return split_result.get("route_macro", split_result["overall"])


def _quantile(split_result: dict[str, Any], metric: str, label: str) -> float | None:
    route_quantiles = split_result.get("route_quantiles", {})
    if metric in route_quantiles:
        return route_quantiles[metric].get(label)
    sample_quantiles = split_result.get("sample_quantiles", {})
    if metric in sample_quantiles:
        return sample_quantiles[metric].get(label)
    return None


def _performance_row(name: str, split_result: dict[str, Any], parameters: int) -> str:
    primary = _preferred_metrics(split_result)
    overall = split_result["overall"]
    worst = split_result.get("baseline_worst_slice", {})
    route_count = split_result.get("route_count")
    route_count_text = str(int(route_count)) if route_count is not None else "—"
    return (
        f"| {name} | {parameters:,} | {route_count_text} | {_number(primary.get('ade'))} | "
        f"{_number(primary.get('fde'))} | {_number(_quantile(split_result, 'ade', 'p90'))} | "
        f"{_number(_quantile(split_result, 'ade', 'p95'))} | "
        f"{_number(primary.get('waypoint_l1'))} | {_number(primary.get('smoothness'))} | "
        f"{_number(primary.get('second_difference_error'))} | {_number(overall.get('ade'))} | "
        f"{_number(worst.get('ade'))} |"
    )


def _comparison_row(name: str, split_result: dict[str, Any]) -> str:
    paired = split_result.get("paired_vs_baseline")
    if isinstance(paired, dict):
        primary_metric = str(paired.get("primary_metric", "ade"))
        delta = paired.get("route_macro_delta", {}).get(primary_metric)
        ci = paired.get("route_bootstrap_ci", {}).get(primary_metric, {})
        if ci:
            ci_text = (
                f"[{_number(ci.get('lower'), signed=True)}, "
                f"{_number(ci.get('upper'), signed=True)}]"
            )
        else:
            ci_text = "—"
        route_fractions = paired.get("route_fractions", {})
        sample_fractions = paired.get("sample_fractions", {})
        return (
            f"| {name} | {_number(delta, signed=True)} | {ci_text} | "
            f"{_percentage(route_fractions.get('improved'))} | "
            f"{_percentage(route_fractions.get('harmed'))} | "
            f"{_percentage(sample_fractions.get('improved'))} | "
            f"{_percentage(sample_fractions.get('harmed'))} |"
        )

    # Legacy results contain a sample-micro ADE delta and improvement fraction.
    return (
        f"| {name} | "
        f"{_number(split_result.get('paired_delta_ade_vs_baseline'), signed=True)} | "
        f"— | — | — | {_percentage(split_result.get('fraction_improved_ade'))} | — |"
    )


def _diagnostic_row(name: str, split_result: dict[str, Any]) -> str:
    return (
        f"| {name} | {_number(split_result.get('mean_gate'))} | "
        f"{_number(split_result.get('mean_raw_residual_l1'))} | "
        f"{_number(split_result.get('mean_applied_residual_l1'))} | "
        f"{_number(split_result.get('mean_latency_ms'))} | "
        f"{_number(split_result.get('p95_latency_ms'))} |"
    )


def render_markdown(results: dict[str, Any]) -> str:
    evidence = results["evidence_level"]
    source = results.get("cache_source", {})
    lines = [
        "# Temporal TransFuser pilot results",
        "",
        f"- Evidence level: `{evidence}`",
        f"- Cache kind: `{source.get('kind', 'unknown')}`",
        f"- Target semantics: `{source.get('target_semantics', 'unspecified')}`",
        f"- Device: `{results['device']}`",
        f"- Evaluation mode: `{results.get('evaluation_mode', 'legacy')}`",
        f"- Test opened: `{results.get('test_opened', 'unknown')}`",
        f"- Selection ID: `{results.get('selection_id', 'legacy')}`",
        f"- Route-safe windows: `{results['dataset_windows']}`",
        "",
    ]
    if evidence == "synthetic_smoke":
        lines.extend(
            [
                "> **SMOKE TEST ONLY.** These numbers validate plumbing and must not be used as research evidence.",
                "",
            ]
        )
    if source.get("target_semantics") == "geometric_path":
        lines.extend(
            [
                "> The released TF++ checkpoint predicts a distance-sampled geometric path, not a time-sampled future ego trajectory.",
                "",
            ]
        )
    report_split = "test" if "test" in results["baseline"] else "val"
    split_title = (
        "Test split (locked final configuration)"
        if report_split == "test"
        else "Validation split (selection only; test unopened)"
    )
    geometric_path = source.get("target_semantics") == "geometric_path"
    ade_label = "Mean checkpoint displacement (m)" if geometric_path else "ADE (m)"
    fde_label = "Last-checkpoint error (m)" if geometric_path else "FDE (m)"
    baseline_split = results["baseline"][report_split]
    route_primary = "route_macro" in baseline_split
    primary_note = (
        "Primary aggregation: equally weighted route macro; each route is one independent cluster."
        if route_primary
        else "Primary aggregation: legacy sample micro (route statistics unavailable)."
    )
    lines.extend(
        [
            f"## {split_title}",
            "",
            primary_note,
            "",
            f"| Method | Trainable params | Routes | {ade_label} | {fde_label} | Route-mean ADE P90 | Route-mean ADE P95 | Point L1 (m) | Smoothness | 2nd-diff error | Sample-micro ADE | Baseline-worst ADE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            _performance_row("Baseline", baseline_split, 0),
        ]
    )
    for variant_name, variant in results["variants"].items():
        lines.append(
            _performance_row(
                variant_name,
                variant["splits"][report_split],
                int(variant["trainable_parameters"]),
            )
        )

    lines.extend(
        [
            "",
            "## Adapter diagnostics",
            "",
            "Latency is the cached adapter path per sample (including device transfer); it excludes frozen TF++ inference.",
            "",
            "| Method | Mean gate | Raw residual L1 | Applied residual L1 | Mean latency (ms) | P95 latency (ms) |",
            "|---|---:|---:|---:|---:|---:|",
            _diagnostic_row("Baseline", baseline_split),
        ]
    )
    for variant_name, variant in results["variants"].items():
        lines.append(
            _diagnostic_row(variant_name, variant["splits"][report_split])
        )

    if results["variants"]:
        lines.extend(
            [
                "",
                "## Paired comparison against frozen baseline",
                "",
                "Negative Δ means lower error than baseline. Bootstrap intervals resample whole routes.",
                "",
                "| Method | Δ route-macro ADE | Route-cluster bootstrap 95% CI | Routes improved | Routes harmed | Windows improved | Windows harmed |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for variant_name, variant in results["variants"].items():
            lines.append(_comparison_row(variant_name, variant["splits"][report_split]))

    lines.extend(
        [
            "",
            f"The worst slice is selected with {report_split} GT and is therefore an oracle diagnostic, not a deployment-time selector.",
            "Offline improvement alone does not establish closed-loop CARLA improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown(results: dict[str, Any], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.write_text(render_markdown(results), encoding="utf-8")
    return destination
