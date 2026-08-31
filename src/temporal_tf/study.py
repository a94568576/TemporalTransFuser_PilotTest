"""Hash-locked multi-seed selection and one-shot final-test orchestration."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import validate_config
from .engine import (
    CHECKPOINT_SELECTION_METRIC,
    SELECTION_MANIFEST,
    STUDY_SELECTION_OWNER_KIND,
    TEST_OPEN_MARKER,
    _STUDY_FINALIZE_CAPABILITY,
    finalize_selection,
    resolve_device,
    run_pilot,
)
from .model import VARIANTS


STUDY_MANIFEST = "study_manifest.json"
STUDY_TEST_OPEN_MARKER = "study_test_opened.marker.json"
STUDY_RESULTS = "study_results.json"
STUDY_REPORT = "STUDY_RESULTS.md"
SELECTION_CHOICE = "selection_choice.json"
DEFAULT_STUDY_SEEDS = (17, 29, 43)
DEFAULT_STUDY_VARIANTS = (
    "current_only",
    "current_only_matched",
    "trajectory_only",
    "current_bev",
    "past_bev",
    "shuffled_past_bev",
    "combined",
)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(
        path, json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    )


def _require_fresh_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"study output must be a fresh directory: {path}")


def _validated_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if len(normalized) < 3:
        raise ValueError("a multi-seed study requires at least three seeds")
    if len(set(normalized)) != len(normalized):
        raise ValueError("study seeds must be unique")
    return normalized


def _validated_variants(variants: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(variant) for variant in variants)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("study variants must be non-empty and unique")
    unknown = [variant for variant in normalized if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown study variants: {unknown}; choose from {sorted(VARIANTS)}")
    return normalized


def _config_without_seed(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(config))
    normalized.pop("seed", None)
    return normalized


def _study_owner_payload(
    *,
    cache_index_sha256: str,
    base_config_without_seed_sha256: str,
    seeds: Sequence[int],
    variants: Sequence[str],
    variant_specs: Mapping[str, Any],
    primary_metric: str,
    checkpoint_selection_metric: Mapping[str, Any] = CHECKPOINT_SELECTION_METRIC,
) -> dict[str, Any]:
    """Fields known before child training that bind every child to one parent."""

    return {
        "kind": STUDY_SELECTION_OWNER_KIND,
        "cache_index_sha256": cache_index_sha256,
        "base_config_without_seed_sha256": base_config_without_seed_sha256,
        "seeds": list(seeds),
        "variants": list(variants),
        "variant_specs": dict(variant_specs),
        "primary_metric": primary_metric,
        "checkpoint_selection_metric": dict(checkpoint_selection_metric),
    }


def _expected_study_owner_id(manifest: Mapping[str, Any]) -> str:
    return _stable_hash(
        _study_owner_payload(
            cache_index_sha256=manifest["cache_index_sha256"],
            base_config_without_seed_sha256=manifest["base_config_without_seed_sha256"],
            seeds=manifest["seeds"],
            variants=manifest["variants"],
            variant_specs=manifest["variant_specs"],
            primary_metric=manifest["primary_metric"],
            checkpoint_selection_metric=manifest["checkpoint_selection_metric"],
        )
    )


def _selection_snapshot(
    *,
    study_dir: Path,
    selection_dir: Path,
    seed: int,
    variants: Sequence[str],
    expected_config: Mapping[str, Any],
    cache_index_hash: str,
    selection_owner: Mapping[str, str],
) -> dict[str, Any]:
    manifest_path = selection_dir / SELECTION_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "selection_complete" or manifest.get("test_opened") is not False:
        raise ValueError(f"seed {seed} did not produce a closed selection manifest")
    if tuple(manifest.get("variants", ())) != tuple(variants):
        raise ValueError(f"seed {seed} selection variants differ from the study variants")
    expected_variant_specs = {
        variant: VARIANTS[variant].input_semantics() for variant in variants
    }
    if manifest.get("variant_specs") != expected_variant_specs:
        raise ValueError(f"seed {seed} selection variant semantics mismatch")
    if manifest.get("checkpoint_selection_metric") != CHECKPOINT_SELECTION_METRIC:
        raise ValueError(f"seed {seed} selection checkpoint metric mismatch")
    expected_config_hash = _stable_hash(expected_config)
    if manifest.get("config_sha256") != expected_config_hash:
        raise ValueError(f"seed {seed} selection config hash mismatch")
    if manifest.get("cache_index_sha256") != cache_index_hash:
        raise ValueError(f"seed {seed} selection cache hash mismatch")
    if manifest.get("selection_owner") != dict(selection_owner):
        raise ValueError(f"seed {seed} selection parent ownership mismatch")
    if (selection_dir / TEST_OPEN_MARKER).exists():
        raise ValueError(f"seed {seed} unexpectedly opened the test split during selection")

    if manifest.get("results") != "results.json":
        raise ValueError(f"seed {seed} selection results path is not canonical")
    results_path = selection_dir / "results.json"
    selection_results = json.loads(results_path.read_text(encoding="utf-8"))
    if selection_results.get("selection_owner") != dict(selection_owner):
        raise ValueError(f"seed {seed} locked results parent ownership mismatch")
    if selection_results.get("checkpoint_selection_metric") != CHECKPOINT_SELECTION_METRIC:
        raise ValueError(f"seed {seed} locked results checkpoint metric mismatch")
    if _stable_hash(selection_results.get("configuration")) != expected_config_hash:
        raise ValueError(f"seed {seed} locked results config mismatch")
    checkpoint_names = set(manifest.get("checkpoints", {}))
    if checkpoint_names != set(variants):
        raise ValueError(f"seed {seed} checkpoint variants differ from the study variants")
    checkpoints: dict[str, dict[str, str]] = {}
    for variant in variants:
        checkpoint_info = manifest["checkpoints"][variant]
        expected_checkpoint_path = f"checkpoints/{variant}.pt"
        if checkpoint_info.get("path") != expected_checkpoint_path:
            raise ValueError(f"seed {seed} checkpoint path is not canonical for {variant}")
        checkpoint_path = selection_dir / checkpoint_info["path"]
        actual_hash = _file_hash(checkpoint_path)
        if actual_hash != checkpoint_info.get("sha256"):
            raise ValueError(f"seed {seed} checkpoint hash mismatch for {variant}")
        checkpoints[variant] = {
            "path": checkpoint_info["path"],
            "sha256": actual_hash,
        }
    return {
        "seed": seed,
        "selection_id": manifest["selection_id"],
        "selection_dir": selection_dir.relative_to(study_dir).as_posix(),
        "selection_manifest": {
            "path": SELECTION_MANIFEST,
            "sha256": _file_hash(manifest_path),
        },
        "selection_results": {
            "path": manifest["results"],
            "sha256": _file_hash(results_path),
        },
        "config_sha256": expected_config_hash,
        "cache_index_sha256": cache_index_hash,
        "checkpoints": checkpoints,
    }


def _study_id_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "study_owner_id": manifest["study_owner_id"],
        "cache_index_sha256": manifest["cache_index_sha256"],
        "base_config_without_seed_sha256": manifest["base_config_without_seed_sha256"],
        "seeds": manifest["seeds"],
        "variants": manifest["variants"],
        "variant_specs": manifest["variant_specs"],
        "primary_metric": manifest["primary_metric"],
        "checkpoint_selection_metric": manifest["checkpoint_selection_metric"],
        "selections": [
            {
                "seed": entry["seed"],
                "selection_id": entry["selection_id"],
                "selection_manifest_sha256": entry["selection_manifest"]["sha256"],
                "selection_results_sha256": entry["selection_results"]["sha256"],
                "config_sha256": entry["config_sha256"],
                "cache_index_sha256": entry["cache_index_sha256"],
                "checkpoint_sha256": {
                    variant: entry["checkpoints"][variant]["sha256"]
                    for variant in manifest["variants"]
                },
            }
            for entry in manifest["selections"]
        ],
    }


def run_multiseed_selection(
    *,
    cache_root: str | Path,
    study_dir: str | Path,
    config: Mapping[str, Any],
    seeds: Iterable[int] = DEFAULT_STUDY_SEEDS,
    variants: Iterable[str] = DEFAULT_STUDY_VARIANTS,
    device_name: str = "auto",
    primary_metric: str = "ade",
) -> dict[str, Any]:
    """Run independent train/val selections without ever opening test records."""

    seeds = _validated_seeds(seeds)
    variants = _validated_variants(variants)
    if not primary_metric:
        raise ValueError("primary_metric must be non-empty")
    base_config = deepcopy(dict(config))
    validate_config(base_config)
    cache_root = Path(cache_root)
    study_dir = Path(study_dir)
    _require_fresh_directory(study_dir)
    cache_index_hash = _file_hash(cache_root / "index.json")
    base_config_without_seed_hash = _stable_hash(_config_without_seed(base_config))
    variant_specs = {
        variant: VARIANTS[variant].input_semantics() for variant in variants
    }
    study_owner_id = _stable_hash(
        _study_owner_payload(
            cache_index_sha256=cache_index_hash,
            base_config_without_seed_sha256=base_config_without_seed_hash,
            seeds=seeds,
            variants=variants,
            variant_specs=variant_specs,
            primary_metric=primary_metric,
            checkpoint_selection_metric=CHECKPOINT_SELECTION_METRIC,
        )
    )
    selection_owner = {
        "kind": STUDY_SELECTION_OWNER_KIND,
        "study_owner_id": study_owner_id,
    }
    study_dir.mkdir(parents=True, exist_ok=True)

    selections: list[dict[str, Any]] = []
    for seed in seeds:
        seed_config = deepcopy(base_config)
        seed_config["seed"] = seed
        selection_dir = study_dir / "selections" / f"seed_{seed}"
        run_pilot(
            cache_root=cache_root,
            output_dir=selection_dir,
            config=seed_config,
            device_name=device_name,
            variants=variants,
            evaluation_mode="selection",
            selection_owner=selection_owner,
        )
        selections.append(
            _selection_snapshot(
                study_dir=study_dir,
                selection_dir=selection_dir,
                seed=seed,
                variants=variants,
                expected_config=seed_config,
                cache_index_hash=cache_index_hash,
                selection_owner=selection_owner,
            )
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "selection_complete",
        "test_opened": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study_owner_id": study_owner_id,
        "cache_index_sha256": cache_index_hash,
        "base_config_without_seed_sha256": base_config_without_seed_hash,
        "seeds": list(seeds),
        "variants": list(variants),
        "variant_specs": variant_specs,
        "primary_metric": primary_metric,
        "checkpoint_selection_metric": deepcopy(CHECKPOINT_SELECTION_METRIC),
        "selections": selections,
        "test_protocol": (
            "One final command opens test once for the complete fixed-seed study; "
            "child selections may not be finalized independently."
        ),
    }
    manifest["study_id"] = _stable_hash(_study_id_payload(manifest))[:20]
    _write_json_atomic(study_dir / STUDY_MANIFEST, manifest)
    return manifest


def _locked_selection_dirs(
    *, study_selection: Path, cache_root: Path, manifest: Mapping[str, Any]
) -> list[tuple[int, Path]]:
    if manifest.get("schema_version") != 1 or manifest.get("status") != "selection_complete":
        raise ValueError("study manifest is not a completed supported selection")
    if manifest.get("test_opened") is not False:
        raise RuntimeError("study test split has already been opened")
    marker_path = study_selection / STUDY_TEST_OPEN_MARKER
    if marker_path.exists():
        raise RuntimeError(f"study test split has already been opened: {marker_path}")
    seeds = _validated_seeds(manifest.get("seeds", ()))
    variants = _validated_variants(manifest.get("variants", ()))
    expected_variant_specs = {
        variant: VARIANTS[variant].input_semantics() for variant in variants
    }
    if manifest.get("variant_specs") != expected_variant_specs:
        raise ValueError("study variant semantics do not match this implementation")
    if manifest.get("checkpoint_selection_metric") != CHECKPOINT_SELECTION_METRIC:
        raise ValueError("study checkpoint metric does not match this implementation")
    expected_owner_id = _expected_study_owner_id(manifest)
    if manifest.get("study_owner_id") != expected_owner_id:
        raise ValueError("study parent ownership identity mismatch")
    expected_selection_owner = {
        "kind": STUDY_SELECTION_OWNER_KIND,
        "study_owner_id": expected_owner_id,
    }
    if _stable_hash(_study_id_payload(manifest))[:20] != manifest.get("study_id"):
        raise ValueError("study manifest identity hash mismatch")
    actual_cache_hash = _file_hash(cache_root / "index.json")
    if actual_cache_hash != manifest.get("cache_index_sha256"):
        raise ValueError("study final cache index hash mismatch")

    entries = manifest.get("selections")
    if not isinstance(entries, list) or len(entries) != len(seeds):
        raise ValueError("study manifest selection count does not match its seeds")
    locked: list[tuple[int, Path]] = []
    common_config_without_seed_hash = manifest.get("base_config_without_seed_sha256")
    for seed, entry in zip(seeds, entries):
        if not isinstance(entry, Mapping) or int(entry.get("seed", -1)) != seed:
            raise ValueError("study selection seed ordering mismatch")
        expected_relative_dir = f"selections/seed_{seed}"
        if entry.get("selection_dir") != expected_relative_dir:
            raise ValueError(f"seed {seed} selection directory is not canonical")
        selection_dir = study_selection / expected_relative_dir
        manifest_info = entry.get("selection_manifest", {})
        if manifest_info.get("path") != SELECTION_MANIFEST:
            raise ValueError(f"seed {seed} child manifest path mismatch")
        child_manifest_path = selection_dir / SELECTION_MANIFEST
        if _file_hash(child_manifest_path) != manifest_info.get("sha256"):
            raise ValueError(f"seed {seed} selection manifest hash mismatch")
        child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
        if child_manifest.get("status") != "selection_complete" or child_manifest.get("test_opened") is not False:
            raise ValueError(f"seed {seed} child selection is not closed")
        if (selection_dir / TEST_OPEN_MARKER).exists():
            raise ValueError(f"seed {seed} child selection was already finalized")
        if child_manifest.get("selection_id") != entry.get("selection_id"):
            raise ValueError(f"seed {seed} child selection ID mismatch")
        if tuple(child_manifest.get("variants", ())) != variants:
            raise ValueError(f"seed {seed} child variant list mismatch")
        if child_manifest.get("variant_specs") != expected_variant_specs:
            raise ValueError(f"seed {seed} child variant semantics mismatch")
        if child_manifest.get("checkpoint_selection_metric") != CHECKPOINT_SELECTION_METRIC:
            raise ValueError(f"seed {seed} child checkpoint metric mismatch")
        if child_manifest.get("selection_owner") != expected_selection_owner:
            raise ValueError(f"seed {seed} child parent ownership mismatch")
        if child_manifest.get("cache_index_sha256") != actual_cache_hash:
            raise ValueError(f"seed {seed} child cache index hash mismatch")
        if child_manifest.get("config_sha256") != entry.get("config_sha256"):
            raise ValueError(f"seed {seed} child config hash mismatch")

        results_info = entry.get("selection_results", {})
        if results_info.get("path") != "results.json" or child_manifest.get("results") != "results.json":
            raise ValueError(f"seed {seed} child results path mismatch")
        child_results_path = selection_dir / results_info["path"]
        if _file_hash(child_results_path) != results_info.get("sha256"):
            raise ValueError(f"seed {seed} child results hash mismatch")
        child_results = json.loads(child_results_path.read_text(encoding="utf-8"))
        if child_results.get("selection_owner") != expected_selection_owner:
            raise ValueError(f"seed {seed} child results parent ownership mismatch")
        if child_results.get("checkpoint_selection_metric") != CHECKPOINT_SELECTION_METRIC:
            raise ValueError(f"seed {seed} child results checkpoint metric mismatch")
        child_config = child_results.get("configuration")
        if not isinstance(child_config, dict) or int(child_config.get("seed", -1)) != seed:
            raise ValueError(f"seed {seed} locked configuration mismatch")
        validate_config(child_config)
        if _stable_hash(child_config) != entry.get("config_sha256"):
            raise ValueError(f"seed {seed} locked configuration hash mismatch")
        if _stable_hash(_config_without_seed(child_config)) != common_config_without_seed_hash:
            raise ValueError(f"seed {seed} changes configuration fields other than seed")

        child_checkpoints = child_manifest.get("checkpoints", {})
        locked_checkpoints = entry.get("checkpoints", {})
        if set(child_checkpoints) != set(variants) or set(locked_checkpoints) != set(variants):
            raise ValueError(f"seed {seed} checkpoint variant set mismatch")
        for variant in variants:
            child_info = child_checkpoints[variant]
            locked_info = locked_checkpoints[variant]
            if child_info != locked_info:
                raise ValueError(f"seed {seed} locked checkpoint metadata mismatch for {variant}")
            if child_info.get("path") != f"checkpoints/{variant}.pt":
                raise ValueError(f"seed {seed} checkpoint path is not canonical for {variant}")
            checkpoint_path = selection_dir / child_info["path"]
            if _file_hash(checkpoint_path) != child_info.get("sha256"):
                raise ValueError(f"seed {seed} checkpoint hash mismatch for {variant}")
        locked.append((seed, selection_dir))
    return locked


def _choice_without_creation_time(choice: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(choice))
    normalized.pop("created_at", None)
    return normalized


def resolve_study_selection_choice(
    selection_choice: str | Path,
    *,
    requested_study_selection: str | Path | None = None,
) -> tuple[Path, dict[str, Any], str]:
    """Recompute a validation-only choice and resolve its hash-locked winner.

    Re-running the comparator validation here prevents a manually substituted
    or edited ``chosen`` entry from opening a non-selected residual weight.
    """

    choice_path = Path(selection_choice).expanduser().resolve()
    if choice_path.name != SELECTION_CHOICE or not choice_path.is_file():
        raise ValueError(f"selection choice must be an existing {SELECTION_CHOICE}: {choice_path}")
    try:
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read selection choice {choice_path}: {error}") from error
    if not isinstance(choice, dict):
        raise ValueError("selection choice must contain a JSON object")
    if (
        choice.get("schema_version") != 1
        or choice.get("status") != "choice_complete"
        or choice.get("comparison_scope") != "validation_only"
        or choice.get("test_data_accessed") is not False
        or choice.get("test_opened") is not False
    ):
        raise ValueError("selection choice is not a completed validation-only closed choice")
    candidates = choice.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("selection choice must contain at least two candidates")
    candidate_dirs: list[Path] = []
    for offset, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("study_dir"), str):
            raise ValueError(f"selection choice candidate {offset} has no study_dir")
        candidate_dirs.append(Path(candidate["study_dir"]))
    primary_variant = choice.get("primary_variant")
    if not isinstance(primary_variant, str) or not primary_variant:
        raise ValueError("selection choice has no primary_variant")

    # Local import avoids a module-load cycle: selection_compare imports the
    # study constants and identity helper used above.
    from .selection_compare import SelectionComparisonError, build_selection_choice

    try:
        recomputed = build_selection_choice(
            candidate_dirs,
            primary_variant=primary_variant,
        )
    except SelectionComparisonError as error:
        raise ValueError(f"selection choice revalidation failed: {error}") from error
    if _choice_without_creation_time(choice) != _choice_without_creation_time(recomputed):
        raise ValueError("selection choice content or chosen study does not match recomputation")

    chosen = choice.get("chosen")
    if not isinstance(chosen, Mapping) or not isinstance(chosen.get("study_dir"), str):
        raise ValueError("selection choice has no chosen study")
    chosen_dir = Path(chosen["study_dir"]).resolve()
    chosen_manifest = chosen_dir / STUDY_MANIFEST
    recorded_manifest = Path(str(chosen.get("study_manifest", ""))).resolve()
    if recorded_manifest != chosen_manifest.resolve():
        raise ValueError("selection choice chosen manifest path is not canonical")
    actual_manifest_hash = _file_hash(chosen_manifest)
    if actual_manifest_hash != chosen.get("study_manifest_sha256"):
        raise ValueError("selection choice chosen manifest SHA256 mismatch")
    if requested_study_selection is not None:
        requested = Path(requested_study_selection).expanduser().resolve()
        if requested != chosen_dir:
            raise ValueError("--study-selection does not match the comparator-chosen study")
    return chosen_dir, choice, _file_hash(choice_path)


def _aggregate(values: Sequence[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("cannot aggregate an empty value list")
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "max": max(numbers),
    }


def _optional_aggregate(values: Sequence[float | None]) -> dict[str, Any]:
    if all(value is None for value in values):
        return {"available": False, "reason": "not_recorded_by_child_engine"}
    if any(value is None for value in values):
        raise ValueError("latency is present for only a subset of study seeds")
    result: dict[str, Any] = {"available": True}
    result.update(_aggregate([float(value) for value in values if value is not None]))
    return result


def _latency_ms(split_result: Mapping[str, Any]) -> float | None:
    for name in ("mean_latency_ms", "mean_inference_latency_ms", "latency_ms"):
        if name in split_result:
            return float(split_result[name])
    return None


def _method_split(result: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    if method == "baseline":
        return result["baseline"]["test"]
    return result["variants"][method]["splits"]["test"]


def aggregate_study_results(
    seed_results: Mapping[str, Mapping[str, Any]],
    *,
    variants: Sequence[str],
    primary_metric: str,
) -> dict[str, Any]:
    """Aggregate test route-macro and diagnostics across fixed random seeds."""

    results = list(seed_results.values())
    if len(results) < 3:
        raise ValueError("at least three seed results are required for study aggregation")
    methods = ("baseline", *variants)
    aggregates: dict[str, Any] = {}
    for method in methods:
        splits = [_method_split(result, method) for result in results]
        route_values = [float(split["route_macro"][primary_metric]) for split in splits]
        method_result: dict[str, Any] = {
            "route_macro_primary": _aggregate(route_values),
            "mean_gate": _aggregate([float(split["mean_gate"]) for split in splits]),
            "mean_raw_residual_l1": _aggregate(
                [float(split["mean_raw_residual_l1"]) for split in splits]
            ),
            "mean_applied_residual_l1": _aggregate(
                [float(split["mean_applied_residual_l1"]) for split in splits]
            ),
            "latency_ms": _optional_aggregate([_latency_ms(split) for split in splits]),
        }
        if method == "baseline":
            method_result["harm_improvement"] = None
        else:
            paired = [split["paired_vs_baseline"] for split in splits]
            method_result["harm_improvement"] = {
                "route_improved_fraction": _aggregate(
                    [float(item["route_fractions"]["improved"]) for item in paired]
                ),
                "route_harmed_fraction": _aggregate(
                    [float(item["route_fractions"]["harmed"]) for item in paired]
                ),
                "sample_improved_fraction": _aggregate(
                    [float(item["sample_fractions"]["improved"]) for item in paired]
                ),
                "sample_harmed_fraction": _aggregate(
                    [float(item["sample_fractions"]["harmed"]) for item in paired]
                ),
            }
        aggregates[method] = method_result
    return aggregates


def _format_mean_std(summary: Mapping[str, Any] | None, *, digits: int = 4) -> str:
    if not summary or summary.get("available") is False:
        return "n/a"
    return f"{float(summary['mean']):.{digits}f} ± {float(summary['std']):.{digits}f}"


def write_study_markdown(results: Mapping[str, Any], path: str | Path) -> Path:
    """Write a compact, explicit account of the single study test-open event."""

    path = Path(path)
    metric = results["primary_metric"]
    checkpoint_metric = results["checkpoint_selection_metric"]
    lines = [
        "# Multi-seed Study Results",
        "",
        (
            f"Study `{results['study_id']}` finalized seeds {results['seeds']} in one permanent "
            "test-open event. All checkpoints were selected on train/validation before this command; "
            "the final command performed no retraining."
        ),
        "",
        (
            "Checkpoint epoch selection: validation "
            f"`{checkpoint_metric['aggregation']}` `{checkpoint_metric['metric']}`."
        ),
        "",
        f"Primary metric: route-macro `{metric}`. Standard deviations are sample SD across seeds.",
        "",
        f"| Method | Route-macro {metric} | Min–max | Route improved | Route harmed | Gate | Applied residual L1 | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, aggregate in results["method_aggregates"].items():
        primary = aggregate["route_macro_primary"]
        harm = aggregate["harm_improvement"]
        improved = "—" if harm is None else _format_mean_std(harm["route_improved_fraction"])
        harmed = "—" if harm is None else _format_mean_std(harm["route_harmed_fraction"])
        lines.append(
            "| "
            + " | ".join(
                (
                    method,
                    _format_mean_std(primary),
                    f"{primary['min']:.4f}–{primary['max']:.4f}",
                    improved,
                    harmed,
                    _format_mean_std(aggregate["mean_gate"]),
                    _format_mean_std(aggregate["mean_applied_residual_l1"]),
                    _format_mean_std(aggregate["latency_ms"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The study-level marker is permanent. A failed child evaluation still consumes this "
            "single test-open event and the study cannot be finalized again.",
            "",
        ]
    )
    _write_text_atomic(path, "\n".join(lines))
    return path


def finalize_multiseed_study(
    *,
    study_selection: str | Path | None,
    cache_root: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    selection_choice: str | Path | None = None,
) -> dict[str, Any]:
    """Open test once for all locked seeds, then evaluate without retraining."""

    choice_provenance: dict[str, Any] | None = None
    if selection_choice is not None:
        chosen_dir, choice, choice_hash = resolve_study_selection_choice(
            selection_choice,
            requested_study_selection=study_selection,
        )
        study_selection = chosen_dir
        choice_provenance = {
            "path": str(Path(selection_choice).expanduser().resolve()),
            "sha256": choice_hash,
            "chosen_study_id": choice["chosen"]["study_id"],
            "chosen_study_manifest_sha256": choice["chosen"]["study_manifest_sha256"],
            "training_residual_weight": choice["chosen"]["training_residual_weight"],
        }
    if study_selection is None:
        raise ValueError("study_selection is required unless selection_choice resolves it")
    study_selection = Path(study_selection).expanduser().resolve()
    cache_root = Path(cache_root)
    output_dir = Path(output_dir)
    manifest_path = study_selection / STUDY_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    locked_selections = _locked_selection_dirs(
        study_selection=study_selection,
        cache_root=cache_root,
        manifest=manifest,
    )
    _require_fresh_directory(output_dir)
    # Reject an unusable device before permanently consuming the study event.
    resolve_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_path = study_selection / STUDY_TEST_OPEN_MARKER
    opened_at = datetime.now(timezone.utc).isoformat()
    marker = {
        "study_id": manifest["study_id"],
        "opened_at": opened_at,
        "cache_index_sha256": manifest["cache_index_sha256"],
        "checkpoint_selection_metric": manifest["checkpoint_selection_metric"],
        "seeds": manifest["seeds"],
        "variants": manifest["variants"],
        "final_output": str(output_dir.resolve()),
        "single_test_open_event": True,
    }
    if choice_provenance is not None:
        marker["selection_choice"] = choice_provenance
    # Exclusive creation is the permanent study-level gate.  It deliberately
    # precedes every child finalize_selection call and therefore every test load.
    with marker_path.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    manifest["test_opened"] = True
    manifest["test_opened_at"] = opened_at
    manifest["final_output"] = marker["final_output"]
    if choice_provenance is not None:
        manifest["selection_choice"] = choice_provenance
    _write_json_atomic(manifest_path, manifest)

    seed_results: dict[str, dict[str, Any]] = {}
    elapsed_by_seed: dict[str, float] = {}
    for selection_index, (seed, selection_dir) in enumerate(locked_selections):
        started = time.perf_counter()
        child_result = finalize_selection(
            selection_dir=selection_dir,
            cache_root=cache_root,
            output_dir=output_dir / "seeds" / f"seed_{seed}",
            config=None,
            device_name=device_name,
            _study_owner_id=manifest["study_owner_id"],
            _study_finalize_capability=_STUDY_FINALIZE_CAPABILITY,
        )
        elapsed_by_seed[str(seed)] = time.perf_counter() - started
        expected_selection = manifest["selections"][selection_index]
        if child_result.get("selection_id") != expected_selection["selection_id"]:
            raise ValueError(f"seed {seed} final result selection ID mismatch")
        expected_selection_owner = {
            "kind": STUDY_SELECTION_OWNER_KIND,
            "study_owner_id": manifest["study_owner_id"],
        }
        if child_result.get("selection_owner") != expected_selection_owner:
            raise ValueError(f"seed {seed} final result parent ownership mismatch")
        if child_result.get("checkpoint_selection_metric") != CHECKPOINT_SELECTION_METRIC:
            raise ValueError(f"seed {seed} final result checkpoint metric mismatch")
        result_config = child_result.get("configuration")
        if not isinstance(result_config, dict) or int(result_config.get("seed", -1)) != seed:
            raise ValueError(f"seed {seed} final result configuration mismatch")
        if _stable_hash(result_config) != expected_selection["config_sha256"]:
            raise ValueError(f"seed {seed} final result config hash mismatch")
        if tuple(child_result.get("variants", {})) != tuple(manifest["variants"]):
            raise ValueError(f"seed {seed} final result variant order mismatch")
        seed_results[str(seed)] = child_result

    study_results: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "evaluation_mode": "final_multiseed",
        "single_test_open_event": True,
        "study_id": manifest["study_id"],
        "study_selection": str(study_selection.resolve()),
        "test_opened_at": opened_at,
        "cache_index_sha256": manifest["cache_index_sha256"],
        "checkpoint_selection_metric": manifest["checkpoint_selection_metric"],
        "seeds": manifest["seeds"],
        "variants": manifest["variants"],
        "primary_metric": manifest["primary_metric"],
        "seed_results": seed_results,
        "finalization_elapsed_seconds_by_seed": elapsed_by_seed,
        "method_aggregates": aggregate_study_results(
            seed_results,
            variants=manifest["variants"],
            primary_metric=manifest["primary_metric"],
        ),
    }
    if choice_provenance is not None:
        study_results["selection_choice"] = choice_provenance
    _write_json_atomic(output_dir / STUDY_RESULTS, study_results)
    write_study_markdown(study_results, output_dir / STUDY_REPORT)
    return study_results
