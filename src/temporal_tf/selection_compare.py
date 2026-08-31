"""Validation-only comparison of locked residual-weight study selections.

This module deliberately reads only selection metadata and the validation
metrics already recorded by each child selection.  It never imports a dataset
or calls a finalization function, so comparing candidates cannot open test
records.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import validate_config
from .engine import (
    CHECKPOINT_SELECTION_METRIC,
    SELECTION_MANIFEST,
    STUDY_SELECTION_OWNER_KIND,
    TEST_OPEN_MARKER,
)
from .study import (
    STUDY_MANIFEST,
    STUDY_TEST_OPEN_MARKER,
    _expected_study_owner_id,
    _study_id_payload,
)


CHOICE_JSON = "selection_choice.json"
CHOICE_REPORT = "SELECTION_CHOICE.md"
DEFAULT_PRIMARY_VARIANT = "past_bev"


class SelectionComparisonError(ValueError):
    """Raised when a candidate is not a compatible, unopened selection."""


@dataclass(frozen=True)
class _Candidate:
    study_dir: Path
    manifest_path: Path
    manifest_sha256: str
    study_id: str
    cache_index_sha256: str
    seeds: tuple[int, ...]
    variants: tuple[str, ...]
    variant_specs: Mapping[str, Any]
    configs_by_seed: Mapping[int, Mapping[str, Any]]
    config_except_weight_sha256: str
    residual_weight: float
    validation_ade_by_variant: Mapping[str, tuple[float, ...]]


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(8 << 20):
                digest.update(chunk)
    except OSError as error:
        raise SelectionComparisonError(f"cannot read locked artifact {path}: {error}") from error
    return digest.hexdigest()


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionComparisonError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise SelectionComparisonError(f"{label} must contain a JSON object: {path}")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SelectionComparisonError(f"{label} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise SelectionComparisonError(f"{label} is not a SHA-256 digest") from error
    return value.lower()


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise SelectionComparisonError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SelectionComparisonError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise SelectionComparisonError(f"{label} must be a finite number")
    return number


def _aggregate(values: Sequence[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise SelectionComparisonError("cannot aggregate an empty validation metric list")
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "max": max(numbers),
    }


def _manifest_path(candidate: str | Path) -> Path:
    path = Path(candidate).expanduser()
    if path.is_dir():
        path = path / STUDY_MANIFEST
    path = path.resolve()
    if not path.is_file():
        raise SelectionComparisonError(f"study manifest does not exist: {path}")
    if path.name != STUDY_MANIFEST:
        raise SelectionComparisonError(
            f"candidate file must be named {STUDY_MANIFEST}: {path}"
        )
    return path


def _without_seed(config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(config))
    value.pop("seed", None)
    return value


def _without_residual_weight(config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(config))
    training = value.get("training")
    if not isinstance(training, dict) or "residual_weight" not in training:
        raise SelectionComparisonError("configuration is missing training.residual_weight")
    del training["residual_weight"]
    return value


def _assert_unopened(path: Path, manifest: Mapping[str, Any], *, label: str) -> None:
    if manifest.get("test_opened") is not False:
        raise SelectionComparisonError(f"{label}.test_opened must be false")
    for marker_name in (STUDY_TEST_OPEN_MARKER, TEST_OPEN_MARKER):
        marker = path / marker_name
        if marker.exists():
            raise SelectionComparisonError(f"{label} has a test-open marker: {marker}")


def _validation_ade(
    results: Mapping[str, Any], *, variant: str, seed: int
) -> float:
    try:
        value = results["variants"][variant]["splits"]["val"]["route_macro"]["ade"]
    except (KeyError, TypeError) as error:
        raise SelectionComparisonError(
            f"seed {seed} variant {variant} lacks val.route_macro.ade"
        ) from error
    return _number(value, label=f"seed {seed} {variant} val route-macro ADE")


def _validate_candidate(candidate: str | Path) -> _Candidate:
    manifest_path = _manifest_path(candidate)
    study_dir = manifest_path.parent
    manifest_sha256 = _file_hash(manifest_path)
    manifest = _read_mapping(manifest_path, label="study manifest")

    if manifest.get("schema_version") != 1 or manifest.get("status") != "selection_complete":
        raise SelectionComparisonError(
            f"study is not a completed supported selection: {manifest_path}"
        )
    _assert_unopened(study_dir, manifest, label="study manifest")
    if manifest.get("primary_metric") != "ade":
        raise SelectionComparisonError("study primary_metric must be preregistered as ade")
    if manifest.get("checkpoint_selection_metric") != CHECKPOINT_SELECTION_METRIC:
        raise SelectionComparisonError(
            "study checkpoints were not selected by validation equal-route macro ADE"
        )

    cache_hash = _require_sha256(
        manifest.get("cache_index_sha256"), label="study cache_index_sha256"
    )
    seeds_value = manifest.get("seeds")
    if not isinstance(seeds_value, list) or len(seeds_value) < 3:
        raise SelectionComparisonError("study must record at least three seeds")
    if any(isinstance(seed, bool) for seed in seeds_value):
        raise SelectionComparisonError("study seeds must be integers")
    try:
        seeds = tuple(int(seed) for seed in seeds_value)
    except (TypeError, ValueError) as error:
        raise SelectionComparisonError("study seeds must be integers") from error
    if list(seeds) != seeds_value or len(set(seeds)) != len(seeds):
        raise SelectionComparisonError("study seeds must be unique integers")

    variants_value = manifest.get("variants")
    if (
        not isinstance(variants_value, list)
        or not variants_value
        or not all(isinstance(variant, str) and variant for variant in variants_value)
    ):
        raise SelectionComparisonError("study variants must be a non-empty string list")
    variants = tuple(variants_value)
    if len(set(variants)) != len(variants):
        raise SelectionComparisonError("study variants must be unique")
    variant_specs = manifest.get("variant_specs")
    if not isinstance(variant_specs, dict) or set(variant_specs) != set(variants):
        raise SelectionComparisonError("study variant_specs do not match variants")

    try:
        expected_owner_id = _expected_study_owner_id(manifest)
        expected_study_id = _stable_hash(_study_id_payload(manifest))[:20]
    except (KeyError, TypeError) as error:
        raise SelectionComparisonError("study identity payload is incomplete") from error
    if manifest.get("study_owner_id") != expected_owner_id:
        raise SelectionComparisonError("study parent ownership identity mismatch")
    if manifest.get("study_id") != expected_study_id:
        raise SelectionComparisonError("study manifest identity hash mismatch")
    expected_selection_owner = {
        "kind": STUDY_SELECTION_OWNER_KIND,
        "study_owner_id": expected_owner_id,
    }

    entries = manifest.get("selections")
    if not isinstance(entries, list) or len(entries) != len(seeds):
        raise SelectionComparisonError("study selection count does not match seeds")

    configs_by_seed: dict[int, Mapping[str, Any]] = {}
    validation_values: dict[str, list[float]] = {variant: [] for variant in variants}
    residual_weights: list[float] = []
    base_config_hash = _require_sha256(
        manifest.get("base_config_without_seed_sha256"),
        label="study base_config_without_seed_sha256",
    )

    for seed, entry in zip(seeds, entries):
        if not isinstance(entry, dict) or entry.get("seed") != seed:
            raise SelectionComparisonError("study selection seed ordering mismatch")
        if entry.get("cache_index_sha256") != cache_hash:
            raise SelectionComparisonError(f"seed {seed} locked cache index hash mismatch")
        expected_selection_dir = f"selections/seed_{seed}"
        if entry.get("selection_dir") != expected_selection_dir:
            raise SelectionComparisonError(
                f"seed {seed} selection directory is not canonical"
            )
        selection_dir = study_dir / expected_selection_dir

        manifest_info = entry.get("selection_manifest")
        if not isinstance(manifest_info, dict) or manifest_info.get("path") != SELECTION_MANIFEST:
            raise SelectionComparisonError(f"seed {seed} child manifest path mismatch")
        recorded_manifest_hash = _require_sha256(
            manifest_info.get("sha256"), label=f"seed {seed} child manifest hash"
        )
        child_manifest_path = selection_dir / SELECTION_MANIFEST
        if _file_hash(child_manifest_path) != recorded_manifest_hash:
            raise SelectionComparisonError(f"seed {seed} child manifest hash mismatch")
        child_manifest = _read_mapping(child_manifest_path, label=f"seed {seed} child manifest")
        if child_manifest.get("status") != "selection_complete":
            raise SelectionComparisonError(f"seed {seed} child selection is incomplete")
        _assert_unopened(selection_dir, child_manifest, label=f"seed {seed} child manifest")
        if child_manifest.get("selection_id") != entry.get("selection_id"):
            raise SelectionComparisonError(f"seed {seed} child selection ID mismatch")
        if child_manifest.get("cache_index_sha256") != cache_hash:
            raise SelectionComparisonError(f"seed {seed} child cache index hash mismatch")
        if child_manifest.get("variants") != list(variants):
            raise SelectionComparisonError(f"seed {seed} child variants mismatch")
        if child_manifest.get("variant_specs") != variant_specs:
            raise SelectionComparisonError(f"seed {seed} child variant_specs mismatch")
        if child_manifest.get("checkpoint_selection_metric") != CHECKPOINT_SELECTION_METRIC:
            raise SelectionComparisonError(f"seed {seed} child checkpoint metric mismatch")
        if child_manifest.get("selection_owner") != expected_selection_owner:
            raise SelectionComparisonError(f"seed {seed} child parent ownership mismatch")
        if child_manifest.get("config_sha256") != entry.get("config_sha256"):
            raise SelectionComparisonError(f"seed {seed} child config hash mismatch")

        results_info = entry.get("selection_results")
        if (
            not isinstance(results_info, dict)
            or results_info.get("path") != "results.json"
            or child_manifest.get("results") != "results.json"
        ):
            raise SelectionComparisonError(f"seed {seed} child results path mismatch")
        recorded_results_hash = _require_sha256(
            results_info.get("sha256"), label=f"seed {seed} child results hash"
        )
        results_path = selection_dir / "results.json"
        if _file_hash(results_path) != recorded_results_hash:
            raise SelectionComparisonError(f"seed {seed} child results hash mismatch")
        results = _read_mapping(results_path, label=f"seed {seed} child results")
        if results.get("status") != "completed" or results.get("evaluation_mode") != "selection":
            raise SelectionComparisonError(
                f"seed {seed} results are not completed selection results"
            )
        if results.get("test_opened") is not False:
            raise SelectionComparisonError(f"seed {seed} results.test_opened must be false")
        if results.get("selection_id") != entry.get("selection_id"):
            raise SelectionComparisonError(f"seed {seed} results selection ID mismatch")
        if results.get("selection_owner") != expected_selection_owner:
            raise SelectionComparisonError(f"seed {seed} results parent ownership mismatch")
        if results.get("checkpoint_selection_metric") != CHECKPOINT_SELECTION_METRIC:
            raise SelectionComparisonError(f"seed {seed} results checkpoint metric mismatch")
        result_variants = results.get("variants")
        if not isinstance(result_variants, dict) or set(result_variants) != set(variants):
            raise SelectionComparisonError(f"seed {seed} result variants mismatch")

        config = results.get("configuration")
        if not isinstance(config, dict) or config.get("seed") != seed:
            raise SelectionComparisonError(f"seed {seed} locked configuration mismatch")
        try:
            validate_config(config)
        except (KeyError, TypeError, ValueError) as error:
            raise SelectionComparisonError(
                f"seed {seed} configuration is invalid: {error}"
            ) from error
        config_hash = _stable_hash(config)
        if config_hash != entry.get("config_sha256"):
            raise SelectionComparisonError(f"seed {seed} recorded config hash mismatch")
        if config_hash != child_manifest.get("config_sha256"):
            raise SelectionComparisonError(f"seed {seed} child config hash mismatch")
        if _stable_hash(_without_seed(config)) != base_config_hash:
            raise SelectionComparisonError(
                f"seed {seed} changes configuration fields other than seed"
            )
        configs_by_seed[seed] = config
        residual_weights.append(
            _number(
                config["training"]["residual_weight"],
                label=f"seed {seed} training.residual_weight",
            )
        )

        child_checkpoints = child_manifest.get("checkpoints")
        locked_checkpoints = entry.get("checkpoints")
        if (
            not isinstance(child_checkpoints, dict)
            or not isinstance(locked_checkpoints, dict)
            or set(child_checkpoints) != set(variants)
            or set(locked_checkpoints) != set(variants)
        ):
            raise SelectionComparisonError(f"seed {seed} checkpoint variants mismatch")
        for variant in variants:
            child_checkpoint = child_checkpoints[variant]
            locked_checkpoint = locked_checkpoints[variant]
            if child_checkpoint != locked_checkpoint or not isinstance(child_checkpoint, dict):
                raise SelectionComparisonError(
                    f"seed {seed} checkpoint metadata mismatch for {variant}"
                )
            expected_path = f"checkpoints/{variant}.pt"
            if child_checkpoint.get("path") != expected_path:
                raise SelectionComparisonError(
                    f"seed {seed} checkpoint path mismatch for {variant}"
                )
            checkpoint_hash = _require_sha256(
                child_checkpoint.get("sha256"),
                label=f"seed {seed} {variant} checkpoint hash",
            )
            if _file_hash(selection_dir / expected_path) != checkpoint_hash:
                raise SelectionComparisonError(
                    f"seed {seed} checkpoint hash mismatch for {variant}"
                )

        for variant in variants:
            validation_values[variant].append(
                _validation_ade(results, variant=variant, seed=seed)
            )

    first_weight = residual_weights[0]
    if any(weight != first_weight for weight in residual_weights[1:]):
        raise SelectionComparisonError("training.residual_weight differs across seeds in one study")
    if first_weight < 0:
        raise SelectionComparisonError("training.residual_weight must be non-negative")

    config_without_weight_by_seed = {
        str(seed): _without_residual_weight(configs_by_seed[seed]) for seed in seeds
    }
    config_except_weight_sha256 = _stable_hash(config_without_weight_by_seed)
    return _Candidate(
        study_dir=study_dir,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        study_id=str(manifest["study_id"]),
        cache_index_sha256=cache_hash,
        seeds=seeds,
        variants=variants,
        variant_specs=variant_specs,
        configs_by_seed=configs_by_seed,
        config_except_weight_sha256=config_except_weight_sha256,
        residual_weight=first_weight,
        validation_ade_by_variant={
            variant: tuple(values) for variant, values in validation_values.items()
        },
    )


def _verify_compatible(candidates: Sequence[_Candidate], *, primary_variant: str) -> None:
    reference = candidates[0]
    if primary_variant not in reference.variants:
        raise SelectionComparisonError(
            f"preregistered primary variant is absent: {primary_variant}"
        )
    for candidate in candidates[1:]:
        if candidate.cache_index_sha256 != reference.cache_index_sha256:
            raise SelectionComparisonError("candidate cache index hashes differ")
        if candidate.seeds != reference.seeds:
            raise SelectionComparisonError("candidate seeds differ")
        if candidate.variants != reference.variants:
            raise SelectionComparisonError("candidate variants differ")
        if candidate.variant_specs != reference.variant_specs:
            raise SelectionComparisonError("candidate variant_specs differ")
        if candidate.config_except_weight_sha256 != reference.config_except_weight_sha256:
            raise SelectionComparisonError(
                "candidate configs differ outside training.residual_weight"
            )
        for seed in reference.seeds:
            left = _without_residual_weight(reference.configs_by_seed[seed])
            right = _without_residual_weight(candidate.configs_by_seed[seed])
            if left != right:
                raise SelectionComparisonError(
                    "candidate configs differ outside training.residual_weight"
                )
        if primary_variant not in candidate.variants:
            raise SelectionComparisonError(
                f"preregistered primary variant is absent: {primary_variant}"
            )


def build_selection_choice(
    studies: Sequence[str | Path], *, primary_variant: str = DEFAULT_PRIMARY_VARIANT
) -> dict[str, Any]:
    """Validate and rank two or more unopened study selection directories."""

    if len(studies) < 2:
        raise SelectionComparisonError("at least two study selections are required")
    if not primary_variant:
        raise SelectionComparisonError("primary_variant must be non-empty")
    manifest_paths = [_manifest_path(study) for study in studies]
    if len(set(manifest_paths)) != len(manifest_paths):
        raise SelectionComparisonError("study selections must be distinct")
    candidates = [_validate_candidate(path) for path in manifest_paths]
    _verify_compatible(candidates, primary_variant=primary_variant)

    serialized_candidates: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: str(item.study_dir)):
        primary_values = candidate.validation_ade_by_variant[primary_variant]
        serialized_candidates.append(
            {
                "study_id": candidate.study_id,
                "study_dir": str(candidate.study_dir),
                "study_manifest": str(candidate.manifest_path),
                "study_manifest_sha256": candidate.manifest_sha256,
                "training_residual_weight": candidate.residual_weight,
                "primary_validation_route_macro_ade_by_seed": {
                    str(seed): value
                    for seed, value in zip(candidate.seeds, primary_values)
                },
                "primary_validation_route_macro_ade": _aggregate(primary_values),
                "validation_diagnostics_by_variant": {
                    variant: {
                        "used_for_choice": False,
                        "route_macro_ade_by_seed": {
                            str(seed): value
                            for seed, value in zip(
                                candidate.seeds,
                                candidate.validation_ade_by_variant[variant],
                            )
                        },
                        "route_macro_ade": _aggregate(
                            candidate.validation_ade_by_variant[variant]
                        ),
                    }
                    for variant in candidate.variants
                },
            }
        )

    chosen = min(
        serialized_candidates,
        key=lambda item: (
            float(item["primary_validation_route_macro_ade"]["mean"]),
            float(item["training_residual_weight"]),
            str(item["study_dir"]),
        ),
    )
    reference = candidates[0]
    return {
        "schema_version": 1,
        "status": "choice_complete",
        "comparison_scope": "validation_only",
        "test_data_accessed": False,
        "test_opened": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_variant": primary_variant,
        "primary_metric": "validation_route_macro_ade",
        "choice_rule": (
            "lowest mean validation route-macro ADE across fixed seeds; ties use lower "
            "training.residual_weight, then lexicographically stable resolved study path"
        ),
        "diagnostic_variants_influence_choice": False,
        "compatibility": {
            "cache_index_sha256": reference.cache_index_sha256,
            "seeds": list(reference.seeds),
            "variants": list(reference.variants),
            "variant_specs": reference.variant_specs,
            "checkpoint_selection_metric": dict(CHECKPOINT_SELECTION_METRIC),
            "config_except_residual_weight_sha256": (
                reference.config_except_weight_sha256
            ),
        },
        "candidates": serialized_candidates,
        "chosen": {
            "study_id": chosen["study_id"],
            "study_dir": chosen["study_dir"],
            "study_manifest": chosen["study_manifest"],
            "study_manifest_sha256": chosen["study_manifest_sha256"],
            "training_residual_weight": chosen["training_residual_weight"],
            "primary_validation_route_macro_ade": chosen[
                "primary_validation_route_macro_ade"
            ],
        },
    }


def _markdown(choice: Mapping[str, Any]) -> str:
    def escaped(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "# Validation-only Residual-weight Choice",
        "",
        (
            f"Chosen study: `{choice['chosen']['study_dir']}` (manifest SHA-256 "
            f"`{choice['chosen']['study_manifest_sha256']}`)."
        ),
        "",
        (
            f"Primary selection signal: mean validation route-macro ADE for preregistered "
            f"variant `{choice['primary_variant']}` across seeds "
            f"{choice['compatibility']['seeds']}. Sample SD is reported across seeds."
        ),
        "",
        "No test data was read and every candidate remained test-closed.",
        "",
        "## Primary comparison",
        "",
        "| Study directory | Residual weight | Mean ADE | Sample SD | Min | Max | Chosen |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    chosen_dir = choice["chosen"]["study_dir"]
    for candidate in choice["candidates"]:
        aggregate = candidate["primary_validation_route_macro_ade"]
        lines.append(
            "| "
            + " | ".join(
                (
                    escaped(candidate["study_dir"]),
                    f"{float(candidate['training_residual_weight']):.8g}",
                    f"{float(aggregate['mean']):.6f}",
                    f"{float(aggregate['std']):.6f}",
                    f"{float(aggregate['min']):.6f}",
                    f"{float(aggregate['max']):.6f}",
                    "yes" if candidate["study_dir"] == chosen_dir else "no",
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Validation diagnostics (not used for selection)",
            "",
            "| Study directory | Variant | Mean route-macro ADE | Sample SD |",
            "|---|---|---:|---:|",
        ]
    )
    for candidate in choice["candidates"]:
        for variant, diagnostic in candidate["validation_diagnostics_by_variant"].items():
            aggregate = diagnostic["route_macro_ade"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        escaped(candidate["study_dir"]),
                        escaped(variant),
                        f"{float(aggregate['mean']):.6f}",
                        f"{float(aggregate['std']):.6f}",
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Only the preregistered primary variant influenced the choice. The other variant "
            "means are shown solely as diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_selection_choice(
    studies: Sequence[str | Path],
    *,
    output_dir: str | Path,
    primary_variant: str = DEFAULT_PRIMARY_VARIANT,
) -> dict[str, Any]:
    """Validate candidates, then write a fresh JSON and Markdown choice artifact."""

    choice = build_selection_choice(studies, primary_variant=primary_variant)
    candidate_dirs = [Path(item["study_dir"]) for item in choice["candidates"]]
    output_dir = Path(output_dir).expanduser().resolve()
    if any(
        output_dir == study_dir or study_dir in output_dir.parents
        for study_dir in candidate_dirs
    ):
        raise SelectionComparisonError(
            "choice output may not be placed inside a candidate study directory"
        )
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"choice output must be a fresh directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(
        output_dir / CHOICE_JSON,
        json.dumps(choice, indent=2, ensure_ascii=False) + "\n",
    )
    _write_text_atomic(output_dir / CHOICE_REPORT, _markdown(choice))
    return choice
