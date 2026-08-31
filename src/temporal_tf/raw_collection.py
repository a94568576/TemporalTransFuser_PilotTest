"""Integrity audit for raw CARLA/Leaderboard data-agent collections.

The audit deliberately treats a successful Leaderboard process exit and a usable
sensor collection as separate conditions.  A collection is accepted only when
both the result metadata and the saved frame streams are internally consistent.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
from typing import Any, Iterable


REQUIRED_MODALITIES = ("rgb", "lidar", "measurements")
_SCENARIO_SUFFIX = re.compile(r"^(?P<label>.+?)_\d+_\d+(?:_(?:rep|retry)\d+)?$")
_COLLECTION_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TFPP_REJECTED_ROUTE_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error_text(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _scenario_label(result_root: Path) -> str:
    match = _SCENARIO_SUFFIX.fullmatch(result_root.name)
    return match.group("label") if match else result_root.name


def load_collection_root_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load an immutable logical-key to collection-root manifest.

    Each non-comment line is ``LOGICAL_KEY ROOT``.  Quoting follows shell
    syntax and relative roots are resolved against the manifest directory.  A
    replacement is explicit by changing only ROOT while retaining LOGICAL_KEY;
    its directory basename must be either the key or ``KEY_*``.
    """

    manifest_path = Path(path).expanduser().resolve()
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise ValueError(f"cannot read collection root manifest {manifest_path}: {_error_text(exc)}") from exc

    entries: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    seen_roots: set[Path] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            fields = shlex.split(raw_line, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"{manifest_path}:{line_number}: {exc}") from exc
        if len(fields) != 2:
            raise ValueError(
                f"{manifest_path}:{line_number}: expected LOGICAL_KEY ROOT, found {len(fields)} fields"
            )
        key, root_text = fields
        if not _COLLECTION_KEY.fullmatch(key):
            raise ValueError(f"{manifest_path}:{line_number}: invalid logical key {key!r}")
        root = Path(root_text).expanduser()
        if not root.is_absolute():
            root = manifest_path.parent / root
        root = root.resolve()
        if key in seen_keys:
            raise ValueError(f"{manifest_path}:{line_number}: duplicate logical key {key!r}")
        if root in seen_roots:
            raise ValueError(f"{manifest_path}:{line_number}: duplicate collection root {root}")
        if root.name != key and not root.name.startswith(f"{key}_"):
            raise ValueError(
                f"{manifest_path}:{line_number}: root basename {root.name!r} must be {key!r} "
                f"or start with {key + '_'!r} for an explicit retry/replacement"
            )
        seen_keys.add(key)
        seen_roots.add(root)
        entries.append({"key": key, "root": str(root)})
    if not entries:
        raise ValueError(f"collection root manifest has no entries: {manifest_path}")
    return entries


def assert_exact_identity_set(
    expected: Iterable[str], actual: Iterable[str], *, stage: str
) -> None:
    """Fail closed when a loader or cache silently omits/substitutes a route."""

    expected_set = set(expected)
    actual_set = set(actual)
    if expected_set == actual_set:
        return
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    raise RuntimeError(
        f"{stage} route-set mismatch: expected={len(expected_set)} actual={len(actual_set)}; "
        f"missing={missing}; unexpected={unexpected}"
    )


def assert_collection_key_matches_route(key: str, route_directory: str | Path) -> None:
    """Bind a logical manifest key to the route seed/index encoded on disk."""

    match = re.search(r"_(\d+)_(\d+)$", key)
    if match is None:
        raise ValueError(
            f"collection key {key!r} must end in _SEED_ROUTE_INDEX for identity binding"
        )
    route_name = Path(route_directory).name
    expected_fragment = f"_{match.group(1)}_{match.group(2)}_route"
    if expected_fragment not in route_name:
        raise RuntimeError(
            f"collection key {key!r} expects route fragment {expected_fragment!r}, "
            f"but found directory {route_name!r}"
        )


def evaluate_tfpp_route_acceptance(route_directory: str | Path) -> dict[str, Any]:
    """Mirror the route filter in upstream ``team_code/data.py::CARLA_Data``.

    Upstream accepts a sub-100 composed score only when every *reported*
    infraction is a minimum-speed infraction.  A collision therefore rejects a
    route even if outer ``result.json`` says Finished and 100% route completion.
    """

    route = Path(route_directory).expanduser().resolve()
    results_path = route / "results.json.gz"
    reasons: list[str] = []
    if route.name.startswith("FAILED_"):
        reasons.append("route directory starts with FAILED_")

    report: dict[str, Any] = {
        "accepted": False,
        "policy": "local_upstream_CARLA_Data_route_filter",
        "route_directory": str(route),
        "results_path": str(results_path),
        "results_sha256": None,
        "status": None,
        "score_composed": None,
        "num_infractions": None,
        "min_speed_infraction_count": None,
        "min_speed_only_exception": None,
        "acceptance_basis": None,
        "reasons": reasons,
    }
    if not results_path.is_file():
        reasons.append("missing results.json.gz")
        return report

    try:
        report["results_sha256"] = _sha256_file(results_path)
        with gzip.open(results_path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    except Exception as exc:
        reasons.append(f"malformed results.json.gz: {_error_text(exc)}")
        return report
    if not isinstance(payload, dict):
        reasons.append("results.json.gz must contain an object")
        return report

    if "status" not in payload:
        reasons.append("results.json.gz is missing status")
        status = None
    else:
        status = payload.get("status")
    report["status"] = status

    scores = payload.get("scores")
    score_composed = scores.get("score_composed") if isinstance(scores, dict) else None
    report["score_composed"] = score_composed
    valid_score = (
        isinstance(score_composed, (int, float))
        and not isinstance(score_composed, bool)
        and math.isfinite(float(score_composed))
    )
    if not valid_score:
        reasons.append(f"score_composed must be a finite number, got {score_composed!r}")

    if status in TFPP_REJECTED_ROUTE_STATUSES:
        reasons.append(f"upstream rejects route status {status!r}")

    if valid_score and float(score_composed) < 100.0:
        num_infractions = payload.get("num_infractions")
        infractions = payload.get("infractions")
        min_speed = (
            infractions.get("min_speed_infractions") if isinstance(infractions, dict) else None
        )
        report["num_infractions"] = num_infractions
        report["min_speed_infraction_count"] = len(min_speed) if isinstance(min_speed, list) else None
        valid_infraction_fields = (
            isinstance(num_infractions, int)
            and not isinstance(num_infractions, bool)
            and isinstance(min_speed, list)
        )
        if not valid_infraction_fields:
            reasons.append(
                "sub-100 result requires integer num_infractions and list "
                "infractions.min_speed_infractions"
            )
        else:
            min_speed_only = num_infractions == len(min_speed)
            report["min_speed_only_exception"] = min_speed_only
            if not min_speed_only:
                reasons.append(
                    "score_composed is below 100 and not all reported infractions are "
                    f"minimum-speed infractions ({num_infractions} total vs {len(min_speed)} min-speed)"
                )

    report["accepted"] = not reasons
    if report["accepted"]:
        report["acceptance_basis"] = (
            "perfect_composed_score"
            if float(score_composed) >= 100.0
            else "sub_100_min_speed_only_exception"
        )
    return report


def _modality_summary(directory: Path) -> dict[str, Any]:
    try:
        files = sorted(path for path in directory.iterdir() if path.is_file())
    except Exception as exc:
        return {
            "count": 0,
            "stems": [],
            "numeric_stems": False,
            "strictly_contiguous": False,
            "first_stem": None,
            "last_stem": None,
            "error": _error_text(exc),
        }

    # Measurements use names such as ``0000.json.gz``.  The frame identifier is
    # therefore the portion before the first dot, not pathlib's final suffix.
    stems = [path.name.split(".", maxsplit=1)[0] for path in files]
    numeric_stems = bool(stems) and all(stem.isdigit() for stem in stems)
    unique_stems = len(set(stems)) == len(stems)
    frame_numbers = sorted(int(stem) for stem in stems) if numeric_stems else []
    contiguous = (
        numeric_stems
        and unique_stems
        and all(right - left == 1 for left, right in zip(frame_numbers, frame_numbers[1:]))
    )
    return {
        "count": len(files),
        "stems": sorted(stems),
        "numeric_stems": numeric_stems,
        "unique_stems": unique_stems,
        "strictly_contiguous": contiguous,
        "first_stem": min(stems) if stems else None,
        "last_stem": max(stems) if stems else None,
        "error": None,
    }


def _empty_modality_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "stems": [],
        "numeric_stems": False,
        "unique_stems": True,
        "strictly_contiguous": False,
        "first_stem": None,
        "last_stem": None,
        "error": "missing directory",
    }


def _route_files(route_directories: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for route_directory in route_directories:
        try:
            files.extend(path for path in route_directory.rglob("*") if path.is_file())
        except Exception:
            # Directory traversal failures are reported by the route audit.  Size
            # accounting remains best-effort so a malformed collection is still
            # rendered as JSON rather than raising from the reporter.
            continue
    return files


def _unique_bytes(paths: Iterable[Path]) -> int:
    total = 0
    seen: set[Path] = set()
    for path in paths:
        try:
            canonical = path.resolve()
            if canonical in seen:
                continue
            seen.add(canonical)
            total += canonical.stat().st_size
        except Exception:
            continue
    return total


def _empty_global() -> dict[str, Any]:
    return {
        "status": None,
        "score_composed": None,
        "score_route": None,
        "score_penalty": None,
        "exceptions": None,
    }


def _parse_global_result(payload: Any, errors: list[str]) -> tuple[Any, dict[str, Any], Any]:
    if not isinstance(payload, dict):
        errors.append("result JSON must contain an object")
        return None, _empty_global(), None

    entry_status = payload.get("entry_status")
    if entry_status != "Finished":
        errors.append(f"entry_status must be 'Finished', got {entry_status!r}")

    checkpoint = payload.get("_checkpoint")
    if not isinstance(checkpoint, dict):
        errors.append("missing or invalid _checkpoint object")
        return entry_status, _empty_global(), None

    global_record = checkpoint.get("global_record")
    if not isinstance(global_record, dict):
        errors.append("missing or invalid _checkpoint.global_record object")
        return entry_status, _empty_global(), checkpoint.get("records")

    scores = global_record.get("scores_mean")
    if not isinstance(scores, dict):
        errors.append("missing or invalid global_record.scores_mean object")
        scores = {}

    global_summary = {
        "status": global_record.get("status"),
        "score_composed": scores.get("score_composed"),
        "score_route": scores.get("score_route"),
        "score_penalty": scores.get("score_penalty"),
        "exceptions": None,
    }

    if not isinstance(global_summary["status"], str) or not global_summary["status"]:
        errors.append(f"global result status must be a non-empty string, got {global_summary['status']!r}")

    for field in ("score_composed", "score_route", "score_penalty"):
        value = global_summary[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            errors.append(f"global {field} must be a finite number, got {value!r}")

    route_score = global_summary["score_route"]
    if isinstance(route_score, bool) or not isinstance(route_score, (int, float)):
        errors.append(f"global route completion must be numeric 100%, got {route_score!r}")
    elif not math.isfinite(float(route_score)) or not math.isclose(
        float(route_score), 100.0, rel_tol=0.0, abs_tol=1e-6
    ):
        errors.append(f"global route completion must be 100%, got {route_score!r}")

    meta = global_record.get("meta")
    if not isinstance(meta, dict) or "exceptions" not in meta:
        errors.append("missing global_record.meta.exceptions")
    else:
        exceptions = meta.get("exceptions")
        global_summary["exceptions"] = exceptions
        if exceptions not in ([], {}, "", None):
            errors.append(f"global result contains exceptions: {exceptions!r}")

    records = checkpoint.get("records")
    if not isinstance(records, list):
        errors.append("missing or invalid _checkpoint.records list")
    else:
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(f"checkpoint record {index} is not an object")
                continue
            record_meta = record.get("meta")
            if isinstance(record_meta, dict):
                for key in ("exception", "exceptions"):
                    value = record_meta.get(key)
                    if value not in (None, "", [], {}):
                        errors.append(f"checkpoint record {index} contains {key}: {value!r}")

    return entry_status, global_summary, records


def _metadata_match(records: Any, route_name: str | None, errors: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checked": False,
        "field": None,
        "value": None,
        "matches": None,
    }
    if route_name is None or not isinstance(records, list):
        return result
    if len(records) != 1:
        errors.append(f"expected exactly one checkpoint record, found {len(records)}")
        return result
    if not isinstance(records[0], dict):
        return result

    # TF++ currently emits ``timestamp``.  ``name`` and ``route_name`` are
    # accepted for compatibility with nearby Leaderboard versions.
    for field in ("timestamp", "name", "route_name"):
        value = records[0].get(field)
        if isinstance(value, str) and value:
            matches = route_name == value or route_name.endswith(value) or value.endswith(route_name)
            result.update({"checked": True, "field": field, "value": value, "matches": matches})
            if not matches:
                errors.append(
                    f"result record {field} {value!r} does not match route directory {route_name!r}"
                )
            return result
    return result


def _audit_result(result_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        sha256 = _sha256_file(result_path)
    except Exception as exc:
        sha256 = None
        errors.append(f"cannot hash result.json: {_error_text(exc)}")

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        payload = None
        errors.append(f"malformed result.json: {_error_text(exc)}")

    entry_status, global_summary, records = _parse_global_result(payload, errors)

    result_root = result_path.parent
    try:
        route_directories = sorted(
            child.resolve()
            for child in result_root.iterdir()
            if child.is_dir() and (child / "measurements").is_dir()
        )
    except Exception as exc:
        route_directories = []
        errors.append(f"cannot enumerate result root: {_error_text(exc)}")

    if len(route_directories) != 1:
        errors.append(
            "result root must contain exactly one route directory with measurements; "
            f"found {len(route_directories)}"
        )

    route_directory = route_directories[0] if len(route_directories) == 1 else None
    route_name = route_directory.name if route_directory is not None else None
    tfpp_loader_acceptance = (
        evaluate_tfpp_route_acceptance(route_directory)
        if route_directory is not None
        else {
            "accepted": False,
            "policy": "local_upstream_CARLA_Data_route_filter",
            "route_directory": None,
            "results_path": None,
            "results_sha256": None,
            "status": None,
            "score_composed": None,
            "num_infractions": None,
            "min_speed_infraction_count": None,
            "min_speed_only_exception": None,
            "acceptance_basis": None,
            "reasons": ["exactly one route directory is required before loader acceptance"],
        }
    )
    if not tfpp_loader_acceptance["accepted"]:
        errors.append(
            "TF++ CARLA_Data would reject route: "
            + "; ".join(tfpp_loader_acceptance["reasons"])
        )
    modality_summaries: dict[str, dict[str, Any]] = {}
    aux_summaries: dict[str, dict[str, Any]] = {}
    frame_count = 0

    if route_directory is not None:
        for modality in REQUIRED_MODALITIES:
            directory = route_directory / modality
            summary = _modality_summary(directory) if directory.is_dir() else _empty_modality_summary()
            modality_summaries[modality] = summary
            if summary["error"] is not None:
                errors.append(f"{modality}: {summary['error']}")
            if summary["count"] == 0:
                errors.append(f"{modality}: no frames")
            if not summary["numeric_stems"]:
                errors.append(f"{modality}: frame stems must be numeric")
            if summary["count"] and not summary["strictly_contiguous"]:
                errors.append(f"{modality}: frame stems must increment by exactly 1")

        counts = {name: summary["count"] for name, summary in modality_summaries.items()}
        if len(set(counts.values())) != 1:
            errors.append(f"required modality frame counts differ: {counts}")

        stem_sets = {
            name: set(summary["stems"]) for name, summary in modality_summaries.items()
        }
        if len({frozenset(stems) for stems in stem_sets.values()}) != 1:
            errors.append("required modality frame stem sets differ")

        frame_count = modality_summaries["measurements"]["count"]
        try:
            auxiliary_directories = sorted(
                directory
                for directory in route_directory.iterdir()
                if directory.is_dir() and directory.name not in REQUIRED_MODALITIES
            )
        except Exception as exc:
            auxiliary_directories = []
            errors.append(f"cannot enumerate auxiliary modalities: {_error_text(exc)}")

        for directory in auxiliary_directories:
            summary = _modality_summary(directory)
            aux_summaries[directory.name] = summary
            # Auxiliary streams are descriptive except for the one condition
            # that makes them unusably misaligned: a present stream with a
            # different number of saved frames.
            if summary["count"] != frame_count:
                errors.append(
                    f"auxiliary modality {directory.name!r} has {summary['count']} frames; "
                    f"expected {frame_count}"
                )

    metadata_match = _metadata_match(records, route_name, errors)
    byte_paths = [result_path, *_route_files(route_directories)]

    return {
        "status": "pass" if not errors else "fail",
        "result_path": str(result_path),
        "result_root": str(result_root),
        "result_sha256": sha256,
        "entry_status": entry_status,
        "global": global_summary,
        "route_directories": [str(path) for path in route_directories],
        "route_directory": str(route_directory) if route_directory is not None else None,
        "route_name": route_name,
        "scenario_label": _scenario_label(result_root),
        "tfpp_loader_acceptance": tfpp_loader_acceptance,
        "record_route_name_match": metadata_match,
        "frame_count": frame_count,
        "required_modalities": modality_summaries,
        "auxiliary_modalities": aux_summaries,
        "bytes": _unique_bytes(byte_paths),
        "errors": errors,
    }


def _discover_result_paths(
    roots: Iterable[str | Path],
) -> tuple[dict[Path, list[str]], list[str], list[str]]:
    discoveries: dict[Path, list[str]] = defaultdict(list)
    input_paths: list[str] = []
    errors: list[str] = []

    for root_value in roots:
        root = Path(root_value).expanduser()
        display = str(root)
        input_paths.append(display)
        try:
            if not root.exists():
                errors.append(f"input does not exist: {display}")
                continue
            if root.is_file():
                candidates = [root] if root.name == "result.json" else []
            elif root.is_dir():
                candidates = sorted(path for path in root.rglob("result.json") if path.is_file())
            else:
                candidates = []
        except Exception as exc:
            errors.append(f"cannot search input {display}: {_error_text(exc)}")
            continue

        if not candidates:
            errors.append(f"no result.json found under input: {display}")
            continue
        for candidate in candidates:
            try:
                discoveries[candidate.resolve()].append(display)
            except Exception as exc:
                errors.append(f"cannot resolve result path {candidate}: {_error_text(exc)}")

    if not input_paths:
        errors.append("at least one input root is required")
    return discoveries, input_paths, errors


def audit_raw_collection(roots: Iterable[str | Path]) -> dict[str, Any]:
    """Audit one or more raw-collection roots and return a JSON-safe report.

    Roots may be containers holding many collection outputs, individual output
    directories containing ``result.json``, or a ``result.json`` file itself.
    Canonical result paths are audited once even when overlapping roots discover
    them repeatedly; repeated discoveries are nevertheless reported as errors.
    """

    discoveries, input_paths, errors = _discover_result_paths(roots)
    duplicate_result_paths = [
        {
            "path": str(path),
            "discoveries": len(sources),
            "source_inputs": sources,
        }
        for path, sources in sorted(discoveries.items(), key=lambda item: str(item[0]))
        if len(sources) > 1
    ]
    for duplicate in duplicate_result_paths:
        errors.append(
            f"result path discovered {duplicate['discoveries']} times: {duplicate['path']}"
        )

    results = [_audit_result(path) for path in sorted(discoveries, key=str)]
    for result in results:
        errors.extend(f"{result['result_path']}: {error}" for error in result["errors"])

    physical_route_groups: dict[str, list[str]] = defaultdict(list)
    route_name_groups: dict[str, list[str]] = defaultdict(list)
    for result in results:
        for route_directory in result["route_directories"]:
            physical_route_groups[route_directory].append(result["result_path"])
            route_name_groups[Path(route_directory).name].append(route_directory)

    duplicate_route_directories = [
        {"path": path, "result_paths": result_paths}
        for path, result_paths in sorted(physical_route_groups.items())
        if len(result_paths) > 1
    ]
    duplicate_route_names = [
        {"name": name, "paths": paths}
        for name, paths in sorted(route_name_groups.items())
        if len(set(paths)) > 1
    ]
    for duplicate in duplicate_route_directories:
        errors.append(f"route directory is referenced by multiple results: {duplicate['path']}")
    for duplicate in duplicate_route_names:
        errors.append(f"duplicate route directory name: {duplicate['name']}")

    unique_route_directories = sorted(physical_route_groups)
    frame_counts_by_route: dict[str, int] = {}
    for result in results:
        route_directory = result["route_directory"]
        if route_directory is not None:
            # A physical route can be reached through more than one result root
            # (for example through symlinks).  It is an error above, but summary
            # totals must still describe unique data rather than inflate it.
            frame_counts_by_route.setdefault(route_directory, result["frame_count"])
    all_byte_paths: list[Path] = list(discoveries)
    all_byte_paths.extend(_route_files(Path(path) for path in unique_route_directories))
    scenario_counts = Counter(result["scenario_label"] for result in results)
    status_counts = Counter(str(result["global"]["status"]) for result in results)

    composed_scores = [
        float(result["global"]["score_composed"])
        for result in results
        if isinstance(result["global"]["score_composed"], (int, float))
        and not isinstance(result["global"]["score_composed"], bool)
    ]
    route_scores = [
        float(result["global"]["score_route"])
        for result in results
        if isinstance(result["global"]["score_route"], (int, float))
        and not isinstance(result["global"]["score_route"], bool)
    ]
    penalty_scores = [
        float(result["global"]["score_penalty"])
        for result in results
        if isinstance(result["global"]["score_penalty"], (int, float))
        and not isinstance(result["global"]["score_penalty"], bool)
    ]
    loader_basis_counts = Counter(
        result["tfpp_loader_acceptance"]["acceptance_basis"]
        for result in results
        if result["tfpp_loader_acceptance"]["accepted"]
    )
    loader_accepted = sum(
        bool(result["tfpp_loader_acceptance"]["accepted"]) for result in results
    )

    return {
        "status": "pass" if not errors and results else "fail",
        "inputs": input_paths,
        "discovery": {
            "result_occurrences": sum(len(sources) for sources in discoveries.values()),
            "unique_result_paths": len(discoveries),
        },
        "duplicates": {
            "result_paths": duplicate_result_paths,
            "route_directories": duplicate_route_directories,
            "route_names": duplicate_route_names,
        },
        "results": results,
        "summary": {
            "results": len(results),
            "routes": len(unique_route_directories),
            "frames": sum(frame_counts_by_route.values()),
            "bytes": _unique_bytes(all_byte_paths),
            "scenario_labels": dict(sorted(scenario_counts.items())),
            "global_statuses": dict(sorted(status_counts.items())),
            "tfpp_loader_acceptance": {
                "accepted": loader_accepted,
                "rejected": len(results) - loader_accepted,
                "acceptance_bases": dict(sorted(loader_basis_counts.items())),
            },
            "score_composed_mean": (
                sum(composed_scores) / len(composed_scores) if composed_scores else None
            ),
            "score_route_mean": sum(route_scores) / len(route_scores) if route_scores else None,
            "score_penalty_mean": (
                sum(penalty_scores) / len(penalty_scores) if penalty_scores else None
            ),
        },
        "errors": errors,
    }
