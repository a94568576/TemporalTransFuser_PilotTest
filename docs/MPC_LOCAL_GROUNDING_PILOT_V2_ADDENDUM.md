# MPC-Local Grounding Pilot V2: Safe-Local Remediation Addendum

## 1. Status and decision boundary

This addendum defines the only permitted collection remediation after the V1
paired-CARLA dataset failed its collision-free integrity precondition. It does
not overwrite, reinterpret, or rescue the V1 dataset. V1 remains an immutable
failed input for the current collision-head-free runner.

The V2 decision was made after inspecting only collection-integrity evidence:

- 2,304 expected V1 records were present;
- 13 records had a nonzero collision label: 8 train, 0 validation, 5 test;
- all 13 occurred in CEM iterations 0 or 1, and final iteration had 0;
- same-state reset, intended/applied controls, hashes, and cleanup passed.

No V1 model was trained for selection, and no V1 prediction MSE, rank
correlation, pair accuracy, regret, variant comparison, or test performance was
observed before freezing this remediation. Looking at the collision bit across
all splits was necessary to evaluate the global zero-collision precondition; it
was not a model-performance test opening.

## 2. Why this is a bounded remediation

The V1 model has no collision-prediction head. Therefore it cannot rank a
candidate using collision without leaking the simulator's real label into the
world-model cost. The runner correctly rejects the entire dataset when any
collision is nonzero, even if validation and final-CEM candidates happen to be
collision-free.

The observed failures were concentrated outside the proposed local steering
neighborhood: every colliding action sequence had `max(abs(steer_0),
abs(steer_1)) >= 0.307197`. V2 restricts only steering support. It does not
change the longitudinal distribution, horizon, real-cost definition, model,
losses, label budget, metrics, or GO thresholds.

| CEM field | V1 | V2 `safe_local_v2` |
|---|---:|---:|
| steering bounds | `[-0.70, +0.70]` | `[-0.20, +0.20]` |
| steering initial std | `0.35` | `0.12` |
| steering minimum std | `0.03` | `0.015` |
| longitudinal bounds | `[-1, +1]` | unchanged |
| longitudinal mean | `0.4` | unchanged |
| longitudinal initial/minimum std | `0.45 / 0.05` | unchanged |
| population / elite / iterations | `24 / 6 / 3` | unchanged |

The ±0.20 cap leaves margin below the least-extreme colliding sequence while
remaining large enough to produce lateral/yaw variation over one second at the
configured initial speeds. This narrows the scientific claim to **local action
ordering**. V2 cannot support conclusions about broad emergency maneuvers or
collision-aware planning.

## 3. A genuinely new state split

The V2 collection seed is `27031`. A seed change alone is insufficient: under
the current Town10HD filter, 18 of the 32 seed-27031 states would overlap V1 and
one sealed-test spawn would repeat. Therefore `safe_local_v2` excludes all 32
V1 `source_spawn_index` values before applying the seeded permutation.

The collector records the exclusion list, selected list, overlap check, and
parent identities in its manifest:

```text
V1 states_sha256:
b8c4e5cebaaf1be629308c70ef6f22c6d529b027b982d0025cc0640c74143ee3

V1 records_sha256:
f3cb11ad2ea8159427a6ff305e582dfb01bbee4860be4341f2a580f4f77f137e
```

All 32 V2 states must be disjoint from all V1 states, not merely assigned to a
different split. V2 again assigns 16 train, 8 validation, and 8 sealed test
states. Test remains inaccessible to selection unless the runner's explicit
one-shot opening conditions are met.

## 4. Pre-training gate and terminal policy

Before any model training or performance aggregation, the complete V2 records
must satisfy all of the following:

1. exactly 2,304 records with the configured state/iteration/population counts;
2. zero collision labels and zero collision events across train/val/test;
3. fresh-actor same-state and per-state initial-state equality attestations;
4. exact intended-versus-applied tick controls within the frozen tolerance;
5. disjoint V1/V2 spawn indices and matching parent/config hashes;
6. cost recomputation, NPZ schema, actor cleanup, and settings restoration.

Any nonzero V2 collision is a terminal NO-GO for this collision-head-free pilot.
There is no V3 search-space shrink, post-hoc spawn deletion, collision-row
filtering, or use of real collision labels in predicted cost. Continuing after
that failure would require a separately preregistered collision model and
held-out calibration, which is a structural redesign rather than remediation.

If the integrity gate passes, all method comparisons occur only within V2, on
the fixed V2 candidate distribution and label budget. V1 and V2 performance
numbers are not treated as paired or directly comparable because both states
and action support changed.

## 5. Frozen collection command

Run with the exact CARLA 0.9.15 client environment and an already-running
traffic-free server:

```bash
../transfuser_test/carla_garage/.venv/bin/python \
  scripts/collect_mpc_local_carla.py \
  --host 127.0.0.1 --port 2100 \
  --action-profile safe_local_v2 \
  --output data/mpc_local_grounding_carla_v2
```

For plumbing only, add `--smoke` and use a new non-evidentiary output. A
non-smoke `safe_local_v2` run is fail-closed to seed `27031`. The authoritative
runner configuration is `configs/mpc_local_grounding_pilot_v2.yaml`.
