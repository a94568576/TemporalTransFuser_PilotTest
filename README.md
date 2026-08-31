# TemporalTransFuser_PilotTest

Curated pilot evidence for three frozen-TransFuser/CARLA research tracks. Negative
results are included intentionally: each track reached a scientifically useful
decision gate even when the measured performance was poor.

## Included pilot lineages

| Track | Status | Main finding |
|---|---|---|
| Temporal residual adapter | `no_go` | Past-BEV ADE was 0.018730 m versus 0.018470 m for the frozen baseline (+1.40% worse); temporal history did not beat current-BEV or shuffled-history controls. |
| Action influence proxy | `no_go` | Centered FiLM improved its warped passive base by 4.73%, but was 22.88% worse than raw latent persistence. This is observational, not counterfactual evidence. |
| MPC-local grounding V1/V2 | `no_go` | The V2 problem-existence gate passed, but elite-query ranking did not consistently beat the equal-budget global-query baseline. A validation-only scale diagnostic improved ranking but not selection regret robustly. |
| MPC-local grounding V3 | terminal collection `no_go` | Three development candidates produced five collision events. Training never started because the frozen model had no collision-risk head. |

The compact reports are under [`public_results/`](public_results/README.md).

## What is in this repository

- Small PyTorch models, losses, metrics, split/leakage guards, and CARLA collectors.
- Frozen experiment configurations and preregistered protocols.
- Unit/contract tests.
- Human-readable results and a small number of machine-readable summaries.

The repository does **not** contain raw CARLA sensor data, cached latents, model
weights, checkpoints, NPZ/NPY arrays, visualizations, sealed/test payloads, or
upstream TransFuser/CARLA code.

## Reproducibility scope

This is an archival pilot package, not a claim of a production driving system.

- The temporal and action-influence studies depend on an external CARLA Garage /
  TransFuser++ checkout and locally acquired data.
- The V2 reports are authoritative, but the exact historical V2 runner/core bytes
  were not preserved; current MPC source is the best available additive successor.
  Do not claim bitwise reproduction of the official V2 run from this export.
- The final Action Influence config is preserved. The pre-persistence-gate run is
  disclosed, but its original config bytes were not archived separately.
- V3 stopped before training. Its result is a collection-integrity finding, not a
  comparison of prediction-only/global/elite model performance.
- Local absolute paths in copied text artifacts were replaced with logical
  placeholders such as `${PROJECT_ROOT}` and `${TFPP_ROOT}`.

## Setup and tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
```

CARLA collection additionally requires CARLA 0.9.15 and the CARLA Python API.
The TF++ cache extractor expects a compatible external CARLA Garage checkout.

## Security and input trust

Read [`SECURITY.md`](SECURITY.md) before loading external caches or CARLA Garage
metadata. The public export uses safe tensor loading where available, but one
legacy TF++ cache extractor must decode trusted upstream `jsonpickle` files.

## License status

No open-source license is granted for this pilot export yet. See
[`LICENSE_STATUS.md`](LICENSE_STATUS.md). External projects and datasets retain
their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
