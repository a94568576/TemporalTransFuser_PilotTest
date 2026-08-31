# MPC-Local Grounding Pilot Results (mpc_local_grounding_pilot_v2_rankscale_diagnostic)

- Mode: `validation`
- Status: **exploratory_diagnostic_gate_fail**
- Generated: `2026-08-28T08:23:47.416682+00:00`

## Exploratory diagnostic disclosure

- Evidence class: **posthoc validation-only scale diagnostic**
- The official V2 decision remains **NO-GO**.
- The sealed test is prohibited and no selection manifest can be generated.
- Global rank-logit temperature: `0.005`; the same value is used for global/elite training queries and common validation.
- This diagnostic cannot qualify, rescue, or replace the official result.

## Exploratory gate replay (non-qualifying)

- Problem-existence gate: `True`
- Selected variant: `None`
- Validation GO: `False`
- Elite non-tied pair fraction: `0.6485507246376812` (required `0.3`)

The test split remains permanently sealed for this diagnostic; a selection manifest is never written.
`elite_rank_inverse` is a diagnostic ablation and cannot qualify or rescue GO.

## Exploratory validation metrics (state macro)

| seed | variant | elite Spearman | elite non-tied order acc. | elite regret | global outcome MSE |
|---:|---|---:|---:|---:|---:|
| 17 | prediction_only | 0.757770 | 0.924459 | 0.004218 | 0.007275 |
| 17 | global_rank | 0.828017 | 0.924856 | 0.002462 | 0.012716 |
| 17 | elite_rank | 0.881127 | 0.972947 | 0.001658 | 0.006791 |
| 17 | elite_rank_inverse | 0.865754 | 0.974821 | 0.002078 | 0.006174 |
| 29 | prediction_only | 0.534922 | 0.777068 | 0.008168 | 0.006937 |
| 29 | global_rank | 0.755497 | 0.905406 | 0.002687 | 0.016027 |
| 29 | elite_rank | 0.840217 | 0.948070 | 0.003361 | 0.006604 |
| 29 | elite_rank_inverse | 0.838263 | 0.946243 | 0.003121 | 0.007186 |
| 43 | prediction_only | 0.672226 | 0.850960 | 0.005611 | 0.008565 |
| 43 | global_rank | 0.826375 | 0.923238 | 0.003601 | 0.012253 |
| 43 | elite_rank | 0.823823 | 0.929770 | 0.003526 | 0.008196 |
| 43 | elite_rank_inverse | 0.828901 | 0.927733 | 0.003169 | 0.008302 |

## Diagnosis

- `ranking_improves_without_selection_regret_reduction` — `structural_no_go`
- `elite_query_not_better_than_equal_budget_global_query` — `structural_no_go`

## Scope and leakage controls

- All outcomes and real costs come from paired CARLA rollouts, not model labels.
- Normalization statistics use train iteration-0 prediction records only.
- Every ranking variant uses the same architecture and global prediction records.
- Global and elite ranking receive the same fixed pair budget per state.
- The mechanism fails closed on collisions; real collision labels are never fed to predicted cost.
- This is a low-dimensional mechanism pilot, not closed-loop driving evidence.

The original machine-readable artifact is intentionally omitted from this public
export because it embeds collection and sealed-split provenance. This summary and
the frozen diagnostic configuration retain the reported scientific result.
