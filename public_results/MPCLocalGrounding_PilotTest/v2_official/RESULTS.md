# MPC-Local Grounding Pilot Results (mpc_local_grounding_pilot_v2)

- Mode: `validation`
- Status: **no_go**
- Generated: `2026-08-28T08:06:39.279860+00:00`

## Validation decision

- Problem-existence gate: `True`
- Selected variant: `None`
- Validation GO: `False`
- Elite non-tied pair fraction: `0.6485507246376812` (required `0.3`)

The test split remained sealed. A test-selection manifest is written only for a non-smoke validation GO.
`elite_rank_inverse` is a diagnostic ablation and cannot qualify or rescue GO.

## Validation metrics (state macro)

| seed | variant | elite Spearman | elite non-tied order acc. | elite regret | global outcome MSE |
|---:|---|---:|---:|---:|---:|
| 17 | prediction_only | 0.757770 | 0.924459 | 0.004218 | 0.007275 |
| 17 | global_rank | 0.726371 | 0.906652 | 0.004603 | 0.007819 |
| 17 | elite_rank | 0.743875 | 0.920016 | 0.004439 | 0.007299 |
| 17 | elite_rank_inverse | 0.750954 | 0.908551 | 0.004161 | 0.007319 |
| 29 | prediction_only | 0.534922 | 0.777068 | 0.008168 | 0.006937 |
| 29 | global_rank | 0.533934 | 0.777370 | 0.008290 | 0.007328 |
| 29 | elite_rank | 0.535734 | 0.784665 | 0.008168 | 0.006929 |
| 29 | elite_rank_inverse | 0.574039 | 0.793056 | 0.007614 | 0.007096 |
| 43 | prediction_only | 0.668946 | 0.847509 | 0.005872 | 0.008452 |
| 43 | global_rank | 0.686646 | 0.847379 | 0.005550 | 0.008734 |
| 43 | elite_rank | 0.678728 | 0.848859 | 0.005868 | 0.008426 |
| 43 | elite_rank_inverse | 0.646283 | 0.838103 | 0.005828 | 0.008338 |

## Diagnosis

- `elite_query_not_better_than_equal_budget_global_query` — `structural_no_go`
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
the frozen configuration retain the reported scientific result.
