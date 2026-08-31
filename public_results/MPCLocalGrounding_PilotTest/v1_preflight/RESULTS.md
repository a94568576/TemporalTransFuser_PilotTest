# MPC-Local Grounding Pilot V1 Results

- Mode: `validation`
- Status: **failed_preflight**
- Generated: `2026-08-28T07:58:50.346444+00:00`

## Validation decision

- Problem-existence gate: `None`
- Selected variant: `None`
- Validation GO: `None`
- Elite non-tied pair fraction: `None` (required `None`)

The test split remained sealed. A test-selection manifest is written only for a non-smoke validation GO.
`elite_rank_inverse` is a diagnostic ablation and cannot qualify or rescue GO.

## Diagnosis

- `collision_head_absent_fail_closed` — `structural_no_go_for_v1`

## Scope and leakage controls

- All outcomes and real costs come from paired CARLA rollouts, not model labels.
- Normalization statistics use train iteration-0 prediction records only.
- Every ranking variant uses the same architecture and global prediction records.
- Global and elite ranking receive the same fixed pair budget per state.
- V1 fails closed on collisions; real collision labels are never fed to predicted cost.
- This is a low-dimensional mechanism pilot, not closed-loop driving evidence.

Machine-readable metrics, checkpoint hashes, and provenance are in `results.json`.
