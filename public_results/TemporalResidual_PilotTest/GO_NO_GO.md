# Exploratory real-pilot verdict

**Exploratory pilot decision: `no_go` (not a paper or closed-loop GO).** At least one authoritative exploratory pilot core gate failed.

The authoritative gate uses `past_bev`; `combined` is diagnostic only and cannot change this verdict.

Study `5ec371a676210a97c0f9`: 3 seeds, 3 test routes per seed.

## Fixed authoritative exploratory gates

| Gate | Result | Observed |
|---|---:|---|
| `ade_3pct_vs_current_only_matched` | PASS | relative improvement 12.73% |
| `ade_vs_baseline` | FAIL | past 0.0187; baseline 0.0185 |
| `ade_vs_current_bev` | FAIL | past 0.0187; current_bev 0.0185 |
| `ade_vs_shuffled_past_bev` | FAIL | past 0.0187; shuffled_past_bev 0.0185 |
| `fde_vs_current_only_matched` | PASS | past 0.0363; current_only_matched 0.0385 |
| `fde_vs_baseline` | FAIL | past 0.0363; baseline 0.0361 |
| `ade_direction_every_seed` | PASS | seed 17: pass, seed 29: pass, seed 43: pass |
| `bootstrap_ci_every_seed` | FAIL | seed 17: fail, seed 29: fail, seed 43: fail |
| `smoothness_vs_baseline` | PASS | past 0.0117; limit 0.0122 |

## Route-macro means across seeds

| Method | ADE | FDE | Smoothness |
|---|---:|---:|---:|
| baseline | 0.0185 | 0.0361 | 0.0116 |
| current_only_matched | 0.0215 | 0.0385 | 0.0135 |
| current_bev | 0.0185 | 0.0361 | 0.0117 |
| past_bev | 0.0187 | 0.0363 | 0.0117 |
| shuffled_past_bev | 0.0185 | 0.0362 | 0.0116 |
| trajectory_only | 0.0185 | 0.0362 | 0.0117 |
| combined | 0.0185 | 0.0361 | 0.0116 |

## Limitations

- Regardless of status, this is an exploratory pilot verdict and never a paper GO or a closed-loop driving GO.
- This is offline cached-path evidence only; it does not establish closed-loop CARLA driving improvement.
- Training seeds repeat the same test routes and are not additional independent route samples.
- The study uses exactly the minimum three seeds; seed-to-seed uncertainty is only coarsely characterized.
- Exactly three independent test routes are present. Repeating them across training seeds does not enlarge the route sample, so this result cannot support a paper-level efficacy or generalization claim.
