# Multi-seed Study Results

Study `5ec371a676210a97c0f9` finalized seeds [17, 29, 43] in one permanent test-open event. All checkpoints were selected on train/validation before this command; the final command performed no retraining.

Checkpoint epoch selection: validation `equal_route_macro` `ade`.

Primary metric: route-macro `ade`. Standard deviations are sample SD across seeds.

| Method | Route-macro ade | Min–max | Route improved | Route harmed | Gate | Applied residual L1 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0185 ± 0.0000 | 0.0185–0.0185 | — | — | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| current_only | 0.0201 ± 0.0024 | 0.0187–0.0229 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.1342 ± 0.0050 | 0.0008 ± 0.0004 | 0.0514 ± 0.0731 |
| current_only_matched | 0.0215 ± 0.0013 | 0.0200–0.0226 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.1423 ± 0.0123 | 0.0014 ± 0.0003 | 0.0105 ± 0.0016 |
| trajectory_only | 0.0185 ± 0.0000 | 0.0185–0.0185 | 0.6667 ± 0.0000 | 0.3333 ± 0.0000 | 0.1288 ± 0.0033 | 0.0003 ± 0.0001 | 0.0190 ± 0.0112 |
| current_bev | 0.0185 ± 0.0001 | 0.0185–0.0186 | 0.3333 ± 0.3333 | 0.6667 ± 0.3333 | 0.1251 ± 0.0043 | 0.0002 ± 0.0002 | 0.0585 ± 0.0702 |
| past_bev | 0.0187 ± 0.0004 | 0.0185–0.0192 | 0.3333 ± 0.3333 | 0.6667 ± 0.3333 | 0.1354 ± 0.0188 | 0.0004 ± 0.0003 | 0.0214 ± 0.0016 |
| shuffled_past_bev | 0.0185 ± 0.0000 | 0.0185–0.0185 | 0.3333 ± 0.3333 | 0.6667 ± 0.3333 | 0.1244 ± 0.0031 | 0.0002 ± 0.0001 | 0.0299 ± 0.0102 |
| combined | 0.0185 ± 0.0001 | 0.0184–0.0185 | 0.5556 ± 0.5092 | 0.4444 ± 0.5092 | 0.1275 ± 0.0058 | 0.0003 ± 0.0001 | 0.0215 ± 0.0016 |

The study-level marker is permanent. A failed child evaluation still consumes this single test-open event and the study cannot be finalized again.
