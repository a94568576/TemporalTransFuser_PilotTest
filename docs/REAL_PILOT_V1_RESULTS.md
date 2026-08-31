# Temporal TransFuser real pilot v1 results

실행일: 2026-08-24 (Asia/Seoul)

## 결론

사전 고정 판정은 **`no_go`**다. 현재의 unwarped pooled past-BEV GRU residual branch는 frozen TF++ baseline을 개선하지 못했다. 따라서 이 결과를 근거로 같은 branch를 cross-attention이나 더 큰 temporal module로 확장하지 않는다.

- 주 방법 `past_bev` test route-macro ADE: **0.018730 m**
- frozen baseline: **0.018470 m**
- baseline 대비 ADE 변화: **+1.40% 악화**
- `current_bev`: 0.018509 m
- `shuffled_past_bev`: 0.018500 m
- 판정: [`no_go`](../artifacts/real_pilot_locked_final/GO_NO_GO.md)
- `paper_go=false`, `closed_loop_go=false`

즉, 실제 과거 BEV는 현재 BEV 반복이나 과거 순서 교란보다도 나빴다. 시간 순서가 유용한 신호로 사용됐다는 증거가 없다.

## 데이터와 프로토콜

| 항목 | 값 |
|---|---:|
| Upstream checkpoint | `all_towns/model_0030_0.pth` |
| Checkpoint SHA-256 | `d6fbdc28f7398354beadc7cf6765d866457c957f7b470c88ba206e73311a3b44` |
| 승인된 실제 Town13 routes | 15 |
| 수집 sensor frames | 1,883 |
| 캐시 records | 1,733 |
| Route split | train/val/test = 9/3/3 |
| Records | 988/298/447 |
| Route-safe history-4 windows | 952/286/435 |
| BEV feature | frozen `backbone_bev [64,8,8]` |
| Prediction/GT | `pred_checkpoint [10,2]`, distance-sampled geometric path |
| Training seeds | 17, 29, 43 |
| Residual weights | 0.01, 0.05, 0.10 |
| Epochs | 20 per variant |
| Variants per seed | 7 |

수집 완료만으로 route를 승인하지 않았다. route-local `results.json.gz`에 upstream `CARLA_Data`와 같은 필터를 적용했다. `HazardAtSideLane_1619_0`은 원본과 retry 모두 같은 차량 충돌을 재현해 제외했고, 두 실패본은 provenance로 보존했다. 100%·Perfect인 `NonSignalizedJunctionRightTurn_73_0`을 사전 감사 후 대체했다.

Route split 뒤 연속 frame만 history window로 묶었다. Best epoch는 validation **equal-route macro ADE**로 선택했다. λ 세 개도 `past_bev`의 세-seed validation route-macro ADE 평균만으로 선택했다. 선택 후 hash-bound final command가 checkpoint를 재학습하지 않고 test를 한 번만 열었다.

## Validation-only λ 선택

| Residual weight | `past_bev` mean ADE | Seed SD | 선택 |
|---:|---:|---:|:---:|
| 0.01 | 0.024681 | 0.000072 |  |
| **0.05** | **0.024586** | **0.000028** | ✓ |
| 0.10 | 0.024600 | 0.000031 |  |

선택 artifact는 [`SELECTION_CHOICE.md`](../artifacts/real_pilot_validation_choice/SELECTION_CHOICE.md)에 있다. λ=0.05의 validation `past_bev`는 baseline 0.024624 m보다 약 0.16% 좋았지만, 이 소폭 개선은 잠긴 test에서 재현되지 않았다.

## Locked test 결과

Primary 값은 세 seed의 equal-route macro ADE 평균이며 `±`는 seed 간 sample SD다. Δ는 baseline 대비 ADE 변화로, 양수는 악화다.

| Method | ADE (m) | Δ vs baseline | FDE (m) | Smoothness | 평균 harmed-route 비율 |
|---|---:|---:|---:|---:|---:|
| frozen baseline | 0.018470 ± 0.000000 | — | 0.036129 | 0.011630 | — |
| current_only | 0.020108 ± 0.002379 | +8.86% | 0.037522 | 0.012864 | 100.0% |
| current_only_matched | 0.021462 ± 0.001333 | +16.20% | 0.038466 | 0.013507 | 100.0% |
| trajectory_only | 0.018487 ± 0.000021 | +0.09% | 0.036153 | 0.011697 | 33.3% |
| current_bev | 0.018509 ± 0.000073 | +0.21% | 0.036144 | 0.011652 | 66.7% |
| **past_bev** | **0.018730 ± 0.000378** | **+1.40%** | **0.036323** | **0.011744** | **66.7%** |
| shuffled_past_bev | 0.018500 ± 0.000036 | +0.16% | 0.036162 | 0.011623 | 66.7% |
| combined | 0.018495 ± 0.000065 | +0.14% | 0.036126 | 0.011643 | 44.4% |

어떤 adapter도 세-seed 평균 ADE에서 baseline을 이기지 못했다. `current_only_matched`는 validation에서 0.024289 m로 좋아 보였지만 test에서는 baseline보다 16.20% 악화돼 작은 validation split에 대한 calibration overfit을 보였다. `past_bev`가 이 control보다 좋은 것은 temporal 성공이 아니라 이 control의 test 악화 때문이다.

## 고정 gate 결과

통과:

- `past_bev`가 `current_only_matched`보다 ADE 12.73% 낮음
- `past_bev`가 `current_only_matched`보다 FDE 낮음
- 세 seed 모두 `current_only_matched` 대비 ADE 방향 통과
- smoothness 악화가 허용치 +5% 이내

실패:

- `past_bev` ADE가 baseline보다 낮지 않음
- `past_bev` ADE가 `current_bev`보다 낮지 않음
- `past_bev` ADE가 shuffled history보다 낮지 않음
- `past_bev` FDE가 baseline보다 낮지 않음
- 세 seed 모두 paired route-bootstrap 95% CI upper bound가 0보다 큼

따라서 matched-current control을 이긴 한 조건만으로 temporal 효과를 주장할 수 없다. authoritative core/temporal gate가 모두 실패했다.

## 무결성 및 재현성

- Raw audit: 15 routes, 1,883 frames, errors 0 — [`real_pilot_town13_v1_raw_audit.json`](../artifacts/real_pilot_town13_v1_raw_audit.json)
- Rejected Hazard retry evidence — [`real_pilot_town13_v1_raw_audit_hazard_retry_rejected.json`](../artifacts/real_pilot_town13_v1_raw_audit_hazard_retry_rejected.json)
- Full cache audit: 1,733/1,733 valid, errors 0 — [`real_pilot_town13_v1_cache_audit_all.json`](../artifacts/real_pilot_town13_v1_cache_audit_all.json)
- Full sanity: 15 routes, frame/hash/shape/finite/cadence errors 0 — [`real_pilot_town13_v1_sanity_all.json`](../artifacts/real_pilot_town13_v1_sanity_all.json)
- Deterministic 20-sample visual audit: pass — [`manifest.json`](../artifacts/real_pilot_town13_v1_visualizations_all/manifest.json)
- Regression suite: 95/95 passed
- Locked final: [`STUDY_RESULTS.md`](../artifacts/real_pilot_locked_final/STUDY_RESULTS.md)
- Fixed decision: [`GO_NO_GO.md`](../artifacts/real_pilot_locked_final/GO_NO_GO.md)

λ=0.05 선택 study에 parent marker 1개와 child marker 3개가 생성됐다. λ=0.01/0.10 studies에는 marker가 없다. Final은 choice의 original manifest SHA-256, checkpoint/config/cache hashes, variant semantics를 검증하고 test tensor를 읽기 전에 marker를 만들었다.

CUDA 학습은 seed, cuDNN deterministic, benchmark/TF32 off, `CUBLAS_WORKSPACE_CONFIG`를 사용했다. 다만 `adaptive_avg_pool2d_backward_cuda`가 deterministic 구현을 제공하지 않아 `torch.use_deterministic_algorithms(..., warn_only=True)` 경고가 발생했다. 따라서 동일 seed의 bitwise 재현성은 주장하지 않는다.

## 해석 한계

- test 독립 단위는 3 routes뿐이다. 세 training seed는 같은 routes를 반복할 뿐 표본 수를 늘리지 않는다.
- 이 결과는 frozen cached output을 사용한 offline adapter 평가다. closed-loop CARLA 개선을 의미하지 않는다.
- `pred_checkpoint`는 시간-sampled ego trajectory가 아니라 거리-sampled geometric path다.
- local `all_towns` checkpoint가 Town13을 pretraining에서 보았을 수 있다. held-out 범위는 adapter이지 전체 driving system OOD가 아니다.
- 입력 past BEV는 current ego frame으로 spatial warp되지 않은 8×8 map을 flatten해 사용했다. 실패는 이 구체적 구조에 대한 결과이며 모든 temporal BEV 방법의 불가능성을 증명하지 않는다.

## 다음 결정

이 실험의 고정 규칙에 따라 **같은 GRU branch의 용량 확대, cross-attention 추가, test 기반 재튜닝은 중단**한다. 현재 locked test를 다시 사용해 구조를 고르면 leakage가 된다.

연구를 계속하려면 별도 가설과 새 untouched test를 갖는 새 study로 시작해야 한다. 가능한 새 질문은 ego-motion-aligned spatial BEV token이 동적 시나리오의 closed-loop 제어에 도움이 되는지이며, 그 경우에도 `13_withheld` checkpoint 또는 새로운 town/log, 충분한 route 수, target-speed/brake를 포함한 closed-loop 평가가 먼저 필요하다. 현재 결과만으로는 논문화 가능한 성능 주장이 없다.

## Artifact hashes

- Cache index: `094d3a9d11f3ad8f733285f3d40207fa82bbcc59de26163724db4d53094ca933`
- Validation choice: `facfa71f3233df74014f3e0c43b22893c9c6fba968495b9bce560c3b0c09c762`
- Locked study results: `216df3aad8ea28421c57310368ca10cc9c3ba205edd093561e4a1665661b4577`
- Go/No-Go JSON: `6e64aa23bfd42c8cbd19986c85bce9cfa2ec572dea4fe65a184cb6ed974eaa24`
