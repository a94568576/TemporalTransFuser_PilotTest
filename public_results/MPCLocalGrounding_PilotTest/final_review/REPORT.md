# MPC-Local Action Grounding 최종 검토 및 파일럿 판정

작성일: 2026-08-28

## 최종 판정

현재 구현과 데이터 범위에서는 **NO-GO이며, 같은 validation에서 추가 튜닝하지 않고 이 경로를 중단하는 것이 맞다.**

사소한 구현 문제였던 rank-logit scale은 수정했고, 순위 학습 자체는 크게 좋아졌다. 그러나 제안의 핵심 주장인 **“CEM elite 후보를 사용한 grounding이 동일 라벨 예산의 global 후보 grounding보다 더 낫다”**는 validation seed 3개 중 1개만 통과했다. 사전 기준은 2개였다.

이 결과는 전체 아이디어가 원리적으로 불가능하다는 뜻은 아니다. 다만 현재의 저차원 terminal-dynamics 모델과 32-state CARLA 파일럿에서 elite-query의 고유 이점은 입증되지 않았다. 따라서 visual world model이나 closed-loop MPC로 바로 확장할 근거도 없다.

## 제안 검토

제안에서 방어 가능한 핵심은 단순 action sensitivity가 아니라 다음 비교다.

> MPC가 실제로 헷갈리는 late-CEM elite 후보를 simulator에서 같은 초기 상태로 실행해 실제 outcome을 얻고, 그 후보로 학습한 모델이 동일 예산의 일반 후보 학습보다 unseen state의 순위와 selection regret를 더 잘 보존하는가?

이렇게 좁혀야 하는 이유는 action-aware latent, inverse dynamics, alternative-action separation, planning-cost ranking이 이미 각각 [Delta-JEPA](https://arxiv.org/abs/2606.31232), [ActSWM](https://arxiv.org/abs/2607.26712), [ACID](https://arxiv.org/abs/2607.02403), [Monotone Planning Costs](https://arxiv.org/abs/2608.09073)에 등장하고, CEM 단계별 decision-metric alignment와 late-elite 붕괴도 [DA-LeWM](https://arxiv.org/abs/2608.18746)이 다루기 때문이다. 일반적인 decision-aware model learning 역시 [VAML](https://proceedings.mlr.press/v54/farahmand17a.html)까지 거슬러 올라간다.

따라서 `WM + MPC + inverse/action loss`나 `action perturbation + ranking`만으로는 노벨티가 약하다. 공정한 equal-budget global-query baseline을 이기는 elite-query의 이점, 반복 re-mining, 새로운 state와 새로운 elite에서의 regret 감소가 필요하다.

## 실제로 시험한 범위

이 파일럿은 전체 visual world model이나 closed-loop driving 실험이 아니다. 핵심 메커니즘만 먼저 검증하기 위해 CARLA 0.9.15에서 다음을 수행했다.

- Town10HD_Opt, traffic 없음, synchronous 20 Hz
- candidate마다 fresh actor를 생성하고 동일 초기 상태를 재현
- 각 state에서 3 CEM iteration × 24 candidate
- 32 state, 총 2,304개의 실제 simulator rollout
- train/validation/sealed-test state = 16/8/8
- exact intended/applied control, reset, initial state, collision, physical-cost 재계산을 fail-closed로 감사
- prediction-only, equal-budget global-rank, elite-rank, elite-rank+inverse를 동일한 8,760-parameter 모델로 비교
- train iteration-0 후보 24개를 prediction 12개와 global-rank query 12개로 deterministic하게 분리
- elite-rank는 iteration-2 query 12개를 사용하고, 양쪽 모두 state당 64 pair로 고정

기존 TF++는 action 입력과 recurrent future transition이 없는 encoder/planner이므로 이 실험의 world model로 재사용하지 않았다. 이전 TF++ latent proxy도 raw persistence보다 나빠 이미 NO-GO였다.

## V1: 데이터 무결성 단계 실패

V1은 2,304개 rollout 수집과 reset/control 감사를 통과했지만 collision record가 13개였다(train 8, validation 0, test 5). 현재 모델에는 collision prediction head가 없어서 simulator의 실제 collision label을 predicted cost에 넣으면 oracle leakage가 된다. 따라서 학습 전에 fail-closed했다.

이 원인은 제한된 범위에서 운영적으로 수정 가능했다. 충돌 action의 최소 `max(abs(steer))`가 0.3072였으므로, 유일한 remediation으로 steer 범위를 ±0.20으로 줄이고 V1과 겹치지 않는 32개 새 spawn state를 사용했다. V1 실패 증거는 덮어쓰지 않았다.

## V2 공식 실험: NO-GO

V2는 2,304개 record, collision/event 0, V1 state overlap 0, intended/applied control 최대 오차 0으로 수집됐다. 문제 자체는 존재했다.

- validation elite의 non-tied pair 비율: 0.6486, 기준 0.30 통과
- prediction-only의 global→elite Spearman 하락: 세 seed 모두 기준 0.05 통과

하지만 원래 구현은 실제 cost 차이를 그대로 BCE logit으로 사용했다. elite real-cost margin 중앙값이 약 0.0079인데 implicit temperature가 1이라, elite rank loss가 0.6883~0.6889에서 거의 움직이지 않았다. 공식 V2는 elite-rank가 0/3 seed만 통과하여 NO-GO였다.

## 사소한 결함 수정: validation-only rank-scale 진단

공식 V2 NO-GO를 보존한 채 한 번만 다음 수정을 적용했다.

```text
temperature = 0.005  # 기존 tie threshold와 동일
rank_logit = (J_pred_j - J_pred_i) / temperature
```

같은 temperature를 global/elite train query와 모든 모델의 common validation objective에 적용했다. rank weight, pair, seed, epoch, architecture, 데이터, 비용, gate는 바꾸지 않았다. 이 진단은 post-hoc exploratory이며 test 접근과 selection manifest 생성을 설정과 코드 양쪽에서 금지했다.

수정은 실제로 작동했다.

- elite train rank loss: 약 0.688 → 0.317/0.321/0.357
- 기존 elite 대비 Spearman 증가: +0.137/+0.304/+0.145
- 기존 elite 대비 regret 감소: 62.6%/58.9%/39.9%
- 최대 pre-clip gradient norm: 0.635 미만; clip threshold 5.0 초과 epoch 0개

즉 최종 실패는 gradient clipping이나 rank loss 미작동 때문이 아니다.

## 최종 primary 비교

Primary gate는 elite-rank를 prediction-only뿐 아니라 **동일 pair 예산의 scaled global-rank와도** 비교한다.

| seed | elite−global Spearman | elite regret 감소 vs global | joint gate |
|---:|---:|---:|---|
| 17 | +0.05311 | +32.67% | 통과 |
| 29 | +0.08472 | **−25.06%** | 실패 |
| 43 | **−0.00255** | +2.09% | 실패 |

필요한 2/3 seed 중 1/3만 통과했다.

- seed 29는 전체 ordering이 좋아졌지만 실제 top selection regret는 global-rank보다 악화됐다.
- seed 43은 Spearman 우위가 없고 regret 개선도 20% 기준에 크게 못 미쳤다.
- paired-state bootstrap 구간도 elite가 global보다 안정적으로 우월하다는 결론을 지지하지 않았다.
- inverse-head variant는 설계상 action으로부터 만들어진 latent delta에서 같은 action을 복원하므로 독립적인 inverse-dynamics 증거가 아니며 GO를 구제하도록 허용하지 않았다.

따라서 남은 문제는 **rank 개선이 실제 선택 regret 감소로 안정적으로 연결되지 않고, elite query가 global query보다 일관되게 낫지 않다**는 구조적 문제다.

## 왜 여기서 더 고치지 않았는가

temperature와 loss-scale 결함은 이미 고쳤다. 이후 가능한 변경인 epoch 연장, rank weight/temperature 재조정, tie 제거, hinge/listwise/top-k loss, 더 유리한 gate 선택은 결과를 본 뒤의 방법 변경 또는 validation overfitting이다. 더 이상 “사소한 버그 수정”으로 볼 수 없다.

Elite checkpoint가 일부 seed에서 250 epoch 경계에 있다는 convergence caveat는 남는다. 그러나 rank supervision은 이미 강하게 작동했고, 실패 지점은 최적화 미작동이 아니라 top-selection regret와 equal-budget global baseline 대비 우위다. 같은 validation에서 추가 탐색할 정당성은 없다.

## 무결성 및 테스트 상태

- V2 sealed test는 공식 실험과 post-hoc 진단 모두 열지 않았다.
- test metric, test label 사용, test record 반환은 모두 없었다.
- selection manifest와 test-open receipt는 생성되지 않았다.
- checkpoint 12개는 감사/재현용으로 보존됐지만 selection/test 권한을 부여하지 않는다.
- 회귀 테스트: `17 passed, 16 subtests passed`
- diagnostic `results.json` SHA-256: `b71676cccecf59b188c8e7791330e60c4a0fb541867fc84899de5aca36ec9489`
- official V2 `results.json` SHA-256: `ac24de4b1eb682d764b26fce8e196deb862582b623acc6904d5a4be96a5f9cf5`
- V2 `records.npz` SHA-256: `ecd0f8c7e9a97b6f8bf7c6b96b0994231a9649eddb83b9935c61f04246f84ff7`

## 권고

1. **현재 elite-query grounding formulation은 중단한다.** 같은 V2 validation에서 더 튜닝하지 않는다.
2. 일반 rank supervision에는 신호가 있었지만 global outcome MSE가 prediction-only 대비 크게 악화되는 seed도 있었고, planning-cost ranking 자체의 선행연구가 강하다. 이를 그대로 새 메인 아이디어로 바꾸는 것은 권하지 않는다.
3. 같은 연구 질문을 다시 시도하려면 별도 연구로 preregister한다. interaction-rich same-state branching data, 실제 sequential action-conditioned semantic/occupancy world model, collision/actor-motion prediction, 더 많은 독립 state, train-only iterative re-mining, fresh validation/test split, closed-loop MPC를 처음부터 포함해야 한다.
4. 더 빠른 다음 프로젝트를 원한다면 이 경로보다 **다른 가설을 먼저 고르는 편이 효율적**이다. 현재 결과에서 가장 약한 고리는 action sensitivity 자체가 아니라 elite-specific data selection의 추가 가치다.

## 재현 산출물

- V1 무결성 실패: `artifacts/mpc_local_grounding_pilot_v1_preflight/RESULTS.md`
- V2 공식 NO-GO: `artifacts/mpc_local_grounding_pilot_v2_validation_r2/RESULTS.md`
- rank-scale 진단: `artifacts/mpc_local_grounding_pilot_v2_rankscale_diagnostic/RESULTS.md`
- V2 remediation 정의: `docs/MPC_LOCAL_GROUNDING_PILOT_V2_ADDENDUM.md`
- collector: `scripts/collect_mpc_local_carla.py`
- runner: `scripts/run_mpc_local_grounding_pilot.py`
- core: `src/temporal_tf/mpc_local_grounding.py`
- tests: `tests/test_mpc_local_grounding.py`, `tests/test_mpc_local_grounding_runner_manifest.py`
