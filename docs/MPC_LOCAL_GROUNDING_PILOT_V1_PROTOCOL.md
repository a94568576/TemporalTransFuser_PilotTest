# MPC-Local Grounding Paired-CARLA Pilot V1

## 1. 질문과 허용되는 결론

이 파일럿은 다음의 좁은 질문만 검사한다.

> 동일한 CARLA 초기 상태에서 실제로 실행해 라벨링한 CEM 후반 후보를 사용한
> pairwise ranking supervision이, 같은 **training-query 라벨 예산**의
> global-candidate ranking보다
> 보지 않은 초기 상태의 late-CEM cost ordering과 action-selection regret를 개선하는가?

이것은 저차원 paired-dynamics **mechanism pilot**이다. RGB/video world model, TF++
latent, 실제 traffic interaction, 완전한 receding-horizon closed-loop 주행을 검증하지
않는다. 성공하더라도 논문 전체나 자율주행 성능의 GO가 아니라 semantic occupancy
world model과 closed-loop 수집으로 진행할 근거만 제공한다.

TF++ native latent는 이 파일럿에 사용하지 않는다. 선행 action-influence V1에서
centered 및 uncentered 모델 모두 raw persistence보다 나빴고, recorded-pose metric
warp 자체가 raw persistence 대비 크게 악화했기 때문이다. 그 표현을 다시 쓰는 것은
경미한 하이퍼파라미터 수정이 아니라 이미 확인된 구조적 오류를 반복하는 일이다.

## 2. 문헌상 novelty 경계

다음은 이 연구의 독립 novelty로 주장하지 않는다.

- world model + CEM/MPC
- latent displacement에서 action 복원
- alternative-action latent separation 또는 context-collapse 진단
- CEM 단계별 latent/real rank alignment 측정
- action perturbation에 대한 monotone/ranking loss
- inverse-action consistency를 planning cost에 추가

관련 직접 선행은 DA-LeWM, Delta-JEPA, ActSWM, Monotone Planning Costs, ACID이다.
따라서 장기적으로 방어 가능한 가설은 단순 `elite hard negative`가 아니라, 제한된
simulator label budget을 planner가 현재 모호해하는 분포에 배분하는 **active paired
querying**과 그 분포가 변한 새 상태/새 elite에서 regret가 실제로 감소한다는 주장이다.
다만 V1 collector는 candidate distribution을 안정적으로 정의하기 위해 oracle CARLA
cost로 CEM을 갱신하며, 그 mining 과정의 simulator 호출은 training-query 12개 예산에
포함하지 않는다. 따라서 V1 성공만으로 active-query sample efficiency나 현재 learned
WM이 스스로 hard candidate를 찾는다고 주장하지 않는다.

## 3. 실제 outcome만 정답으로 사용

각 candidate sequence는 CARLA 0.9.15의 동일 초기 vehicle transform 및 velocity에서
20 Hz로 1초간 실제 실행한다. Action은 두 0.5초 segment의
`[steer, longitudinal]` 네 파라미터이며, 저장되는 저수준 sequence는 매 tick의
`[steer, throttle, brake]`이다. 양의 longitudinal은 throttle, 음수는 brake로
변환한다.

World-model prediction은 candidate mining에는 사용할 수 있지만 다음 값의 정답이 될
수 없다.

- physical outcome distance
- real planning cost
- pair ordering
- selection regret

이 값들은 반드시 simulator rollout에서 계산한다. 같은 모델의 prediction으로 mining과
labeling을 모두 하는 circular self-training은 fail-closed로 금지한다.

## 4. 상태 분리와 test 정책

Town10HD의 서로 다른 비-junction spawn state 32개를 고정된 seed로 선택한다.

```text
train: 16 states
val:    8 states
test:   8 states
```

한 initial state의 모든 CEM iteration/candidate는 같은 split에만 속한다. State ID를
넘는 record-level random split은 금지한다. Collector가 test rollout도 한 NPZ에
materialize할 수는 있지만, runner는 `allow_test=False`가 기본이며 model selection,
loss weight, remediation에는 train/val만 사용한다.

Validation에서 variant와 허용된 remediation을 확정하고 config/result hash를 기록한
뒤 test를 한 번만 연다. Test 결과를 보고 variant, seed, threshold 또는 cost를 바꾸면
그 결과는 confirmation이 아니라 exploratory 재사용으로 표시한다.

## 5. CEM과 물리 cost

Horizon은 20 tick(1초), population은 24, elite는 6, iteration은 3이다. Candidate
parameter 순서는 다음과 같다.

```text
[steer_0, longitudinal_0, steer_1, longitudinal_1]
```

Terminal outcome은 초기 lane frame 기준으로 정규화한 다음 네 값이다.

```text
[progress/10m, lateral/3m, yaw_error/pi, (speed-8m/s)/8m/s]
```

Real cost는 terminal progress, lateral/yaw/speed error, 두 segment 각각의 steering 및
longitudinal **mean squared magnitude**, collision로 구성하며 authoritative weight는
config에 있다. `abs(cost_i-cost_j) <= 0.005`인 pair는
tie이다. Tie를 억지로 latent에서 분리하거나 어느 쪽이 낫다고 학습하지 않는다.

V1 model은 collision head를 갖지 않으므로 primary ranking은 **모든 collected
collision label이 0인 bounded scene에서만** 유효하다. Runner는 nonzero collision을
하나라도 발견하면 fail-closed하고, predicted cost에 simulator의 real collision label을
절대 넣지 않는다. Nonzero collision scene으로 확장할 때는 train-only collision head와
held-out calibration을 먼저 추가해야 한다.

## 6. 비교 모델과 label budget

모든 variant는 동일 architecture, seed, global prediction records, normalization 및
optimizer를 사용한다. 상태별 iteration-0 후보 24개를 outcome/cost를 보지 않는
`SHA256(state_id, raw_action_bytes, collection_seed)` 순서로 고정 정렬한 뒤 처음
12개만 공통 prediction target으로 사용한다. 나머지 12개는
`global_rank`의 별도 query이고, `elite_rank`는 iteration-2에서 같은 방식으로 고른
12개를 query한다. 따라서 prediction label과 global-rank query가 겹치지 않으며,
global/elite 비교는 상태당 12개의 새로운 simulator candidate label과 64개의
deterministic tie-aware pair로 동일하다. Pair budget에는 real-cost tie도 포함하지만
target을 0.5로 두어 임의의 separation을 강제하지 않으며, non-tied fraction과
non-tied ordering accuracy는 별도로 보고한다.

1. `prediction_only`: iteration-0의 고정 prediction 절반 outcome만 사용.
2. `global_rank`: 겹치지 않는 iteration-0 나머지 절반에서 ranking pair를 추가.
3. `elite_rank`: 동일 pair 예산을 final CEM iteration에서 active query.
4. `elite_rank_inverse`: elite ranking에 model-generated latent-delta action
   reconstruction을 추가하는 **비선택 diagnostic ablation**.

Primary novelty 비교는 `elite_rank` 대 `global_rank`이다. `prediction_only`는 standard
WM baseline이다. 현재 inverse head의 입력 delta는 action을 직접 입력받아 생성되므로,
그 reconstruction은 실제 endpoint에서 action을 복원하는 masked inverse dynamics가
아니다. 따라서 `elite_rank_inverse`는 optimization diagnostic일 뿐 GO를 구제하거나
test 대상 variant로 선택될 수 없다. Proposed variant만 더 많은 pair label이나 다른
prediction target을 받지 않는다.

## 7. Primary metrics와 사전 gate

모든 metric은 record micro-average보다 initial-state macro-average를 우선한다.

- CEM iteration별 real-cost 대 predicted-cost Spearman
- tie-aware pair ordering accuracy
- candidate-set simple selection regret
- terminal outcome MSE
- non-tied pair fraction과 real-cost margin distribution

문제 존재 gate는 final-iteration non-tied pair fraction이 30% 이상이고,
prediction-only의 global-to-elite Spearman이 최소 0.05 하락하는 것이다. Collapse가
없다면 이 데이터/모델에서 제안의 동기가 없으므로 method score와 무관하게 NO-GO다.

Validation method GO는 3 seed 중 2개 이상에서 `elite_rank`가 prediction-only와
global-rank 모두보다 다음을 만족해야 한다.

- final-iteration Spearman +0.05 이상
- simple regret 20% 이상 감소
- global outcome MSE 악화 10% 이하

Sealed test의 한 번짜리 확인 기준은 Spearman +0.03, regret 10% 감소, 2/3 seed다.
Offline gate가 모두 통과해야만 semantic occupancy와 receding-horizon closed-loop
Stage로 진행한다.

Bootstrap은 initial state를 단위로 method-minus-baseline 차이를 진단하지만, val/test가
각 8 state이고 세 optimizer seed는 독립 데이터 반복이 아니다. 따라서 위 point gate는
통계적 유의성 기준이 아니라 다음 단계 비용을 결정하는 운영 heuristic이다. CI가 0을
포함하면 결과 보고서에 그대로 표시하며 강한 일반화 증거로 표현하지 않는다.

## 8. NO-GO 원인과 허용된 한 번의 수정

다음은 경미한 구현/운영 실패로 보고 validation에서 한 번만 수정할 수 있다.

- train/validation loss가 함께 높은 명확한 optimization underfit
- cost normalization, sign, unit의 구현 오류
- horizon/scale 오류 때문에 real candidate가 사실상 전부 tie인 경우

허용 수정은 오류의 원인과 변경 전후 config를 모두 기록하고 전체 train/val을 처음부터
다시 실행한다. Test를 이미 열었으면 수정 결과는 새 test split 없이는 확인 결과가 아니다.

다음은 구조적 NO-GO이며 성능이 좋아 보일 때까지 튜닝하지 않는다.

- baseline에서 late-CEM collapse가 존재하지 않음
- elite query가 동일 예산 global query보다 낫지 않음
- rank만 좋아지고 selection regret가 줄지 않음
- same-state reset이 재현되지 않음
- simulator 대신 model prediction을 real label로 사용해야만 효과가 남음
- TF++ metric-warp substrate를 되살려야만 개선됨
- offline 개선이 closed-loop로 이어지지 않음

## 9. 재현성 산출물

Collector와 runner는 config, source, dataset, split/state list, CARLA/map/version,
checkpoint 및 result SHA-256을 기록한다. 기존 locked/frozen data root는 읽거나 수정하지
않으며 새 output root만 사용한다.
