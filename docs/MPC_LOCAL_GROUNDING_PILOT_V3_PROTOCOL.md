# MPC-Local Grounding Pilot V3: Fresh Top-Selection Protocol

## 1. 상태와 연구 질문

이 문서는 V2 공식 `NO-GO`와 사후 rank-scale diagnostic을 본 뒤 설계한 **새 확인
실험**을 사전 등록한다. V2의 판정, 산출물, validation 및 sealed test는 변경하거나
재해석하지 않는다. V3 non-smoke 수집 전에 이 문서와
`configs/mpc_local_grounding_pilot_v3.yaml`의 SHA-256을 동결해야 한다.

V3가 검사하는 질문은 다음 하나다.

> 같은 수의 실제 CARLA query rollout을 사용할 때, late-CEM 후보 전체에 대해
> exact-best action 선택을 직접 학습한 `elite_listwise`가 early/global 후보 전체를
> 학습한 `global_listwise`보다, 완전히 새로운 outer-validation state의
> deterministic exact-argmin simple regret를 안정적으로 줄이는가?

이는 traffic-free, 1초 horizon, 4차원 action과 4차원 terminal outcome을 사용하는
저차원 paired-dynamics **mechanism study**다. RGB/video/semantic occupancy world model,
다른 actor 예측, collision-aware planning, receding-horizon closed-loop MPC 또는 실제
자율주행 성능을 검증하지 않는다. GO도 그 다음 단계에 투자할 근거일 뿐이다.

## 2. V2 실패에서 사전에 고친 항목

V2에서 rank loss logit은 약 `0.008` cost 차이를 temperature 1로 사용해 학습 신호가
거의 없었다. 사후 diagnostic에서 temperature `0.005`로 바로잡았지만 다음 문제가
남았다.

1. pairwise 평균 ordering과 실제 top action 선택 목적이 달랐다.
2. tie target은 후보를 같게 만들었지만 기존 regret는 그 후보군을 다시 벌점화했다.
3. 12개 후보를 hash로 고르면서 실제 best 후보가 query에서 빠질 수 있었다.
4. train과 checkpoint/outer validation의 역할이 분리되지 않았고 state coverage가
   작았다.
5. 모든 scaled run이 epoch 250 cap에서 끝나 convergence 여부가 불명확했다.

V3는 이 결과를 본 뒤 온도나 loss를 더 탐색하는 V2 구제 실험이 아니다. 아래 변경을
한 번에 고정하고 새 map/state/outer validation에서 처음부터 검사한다.

## 3. 완전히 새로운 state와 split

V1/V2는 `Town10HD_Opt`의 eligible spawn 70개 중 64개를 이미 사용했다. 동일 조건에서
충분한 새 spawn을 만들 수 없으므로 V3는 CARLA 0.9.15의
`/Game/Carla/Maps/Town05_Opt`를 사용한다. Map이 다르므로 V1/V2와 초기 transform이
구조적으로 겹치지 않는다. V1/V2 hash는 데이터 재사용이 아니라 이 사실을 감사하기
위한 parent provenance다.

Town05의 literal spawn point는 대부분 직선이고 현재 filter에서는 curved point가
17개뿐이므로 그것만 사용하면 curvature coverage를 만들 수 없다. 따라서
`map.generate_waypoints(5.0)`으로 결정론적 waypoint pool을 만들고, 비-junction이며
앞으로 25 m 이상 유효한 driving lane만 남긴다. 고정 seed `37031`로 서로 다른 base
waypoint 60개를 선택한다. State identity는
`(map, road, section, lane, s, lateral offset, speed)` 전체이며 manifest에 저장한다.

```text
train pool:                 32 states
  fit:                      24 states
  inner checkpoint-val:      8 states
fresh outer gate-val:       16 states
sealed test:                12 states
```

Train 32개를 fit 24개와 inner-val 8개로 나누는 순서는 outcome/cost를 읽기 전에
`SHA256(state identity, collection seed)`로 결정한다. Normalization, gradient 및
optimizer update에는 fit 24개만 사용한다. Inner-val은 early stopping과 checkpoint
선택에만 사용한다. Outer-val 16개는 모든 seed/variant checkpoint와 hash가 동결된 뒤
한 번만 평가한다. Test access는 V3 전체에서 금지되며 outer GO가 나와도 열지 않는다.

Test seal은 split filter에만 의존하지 않는다. Collector는 payload를 다음처럼 물리적으로
분리한다.

```text
development_records.npz:      train + outer-val, 48 states, 3,456 records
development_diagnostics.npz:  development diagnostics only
test_records_sealed.npz:      test only, 12 states, 864 records
test_diagnostics_sealed.npz:  test diagnostics only
```

Public manifest의 `dataset_files`에는 `development_records`, `sealed_test_records`,
`development_diagnostics`, `sealed_test_diagnostics` entry를 두고 각각
`path, role, sha256, records, states, split_codes, sealed`를 기록하고,
`sealed_test_records`에만 `states_sha256`을 추가한다. Test entry의
허용되는 정보는 role/path와 aggregate count 및 collector가 계산한 SHA-256 attestation뿐이다.
Test state/record/outcome/collision/diagnostic 값은 development 파일이나 public manifest에
복제하지 않는다.

구체적으로 top-level `states`에는 train/outer-val 48개 description만 있고,
`split_state_ids`에는 `train`, `val`만 존재한다. Test의 개별 ID, transform, waypoint,
speed, lateral offset, curvature 또는 initial feature는 공개하지 않는다.
`dataset_files.sealed_test_records.states_sha256`에는 collector가 canonical test-state
description 12개 전체에서 계산한 64자리 SHA-256만 둔다. Top-level
`sealed_test_integrity`가 가질 수 있는 필드는 다음으로 제한한다.

```text
states: 12
records: 864
split_code: 2
states_sha256: <64 lowercase hex>
schema_finite_passed: true
reset_passed: true
control_execution_passed: true
individual_state_metadata_redacted: true
sealed_test_stratification_passed: true
```

마지막 값은 frozen speed/lateral/curvature allocation을 collector가 내부 확인했다는
단일 boolean일 뿐이며 test stratum별 count나 범위는 공개하지 않는다. Runner는 이
block의 정확한 key set, 고정 count와 hash 형식만 검사하며 test identity를 복원하거나
개별 state balance를 다시 계산하지 않는다.

`collection.sealed_test_redaction=true`다. Collector의 publish 결정과 collision-free
gate는 development rows에만 적용한다. Sealed rows에서는 파일 schema, 모든 값의
finite 여부, same-state reset 및 intended/applied control 무결성만 내부 검사한다.
Sealed outcome/cost/collision의 최소·최대·평균·개수·통과 여부 등 어떤 aggregate도
public manifest, stdout 또는 development sidecar에 노출하지 않는다. 따라서 sealed
collision 유무는 V3 실행 및 publish 성공 여부로도 추론할 수 없다. 향후 별도
preregistration이 sealed payload를 연다면 그때 collision applicability를 별도로
판정해야 한다.

수집 progress 로그도 같은 규칙을 따른다. Sealed state에서는 state 진행률과
`outcomes_redacted=true`만 출력할 수 있고, 실제 best cost, collision count, candidate
rank 또는 terminal outcome을 stdout/stderr에 출력하지 않는다.

Runner는 CLI `--data`가 manifest의 `development_records` path/role/hash와 정확히
일치하는지 검사하고, development NPZ에 split code 0/1만 있는지 확인한다. Sealed
entry는 manifest 문자열 형식, count, role, `sealed=true`와 파일 존재만 확인한다.
Runner는 sealed file을 열거나 자체 hash 계산, mmap, ZIP header 검사 또는 `np.load`를
하지 않는다. 즉 test record 수와 SHA는 collector의 manifest attestation이며 runner가
payload를 decode해 재확인한 값이 아니다. 나중의 별도 confirmatory opener만 manifest
hash를 실제 sealed bytes와 비교할 수 있다.

각 split에서 다음 coverage를 결과와 무관하게 균형화한다.

- initial speed: `4, 6, 8 m/s`;
- lane-center 기준 lateral offset: `-0.25, 0, +0.25 m`;
- 5/10/20 m 앞의 absolute curvature 최대값을 stable-rank로 나눈 4개 stratum.

각 factor의 marginal count 차이는 최대 1이고, state마다 base waypoint는 유일해야 한다.
요청한 lateral/speed와 실제 post-warm-up t=0 값도 manifest에 기록하고 각각
`0.05 m`, `0.20 m/s` 이내인지 검사한다. 이 tolerance는 V2에서 neutral physics
activation 뒤 요청 speed가 약 `0.06~0.12 m/s`, lateral이 최대 약 `0.032 m` 변한
감사 결과를 포함하도록 사전에 고정했다. 같은 state의 candidate 간 실제 t=0
transform/speed/features equality는 별도의 기존 strict reset tolerance로 검사한다.
특정 OOD state를 보고 삭제하거나 split을 다시 만드는 것은 금지한다.

## 4. 실제 rollout과 동일 query 예산

각 candidate는 동일 state에서 fresh actor/sensor로 실제 실행한다. 20 Hz, 20 tick,
두 0.5초 segment의 `[steer, longitudinal]` action과 물리 cost는 collision-free V2와
같다. Steering support `[-0.20,+0.20]`, longitudinal support `[-1,+1]`, CEM
population/elite/iteration `24/6/3`도 변경하지 않는다.

Collector는 CEM pool을 만들기 위해 모든 candidate를 실행하지만 runner는 아래에
정한 label만 각 arm에 노출한다.

| 용도 | CEM iteration | state당 고유 candidate |
|---|---:|---:|
| 모든 variant의 공통 outcome fit | 1 | 전체 24개 |
| `global_listwise` query | 0 | 전체 24개 |
| `elite_listwise` query | 2 | 전체 24개 |

따라서 두 ranking arm은 각각 `공통 24 + 고유 query 24 = 48`개의 labeled rollout을
fit state마다 사용한다. Candidate 24개를 전부 써서 hash subset 때문에 true best가
빠지는 문제를 제거한다. SHA ordering은 exact tie-break와 감사용일 뿐 subsampling에
사용하지 않는다. Pair 개수를 부풀려 rollout 예산으로 세지 않는다.

Oracle real cost로 CEM pool 전체를 만든 simulator 호출은 실제로 발생한다. 그러므로
V3는 **global 대 elite의 동일 query-exposure 비교**는 할 수 있지만 end-to-end active
learning label efficiency나 learned model이 스스로 후보를 발굴했다고 주장할 수 없다.

## 5. 비교군과 top-aligned listwise loss

Architecture, initialization seed, fit outcome labels, normalization, optimizer 및 schedule은
모든 variant에서 같다.

1. `prediction_only`: iteration-1의 24개 outcome regression만 사용한다.
2. `global_listwise`: 공통 regression과 iteration-0 전체 후보 top-1 loss를 사용한다.
3. `elite_listwise`: 공통 regression과 iteration-2 전체 후보 top-1 loss를 사용한다.

한 state의 query candidate 24개에 대해 real cost의 deterministic exact argmin을
one-hot target으로 만든다. Real cost가 bitwise exact tie이면 raw action bytes의
SHA-256가 작은 candidate를 고른다. Predicted terminal outcome과 action에서 계산한
predicted physical cost를 `J_hat`라 할 때 loss는 다음과 같다.

```text
target = one_hot(deterministic_exact_argmin(J_real))
logits = -J_hat / 0.005
L_top1 = cross_entropy(logits, target)
L_fit  = L_outcome + 0.25 * L_top1
```

`0.005`와 weight `0.25`는 V2 failure audit를 본 뒤, V3 outer 결과를 보기 전에 한 번
고정한다. Variant별 temperature, real-margin median으로 정한 scale, pair resampling,
inverse ablation 및 추가 loss arm은 없다. Collision의 real label을 predicted cost에
주입하지 않는다.

One-hot target은 작은 real-cost 차이도 구별한다. 이는 아래 primary exact-argmin
목표와 의도적으로 일치하지만, 물리적으로 거의 동등한 action까지 강하게 구별할 수
있다. 이 민감도는 epsilon regret를 의무적으로 함께 보고해 드러낸다.

## 6. 고정 convergence와 checkpoint 선택

각 variant/seed는 최대 1,200 epoch, 최소 300 epoch를 실행한다. Inner-val의 공통
`outcome MSE + 0.25 * final-iteration top1 CE`가 `1e-6` 이상 개선되지 않는 상태가
200 epoch 지속되면 멈춘다. 동일 값이면 더 이른 epoch를 선택한다. Gradient clip은
5.0이며 clip 횟수와 최대 pre-clip norm을 보고한다.

Inner common objective는 variant가 어떤 query loss를 학습했는지와 무관하게 같은
late-CEM candidate 전체에서 계산한다. Outer-val의 regret, Spearman, MSE 또는 GO
결과로 epoch를 고르거나 다시 학습하지 않는다. 세 seed 중 잘 나온 seed만 선택하는
것도 금지한다.

## 7. Primary metric과 secondary epsilon metric

각 state에서 predicted cost가 가장 낮은 candidate **하나**를 고른다. Predicted-cost
exact tie는 raw action SHA-256로 결정한다.

```text
selected = deterministic_exact_argmin(J_pred)
R_exact  = J_real[selected] - min(J_real)
R_eps    = max(0, R_exact - 0.005)
```

Primary는 outer state-macro `R_exact`다. `R_eps`는 `0.005` 이내의 행동을 실질적으로
동등하다고 보는 secondary sensitivity metric이다. V2처럼 predicted minimum 주변의
여러 후보 real cost를 평균하지 않으므로 loss가 만든 tie 집합에 벌점을 주는 모순이
없다. Exact top-1 accuracy, Spearman, outcome MSE와 best-second gap도 모두 보고하지만
primary gate를 대체할 수 없다.

## 8. 사전 GO/NO-GO gate

### 8.1 무결성과 문제 존재

Performance 평가 전에 다음을 모두 만족해야 한다.

- development에 정확히 `(32+16) x 3 x 24 = 3,456` records와 split code 0/1만 존재;
- development collision record/event 0개;
- sealed-test manifest가 `12 x 3 x 24 = 864` records, split code 2, file SHA-256과
  sealed role을 attest하고 payload가 별도 파일로 존재;
- 전체 `4,320` records는 두 manifest count의 합으로만 확인하며 runner가 sealed
  payload를 decode해 다시 세지 않음;
- fresh actor same-state, intended/applied control, cost 재계산 및 cleanup 통과;
- 60개의 unique base waypoint, speed/offset/curvature 균형 및 요청/실제 t=0 일치;
- Town05 map identity와 V1/V2 Town10 parent provenance 일치;
- global/elite 각각 state당 전체 24개 unique query, 두 arm의 동일 query budget.

또한 outer state 중 75% 이상에서 final candidate real-cost range가 `0.005`보다 커야
한다. 그렇지 않으면 exact-best 선택 문제 자체가 이 pool에서 너무 퇴화한 것이므로
method 성능과 무관하게 NO-GO다. Best-second gap은 삭제 기준이 아니라 그대로 보고한다.

### 8.2 Method GO

한 seed가 통과하려면 `elite_listwise`가 outer-val에서 다음을 **모두** 만족해야 한다.

1. `global_listwise` 대비 state-macro exact simple regret 20% 이상 감소;
2. `prediction_only` 대비 exact simple regret 20% 이상 감소;
3. global 대비 state-macro epsilon regret의 absolute 악화가 `0.00025` 이하;
4. prediction-only 대비 global outcome MSE 악화가 10% 이하.

Baseline exact regret가 수치적으로 0이면 20% 개선을 주장할 수 없으므로 해당 비교는
통과하지 않는다. 세 optimizer seed `[17,29,43]` 중 2개 이상이 통과해야 V3 mechanism
GO다. Initial state 단위 5,000회 paired bootstrap 95% interval은 의무 보고하지만,
16개 state와 optimizer seed 3개는 독립적인 대규모 반복이 아니므로 CI를 사후 대체
gate로 사용하지 않는다.

## 9. 한 번짜리 판정과 금지사항

V3에는 post-hoc remediation round가 0개다. Non-smoke 실행 뒤 collision, coverage,
수렴 또는 outer gate가 실패하면 이 protocol의 결론은 terminal NO-GO다. 다음 변경으로
같은 outer validation을 다시 사용하는 것은 금지한다.

- state 재수집/삭제/재배치 또는 유리한 curvature bin만 선택;
- loss, temperature, weight, epoch, patience, seed, metric 또는 threshold 변경;
- outer validation으로 checkpoint 선택;
- V1/V2 record/checkpoint/validation/test 재사용;
- V1/V2/V3 sealed test 열기;
- V3 sealed records/diagnostics를 열기, hash 계산, mmap 또는 decode하기;
- secondary epsilon regret, Spearman 또는 특정 seed로 primary 실패 구제.

Smoke run은 schema, reset, control 및 실행 가능성만 확인하는 비증거 출력이다. Smoke의
성능 숫자로 frozen 설정을 바꾸지 않는다. V3 GO가 나와도 selection manifest나 test
결과를 만들지 않으며, 별도 preregistration과 새 split 없이 visual world model 또는
closed-loop 단계 성공으로 확대 해석하지 않는다.

## 10. 동결 및 실행 순서

1. 이 protocol, config, collector/runner source 및 tests를 먼저 완료한다.
2. Unit test와 non-evidentiary smoke만 실행한다.
3. Freeze artifact에 config, 이 protocol, shared base collector, V3 collector,
   V3 runner, grounding core 및 두 V3 contract test의 SHA-256을 각각 기록한다.
   대상은 `configs/mpc_local_grounding_pilot_v3.yaml`, 이 문서,
   `scripts/collect_mpc_local_carla.py`, `scripts/collect_mpc_local_carla_v3.py`,
   `scripts/run_mpc_local_grounding_pilot_v3.py`,
   `src/temporal_tf/mpc_local_grounding.py`,
   `tests/test_mpc_local_carla_v3_collector.py`,
   `tests/test_mpc_local_grounding_v3.py`다. Frozen timestamp, CARLA client/server
   `0.9.15`, canonical map name, Python/NumPy/PyTorch 환경, 최종 contract-test 결과와
   smoke manifest SHA-256도 함께 기록한다.
4. 새 output root에 V3 non-smoke collection을 정확히 한 번 실행하고 development/test
   records와 diagnostics를 각각 별도 파일로 publish한다.
5. Runner에는 `development_records.npz`만 전달한다. 무결성 gate 통과 후
   fit/inner-val만 사용해 모든 checkpoint를 고정한다.
6. Outer-val을 한 번 열어 frozen gate를 계산한다.
7. Test를 열지 않고 GO/NO-GO와 모든 실패 원인을 보고한다.
