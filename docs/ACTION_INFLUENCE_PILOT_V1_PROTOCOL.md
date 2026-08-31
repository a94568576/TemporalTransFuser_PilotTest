# Action Influence Pilot V1 Protocol

## 1. 목적과 허용되는 결론

이 파일럿은 frozen TF++ native BEV latent의 1초 endpoint 변화를 예측할 때, 로그에 저장된 ego control이 **관측적 예측 신호**로 유용한지 진단한다. Action-free passive predictor를 먼저 학습해 동결한 뒤, 같은 frozen base 위에서 equal-capacity uncentered/centered FiLM residual을 비교한다.

고정된 질문은 다음과 같다.

> V1 로그의 train/validation 분포에서 4 Hz `steer/throttle/brake` proxy를 입력한 K=4 centered FiLM residual이 frozen passive base를 개선하고, 올바른 action과 shuffled action을 구별하는가?

이 실험은 full action-conditioned future prediction이 아니다. Recorded future ego pose로 현재 BEV를 미래 ego frame에 미리 정렬한 뒤 action residual을 평가하므로, 정확한 명칭은 **true-geometry-conditioned interaction residual diagnostic**이다.

허용되는 결론은 validation 내부의 observational action utility와 model sensitivity뿐이다. 결과와 관계없이 다음을 주장하지 않는다.

- same-state multi-action counterfactual 정확도 또는 action의 인과 효과
- deployable future ego-motion prediction
- 다른 actor의 causal interaction을 분리했다는 결론
- spatial map이 causal action-influence 영역이라는 결론
- offline latent 개선이 closed-loop driving 개선을 뜻한다는 결론
- TF++ perception backbone이나 locally trained passive predictor가 pretrained world model이라는 주장

### 1.1 Protocol revision provenance

첫 full run이 끝난 뒤, learned/oracle-warped pipeline을 단순 raw-latent persistence와 비교하는 trivial sanity baseline이 원 protocol에서 누락됐음을 발견했다. Final rerun 전에 `gates.require_beat_raw_persistence: true`를 mandatory gate로 추가했다.

- 이 correction 전후로 model architecture와 training hyperparameter는 변경하지 않았다.
- Final rerun은 model과 training hyperparameter를 그대로 두고 verdict config에 이 gate만 추가해 다시 수행한다.
- 이 gate는 첫 full-run 결과를 본 뒤 추가한 protocol correction이다.
- 따라서 이 gate가 원래부터 preregistered 또는 사전 고정돼 있었다고 주장하지 않는다.
- 결과 보고서에는 이 revision의 시점, 이유, 변경 범위를 그대로 공개한다.

Correction 이후의 유효 verdict에서는 raw-persistence gate가 다른 centered gate보다 우선한다.

## 2. 실제 데이터와 test 경계

### 2.1 Native cache

실행 config의 cache는 다음이다.

```text
data/cache_action_latent_town13_v1_native64
```

- feature source: frozen TF++ `backbone_bev`
- record당 latent: `[64,64,64]`
- cache schema: v3
- 분석 split: `train`, `val`

Model fitting, checkpoint selection, metric, bootstrap, verdict에는 train/val만 사용하고 test record를 평가하지 않는다. `ActionTransitionDataset(..., allow_test=False)`가 이 경계를 강제한다.

그러나 이 cache는 extraction 시 test를 포함해 materialize되었고, deep audit도 `selected_splits=all`, `deep_splits=all`로 1,733개 전체 record를 읽었다. 감사 결과는 `artifacts/action_influence_pilot_v1_native_cache_audit.json`에 있으며 split record 수는 train 988, val 298, test 447이다. 따라서 이 연구는 다음과 같이만 표현한다.

> Test는 action-influence 학습·선택·평가에 사용하지 않았다.

`untouched test`, `unopened test`, `first test open`은 주장하지 않는다.

### 2.2 Endpoint transition

한 sample은 중간 네 latent를 예측하지 않는다. 저장 frame cadence 0.25초에서 `t→t+4` endpoint 하나만 예측한다.

```text
raw current:  Z_t                         [64,64,64]
action proxy: A_t, A_{t+1}, A_{t+2}, A_{t+3}  [4,3]
raw target:   Z_{t+4}                     [64,64,64]
nominal span: 4 saved frames = 1.0 s
```

Window는 같은 route의 연속 frame에서만 만들고 `max_frame_gap=1`, expected cadence `0.25 s`를 요구한다. 조건을 만족하지 않는 window는 보간하지 않고 제외한다. Train/val route 경계를 넘는 window는 만들지 않는다.

`horizon: 4`는 action sequence 길이와 endpoint 간격을 뜻하며, 출력 horizon 네 개를 뜻하지 않는다.

## 3. Saved action의 실제 의미

각 saved frame의 raw measurement에서 다음 proxy를 읽는다.

```text
a_t = [saved steer, saved throttle, bool(brake OR control_brake)]
```

구체적 의미는 다음과 같다.

- `steer`: actuation noise가 더해지기 전에 저장된 steer
- `throttle`: 일부 runtime override가 적용되기 전에 저장된 throttle
- `brake`: measurement의 `brake`와 `control_brake` 중 하나라도 참이면 `1`, 아니면 `0`
- 저장 cadence: 4 Hz
- 실제 controller actuation cadence: 20 Hz

따라서 `[4,3]` sequence는 1초 구간의 exact executed 20 Hz actuator sequence가 아니다. 저장 시점 사이의 control과 후속 override를 잃은 observational proxy다. 문서와 결과에서 `executed action`, `exact low-level actuation`이라고 부르지 않는다.

Raw bounds는 `steer∈[-1,1]`, `throttle∈[0,1]`, brake boolean/0/1로 검사한다. Action normalization은 train transition의 네 saved action 전체에서 차원별 mean/std를 계산해 적용한다.

## 4. Oracle geometry alignment와 loss support

### 4.1 실제 warp

각 transition에서 recorded current/future pose를 사용해 다음 호출을 먼저 수행한다.

```python
warped_current, validity = warp_bev_to_current(
    raw_current,
    current_pose,
    future_pose,
)
```

여기서 `warped_current`는 `Z_t`를 **recorded `t+4` ego frame**으로 옮긴 tensor다. Raw target `Z_{t+4}`는 이미 같은 future ego frame에 있다. Passive predictor의 입력은 `warped_current`이고, adapter의 scene 입력도 `warped_current + frozen base`이므로 두 입력 모두 target과 같은 frame이다.

Grid sampling support 밖의 warped cell은 실제 zero latent가 아니다. 구현은 그 cell을 train target channel mean으로 채운 뒤 표준화해 0이 되게 한다.

### 4.2 Train-target standardization

Channel statistics는 오직 train split의 **raw future endpoint target** 전체 cell에서 계산한다.

```text
Z_std[c,y,x] = (Z[c,y,x] - latent_mean[c]) / latent_std[c]
latent_std = sqrt(max(train variance, 1e-8))
```

같은 통계를 warped current와 raw future target에 적용한다. Action은 train action mean과 `max(std,1e-6)`으로 별도 표준화한다. Validation이나 test로 normalization 통계를 다시 계산하지 않는다.

### 4.3 Masked standardized latent MSE

모든 train loss와 validation metric은 oracle warp의 overlap validity mask `[B,1,64,64]` 안에서만 계산한다.

```text
sample_mse =
  sum((prediction - target)^2 * validity)
  / max(sum(validity) * 64, 1)
```

Primary metric은 sample MSE를 route별로 먼저 평균한 뒤 route에 동일 가중치를 주는 validation route-macro standardized latent MSE다. Micro MSE와 mean valid fraction도 diagnostic으로 보고한다.

### 4.4 과학적 의미의 제한

모든 action control은 실제 기록된 동일한 `future_pose`와 동일한 validity mask를 공유한다. 예를 들어 `zero`나 `shuffled` action을 넣어도 geometry는 그 action이 만들었을 pose로 바뀌지 않는다. Ego-motion의 큰 부분이 true future geometry로 이미 제공되므로 이 실험은 다음을 측정하지 못한다.

- action 후보별 full future state
- action 후보별 ego trajectory
- counterfactual ego-motion 및 그에 따른 world response

측정 대상은 oracle geometry 뒤에 남은 latent error를 logged action proxy가 얼마나 보정하는지다.

## 5. 고정 config

Authoritative config는 `configs/action_influence_pilot_v1.yaml`이다. 실제 key와 값은 다음과 같다.

```yaml
torch_num_threads: 4

data:
  cache: data/cache_action_latent_town13_v1_native64
  analysis_splits: [train, val]
  horizon: 4
  horizon_seconds: 1.0
  prepare_batch_size: 8
  oracle_current_to_future_se2_warp: true

base:
  seed: 17
  hidden_channels: 16
  epochs: 20
  patience: 5
  batch_size: 16
  learning_rate: 0.001
  weight_decay: 0.0001

adapter:
  hidden_channels: 16
  action_hidden_dim: 64
  gate_bias: -5.0
  residual_budget_weight: 0.0001
  reference_count: 4
  epochs: 25
  patience: 6
  batch_size: 8
  learning_rate: 0.001
  weight_decay: 0.0001

seeds: [17, 29, 43]

evaluation:
  bootstrap_samples: 2000
  finite_difference_samples: 64
  finite_difference_perturbation: 0.1

gates:
  improvement_percent: 3.0
  shuffled_penalty_percent: 2.0
  required_seeds: 2
  require_beat_raw_persistence: true
```

`3%`, `2%`, `2/3 seeds`, bootstrap 수와 sensitivity threshold는 인과적·통계적 사실이 아니라 첫 full run 전에 고정한 운영 가정이다. 반면 `require_beat_raw_persistence`는 위 revision note에 적은 사후 protocol correction이며 원 preregistration의 일부가 아니다. `--smoke`는 base/adapter 2 epoch, seed 17 하나, bootstrap 200, finite-difference sample 8로 축소하는 plumbing 검사이며 연구 결과로 사용하지 않는다.

## 6. Passive predictor와 Stage 0

### 6.1 Action-free passive predictor

Passive predictor는 warped current endpoint-aligned latent만 받는다.

```text
delta = Conv3x3(64 -> 16)
        -> GELU
        -> Conv3x3(16 -> 16)
        -> GELU
        -> Conv1x1(16 -> 64)

base = warped_current + delta
```

마지막 `1x1` weight/bias는 zero initialization이므로 초기 출력은 정확히 persistence다. Base는 seed 17, AdamW, 최대 20 epoch로 한 번 학습한다. Masked train MSE로 update하고 validation route-macro MSE로 best epoch를 선택하며 patience는 5다. Gradient norm은 5로 clip한다.

Best checkpoint를 불러온 뒤 `requires_grad_(False)`와 `eval()`로 동결하고 모든 Stage 1 seed/variant가 같은 checkpoint를 공유한다.

### 6.2 Stage 0의 구조적 N/A

Passive predictor에는 action input path가 없으므로 action-neglect 진단은 구조적으로 `N/A`이고 finite-difference sensitivity는 0이다. Stage 0은 다음 invariance만 기록한다.

- `true`, `shuffled`, `zero`, `hold`, `reverse`, `other` label에서 같은 base output/MSE
- `max_control_output_difference = 0`
- `action_sensitivity = structurally_zero_no_action_input`
- native shape, route split, frame cadence, measurement join, normalization 및 warp audit 통과

Stage 0 결과로 pretrained model이 action을 무시한다고 결론 내리지 않는다. 이 base는 애초에 action-free로 설계하고 이 데이터에서 새로 학습했다.

## 7. Stage 1 모델

### 7.1 비교군

| Variant | Prediction | 역할 |
|---|---|---|
| `frozen_base` | `base` | Action-free 기준 |
| `uncentered_film` | `base + gate*r_raw(query)` | Generic residual/action conditioning 대조군 |
| `centered_film_k4` | `base + gate*(r_raw(query)-mean_k r_raw(ref_k))` | Primary method |

두 FiLM variant는 같은 architecture와 trainable parameter 수를 가지며 centering만 parameter-free다. 구현은 parameter count equality를 검사하고 다르면 중단한다.

### 7.2 실제 FiLM branch

Warped current와 detached frozen base를 channel 방향으로 연결한다.

```text
state = concat(warped_current, base)                  [128,64,64]
state_encoder:
  Conv3x3(128 -> 16) -> GELU -> Conv3x3(16 -> 16) -> GELU

action_encoder:
  flatten standardized action [4,3] -> [12]
  Linear(12 -> 64) -> GELU -> Linear(64 -> 32)
  split -> gamma[16], beta[16]
  gamma = tanh(gamma)

FiLM = state_feature * (1 + gamma) + beta
output_head:
  Conv3x3(16 -> 16) -> GELU -> Conv1x1(16 -> 64)
```

Output head의 마지막 `1x1`은 zero initialization이다. Scalar gate는 `sigmoid(gate_logit)`이며 `gate_logit=-5.0`, 즉 약 `0.0067`에서 시작한다. Prediction은 frozen base와 같은 single endpoint `[64,64,64]`이다.

### 7.3 Query별 K=4 centering

Centered variant는 action vector 평균을 빼지 않는다. 각 query sample마다 고정된 train-only reference action sequence 네 개를 FiLM branch에 통과시킨 뒤 **raw residual의 평균**을 뺀다.

```text
r_query = r_raw(warped_current, base, A_query)
r_ref_k = r_raw(warped_current, base, A_ref_k),  k=1..4
r_centered = r_query - (1/4) * sum_k r_ref_k
prediction = base + sigmoid(gate_logit) * r_centered
```

- Reference action은 raw physical-unit train action pool에서만 고른 뒤 train 통계로 표준화한다.
- Train query와 validation query 모두 train-only references를 사용한다.
- `sha256("action_reference_v1|sample_id|reference_index")`로 시작 index를 정하고, 가능한 경우 query와 다른 route의 train sample을 선택한다.
- Reference mapping은 학습 전에 고정하며 seed와 control 사이에서 바꾸지 않는다.
- Reference 선택에 future target이나 model error를 사용하지 않는다.

### 7.4 학습

- Seed: `17`, `29`, `43`
- 각 seed에서 uncentered와 centered를 독립 학습
- Optimizer: AdamW, learning rate `0.001`, weight decay `0.0001`
- 최대 25 epoch, patience 6, batch size 8
- Gradient norm clip: 5
- Checkpoint selection: true-action validation route-macro masked standardized MSE
- Train input: true action만 사용

Loss는 다음과 같다.

```text
applied_residual = prediction_true - frozen_base
loss = masked_standardized_future_mse
       + 0.0001 * masked_mean(applied_residual^2)
```

같은-state paired target이 없으므로 counterfactual-difference loss는 사용하지 않는다.

## 8. Action controls

구현의 candidate key 순서는 다음과 같다.

```text
true, shuffled, zero, hold, reverse, other
```

| Key | 실제 정의 | 판정 역할 |
|---|---|---|
| `true` | 해당 sample의 intact `[4,3]` proxy | Primary 입력 |
| `shuffled` | validation sequence 전체의 고정 cyclic derangement, seed 1701 | Authoritative action-use control |
| `zero` | raw physical space의 literal `[0,0,0]` 네 개 | Zero diagnostic |
| `hold` | 첫 saved action을 네 위치에 반복 | Temporal-change diagnostic |
| `reverse` | 네 saved action의 시간 순서를 뒤집음 | Order diagnostic |
| `other` | shuffled와 다른 고정 cyclic derangement, seed 2903; 같으면 2904 | 두 번째 mismatch diagnostic |

`zero`는 standardized zero가 아니라 raw literal zero를 만든 후 train 통계로 normalize한다. `shuffled`와 `other`는 fixed point가 없어야 하며 서로 다른 permutation이어야 한다. 모든 control은 같은 warped current, frozen base, K=4 references, recorded future pose, target과 validity mask를 공유한다.

## 9. 평가 지표

### 9.1 Control MSE와 primary improvement

각 model/seed/control에 대해 validation micro MSE와 route-macro MSE를 저장한다. Seed별 centered improvement는 다음이다.

```text
improvement_seed(%) =
  100 * (base_route_macro_mse - centered_true_route_macro_mse)
      / base_route_macro_mse
```

Uncentered 결과는 equal-capacity diagnostic이고, gate의 primary는 `centered_film_k4`다.

### 9.2 Shuffled penalty

```text
shuffled_penalty(%) =
  100 * (centered_shuffled_mse - centered_true_mse)
      / centered_true_mse
```

양수이면 correct logged proxy가 fixed mismatched proxy보다 낮은 observational prediction error를 만들었다는 뜻이다. Causal action correctness를 뜻하지 않는다.

### 9.3 Pooled route bootstrap

- `evaluation.bootstrap_samples: 2000`
- 세 centered seed의 sample error를 먼저 평균
- 동일 sample의 paired error를 유지
- route별 평균을 만든 뒤 route를 replacement sampling
- percentile 95% CI
- improvement bootstrap RNG seed 701
- shuffled penalty bootstrap RNG seed 1701

Improvement percent denominator는 base MSE이고 shuffled penalty denominator는 true-action MSE다. Validation route가 세 개뿐이므로 CI는 매우 거친 exploratory uncertainty다. 세 training seed를 독립 route로 세지 않는다.

### 9.4 Mandatory raw-persistence sanity

단순 baseline은 warp나 learned parameter 없이 raw current latent를 그대로 endpoint target과 비교한다.

```text
raw_persistence_error_i = masked_mse(
  normalize(raw Z_t),
  normalize(raw Z_{t+4}),
  oracle_warp_overlap_validity_mask,
)
```

비교 양쪽은 동일한 train-target normalization과 동일한 oracle-warp overlap validity mask를 사용한다. Raw current 자체에는 SE(2) warp를 적용하지 않는다.

Centered 값은 세 seed의 true-action sample error를 sample별로 먼저 평균한 뒤 equal-route macro mean을 계산한다.

```text
centered_three_seed_mean_true_route_macro_mse
  = route_macro_mean(mean_seed(centered_true_sample_error))

raw_persistence_pass =
  centered_three_seed_mean_true_route_macro_mse
  < raw_unaligned_persistence_route_macro_mse
```

Strict `<` 비교이며 동률은 실패다. `gates.require_beat_raw_persistence: true`이므로 실패하면 improvement, shuffled penalty, bootstrap, sensitivity가 좋아도 최종 verdict는 `no_go`다.

이 baseline은 첫 full run 뒤 누락을 발견해 final rerun 전에 추가한 mandatory sanity correction이다. 원래부터 preregistered였다고 소급해 표현하지 않는다.

### 9.5 Action separation

동일 sample의 여섯 candidate prediction에서 가능한 15개 pair의 masked standardized RMS difference를 계산하고, pair와 sample 전체 평균을 `action_separation_rms`로 보고한다.

```text
separation(c1,c2) = sqrt(masked_mse(prediction_c1, prediction_c2))
```

Separation은 반응량이지 방향의 정확성이나 인과성 지표가 아니다.

### 9.6 Finite-difference sensitivity

Validation dataset 순서의 앞 `min(64,N)` sample에서 standardized action의 12개 좌표 각각에 `±0.1`을 적용한다.

```text
derivative_j = (prediction(A + 0.1 e_j) - prediction(A - 0.1 e_j)) / 0.2
S_fd = mean_j,sample sqrt(masked_mse(derivative_j, 0))
```

구현은 standardized 좌표를 직접 perturb하며 raw bound로 clip하지 않는다. Gate는 centered 세 seed sensitivity 평균이 `1e-6`보다 큰지만 검사한다. Sensitivity CI는 현재 gate에 없다.

### 9.7 Spatial response map

각 control에서 true prediction과의 차이를 overlap mask 안에서 channel/sample RMS로 모아 `[64,64]` `.npy`를 저장한다.

```text
map(control)[y,x] = RMS_channel,sample(prediction_control - prediction_true)
```

파일명은 `true_vs_<control>.npy`다. 이 map은 oracle-warped model response map이며 causal influence, actor interaction 또는 counterfactual difference map이 아니다.

### 9.8 추가 diagnostic

- applied residual RMS / frozen-base error RMS
- learned sigmoid gate
- oracle-warp persistence MSE와 learned passive MSE
- oracle-warp overlap mean valid fraction
- State-only ridge probe와 state+predicted-residual ridge probe의 action recovery `mean_r2`, normalized MAE

이 절의 diagnostic은 GO boolean을 바꾸지 않는다. Raw unaligned persistence는 9.4의 mandatory gate이므로 diagnostic-only가 아니다.

## 10. 실제 GO / NO-GO / INCONCLUSIVE gate

`3%`, `2%`, `2/3 seeds`, bootstrap 수와 `1e-6` sensitivity threshold는 첫 full run 전에 고정한 운영 가정이며 결과를 본 뒤 완화하지 않는다. Raw-persistence sanity는 첫 full run 뒤 추가한 공개된 protocol correction이며, 이 provenance를 다른 사전 gate와 구분한다.

### 10.1 실행 전제

다음이 실패하면 verdict를 계산하지 않고 run을 실패시킨다.

- Native cache feature source와 `[64,64,64]` shape 확인
- `analysis_splits == [train,val]`
- `oracle_current_to_future_se2_warp == true`
- Action/route/cadence/reference/control manifest 검사
- Frozen base 유지
- Uncentered/centered trainable parameter count 동일

### 10.2 Proxy GO

최종 Proxy GO status는 다음을 모두 요구한다. 구현 내부의 기존 centered `go` boolean이 1~5를 계산하고, raw-persistence 우선 분기가 6을 강제한다.

1. 세 seed 중 최소 `gates.required_seeds: 2`개에서 centered true MSE가 base보다 `gates.improvement_percent: 3.0%` 이상 개선된다.
2. Pooled centered improvement의 route-bootstrap 95% CI lower bound가 `>0`이다.
3. Pooled shuffled penalty point estimate가 `gates.shuffled_penalty_percent: 2.0%` 이상이다.
4. Pooled shuffled penalty의 95% CI lower bound가 `>0`이다.
5. 세 centered seed의 finite-difference sensitivity 평균이 `>1e-6`이다.
6. Centered 3-seed mean true route-macro MSE가 raw unaligned `Z_t` persistence route-macro MSE보다 엄격히 낮다.

통과 status의 실제 이름은 다음이다.

```text
go_to_paired_counterfactual_data_not_stage2_model_yet
```

즉 proxy GO는 query adapter를 즉시 허가하지 않는다. 먼저 작은 CARLA same-state multi-action paired dataset을 사전등록하고 counterfactual difference target으로 Stage 1을 재검증해야 한다.

### 10.3 NO-GO

GO가 아니면서 다음 중 하나면 구현은 `no_go`로 판정한다.

- Mandatory raw-persistence sanity 실패. 이 조건은 최우선이며 다른 centered gate 신호를 모두 override한다.
- pooled centered improvement point estimate `<1%`
- pooled improvement 95% CI upper bound `<=0`
- `abs(pooled shuffled penalty)<1%`인데 pooled improvement point estimate가 양수인 generic-correction pattern

NO-GO에서는 current observational result를 근거로 query/cross-attention Stage 2를 확장하지 않는다.

### 10.4 INCONCLUSIVE

Mandatory raw-persistence sanity를 통과했지만 GO와 나머지 NO-GO 조건 모두에 해당하지 않으면 `inconclusive`다. 일부 개선 신호가 있지만 route-bootstrap 또는 action-use gate가 모두 충족되지 않은 경우다. 현재 validation에 맞춰 추가 tuning하지 않고 paired data 또는 독립 route를 확보한 뒤 같은 고정 설정을 재검증한다.

현재 구현은 status와 관계없이 다음을 기록한다.

```text
stage2_query_adapter_authorized: false
```

Proxy GO는 Stage 2의 필요조건일 수 있으나 충분조건은 아니다.

## 11. 재현성과 산출물

실행기는 Python/NumPy/PyTorch seed를 고정하고 `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms `warn_only=True`, cuDNN deterministic, benchmark off, TF32 off를 사용한다. `warn_only=True` 경고가 있는 run은 bitwise reproducible하다고 주장하지 않는다.

필수 산출물은 다음이다.

```text
results.json
RESULTS.md
data_manifest.json
control_manifest.json
train_normalization.pt + SHA-256
base/best.pt + SHA-256
uncentered_film/seed_{17,29,43}/best.pt
centered_film_k4/seed_{17,29,43}/best.pt
variant/seed별 spatial_maps/true_vs_<control>.npy
config/cache/control/sample manifest hashes
seed/control/sample/route metric과 pooled bootstrap 결과
machine-readable verdict
```

기존 non-empty output directory는 덮어쓰지 않는다. 모든 paired metric은 같은 validation sample ID와 순서를 유지한다. 실행은 다음 entry point를 사용한다.

```bash
cd ${PROJECT_ROOT}
PYTHONPATH="$PWD/src" python scripts/run_action_influence_pilot.py \
  --config configs/action_influence_pilot_v1.yaml \
  --output artifacts/action_influence_pilot_v1 \
  --device cuda
```

## 12. 다음 단계

Stage 1이 proxy GO여도 바로 기존 제안의 K-query/spatial adapter를 구현하지 않는다. 우선 동일 초기 state에서 여러 action을 실행한 simulator rollout과 action별 ego pose를 확보하고, 각 candidate를 자기 geometry로 정렬한 새 paired protocol을 작성한다.

그 paired 재검증까지 통과한 경우에만 Stage 2 architecture를 별도 protocol로 검토한다. 현재 V1 native cache와 oracle geometry diagnostic만으로는 counterfactual action-influence contribution을 뒷받침할 수 없다.
