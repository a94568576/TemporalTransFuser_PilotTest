# MPC-Local Grounding 실패 원인 및 개선 가능성 감사

작성일: 2026-08-28

이 문서는 공식 V2 `NO-GO`와 post-hoc rankscale diagnostic을 변경하거나 구제하지 않는다. 기존 sealed test는 열지 않았으며, 아래 대안 metric 계산은 원인 파악을 위한 sensitivity analysis일 뿐이다.

## 결론

초기 실패에는 실제로 경미한 구현 문제가 있었다. real-cost 차이가 약 `0.008`인 elite pair를 temperature 1.0의 BCE logit으로 사용해 rank loss가 `log(2)` 근처에서 거의 작동하지 않았다. 이를 전역 temperature `0.005`로 고치자 elite Spearman과 regret가 크게 개선됐다.

수정 후 남은 실패는 하나의 코드 버그가 아니라 다음 네 요소의 결합이다.

1. elite query의 ordering 정보 밀도가 global query보다 낮다.
2. tie-aware 학습 목표와 selection-regret 정의가 충돌한다.
3. pairwise 평균 ordering 개선이 실제 top candidate 선택으로 안정적으로 이어지지 않는다.
4. validation state가 8개뿐이고, 그중 state 17은 train feature support 밖의 lateral OOD다.

추가로 scaled elite 모델은 모두 epoch 250 cap에서 선택되어 완전히 수렴하지 않았다. 개선 가능성은 있으나, 현재 validation을 본 뒤 epoch/loss/metric을 바꾸는 것은 사후 튜닝이다. 공식 판정을 바꾸려면 fresh preregistered split이 필요하다.

## 1. 고친 문제: rank-logit scale

공식 V2에서 elite real-cost pair margin 중앙값은 `0.007913`이었지만 logit은 cost 차이를 그대로 사용했다. 세 seed의 elite train rank loss는 `0.68849 / 0.68888 / 0.68830`으로 `log(2)=0.69315`에 가까웠다.

Diagnostic에서는 다음 하나만 바꿨다.

```text
rank_logit = (J_pred_j - J_pred_i) / 0.005
```

그 결과 elite train rank loss는 `0.31738 / 0.32132 / 0.35697`로 내려갔다. Gradient clipping은 0회였고 최대 pre-clip norm도 `0.635 < 5.0`이었다. 따라서 scale 수정 이후 실패를 gradient 폭주나 rank loss 미작동으로 설명할 수 없다.

## 2. Elite는 같은 pair 수를 받아도 ordering 정보가 적었다

Train에서 두 arm 모두 16 state × 64 pair = 1,024 pair를 받았다.

| Query | Non-tie | Tie | 실제 ordering label 비율 |
|---|---:|---:|---:|
| global | 937 | 87 | 91.5% |
| elite | 665 | 359 | 64.9% |

Elite는 global보다 informative non-tie pair가 272개, 약 29% 적었다. Late-CEM 후보가 실제로 가까워지는 현상 자체가 원인이다. Elite margin 중앙값은 `0.0079`, global은 `0.0704`였다.

따라서 현재 실험은 simulator candidate label 수는 같지만 ordering information budget까지 같은 비교는 아니다. 그렇다고 tie를 임의로 better/worse로 나누면 물리적으로 동등한 action을 거짓으로 분리하게 된다. 해결하려면 새 실험에서 rollout budget 대비 성능을 명시하고, uncertainty/disagreement 또는 predicted near-boundary 기반 acquisition을 사전 정의해야 한다.

## 3. Tie 학습과 regret metric의 의미가 충돌했다

Loss는 `abs(J_i-J_j) <= 0.005`를 tie target 0.5로 두어 두 predicted cost를 같게 만들도록 학습한다. 반면 protocol의 regret는 다음처럼 계산된다.

```text
predicted minimum ±0.005 안의 모든 후보를 선택 후보군으로 간주
regret = 그 후보들의 실제 cost 평균 - exact real minimum
```

즉 loss가 near-equal elite 후보를 올바르게 모을수록 predicted-min 후보군이 커지고, metric은 그 후보군의 평균을 exact 단일 최솟값과 비교해 다시 벌점화할 수 있다.

Validation 8개 state 모두 real best와 second-best의 차이가 `0.005`보다 작았다. State별 best-second gap 범위는 `0.000238~0.004964`였다.

이 영향이 가장 큰 예는 seed 29/state 22다.

- global protocol regret: `0`
- elite protocol regret: `0.005137`
- elite도 exact top-1 후보는 실제 최적 후보를 맞혔다.
- 다만 predicted minimum ±0.005 후보가 3개라 그 실제 cost 평균 때문에 regret가 생겼다.
- 이 한 state가 seed 29의 elite-global regret 초과분 약 95%를 설명한다.

Read-only sensitivity 결과는 다음과 같다.

| Seed | Protocol elite regret 개선 vs global | Exact-argmin 기준 | 0.005 tolerance-adjusted 기준 |
|---:|---:|---:|---:|
| 17 | +32.67% | +84.00% | 양쪽 모두 거의 0 |
| 29 | **−25.06%** | +14.53% | +55.92% |
| 43 | +2.09% | +47.06% | +83.45% |

이 결과는 elite가 완전히 무의미하다는 해석을 약화하지만 공식 NO-GO를 뒤집지는 않는다. Exact-argmin으로도 seed 29는 사전 기준 20%에 못 미치고, seed 43은 Spearman gate가 실패한다. Tolerance-adjusted regret는 결과를 본 뒤 계산한 새 metric이므로 confirmatory evidence가 아니다.

## 4. State 17 OOD가 seed 43 Spearman 실패를 지배했다

Seed 43에서 state 17의 elite-global Spearman 차이는 `−0.38531`이었다.

- global Spearman: `0.79509`
- elite Spearman: `0.40978`
- state 17을 사후 제외하면 평균 elite-global 차이는 `+0.05213`이지만, state 삭제는 허용할 수 없다.
- state 17의 real-cost 전체 범위는 `0.017168`뿐이고 candidate pair의 약 60.5%가 tie다.
- global predicted-min 후보군은 5개, elite는 14개였다.

State 17의 normalized lateral feature는 `0.06738`이고 train state 평균/표준편차는 `−0.0105 / 0.0196`이다. 즉 약 `+3.98σ`이며 실제 lateral offset은 약 `0.202 m`다. Train의 최대 양의 normalized lateral은 `0.01516`뿐이었다.

이는 state 17을 제거해야 한다는 뜻이 아니라, train state 수와 lateral/curvature coverage가 부족했다는 뜻이다. Validation 8개 중 한 state가 macro 평균의 12.5%이므로 현재 표본에서 seed 안정성도 약하다.

## 5. Pairwise ranking과 top-choice 목적이 다르다

Scale 수정 후 seed 29에서는 elite Spearman이 global보다 `+0.08472` 높았지만 protocol regret는 25.06% 악화됐다. 전체 후보 순서를 더 잘 맞히는 것과 최상위 하나를 고르는 것은 같은 목적이 아니다.

현재 BCE는 모든 sampled pair를 거의 같은 비중으로 본다. Top candidate 주변의 오류를 직접 줄이는 objective가 아니다. 이를 해결하려면 top-weighted listwise loss, calibrated cost regression, differentiable regret surrogate 등을 새 방법으로 비교해야 한다. 이는 구현 버그 수정이 아니라 연구 방법 변경이다.

## 6. 수렴 한계

Scaled elite 세 run은 모두 `best_epoch=250`이고 early stop되지 않았다. 마지막 30 epoch의 common validation objective 변화는 다음과 같다.

```text
seed 17: -0.044
seed 29: -0.043
seed 43: -0.022
```

따라서 충분히 수렴했다고 단정할 수 없다. 다만 common objective에는 full-set selection regret가 없기 때문에 epoch를 늘린다고 핵심 실패가 반드시 해결되지는 않는다. 기존 validation curve를 본 뒤 max epoch를 늘려 성공 판정하는 것은 validation-adaptive tuning이다.

## 개선 가능성 판정

### 현재 V2 결과를 구제할 수 있는가?

아니다. 현재 code는 frozen protocol을 그대로 수행했고 sign, index remapping, cost 재계산, reset/control, pair budget에서 추가 런타임 버그가 발견되지 않았다. Metric이나 epoch를 지금 바꾸면 사후 변경이다.

### 같은 연구 질문을 새 실험에서 개선할 수 있는가?

가능성은 있다. 다음을 모두 결과 확인 전에 고정해야 한다.

1. `raw simple regret`, deterministic exact-argmin regret, `epsilon=0.005` regret, top-k recall을 함께 보고하고 primary metric을 하나만 사전 선택한다.
2. Tie를 동등 행동으로 정의한다면 loss와 primary regret에 같은 epsilon 의미를 적용한다.
3. Candidate label 수, non-tie pair 수, 전체 CARLA rollout 수를 각각 회계하고 rollout-budget curve로 global/elite를 비교한다.
4. Hash-random elite 절반 대신 base-model uncertainty/disagreement와 predicted near-optimality를 결합한 query rule을 cost label 확인 전에 고정한다.
5. 모든 variant에 충분히 큰 동일 max epoch와 convergence-based early stopping을 미리 고정한다.
6. Lateral offset, speed, curvature를 층화해 더 많은 독립 train/validation state를 수집한다. 기존 state 17 같은 OOD를 삭제하지 않는다.
7. 현재 traffic-free smooth ego dynamics를 넘어가려면 collision/actor-motion head와 sequential action-conditioned semantic/occupancy model을 먼저 추가한다.
8. 기존 V2 sealed test는 계속 열지 않고, 새 outer validation/test split으로 확인한다.

## 권고

현재 validation에서 추가 temperature, epoch, tie threshold, loss를 탐색하지 않는다. 가장 저렴한 정당한 다음 단계는 **새 state split을 가진 preregistered V3 mechanism study**이며, 위 tie-consistent metric과 convergence 정책을 먼저 고정해야 한다.

다만 이는 현재 결과의 사소한 패치가 아니라 새 확인 실험이다. 비용을 아끼는 것이 우선이면 elite-query 아이디어를 중단하고 다른 가설로 이동하는 기존 권고가 여전히 합리적이다.
