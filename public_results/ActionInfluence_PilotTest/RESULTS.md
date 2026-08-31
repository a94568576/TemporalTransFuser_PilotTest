# Action Influence Pilot V1 Results

실행 완료: `2026-08-28T06:32:33.759636+00:00`

## 판정: `no_go`

Centered residual은 warped passive base보다 개선됐지만, 전체 pipeline이 단순 raw-latent persistence보다 나빴다. TF++ latent의 metric-warp 기반 spatial decomposition은 현재 형태로 확장하지 않는다.

이 결과는 frozen TF++ BEV latent와 단일 logged-policy action을 이용한 **observational proxy**다. same-state multi-action counterfactual 정확도나 closed-loop 개선을 입증하지 않는다.

## Stage 0

Passive base는 action 입력이 없으므로 action neglect 진단은 구조적으로 `N/A`다. 모든 action label의 출력 불변성만 확인했다.

- frozen passive validation route-macro standardized MSE: `0.550359`
- action sensitivity: `structurally_zero_no_action_input`
- control output max difference: `0.0`

## Stage 1 validation 결과

| Variant | Seed | true MSE | base 대비 개선 | shuffled penalty | sensitivity |
|---|---:|---:|---:|---:|---:|
| uncentered_film | 17 | 0.493728 | +10.29% | +5.02% | 0.046101 |
| uncentered_film | 29 | 0.491582 | +10.68% | +4.72% | 0.040993 |
| uncentered_film | 43 | 0.486030 | +11.69% | +4.87% | 0.049738 |
| centered_film_k4 | 17 | 0.523762 | +4.83% | +4.04% | 0.027189 |
| centered_film_k4 | 29 | 0.524185 | +4.76% | +4.22% | 0.024708 |
| centered_film_k4 | 43 | 0.525058 | +4.60% | +3.42% | 0.024307 |

## Centered primary gate

- 3% 이상 개선 seed: `3`
- pooled improvement: `{'point_percent': 4.7285474150567754, 'ci95_lower_percent': -0.6233894722978667, 'ci95_upper_percent': 9.290821063024795, 'bootstrap_unit': 'route', 'routes': 3, 'samples': 2000, 'ratio_denominator': 'numerator'}`
- pooled shuffled penalty: `{'point_percent': 3.8927888191280524, 'ci95_lower_percent': 3.2113738063395965, 'ci95_upper_percent': 4.843406445676593, 'bootstrap_unit': 'route', 'routes': 3, 'samples': 2000, 'ratio_denominator': 'denominator'}`

## Mandatory persistence sanity

- raw unaligned persistence MSE: `0.426708`
- oracle-warp persistence MSE: `0.855361` (`+100.46%` vs raw)
- uncentered 3-seed mean: `0.490447` (`+14.94%` vs raw)
- centered 3-seed mean: `0.524335` (`+22.88%` vs raw)
- baseline gate: `fail`

Spatial `.npy` 파일은 모델 출력이 action control에 반응한 위치를 나타낼 뿐 causal influence map이 아니다.

## 다음 단계

Query adapter는 중단한다. 계속하려면 warp-equivariant occupancy/semantic latent 또는 실제 action-conditioned world model을 선택하고, CARLA same-state paired action 데이터에서 처음부터 재검증한다.
