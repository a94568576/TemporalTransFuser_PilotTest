# Offline pilot protocol

## 1. 고정된 연구 질문

Frozen single-frame planner가 만든 현재 path의 오차를 과거 frozen prediction과 과거 scene feature가 보정할 수 있는지 확인합니다. Adapter 외 parameter는 학습하지 않습니다.

로컬 released TF++ checkpoint에서는 target이 시간 trajectory가 아니라 geometric checkpoint path입니다. 문서와 결과에서 이를 trajectory로 바꿔 부르지 않습니다. 진짜 future ego trajectory 연구는 대응 prediction head/checkpoint를 별도로 확보한 뒤 새 protocol로 분리합니다.

## 2. Data contract

각 frame record에는 다음만 둡니다.

- `bev_feature [C,H,W]`: frozen forward hook 결과
- `pred_trajectory [N,2]`: frozen `pred_wp` 또는 `pred_checkpoint`
- `gt_trajectory [N,2]`: 현재 frame loss/evaluation label
- `ego_pose [x,y,yaw]`: feature/prediction과 동일 timestamp
- `route_id`, `frame_id`, `timestamp`
- `speed_t` scalar (m/s), `command_t` 6-way one-hot: frozen planner에 실제로 공급한 causal input의 분석용 provenance
- `trajectory_source=frozen_model_prediction`

Dataset은 history를 만들 때 `pred_trajectory`만 stack합니다. `speed_t`와 `command_t`는 cache에만 보존하고 현재 adapter에는 노출하지 않습니다. 과거 `gt_trajectory`, future pose, actor GT, oracle/failure label은 adapter 입력으로 금지합니다.

## 3. Split and coordinate policy

- 먼저 route 전체를 train/val/test에 배정하고 그 안에서만 sliding window를 만듭니다.
- 동일 `route_id`가 여러 split에 등장하면 audit failure입니다.
- frame은 strictly increasing이고 기본 gap은 1입니다.
- past predicted path는 full SE(2)로 current ego frame에 변환합니다.
- 현재 Pilot의 pooled BEV는 warp하지 않습니다. 이것은 명시된 limitation입니다.

## 4. Model-selection policy

- epoch, learning rate, residual λ, gate, architecture 선택은 validation만 사용합니다.
- 각 variant의 best epoch는 validation window의 sample-micro 평균이 아니라 route별 ADE를 먼저 평균한 **equal-route macro ADE** 최소값으로 고릅니다. 매 epoch의 validation `sample_id`와 `route_id` 순서가 고정 baseline 순서와 정확히 같지 않으면 selection을 중단합니다.
- test는 선택 종료 후 최종 1회 보고합니다.
- Cache index는 각 record SHA256을 포함하고, selection artifact는 config/cache-index/checkpoint hash를 고정합니다. Final은 이 checkpoint를 재학습하지 않고 로드하며, test tensor를 읽기 전에 permanent atomic open marker를 남깁니다.
- λ 후보는 `0.01, 0.05, 0.1`; 후보별 별도 output directory와 `selection` mode를 사용합니다. 확정 설정만 `final` mode로 test를 한 번 엽니다.
- frozen baseline, current-only, parameter-matched current-only, trajectory-only, current-BEV, correct past-BEV, within-window shuffled past-BEV, combined를 같은 split에서 비교합니다.
- 세 고정 seed `17/29/43`의 selection을 모두 validation에서 완료한 뒤, study-level final command 하나로 test를 엽니다. Seed별 final 사이에서 판단하거나 설정을 바꾸지 않습니다.
- Multi-seed child selection은 parent-owned로 잠겨 단독 final을 거부합니다. λ comparator의 `selection_choice.json`을 final 입력으로 사용하고 chosen study path·manifest hash·λ를 재검증하므로 사람이 임의 study path를 치환하지 않습니다.
- CUDA study는 새 process를 시작하기 전에 `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 export합니다. Engine은 Python/NumPy/PyTorch/CUDA seed, `cudnn.deterministic=True`, `cudnn.benchmark=False`, TF32 off를 고정하고 `torch.use_deterministic_algorithms(True, warn_only=True)`를 사용합니다. `warn_only=True`는 현재 `AdaptiveAvgPool2d` CUDA backward가 strict deterministic 구현을 제공하지 않기 때문에 선택한 실행 가능성 절충이며, 경고가 발생한 run을 bitwise reproducible이라고 주장하지 않습니다. CUBLAS 변수는 첫 CUDA context 뒤에 설정하면 늦으므로 결과 메타데이터만으로 이 launch-time 조건을 복구할 수 없습니다.

## 5. Metrics and interpretation

Primary offline metrics are ADE and FDE. Also report waypoint L1, second-difference path smoothness, paired ΔADE, improvement fraction, residual magnitude, and gate mean.

`sigmoid(gate) * delta`만 실제 correction입니다. 따라서 raw `|delta|`와 applied residual `|gate * delta|`를 함께 보고합니다. Loss가 raw delta를 regularize하더라도 gate/delta scale은 완전히 식별되지 않으므로 gate 평균을 보정 필요성의 calibrated probability로 해석하지 않습니다.

Released all-towns checkpoint의 원 학습 town/route와 겹칠 수 있는 데이터를 adapter test로 쓰면 held-out은 adapter에 대해서만 성립합니다. 특히 Town13 subset을 `all_towns` checkpoint로 평가한 결과는 전체 system OOD 근거가 아닙니다. 전체 system 일반화 주장은 checkpoint pretraining에도 사용되지 않은 새 route/log split 또는 `13_withheld` checkpoint protocol이 있어야 합니다.

Baseline worst 20%는 test GT를 보고 고른 oracle diagnostic slice입니다. 전체 test 결과를 대신할 수 없습니다. `combined`가 frozen baseline만 이기고 current-only/shuffled-history control을 못 이기면 temporal evidence가 아닙니다. Route/log 단위 bootstrap CI와 여러 training seed는 실제 claim 전 필수입니다.

## 6. Exploratory pilot continuation gate

현재 자동 판정의 유일한 authoritative primary method는 `past_bev`입니다. `past_bev`가 route-macro ADE에서 `current_only_matched`보다 seed 평균 3% 이상 낮고, baseline/current-BEV/within-window shuffled-past-BEV보다 낮으며, FDE도 개선하고, 세 seed의 ADE 방향이 일치하고, 각 seed의 route-bootstrap ADE 95% CI upper bound가 0 이하이며, smoothness 악화가 baseline 대비 5% 이내일 때만 exploratory `go`입니다. Cache/split/coordinate audit 통과는 실행 전제입니다.

`combined`와 `trajectory_only` 결과는 diagnostic입니다. 특히 `combined`가 `past_bev`/`trajectory_only`보다 나은지는 modality complementarity와 후속 path branch 설계에 참고하지만, 현재 exploratory verdict를 뒤집지 않습니다. Reversed/replaced history는 현재 구현된 gate가 아니며 within-window non-identity `shuffled_past_bev`가 등록된 temporal-order control입니다.

`go`는 aligned spatial BEV/cross-attention 실험을 한 번 진행해도 된다는 내부 continuation 신호일 뿐입니다. `ambiguous`와 `no_go`도 같은 범위의 exploratory verdict입니다. 어떤 status도 paper GO, 전체 system OOD/generalization, 또는 closed-loop CARLA 개선을 뜻하지 않습니다. 특히 고정 test route가 3개이면 세 training seed가 독립 route 수를 늘리지 않으므로 논문 수준의 효능 근거가 될 수 없습니다. Paper/closed-loop 주장 전에는 더 많은 독립 route/log, pretraining-withheld 평가, 그리고 별도 closed-loop protocol이 필요합니다.

Synthetic cache와 기존 `routmem/data*`는 이 gate의 정량 근거가 될 수 없습니다.
