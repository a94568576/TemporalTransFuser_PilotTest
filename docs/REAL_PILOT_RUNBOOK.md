# Real TF++ temporal-BEV pilot runbook

이 문서는 synthetic 결과를 더 조정하지 않고, 실제 sequential CARLA route에서 `past_bev`의 추가 정보를 검증하는 고정 실행 순서입니다. 로컬 released TF++의 target은 시간 trajectory가 아닌 거리-sampled geometric checkpoint path입니다.

## 1. 고정 상태

- Synthetic selection `6088254dc98bcacf18c7`은 `synthetic_smoke_complete`로 동결했습니다.
- 실제 cache schema는 v3이며 `BEV [64,8,8]` float16, `pred/GT [10,2]`, 동일 timestamp의 pose/speed/command를 저장합니다.
- 첫 GRU/MLP pilot에서는 BEV를 ego-motion warp하지 않습니다. 이 limitation 때문에 성공하더라도 곧바로 spatial-temporal alignment 주장을 하지 않습니다.
- Test는 아래 6절의 study final 명령 전까지 열지 않습니다.

## 2. 실제 route 수집

터미널 1:

```bash
cd ${TFPP_ROOT}
./scripts/run_carla_server.sh
```

터미널 2:

```bash
cd ${PROJECT_ROOT}
./scripts/collect_tfpp_route_manifest.sh \
  configs/real_pilot_routes_town13_v1.txt \
  data/raw_real_pilot_town13_v1_routes14_retry
```

수집기는 non-empty output을 덮어쓰지 않고, 이미 100% 완료된 route만 재실행에서 건너뜁니다. 기존 one-route smoke `data/raw_real_smoke/Accident_1044_0`을 합쳐 총 15 routes를 사용합니다.

## 3. Raw collection audit와 feature cache

```bash
export PYTHONPATH="$PWD/src"
TFPP_PY=../transfuser_test/carla_garage/.venv/bin/python

$TFPP_PY scripts/audit_raw_collection.py \
  --root-manifest configs/real_pilot_cache_roots_town13_v1.txt \
  --output artifacts/real_pilot_town13_v1_raw_audit.json

./scripts/build_real_pilot_cache.sh data/cache_real_pilot_town13_v1
```

`result.json`의 Finished/100%/no-exception만으로는 cache 입력을 승인하지 않습니다. Audit은 route 내부 `results.json.gz`에도 local upstream `CARLA_Data`와 같은 filter를 적용합니다. Composed score가 100 미만이면 reported infraction 전부가 min-speed일 때만 승인하므로, min-speed-only route는 사용하지만 collision이 있는 기존 `HazardAtSideLane_1619_0`은 100% 완주여도 거부합니다.

Hazard 재수집은 원본을 덮어쓰지 않고 `HazardAtSideLane_1619_0_retry1`에 저장했지만 동일 차량 충돌이 재현되어 두 수집본 모두 제외했습니다. 실패 데이터는 provenance 증거로 보존합니다. [root manifest](../configs/real_pilot_cache_roots_town13_v1.txt)는 사전에 감사를 거쳐 100%·Perfect로 끝난 대체 경로 `NonSignalizedJunctionRightTurn_73_0`을 명시적으로 가리킵니다. Broad directory discovery, 결과 JSON 수정, 실패 root 삭제로 route 수를 맞추지 않습니다.

Cache build는 manifest의 정확한 15개 logical key/root와 manifest SHA256을 provenance에 고정합니다. GPU forward 전에 raw preflight를 통과해야 하며, upstream `CARLA_Data`가 실제로 연 route directory set과 cache가 실제로 쓴 route ID set이 expected set과 완전히 같아야 finalize됩니다. 따라서 upstream filter에 의한 silent omission은 build 실패가 됩니다. Frozen `all_towns/model_0030_0.pth`를 CUDA inference로 실행한 뒤 route split은 seed 17, train/val/test = 0.6/0.2/0.2로 route를 먼저 나누고 window를 만듭니다.

## 4. Cache audit와 20-sample visual review

```bash
$TFPP_PY scripts/audit_cache.py \
  --cache data/cache_real_pilot_town13_v1 --deep --splits train val \
  --output artifacts/real_pilot_town13_v1_cache_audit_train_val.json

$TFPP_PY scripts/sanity_check_cache.py \
  --cache data/cache_real_pilot_town13_v1 \
  --splits train val \
  --output artifacts/real_pilot_town13_v1_sanity_train_val.json

$TFPP_PY scripts/visualize_cache.py \
  --cache data/cache_real_pilot_town13_v1 \
  --raw-root data \
  --splits train val \
  --output-dir artifacts/real_pilot_town13_v1_visualizations_train_val \
  --max-samples 20 --history-length 4
```

Final 전 QC는 `train val` entry만 tensor-load/metric/샘플링합니다. Index 자체의 split/route 구조는 전체 cache에 대해 검증되지만 test tensor와 GT는 열지 않습니다. Deep audit, cadence/gap/hash/shape/finite check 중 하나라도 실패하면 학습하지 않습니다. Figure에서는 train/val RGB, GT/current frozen path, 현재 ego frame으로 SE(2) 정렬한 과거 frozen paths를 직접 확인합니다.

## 5. Validation-only residual λ selection

각 후보는 같은 split, seeds, variants를 사용하고 test를 읽지 않습니다.

CUDA 재현성 설정은 Python process를 시작하기 전에 해야 합니다. 아래 export를 한 새 shell에서 study command를 실행합니다.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Engine은 seed, cuDNN deterministic/benchmark flags와 TF32-off를 고정하고 PyTorch deterministic algorithms를 `warn_only=True`로 요청합니다. 이는 현재 `AdaptiveAvgPool2d` CUDA backward 때문에 strict mode가 실제 학습을 중단시키는 것을 피한 명시적 절충입니다. 발생한 nondeterministic-op warning을 보존하고, warning이 있는 run은 bitwise reproducibility를 주장하지 않습니다. 각 best epoch는 validation equal-route macro ADE로 선택되며 validation sample/route ID 순서가 shared baseline과 달라지면 실패합니다.

```bash
$TFPP_PY scripts/run_multiseed_study.py \
  --evaluation-mode selection --config configs/pilot.yaml \
  --cache data/cache_real_pilot_town13_v1 --device cuda \
  --seeds 17 29 43 --residual-weight 0.01 \
  --output artifacts/real_pilot_rw001_selection

$TFPP_PY scripts/run_multiseed_study.py \
  --evaluation-mode selection --config configs/pilot.yaml \
  --cache data/cache_real_pilot_town13_v1 --device cuda \
  --seeds 17 29 43 --residual-weight 0.05 \
  --output artifacts/real_pilot_rw005_selection

$TFPP_PY scripts/run_multiseed_study.py \
  --evaluation-mode selection --config configs/pilot.yaml \
  --cache data/cache_real_pilot_town13_v1 --device cuda \
  --seeds 17 29 43 --residual-weight 0.10 \
  --output artifacts/real_pilot_rw010_selection
```

기본 variants는 `current_only`, `current_only_matched`, `trajectory_only`, `current_bev`, `past_bev`, `shuffled_past_bev`, `combined`입니다.

```bash
$TFPP_PY scripts/compare_study_selections.py \
  artifacts/real_pilot_rw001_selection \
  artifacts/real_pilot_rw005_selection \
  artifacts/real_pilot_rw010_selection \
  --primary-variant past_bev \
  --output artifacts/real_pilot_validation_choice
```

Comparator는 test-closed 상태와 모든 hash/config invariant를 확인하고, `past_bev`의 세 seed validation route-macro ADE 평균만으로 후보 하나를 고정합니다.

## 6. 최종 test 1회

Comparator가 만든 `selection_choice.json`을 그대로 입력해 정확히 한 번 실행합니다. Final은 choice의 후보들을 validation-only 규칙으로 다시 검증하고, chosen study path·manifest SHA256·residual λ가 재계산 결과와 정확히 같을 때만 진행합니다.

```bash
$TFPP_PY scripts/run_multiseed_study.py \
  --evaluation-mode final \
  --cache data/cache_real_pilot_town13_v1 \
  --selection-choice artifacts/real_pilot_validation_choice/selection_choice.json \
  --output artifacts/real_pilot_locked_final \
  --device cuda
```

이 명령은 임의의 비선택 λ path를 받을 수 없고, 3개 seed checkpoint를 재학습하지 않고 평가하며, test tensor를 읽기 전에 영구 study marker를 생성합니다. Study child selection은 parent-owned라 단독 `run_pilot.py final`로 열 수 없습니다. 실패해도 test-open event는 소비되며 재호출하지 않습니다.

Final이 성공한 뒤에만 전체 split QC를 별도 artifact로 생성합니다.

```bash
$TFPP_PY scripts/audit_cache.py \
  --cache data/cache_real_pilot_town13_v1 --deep --splits train val test \
  --output artifacts/real_pilot_town13_v1_cache_audit_all.json

$TFPP_PY scripts/sanity_check_cache.py \
  --cache data/cache_real_pilot_town13_v1 \
  --splits train val test \
  --output artifacts/real_pilot_town13_v1_sanity_all.json

$TFPP_PY scripts/visualize_cache.py \
  --cache data/cache_real_pilot_town13_v1 \
  --raw-root data \
  --splits train val test \
  --output-dir artifacts/real_pilot_town13_v1_visualizations_all \
  --max-samples 20 --history-length 4
```

## 7. 사전 고정 exploratory pilot 판정

```bash
$TFPP_PY scripts/evaluate_go_no_go.py \
  artifacts/real_pilot_locked_final/study_results.json
```

`past_bev`가 authoritative primary gate입니다. `go`는 `past_bev`가 `current_only_matched`보다 route-macro ADE를 평균 3% 이상 개선하고 baseline/current-BEV/shuffled-history를 모두 이기며, FDE·세 seed 방향·route-bootstrap CI·smoothness 조건까지 모두 만족할 때만 나옵니다. `ambiguous`는 BEV refinement 신호는 있지만 current/shuffled control을 이기지 못한 경우입니다. `combined`는 modality/path-branch diagnostic이며 status를 바꾸지 않습니다.

출력의 `go/ambiguous/no_go`는 모두 **exploratory offline pilot verdict**입니다. 특히 이 split의 test route가 3개라면 seed 17/29/43은 같은 3개 route를 반복할 뿐 독립 표본을 늘리지 않습니다. 따라서 `go`도 paper GO나 closed-loop GO가 아니며, 논문 효능·일반화 주장의 근거로 쓰지 않습니다.

## 8. 다음 구조 gate

- `go`: 내부 continuation으로 aligned spatial BEV token + current-path query cross-attention을 한 번 구현합니다.
- `ambiguous`: 연구 주장을 temporal correction이 아니라 post-planning BEV refinement로 제한하거나, alignment 구조를 한 번만 시도합니다.
- `no_go`: 지금 GRU branch를 성능 근거로 확장하지 않습니다. `trajectory_only ≈ current_only` 또는 `combined`가 `past_bev`보다 나쁘면 diagnostic에 따라 과거 geometric-path branch를 제거합니다.

Town13은 local `all_towns` checkpoint pretraining과 겹칠 수 있으므로, 이 결과의 held-out 범위는 adapter입니다. 전체 system OOD/generalization과 closed-loop CARLA 개선은 별도 `13_withheld`/새 log 및 closed-loop 평가가 필요합니다.
