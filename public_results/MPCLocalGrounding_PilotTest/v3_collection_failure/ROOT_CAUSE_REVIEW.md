# MPC-local grounding V3 root-cause review

## Decision

The official V3 result is **terminal NO-GO at the development collection-integrity gate**.
It is not a model-performance result: training, checkpoint selection, and outer
validation never started.

The frozen run observed 3 collision-positive candidate rollouts among 3,456
development records (`0.0868056%`), comprising 5 sensor events with summed impulse
`37087.6103515625`. They occurred in train states 5, 28, and 31, all during CEM
iteration 0. Atomic publication was withheld and the requested dataset root does
not exist.

## Why the gate failed

The real physical cost includes a collision penalty of 10, but the frozen model
has only a four-dimensional terminal-outcome head. Its predicted-cost path sets
collision to zero for every candidate. The three colliding candidates therefore
cannot be scored under the same objective used by their real labels.

This mismatch is too large to hide inside another outcome. The maximum frozen
action regularizer is only `0.0108`; collision adds `10`. A full stop contributes
only `0.4` through the speed term. Making the four-output head emulate collision
would require physically false predictions, such as about 7.75 m normalized
lateral error or roughly -500 m progress.

The collision sensor is attached after neutral warm-up and recreated for every
candidate. A spawn-settling or sensor carry-over explanation is therefore
unlikely. The pattern—three isolated iteration-0 candidates and zero collision
labels in their later CEM iterations—is consistent with rare unsafe exploratory
actions that CEM subsequently rejects.

## Exploratory state audit

This section is post-hoc, development-only context and is not a preregistered
causal attribution.

- State 5: speed 8 m/s, lateral offset -0.25 m, curvature score 0.009902 1/m,
  road 34 lane 2.
- State 28: speed 8 m/s, lateral offset -0.25 m, straight, road 34 lane -1.
- State 31: speed 6 m/s, center offset, straight, road 37 lane 2.

All three lanes have a left-adjacent shoulder. Three of 8 such train states had
a colliding candidate, versus 0 of the other 24, but this is a small post-hoc
association, not proof. Two states were straight, so curvature alone is not the
cause. The frozen state filter checked centerline forward-road length but not a
lateral static-object/guardrail envelope.

## Why the obvious rescues are invalid

- Removing three rows produces incomplete 23-candidate groups and unequal query
  exposure.
- Removing or replacing their states conditions the split on observed outcomes
  and breaks the frozen state balance.
- Supplying the real collision label at scoring time is simulator-oracle leakage.
- Shrinking steering support or changing the seed after observing the collisions
  changes the CEM distribution and the research question.
- Relaxing the gate leaves real cost collision-aware while predicted cost remains
  collision-blind.

The protocol froze zero post-hoc remediation rounds, so V3 must not be retried or
continued.

## Feasible successor

A collision-aware successor is technically possible, but it is a new study, not
a minor repair. The minimum defensible design is:

1. add a shared scalar collision-probability head to every variant;
2. define `J_pred = J_outcome + 10 * P_hat(collision | state, action)` before
   collection;
3. retain collided rows and train the head with a frozen, common loss and an
   independent train-only calibration split;
4. use fresh, state-grouped fit/inner/outer states disjoint from V1/V2/V3;
5. give every comparison arm identical collision/outcome supervision and query
   budgets;
6. preregister collision-inclusive regret, selected-action collision rate,
   false-negative safety, AUPRC, Brier score, and calibration gates;
7. keep the sealed test byte-separated and unopened.

Only 3 independent positive rollouts were observed, an imbalance of 1:1,151.
At the natural rate, about 57,600 rollouts are needed for 50 expected positives.
A preregistered collision-enriched training mixture near 5% prevalence reduces
that to roughly 1,000–2,000 rollouts for 50–100 positives, followed by a separate
representative outer validation. The existing 16-state outer design is not large
enough for a collision-safety claim.

## Recommendation

Do not tune or rerun V3. If the research goal is MPC safety, proceed only with the
collision-aware successor and fresh preregistration. If that data/validation
budget is not acceptable, pivot away from a safety-cost claim rather than hiding
unsafe candidates with a narrower post-hoc support.
