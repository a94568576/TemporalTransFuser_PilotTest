# MPC-local grounding V3 collection result

- Status: **terminal NO-GO before training**
- Failure gate: development collision-free integrity
- Observed: 3 colliding candidate records and 5 collision events
- Affected development candidates: train states 5, 28, and 31, all at CEM iteration 0
- Published dataset: none (atomic output was withheld)
- Training / checkpointing / outer validation: not started
- Sealed test: values remained redacted; no sealed payload was published or accessed
- CARLA cleanup: settings restored and no experiment actors remained

This is not a software crash. The frozen predictor has no collision head and its
predicted physical cost assumes zero collision. Filtering the three candidates,
deleting their states, narrowing support after seeing the data, or injecting the
simulator collision label would change the preregistered comparison. The valid
next design is therefore a fresh collision-aware study, not a retry of V3.
