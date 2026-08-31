# Public pilot results

Only compact, non-binary evidence is included. Raw records, checkpoints, spatial
arrays, images, and sealed/test payloads are intentionally excluded.

| Directory | Evidence class | Authoritative outcome |
|---|---|---|
| `TemporalResidual_PilotTest/` | Locked three-seed offline pilot on 15 routes | `no_go` |
| `ActionInfluence_PilotTest/` | Train/validation observational latent-transition proxy | `no_go` |
| `MPCLocalGrounding_PilotTest/v1_preflight/` | Collision-representation preflight | structural preflight failure |
| `MPCLocalGrounding_PilotTest/v2_official/` | Equal-budget global vs elite ranking validation | `no_go` |
| `MPCLocalGrounding_PilotTest/v2_rankscale_diagnostic/` | Post-hoc validation-only scale diagnostic | diagnostic gate failure; cannot rescue V2 |
| `MPCLocalGrounding_PilotTest/v3_collection_failure/` | Fresh preregistered collection integrity | terminal `no_go` before training |

The initial Action Influence run is retained under `initial_full_run/` to disclose
the result before the mandatory raw-persistence sanity gate was added.

