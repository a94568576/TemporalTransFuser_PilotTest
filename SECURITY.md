# Security and trusted-input boundary

This repository contains research tooling, not a hardened data-ingestion service.

- Never load checkpoints, cache records, manifests, or CARLA metadata from an
  untrusted source.
- Tensor/checkpoint reads in the curated code use `weights_only=True` where the
  stored schema permits it.
- `scripts/cache_tfpp_dataset.py` decodes `jsonpickle` metadata emitted by the
  external CARLA Garage data pipeline. `jsonpickle` may construct Python objects;
  run this extractor only on data you generated or independently trust.
- Output directories are generally fail-closed and should not be shared between
  mutually untrusted users.
- Raw data, model weights, and sealed evaluation payloads are intentionally absent.

Do not report a vulnerability by committing secrets or private data to the
repository. Rotate any credential immediately if it is accidentally exposed.
