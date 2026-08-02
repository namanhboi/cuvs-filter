# Exact-state FAVOR progress termination diagnostic

Disposition: **reject_shadow_rule**

The shadow rule never changes kernel termination. It measures evidence accumulated since
the ordered passing top-10 last changed and is eligible only at or after B0.

## Controls

| Dataset | B0 recall | B0 work | Fixed >=.905 iteration | Fixed recall | Fixed work | Cap recall | Hash-full |
|---|---:|---:|---:|---:|---:|---:|---:|
| sift | 0.95008 | 9598.4 | 97 | 0.91639 | 7189.2 | 0.98191 | 0 |
| gist | 0.83094 | 11292.3 | 237 | 0.90503 | 18832.3 | 0.91924 | 0 |
| bigann1m | 0.94291 | 8240.1 | 385 | 0.91450 | 6415.0 | 0.96949 | 0 |
| bigann10m | 0.92280 | 10224.7 | 449 | 0.90687 | 8966.7 | 0.96757 | 0 |
| msturing1m | 0.72224 | 12004.8 | 1669 | 0.90618 | 36211.5 | 0.92728 | 0 |
| msturing10m | 0.69609 | 11076.9 | 2694 | 0.90582 | 59110.8 | 0.92653 | 0 |

## Per-dataset admissible rules

These are diagnostic optima, not deployable choices. Their disagreement is the reason
the dataset-independent gate fails.

| Dataset | Lowest-work admissible rule | Recall | Mean work / limit |
|---|---|---:|---:|
| sift | evidence2_gapoff | 0.95372 | 10255.9 / 11997.9 |
| gist | evidence8_gapoff | 0.90824 | 19027.0 / 20715.5 |
| bigann1m | evidence2_gapoff | 0.95336 | 9389.0 / 10300.1 |
| bigann10m | evidence2_gapoff | 0.93208 | 11049.2 / 12780.8 |
| msturing1m | evidence16_gapoff | 0.91232 | 38159.5 / 39832.7 |
| msturing10m | evidence32_gapoff | 0.91310 | 63178.6 / 65021.9 |

## Progress-signal dynamics

| Dataset | B0 evidence p50 / p90 | Max evidence p50 | Top-10 changes p50 / p90 |
|---|---:|---:|---:|
| sift | 28.3 / 56.1 | 93.4 | 6 / 9 |
| gist | 18.0 / 50.0 | 81.0 | 8 / 12 |
| bigann1m | 17.8 / 41.8 | 76.0 | 7 / 11 |
| bigann10m | 25.1 / 53.6 | 95.4 | 7 / 10 |
| msturing1m | 1.0 / 28.9 | 193.0 | 11 / 17 |
| msturing10m | 1.0 / 27.0 | 452.0 | 12 / 18 |

## Leave-one-family-out validation

| Held-out family | Selected on training families | Held-out result | Pass |
|---|---|---|---|
| bigann | none | no qualifying training rule | False |
| gist | none | no qualifying training rule | False |
| msturing | none | no qualifying training rule | False |
| sift | none | no qualifying training rule | False |

A DEEP-image1M holdout may be run only when the disposition is `run_frozen_holdout`.
Failure leaves the rule shadow-only; it must not be retuned on the holdout.
