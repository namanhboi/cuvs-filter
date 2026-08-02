# FAVOR termination + forgetful-hash shadow diagnostic

Disposition: **reject_forgetful_hash_no_live_v2**

The policy gate uses instantaneous checkpoint top-10 recall. For a selected policy,
accumulated checkpoint recall is computed afterward as a sensitivity diagnostic only.

## Deep hash controls

| Dataset | Hash | B0 recall | B0 underfill | Final recall | Candidate evals/query | Duplicates/query | Hash full | Hash bits | Small bits | Reset |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gist | exact | 0.83093 | 0.0020 | 0.91928 | 20815.7 | 13104.3 | 0 | 17 | 0 | 1048576 |
| gist | forgetful | 0.28780 | 0.7285 | 0.29276 | 13731.0 | 20189.0 | 0 | 11 | 11 | 4 |
| msturing1m | exact | 0.72224 | 0.0862 | 0.92728 | 44528.7 | 21615.3 | 0 | 18 | 0 | 1048576 |
| msturing1m | forgetful | 0.16266 | 0.9496 | 0.16266 | 16199.7 | 49944.3 | 0 | 11 | 11 | 16 |
| msturing10m | exact | 0.69609 | 0.0903 | 0.92653 | 80017.2 | 35982.8 | 0 | 18 | 0 | 1048576 |
| msturing10m | forgetful | 0.17353 | 0.9195 | 0.17353 | 21716.9 | 94283.1 | 0 | 11 | 11 | 16 |

## Uninstrumented factorial edge

The first repetition may include JIT loading, so the table uses the three-repetition
median. V2 cells were not run because the shadow gate rejected implementation.

| Dataset | Hash | Termination | Recall | QPS | QPS/exact-current | Underfill | Accepted |
|---|---|---|---:|---:|---:|---:|---|
| gist | exact | current_adaptive | 0.92540 | 2023.7 | 1.000 | 0.0000 | False |
| gist | forgetful | current_adaptive | 0.29670 | 1395.3 | 0.689 | 0.7409 | False |
| gist | exact | v2 | — | — | — | — | False |
| gist | forgetful | v2 | — | — | — | — | False |
| msturing1m | exact | current_adaptive | 0.94560 | 5805.7 | 1.000 | 0.0000 | False |
| msturing1m | forgetful | current_adaptive | 0.16266 | 7985.1 | 1.375 | 0.9481 | False |
| msturing1m | exact | v2 | — | — | — | — | False |
| msturing1m | forgetful | v2 | — | — | — | — | False |
| msturing10m | exact | current_adaptive | 0.89432 | 5800.4 | 1.000 | 0.0000 | False |
| msturing10m | forgetful | current_adaptive | 0.17353 | 8336.2 | 1.437 | 0.9177 | False |
| msturing10m | exact | v2 | — | — | — | — | False |
| msturing10m | forgetful | v2 | — | — | — | — | False |

## Gate

```json
{
  "recall_floor": 0.905,
  "selected": null,
  "live_v2_justified": false,
  "forgetful_hash_recall_safe_all": false,
  "disposition": "reject_forgetful_hash_no_live_v2",
  "candidates": [
    {
      "policy": "top10_stable1",
      "qualifies": false,
      "geometric_mean_iterations": 1149.531519013018,
      "recall_by_dataset": {
        "gist": 0.2889,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "top10_stable2",
      "qualifies": false,
      "geometric_mean_iterations": 1164.5047825361146,
      "recall_by_dataset": {
        "gist": 0.28997,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "top10_stable4",
      "qualifies": false,
      "geometric_mean_iterations": 1181.426041178671,
      "recall_by_dataset": {
        "gist": 0.29128000000000004,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "top10_stable8",
      "qualifies": false,
      "geometric_mean_iterations": 1200.4741009080055,
      "recall_by_dataset": {
        "gist": 0.29178000000000004,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "gap1.00_stable1",
      "qualifies": false,
      "geometric_mean_iterations": 1245.915608769037,
      "recall_by_dataset": {
        "gist": 0.2904,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "gap1.00_stable2",
      "qualifies": false,
      "geometric_mean_iterations": 1246.8851606118649,
      "recall_by_dataset": {
        "gist": 0.29097,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "gap1.05_stable1",
      "qualifies": false,
      "geometric_mean_iterations": 1247.1561594011503,
      "recall_by_dataset": {
        "gist": 0.2906,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "gap1.05_stable2",
      "qualifies": false,
      "geometric_mean_iterations": 1248.3247980311,
      "recall_by_dataset": {
        "gist": 0.29104,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "gap1.10_stable1",
      "qualifies": false,
      "geometric_mean_iterations": 1248.9064548426916,
      "recall_by_dataset": {
        "gist": 0.29117000000000004,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "gap1.00_stable4",
      "qualifies": false,
      "geometric_mean_iterations": 1248.983530197655,
      "recall_by_dataset": {
        "gist": 0.29178000000000004,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "gap1.10_stable2",
      "qualifies": false,
      "geometric_mean_iterations": 1249.675500305872,
      "recall_by_dataset": {
        "gist": 0.29134,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "gap1.05_stable4",
      "qualifies": false,
      "geometric_mean_iterations": 1250.1238066765127,
      "recall_by_dataset": {
        "gist": 0.29185,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "gap1.10_stable4",
      "qualifies": false,
      "geometric_mean_iterations": 1252.0187543904412,
      "recall_by_dataset": {
        "gist": 0.29225,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "gap1.00_stable8",
      "qualifies": false,
      "geometric_mean_iterations": 1252.8389267185123,
      "recall_by_dataset": {
        "gist": 0.29205,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "gap1.05_stable8",
      "qualifies": false,
      "geometric_mean_iterations": 1253.458562127811,
      "recall_by_dataset": {
        "gist": 0.29205,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "gap1.10_stable8",
      "qualifies": false,
      "geometric_mean_iterations": 1254.7439463940784,
      "recall_by_dataset": {
        "gist": 0.29234,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "prefix12_of32",
      "qualifies": false,
      "geometric_mean_iterations": 1258.497590606287,
      "recall_by_dataset": {
        "gist": 0.29266000000000003,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "current_prefix16_of32",
      "qualifies": false,
      "geometric_mean_iterations": 1258.497590606287,
      "recall_by_dataset": {
        "gist": 0.29266000000000003,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "prefix14_of32",
      "qualifies": false,
      "geometric_mean_iterations": 1258.497590606287,
      "recall_by_dataset": {
        "gist": 0.29266000000000003,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    },
    {
      "policy": "prefix10_of32",
      "qualifies": false,
      "geometric_mean_iterations": 1258.497590606287,
      "recall_by_dataset": {
        "gist": 0.29266000000000003,
        "msturing1m": 0.16266,
        "msturing10m": 0.17353
      }
    }
  ]
}
```
