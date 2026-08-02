# FAVOR saved-state retry diagnostic

Decision: **full_in_kernel_state_required**

These are diagnostic recalls, not benchmark throughput results. Every restart strategy
uses a fresh hash table; the oracle instead reruns one uninterrupted traversal to the
same cumulative iteration budget.

| Dataset | Strategy | Rounds | Final recall | Last-round recall | New GT/query | Jaccard |
|---|---|---:|---:|---:|---:|---:|
| gist | independent | 2 | 0.85067 | 0.83120 | 0.1970 | 0.9159 |
| gist | passing | 2 | 0.84891 | 0.84581 | 0.1795 | 0.9507 |
| gist | frontier | 2 | 0.86338 | 0.75302 | 0.3247 | 0.6859 |
| gist | combined | 2 | 0.86383 | 0.74735 | 0.3307 | 0.6574 |
| gist | oracle | 2 | 0.92247 | 0.91932 | 0.9151 | 0.8211 |
| msturing1m | independent | 4 | 0.73792 | 0.72209 | 0.0212 | 0.9123 |
| msturing1m | passing | 4 | 0.73311 | 0.71793 | 0.0000 | 0.9998 |
| msturing1m | frontier | 4 | 0.74867 | 0.68512 | 0.0053 | 0.7088 |
| msturing1m | combined | 4 | 0.74833 | 0.67997 | 0.0060 | 0.7948 |
| msturing1m | oracle | 4 | 0.93077 | 0.92728 | 0.2913 | 0.9403 |
| msturing10m | independent | 7 | 0.72170 | 0.69527 | 0.0176 | 0.9013 |
| msturing10m | passing | 7 | 0.70839 | 0.69366 | 0.0000 | 1.0000 |
| msturing10m | frontier | 7 | 0.72464 | 0.65918 | 0.0008 | 0.6980 |
| msturing10m | combined | 7 | 0.72438 | 0.65338 | 0.0008 | 0.7825 |
| msturing10m | oracle | 7 | 0.93026 | 0.92653 | 0.1024 | 0.9782 |

```json
{
  "conclusion": "full_in_kernel_state_required",
  "oracle_reaches_0_90_all": true,
  "reseed_reaches_0_90_all": {
    "passing": false,
    "frontier": false,
    "combined": false
  },
  "partial_gain_gate": {
    "passing": {
      "passes": false,
      "details": [
        {
          "dataset": "gist",
          "candidate_gain": 0.017940000000000067,
          "oracle_gain": 0.09150000000000003,
          "fraction_of_oracle_gain": 0.19606557377049247,
          "margin_vs_independent": -0.0017599999999999838,
          "passes": false
        },
        {
          "dataset": "msturing1m",
          "candidate_gain": 0.01094000000000006,
          "oracle_gain": 0.2086,
          "fraction_of_oracle_gain": 0.052444870565676226,
          "margin_vs_independent": -0.004809999999999981,
          "passes": false
        },
        {
          "dataset": "msturing10m",
          "candidate_gain": 0.012409999999999921,
          "oracle_gain": 0.23427999999999993,
          "fraction_of_oracle_gain": 0.05297080416595495,
          "margin_vs_independent": -0.013310000000000044,
          "passes": false
        }
      ]
    },
    "frontier": {
      "passes": false,
      "details": [
        {
          "dataset": "gist",
          "candidate_gain": 0.03241000000000005,
          "oracle_gain": 0.09150000000000003,
          "fraction_of_oracle_gain": 0.3542076502732245,
          "margin_vs_independent": 0.012709999999999999,
          "passes": false
        },
        {
          "dataset": "msturing1m",
          "candidate_gain": 0.026499999999999968,
          "oracle_gain": 0.2086,
          "fraction_of_oracle_gain": 0.12703739213806312,
          "margin_vs_independent": 0.010749999999999926,
          "passes": false
        },
        {
          "dataset": "msturing10m",
          "candidate_gain": 0.028659999999999908,
          "oracle_gain": 0.23427999999999993,
          "fraction_of_oracle_gain": 0.12233225200614613,
          "margin_vs_independent": 0.0029399999999999427,
          "passes": false
        }
      ]
    },
    "combined": {
      "passes": false,
      "details": [
        {
          "dataset": "gist",
          "candidate_gain": 0.03286,
          "oracle_gain": 0.09150000000000003,
          "fraction_of_oracle_gain": 0.35912568306010917,
          "margin_vs_independent": 0.01315999999999995,
          "passes": false
        },
        {
          "dataset": "msturing1m",
          "candidate_gain": 0.026160000000000072,
          "oracle_gain": 0.2086,
          "fraction_of_oracle_gain": 0.125407478427613,
          "margin_vs_independent": 0.01041000000000003,
          "passes": false
        },
        {
          "dataset": "msturing10m",
          "candidate_gain": 0.02839999999999998,
          "oracle_gain": 0.23427999999999993,
          "fraction_of_oracle_gain": 0.12122246884070338,
          "margin_vs_independent": 0.0026800000000000157,
          "passes": false
        }
      ]
    }
  }
}
```
