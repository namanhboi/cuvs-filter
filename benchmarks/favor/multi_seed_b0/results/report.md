# Independent multi-seed B0 result

Overall gate: **FAIL**

The gate requires either the two- or three-mask variant to reach recall ≥ 0.90 on all
three 10,000-query workloads and exceed the same-machine adaptive-termination QPS.

| Dataset | Variant | Recall | QPS | QPS/adaptive | Marginal recall |
|---|---:|---:|---:|---:|---:|
| gist | adaptive_termination | 0.92545 | 2,016 | 1.000 |  |
| gist | automatic_retention | 0.83091 | 5,399 | 2.677 |  |
| gist | multi_seed_1 | 0.83093 | 5,397 | 2.677 | +0.00002 |
| gist | multi_seed_2 | 0.85061 | 2,694 | 1.336 | +0.01968 |
| gist | multi_seed_3 | 0.85934 | 1,795 | 0.890 | +0.00873 |
| msturing10m | adaptive_termination | 0.89432 | 5,560 | 1.000 |  |
| msturing10m | automatic_retention | 0.69601 | 28,601 | 5.144 |  |
| msturing10m | multi_seed_1 | 0.69601 | 28,616 | 5.146 | +0.00000 |
| msturing10m | multi_seed_2 | 0.70795 | 14,206 | 2.555 | +0.01194 |
| msturing10m | multi_seed_3 | 0.71293 | 9,328 | 1.678 | +0.00498 |
| msturing1m | adaptive_termination | 0.94560 | 5,506 | 1.000 |  |
| msturing1m | automatic_retention | 0.72220 | 28,785 | 5.228 |  |
| msturing1m | multi_seed_1 | 0.72220 | 28,880 | 5.246 | +0.00000 |
| msturing1m | multi_seed_2 | 0.73272 | 14,328 | 2.602 | +0.01052 |
| msturing1m | multi_seed_3 | 0.73580 | 9,414 | 1.710 | +0.00308 |

```json
{
  "pass": false,
  "rounds": {
    "2": {
      "pass": false,
      "details": [
        {
          "dataset": "gist",
          "recall_ok": false,
          "qps_ok": true
        },
        {
          "dataset": "msturing1m",
          "recall_ok": false,
          "qps_ok": true
        },
        {
          "dataset": "msturing10m",
          "recall_ok": false,
          "qps_ok": true
        }
      ]
    },
    "3": {
      "pass": false,
      "details": [
        {
          "dataset": "gist",
          "recall_ok": false,
          "qps_ok": false
        },
        {
          "dataset": "msturing1m",
          "recall_ok": false,
          "qps_ok": true
        },
        {
          "dataset": "msturing10m",
          "recall_ok": false,
          "qps_ok": true
        }
      ]
    }
  }
}
```
