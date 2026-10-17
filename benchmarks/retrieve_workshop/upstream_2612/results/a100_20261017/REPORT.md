# Upstream integration validation

All benchmark rows use the same first 2,048 queries and three repetitions.
Settings are frozen paper selections or maximum-observed-recall configurations; no retuning is performed.

| Dataset | k | Method | Old recall | Integrated recall | Old QPS | Integrated QPS | QPS ratio | Roaring QPS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| yfcc | 10 | default_cagra | 0.655371 | 0.655371 | 3497.2 | 3537.5 | 1.012 | 3162.8 |
| yfcc | 10 | default_cagra_accumulator | 0.799854 | 0.799854 | 46213.7 | 46817.0 | 1.013 | 37469.1 |
| em | 10 | default_cagra | 0.956738 | 0.956738 | 31729.9 | 31753.0 | 1.001 | 31983.4 |
| em | 10 | default_cagra_accumulator | 0.951514 | 0.951514 | 42793.9 | 42725.6 | 0.998 | 42540.5 |
| emis | 10 | default_cagra | 0.700781 | 0.700781 | 580.5 | 580.1 | 0.999 | 573.1 |
| emis | 10 | default_cagra_accumulator | 0.943408 | 0.943408 | 1142.4 | 1139.8 | 0.998 | 1120.0 |
| r | 10 | default_cagra | 0.942480 | 0.942480 | 17570.6 | 17584.6 | 1.001 | 17685.2 |
| r | 10 | default_cagra_accumulator | 0.946875 | 0.946875 | 32577.5 | 32541.3 | 0.999 | 32682.2 |
| yfcc | 10 | navix_reference | 0.801904 | 0.801904 | 44829.5 | 44756.4 | 0.998 | 25760.0 |
| em | 10 | navix_reference | 0.944971 | 0.944971 | 36851.0 | 36769.0 | 0.998 | 30899.1 |
| emis | 10 | navix_reference | 0.949951 | 0.949951 | 19047.3 | 19011.7 | 0.998 | 16418.5 |
| r | 10 | navix_reference | 0.946631 | 0.946631 | 41581.0 | 41517.6 | 0.998 | 40023.5 |
| yfcc | 100 | default_cagra | 0.337299 | 0.337299 | 3495.9 | 3471.2 | 0.993 | 3169.9 |
| yfcc | 100 | default_cagra_accumulator | 0.781745 | 0.781745 | 4609.4 | 4662.1 | 1.011 | 3909.8 |
| yfcc | 100 | navix_reference | 0.797244 | 0.797254 | 35848.1 | 35787.6 | 0.998 | 16865.2 |
| em | 100 | default_cagra | 0.956235 | 0.956235 | 4980.8 | 4983.6 | 1.001 | 4956.2 |
| em | 100 | default_cagra_accumulator | 0.955156 | 0.955156 | 12856.9 | 12873.9 | 1.001 | 12870.3 |
| em | 100 | navix_reference | 0.948745 | 0.948745 | 17015.6 | 16900.9 | 0.993 | 14289.9 |
| emis | 100 | default_cagra | 0.465356 | 0.465356 | 579.1 | 578.7 | 0.999 | 572.4 |
| emis | 100 | default_cagra_accumulator | 0.906597 | 0.906597 | 683.6 | 683.9 | 1.000 | 656.4 |
| emis | 100 | navix_reference | 0.949453 | 0.949453 | 11177.5 | 11223.3 | 1.004 | 10279.7 |
| r | 100 | default_cagra | 0.931187 | 0.931187 | 1310.8 | 1296.3 | 0.989 | 1301.0 |
| r | 100 | default_cagra_accumulator | 0.945898 | 0.945898 | 9079.9 | 9078.6 | 1.000 | 9060.0 |
| r | 100 | navix_reference | 0.947661 | 0.947661 | 18909.9 | 19085.2 | 1.009 | 18085.1 |
| yfcc | 10 | ivf_flat | 0.800928 | 0.800928 | 179937.4 | 179544.3 | 0.998 | 0.0 |
| yfcc | 100 | ivf_flat | 0.800245 | 0.800245 | 33947.5 | 33869.9 | 0.998 | 0.0 |
| em | 10 | ivf_flat | 0.954932 | 0.954932 | 3676.7 | 3670.7 | 0.998 | 0.0 |
| em | 100 | ivf_flat | 0.952422 | 0.952422 | 2166.6 | 2168.5 | 1.001 | 0.0 |
| emis | 10 | ivf_flat | 0.937061 | 0.937061 | 669.4 | 667.1 | 0.997 | 0.0 |
| emis | 100 | ivf_flat | 0.940381 | 0.940381 | 343.9 | 347.1 | 1.009 | 0.0 |
| r | 10 | ivf_flat | 0.949707 | 0.949707 | 5562.3 | 5575.5 | 1.002 | 0.0 |
| r | 100 | ivf_flat | 0.947534 | 0.947534 | 3142.8 | 3152.8 | 1.003 | 0.0 |
| synthetic_low_0p0001_random | 10 | default_cagra | 0.085303 | 0.085303 | 430.4 | 431.5 | 1.002 | 428.0 |
| synthetic_low_0p0001_random | 10 | default_cagra_accumulator | 0.868652 | 0.868652 | 430.0 | 428.1 | 0.995 | 428.1 |
| synthetic_low_0p0001_random | 10 | navix_reference | 0.471777 | 0.471777 | 24299.9 | 24289.6 | 1.000 | 14774.3 |
| synthetic_low_0p0001_random | 10 | ivf_flat | 0.950000 | 0.950000 | 28792.9 | 28823.4 | 1.001 | 0.0 |
| synthetic_low_0p0001_random | 10 | exact | 1.000000 | 1.000000 | 134518.5 | 132281.9 | 0.983 | 0.0 |
| synthetic_low_0p0003_random | 10 | default_cagra | 0.187939 | 0.187939 | 419.9 | 420.8 | 1.002 | 418.0 |
| synthetic_low_0p0003_random | 10 | default_cagra_accumulator | 0.950049 | 0.950049 | 635.7 | 635.8 | 1.000 | 628.4 |
| synthetic_low_0p0003_random | 10 | navix_reference | 0.208105 | 0.208105 | 12566.9 | 12582.5 | 1.001 | 8191.5 |
| synthetic_low_0p0003_random | 10 | ivf_flat | 0.950000 | 0.950000 | 30011.9 | 30074.5 | 1.002 | 0.0 |
| synthetic_low_0p0003_random | 10 | exact | 1.000000 | 1.000000 | 76468.5 | 76487.0 | 1.000 | 0.0 |
| synthetic_low_0p001_random | 10 | default_cagra | 0.280859 | 0.280859 | 420.8 | 421.0 | 1.000 | 415.5 |
| synthetic_low_0p001_random | 10 | default_cagra_accumulator | 0.951611 | 0.951611 | 1608.3 | 1607.6 | 1.000 | 1590.7 |
| synthetic_low_0p001_random | 10 | navix_reference | 0.852393 | 0.852393 | 6928.8 | 6910.4 | 0.997 | 4126.5 |
| synthetic_low_0p001_random | 10 | ivf_flat | 0.950000 | 0.950000 | 24073.0 | 23969.3 | 0.996 | 0.0 |
| synthetic_low_0p001_random | 10 | exact | 1.000000 | 1.000000 | 30595.3 | 30586.7 | 1.000 | 0.0 |
| synthetic_low_0p003_random | 10 | default_cagra | 0.424902 | 0.424902 | 430.9 | 431.7 | 1.002 | 427.4 |
| synthetic_low_0p003_random | 10 | default_cagra_accumulator | 0.950049 | 0.950049 | 3908.9 | 3891.1 | 0.995 | 3833.8 |
| synthetic_low_0p003_random | 10 | navix_reference | 0.950684 | 0.950684 | 29299.5 | 29290.2 | 1.000 | 15992.8 |
| synthetic_low_0p003_random | 10 | ivf_flat | 0.950049 | 0.950049 | 16334.6 | 16340.3 | 1.000 | 0.0 |
| synthetic_low_0p003_random | 10 | exact | 1.000000 | 1.000000 | 11320.7 | 11337.0 | 1.001 | 0.0 |
| synthetic_low_0p01_random | 10 | default_cagra | 0.779004 | 0.779004 | 419.2 | 418.4 | 0.998 | 413.9 |
| synthetic_low_0p01_random | 10 | default_cagra_accumulator | 0.950391 | 0.950391 | 8741.9 | 8747.1 | 1.001 | 8562.3 |
| synthetic_low_0p01_random | 10 | navix_reference | 0.950098 | 0.950098 | 60421.9 | 60404.4 | 1.000 | 24940.8 |
| synthetic_low_0p01_random | 10 | ivf_flat | 0.950488 | 0.950488 | 9184.3 | 9182.3 | 1.000 | 0.0 |
| synthetic_low_0p01_random | 10 | exact | 1.000000 | 1.000000 | 3532.1 | 3523.0 | 0.997 | 0.0 |

Threshold flags requiring investigation: 1 benchmark rows; 13 native-control rows.
Flags are not automatically waived. Native-control timing is separate from cuVS-bench timing.
No push or main-branch update is performed by this runner.
