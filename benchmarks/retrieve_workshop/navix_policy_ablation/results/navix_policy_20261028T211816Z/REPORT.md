# NaviX directed-expansion ablation

Dense bitmaps; first 2,048 queries; single-CTA batched throughput. Three repetitions per reported point.
Fixed settings replay the paper configuration; achieved subset recall is reported explicitly.
Matched settings require all three recalls inside YFCC [0.800,0.802] or ArXiv [0.950,0.952].

| Stage | Workload | Policy | Recall | QPS | Matched |
|---|---|---|---:|---:|---|
| fixed | yfcc_k10 | adaptive_kuzu | 0.801904 | 44824.5 | False |
| fixed | yfcc_k10 | adaptive_blind | 0.795020 | 45045.8 | False |
| fixed | em_k10 | adaptive_kuzu | 0.944971 | 36963.2 | False |
| fixed | em_k10 | adaptive_blind | 0.902979 | 53143.1 | False |
| fixed | emis_k10 | adaptive_kuzu | 0.950000 | 19161.3 | False |
| fixed | emis_k10 | adaptive_blind | 0.929688 | 26407.2 | False |
| fixed | r_k10 | adaptive_kuzu | 0.946631 | 41863.0 | False |
| fixed | r_k10 | adaptive_blind | 0.918604 | 56044.1 | False |
| fixed | yfcc_k100 | adaptive_kuzu | 0.797239 | 35792.5 | False |
| fixed | yfcc_k100 | adaptive_blind | 0.789263 | 36079.5 | False |
| fixed | em_k100 | adaptive_kuzu | 0.948745 | 17078.5 | False |
| fixed | em_k100 | adaptive_blind | 0.911514 | 25298.4 | False |
| fixed | emis_k100 | adaptive_kuzu | 0.949453 | 11369.5 | False |
| fixed | emis_k100 | adaptive_blind | 0.934541 | 15218.7 | False |
| fixed | r_k100 | adaptive_kuzu | 0.947661 | 19045.5 | False |
| fixed | r_k100 | adaptive_blind | 0.926567 | 26212.6 | False |
| matched | yfcc_k10 | adaptive_kuzu | 0.801904 | 44796.0 | True |
| matched | yfcc_k10 | adaptive_blind | 0.800977 | 49919.0 | True |
| matched | em_k10 | adaptive_kuzu | 0.950586 | 34931.8 | True |
| matched | em_k10 | adaptive_blind | 0.950635 | 34907.7 | True |
| matched | emis_k10 | adaptive_kuzu | 0.950244 | 18809.3 | True |
| matched | emis_k10 | adaptive_blind | 0.950244 | 20049.9 | True |
| matched | r_k10 | adaptive_kuzu | 0.951172 | 39602.6 | True |
| matched | r_k10 | adaptive_blind | 0.950195 | 39389.4 | True |
| matched | yfcc_k100 | adaptive_kuzu | 0.801448 | 35086.4 | True |
| matched | yfcc_k100 | adaptive_blind | 0.800118 | 28561.5 | True |
| matched | em_k100 | adaptive_kuzu | 0.950942 | 16631.4 | True |
| matched | em_k100 | adaptive_blind | 0.950332 | 17225.3 | True |
| matched | emis_k100 | adaptive_kuzu | 0.950044 | 11111.8 | True |
| matched | emis_k100 | adaptive_blind | 0.950156 | 14450.4 | True |
| matched | r_k100 | adaptive_kuzu | 0.951064 | 18172.5 | True |
| matched | r_k100 | adaptive_blind | 0.950195 | 20415.0 | True |
