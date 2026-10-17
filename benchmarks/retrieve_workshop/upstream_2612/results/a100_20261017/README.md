# A100 pinned-upstream validation records

Frozen first-2,048-query replay; three repetitions; no parameter retuning.
Start with ASSESSMENT.md; REPORT.md, comparison.csv, and investigations/ retain
the full measurements and findings, including raw automated flags.
The source revisions, exact configurations, raw repetitions, environment package
versions, correctness results, and lossless-conversion manifests are included.

Large datasets, converted index copies, compiled binaries, logs, and native result
arrays stay under /data/cuvs-upstream-validation on the validation host. Checksums
identify the native arrays. Paths in archived configurations and commands describe
that host; these files do not pretend to embed the underlying datasets.

The control CAGRA comparisons use separate public-API probe binaries,
compiled with each checkout's headers. Their timing is compared only with each
other, separately from the cuVS-bench results. The 13 control cases cover eight
primary dataset/k combinations and five sparse datasets.

CSV line endings and GPU inventory whitespace are normalized for Git; numeric
measurements and raw benchmark/native JSON are unchanged.
