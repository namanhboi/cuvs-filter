# Retired FAVOR experiments

FAVOR is deprecated and unavailable in the active cuVS integration. Its search mode,
penalty parameters, delta-d estimation/sidecars, and experimental launchers are no
longer supported. Legacy launchers exit with a retirement message; FAVOR kernels
and FAVOR-only GPU tests are excluded from the build. Public legacy API symbols
remain only to report explicit errors, never to substitute a different algorithm.

Historical configurations, reports, and measurements in this directory are retained
as experiment records. They do not describe the supported algorithm set.

Use `benchmarks/retrieve_workshop` for CAGRA-Base, CAGRA-Retain, CAGRA-NaviX, and IVF.
The shared NaviX bitmap preparation tools moved to
`benchmarks/retrieve_workshop/navix_bitmap`.

Some frozen paper configurations use `favor_udf_passing_accumulator` and
`favor_diagnostics_*` for shared Retain and search diagnostics. These compatibility
names do not enable FAVOR. New configurations can use `passing_result_accumulator`
for the retention control. Shared C++ search diagnostics have generic names.
