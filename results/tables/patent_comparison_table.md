# Patent Comparison Table

| Method | Recovery Success (%) | Mean Recovery Latency (ms) | Median Recovery Latency (ms) | States Reconstructed (mean) | States Reconstructed (%) | Duplicate Side Effects (mean) | Mean CIV Score | Recovery Iterations (mean) | CPU Time (ms, mean) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full Rollback | 100.0 ± 0.0 | 13.76 | 3.37 | 185.0 | 100.0 | 36.45 | 1.0000 | 1.00 | 13.64 |
| Checkpoint Recovery | 100.0 ± 0.0 | 0.02 | 0.01 | 178.1 | 91.4 | 35.25 | 1.0000 | 1.00 | 0.03 |
| Dependency-Guided | 100.0 ± 0.0 | 0.86 | 0.15 | 176.4 | 88.8 | 34.60 | 1.0000 | 1.00 | 0.86 |
| BCRF (Proposed) | 100.0 ± 0.0 | 54.29 | 11.81 | 87.8 | 46.5 | 0.00 | 0.8961 | 1.00 | 53.95 |