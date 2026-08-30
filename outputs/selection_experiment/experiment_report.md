# Stock-Selection Experiment Report

**Preregistered gate outcome:** Partitioning Selection: feasible-but-not-promising, Density Selection: feasible-but-not-promising.

This is a historical walk-forward result over a retrospective 30-stock universe, not an investment recommendation or a strict constituent-level out-of-sample test.

## Full-Universe Metrics

| Strategy | Initial Value ($) | Final Value ($) | Return Amount ($) | Total Return | CAGR | Volatility | Sharpe Ratio | Sortino Ratio | Maximum Drawdown | Calmar Ratio | Beta | Alpha | Tracking Error | Information Ratio | Turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Density Selection | 10000 | 32591.1 | 22591.1 | 2.25911 | 0.218236 | 0.182558 | 0.949304 | 0.946821 | -0.254011 | 0.859161 | 0.709049 | 0.091488 | 0.137767 | 0.420391 | 0.438219 |
| Eligible Universe Equal Weight | 10000 | 27264.4 | 17264.4 | 1.72644 | 0.182448 | 0.162309 | 0.864003 | 0.787864 | -0.272503 | 0.669524 | 0.710918 | 0.0582047 | 0.108941 | 0.22809 | 0.135413 |
| Partitioning Selection | 10000 | 25065.6 | 15065.6 | 1.50656 | 0.165951 | 0.167364 | 0.759136 | 0.683305 | -0.254393 | 0.652343 | 0.749063 | 0.0406195 | 0.104482 | 0.111641 | 0.440611 |
| Equal Weight | 10000 | 23932.9 | 13932.9 | 1.39329 | 0.156977 | 0.15786 | 0.746385 | 0.681414 | -0.252243 | 0.622327 | 0.700458 | 0.0370001 | 0.105648 | 0.0230646 | 0.13491 |
| SPY | 10000 | 22906.4 | 12906.4 | 1.29064 | 0.148533 | 0.185248 | 0.622879 | 0.54297 | -0.286448 | 0.518536 | 1 | 0 | 0 |  | 0 |

## Calendar-Year Returns

| Year | Equal Weight | Eligible Universe Equal Weight | Partitioning Selection | Density Selection | SPY | Partitioning Selection vs Equal Weight | Partitioning Selection vs Eligible Universe Equal Weight | Partitioning Selection vs SPY | Density Selection vs Equal Weight | Density Selection vs Eligible Universe Equal Weight | Density Selection vs SPY |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 0.635626 | 0.650658 | 0.226873 | 1.03066 | 0.172352 | -0.408753 | -0.423785 | 0.0545211 | 0.39503 | 0.379998 | 0.858304 |
| 2021 | 0.0887372 | 0.149682 | 0.190285 | 0.155593 | 0.268261 | 0.101548 | 0.0406036 | -0.077976 | 0.0668555 | 0.00591111 | -0.112669 |
| 2022 | -0.106693 | -0.109871 | -0.0718771 | -0.125708 | -0.176102 | 0.0348154 | 0.0379935 | 0.104225 | -0.0190158 | -0.0158378 | 0.0503935 |
| 2023 | 0.223385 | 0.270757 | 0.291474 | 0.258348 | 0.250414 | 0.0680893 | 0.020717 | 0.0410603 | 0.0349633 | -0.012409 | 0.00793435 |
| 2024 | 0.0872119 | 0.0982373 | 0.185352 | 0.115888 | 0.267204 | 0.09814 | 0.0871145 | -0.0818526 | 0.0286757 | 0.0176503 | -0.151317 |
| 2025 | 0.131124 | 0.156508 | 0.208062 | 0.131305 | 0.18009 | 0.0769379 | 0.051554 | 0.0279719 | 0.00018122 | -0.0252026 | -0.0487848 |

Only six annual holding periods are available, so no reliable statistical significance or p-value claim is made. Full and ex-CELH results, coverage, turnover, and integrity gates are recorded in `decision.json` and the companion CSV files.