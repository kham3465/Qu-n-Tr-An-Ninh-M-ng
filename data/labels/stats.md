# Curated label stats

- slither ok: 141/143
- fail (Slither 0.11 assert on tuple-assign): reentrancy_bonus.sol, reentrancy_cross_function.sol
- in-scope RAC alerts: 343
- SFT: R=164, A=77, C=83, J=112 (unknown excluded)

| family | keep | drop | unknown |
| --- | --- | --- | --- |
| R | 28 | 136 | 3 |
| A | 9 | 68 | 1 |
| C | 56 | 27 | 15 |

Keep by detector: unchecked-lowlevel 46, reentrancy-eth 27, unchecked-send 10, controlled-array-length 3, arbitrary-send-eth 2, tx-origin 2, suicidal 2, reentrancy-no-eth 1.

A keep is sparse because Curated access_control (constructor name, mapping write) rarely matches Slither A detectors — those bugs land ABSENT on the closed list, not as missed keeps.
