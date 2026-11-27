# 4-hour RSI + 5-day options study

Run 2026-11-26 20:35 ET. Test window: 2026-05-28 to today. Check window: the 18 months before.
Option prices are ESTIMATED (Black-Scholes, realized vol x 1.2, 1.5% spread, $0.65/side). Not real quotes.
Returns are % on the premium paid, per trade. $ is per 1 contract.

## SPY — RSI(14) 30/70

| Window | Side | Exit | Trades | Win % | Avg % | Median % | $ total (1 ct) | Underlying avg | Benchmark avg % | p |
|---|---|---|---|---|---|---|---|---|---|---|
| TEST (6 mo) | calls | next | 1 | 100 | +16.7 | +16.7 | +231 | +1.24% | -9.7 | 0.264 |
| TEST (6 mo) | calls | 3d | 1 | 0 | -20.2 | -20.2 | -280 | +0.45% | -20.5 | 0.360 |
| TEST (6 mo) | calls | expiry | 0 | | | | | | | |
| TEST (6 mo) | puts | next | 4 | 25 | -18.6 | -3.8 | -445 | -0.01% | +7.4 | 0.764 |
| TEST (6 mo) | puts | 3d | 4 | 25 | -14.9 | -17.0 | -369 | +0.32% | +6.7 | 0.616 |
| TEST (6 mo) | puts | expiry | 4 | 50 | +49.2 | +103.9 | +861 | +0.99% | -1.0 | 0.211 |
| CHECK (18 mo before) | calls | next | 11 | 36 | -8.2 | -44.9 | -494 | -0.19% | +0.0 | 0.706 |
| CHECK (18 mo before) | calls | 3d | 11 | 64 | +39.7 | +53.7 | +3,584 | +1.10% | -1.3 | 0.057 |
| CHECK (18 mo before) | calls | expiry | 11 | 36 | +23.8 | -18.9 | +2,474 | +0.66% | -2.9 | 0.215 |
| CHECK (18 mo before) | puts | next | 32 | 25 | -4.9 | -12.7 | -1,333 | +0.00% | -6.1 | 0.433 |
| CHECK (18 mo before) | puts | 3d | 32 | 22 | -40.0 | -50.1 | -6,237 | -0.27% | -20.1 | 0.855 |
| CHECK (18 mo before) | puts | expiry | 32 | 25 | -27.9 | -100.2 | -5,563 | -0.07% | -34.1 | 0.379 |

## SPY — RSI(5) 20/80

| Window | Side | Exit | Trades | Win % | Avg % | Median % | $ total (1 ct) | Underlying avg | Benchmark avg % | p |
|---|---|---|---|---|---|---|---|---|---|---|
| TEST (6 mo) | calls | next | 5 | 20 | -33.5 | -45.8 | -1,490 | -0.67% | -9.7 | 0.854 |
| TEST (6 mo) | calls | 3d | 5 | 20 | -41.8 | -61.3 | -1,652 | -0.53% | -20.5 | 0.655 |
| TEST (6 mo) | calls | expiry | 5 | 20 | -17.9 | -53.4 | -1,015 | +0.21% | -37.8 | 0.279 |
| TEST (6 mo) | puts | next | 6 | 33 | -18.1 | -0.7 | -622 | -0.22% | +7.4 | 0.818 |
| TEST (6 mo) | puts | 3d | 6 | 33 | -10.2 | -17.0 | -382 | +0.11% | +6.7 | 0.614 |
| TEST (6 mo) | puts | expiry | 6 | 33 | -1.2 | -96.0 | -268 | +0.35% | -1.0 | 0.444 |
| CHECK (18 mo before) | calls | next | 30 | 53 | +11.8 | +1.4 | +2,555 | +0.22% | +0.0 | 0.115 |
| CHECK (18 mo before) | calls | 3d | 30 | 63 | +43.6 | +31.1 | +9,558 | +0.88% | -1.3 | 0.004 |
| CHECK (18 mo before) | calls | expiry | 30 | 60 | +72.2 | +90.4 | +12,985 | +1.01% | -2.9 | 0.000 |
| CHECK (18 mo before) | puts | next | 49 | 31 | -9.5 | -13.9 | -3,163 | -0.06% | -6.1 | 0.618 |
| CHECK (18 mo before) | puts | 3d | 49 | 16 | -37.9 | -56.0 | -10,259 | -0.33% | -20.1 | 0.899 |
| CHECK (18 mo before) | puts | expiry | 49 | 14 | -46.1 | -100.2 | -12,942 | -0.27% | -34.1 | 0.739 |

## QQQ — RSI(14) 30/70

| Window | Side | Exit | Trades | Win % | Avg % | Median % | $ total (1 ct) | Underlying avg | Benchmark avg % | p |
|---|---|---|---|---|---|---|---|---|---|---|
| TEST (6 mo) | calls | next | 6 | 83 | +36.5 | +29.6 | +2,910 | +1.57% | -9.2 | 0.035 |
| TEST (6 mo) | calls | 3d | 6 | 83 | +54.9 | +65.3 | +3,168 | +1.49% | -17.8 | 0.058 |
| TEST (6 mo) | calls | expiry | 5 | 60 | +108.1 | +103.6 | +5,660 | +2.73% | -32.8 | 0.022 |
| TEST (6 mo) | puts | next | 3 | 0 | -41.2 | -39.5 | -849 | -0.73% | +5.4 | 0.923 |
| TEST (6 mo) | puts | 3d | 3 | 33 | -33.3 | -40.2 | -746 | -0.27% | +10.4 | 0.731 |
| TEST (6 mo) | puts | expiry | 3 | 33 | -50.0 | -61.5 | -1,092 | +0.26% | +10.0 | 0.717 |
| CHECK (18 mo before) | calls | next | 13 | 46 | -2.6 | -16.3 | +125 | +0.22% | +1.5 | 0.605 |
| CHECK (18 mo before) | calls | 3d | 13 | 54 | +34.7 | +6.8 | +4,283 | +1.17% | +3.2 | 0.115 |
| CHECK (18 mo before) | calls | expiry | 13 | 54 | +27.7 | +18.3 | +3,309 | +0.73% | +5.4 | 0.255 |
| CHECK (18 mo before) | puts | next | 29 | 34 | -13.6 | -11.5 | -2,856 | -0.22% | -5.2 | 0.737 |
| CHECK (18 mo before) | puts | 3d | 29 | 24 | -10.6 | -47.1 | -2,671 | -0.13% | -18.2 | 0.311 |
| CHECK (18 mo before) | puts | expiry | 29 | 31 | -27.4 | -100.2 | -5,117 | -0.30% | -31.4 | 0.424 |

## QQQ — RSI(5) 20/80

| Window | Side | Exit | Trades | Win % | Avg % | Median % | $ total (1 ct) | Underlying avg | Benchmark avg % | p |
|---|---|---|---|---|---|---|---|---|---|---|
| TEST (6 mo) | calls | next | 12 | 67 | +1.7 | +12.0 | +449 | +0.40% | -9.2 | 0.229 |
| TEST (6 mo) | calls | 3d | 12 | 33 | -6.4 | -44.6 | -1,159 | +0.01% | -17.8 | 0.322 |
| TEST (6 mo) | calls | expiry | 11 | 18 | +5.1 | -29.0 | +689 | +0.63% | -32.8 | 0.150 |
| TEST (6 mo) | puts | next | 5 | 20 | -27.8 | -39.5 | -927 | -0.31% | +5.4 | 0.893 |
| TEST (6 mo) | puts | 3d | 5 | 40 | -15.9 | -28.5 | -526 | +0.35% | +10.4 | 0.665 |
| TEST (6 mo) | puts | expiry | 5 | 60 | +55.5 | +11.7 | +2,382 | +1.88% | +10.0 | 0.228 |
| CHECK (18 mo before) | calls | next | 33 | 48 | +18.1 | -3.1 | +4,040 | +0.50% | +1.5 | 0.044 |
| CHECK (18 mo before) | calls | 3d | 33 | 76 | +73.2 | +65.2 | +17,188 | +1.91% | +3.2 | 0.000 |
| CHECK (18 mo before) | calls | expiry | 33 | 61 | +67.7 | +39.5 | +14,569 | +1.45% | +5.4 | 0.004 |
| CHECK (18 mo before) | puts | next | 51 | 25 | -16.9 | -22.1 | -5,918 | -0.21% | -5.2 | 0.905 |
| CHECK (18 mo before) | puts | 3d | 51 | 24 | -26.9 | -48.5 | -9,512 | -0.29% | -18.2 | 0.720 |
| CHECK (18 mo before) | puts | expiry | 51 | 22 | -39.6 | -100.2 | -14,146 | -0.46% | -31.4 | 0.671 |

## How to read this

- Under 30 trades in a row means nothing yet, whatever the win rate.
- 'Benchmark avg' is the same option bought at every 4-hour bar. Buying short-dated options loses to decay and spread on average, so the benchmark is usually negative. Beating it is the bar, not beating zero.
- p under 0.05 is only interesting if the CHECK window agrees. This table has 48 cells; about 2 will look significant by luck.
- A 100% win rate or a huge average on a handful of trades is a red flag, not a result.
- Only the 'next' exit fits the 60-day rule exemption (SPY/QQQ sold the same or next session).
