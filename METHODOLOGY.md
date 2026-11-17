# Forecast methodology

## Point-in-time inputs

- **Player form.** Use points, minutes, starts, goals, assists, bonus, BPS and available expected statistics from the last 3, 6 and 20 eligible fixtures. Stable FPL player codes connect seasons. In periods with no observed start field, at least 45 minutes serves as a proxy.
- **Team form.** Use each side’s past goals and expected goals for and against, with home and away splits for the opponent. Aggregate a team’s match xG from its players’ xG when at least 80% of recorded appearances have xG. Missing seasons remain missing. Club codes connect promoted, relegated and returning teams where prior Premier League results are available.
- **Availability and set pieces.** Archived official FPL snapshots supply the availability flag, age of the published news, and the order for penalties, direct free kicks, and corners and indirect free kicks. An archived snapshot must identify the target round as upcoming, match the recovered official deadline, and have been captured before it. Current forecasts use the latest eligible official FPL capture. [Snapshot archive](https://github.com/Randdalf/fplcache)
- **Cup and European congestion.** Count completed FA Cup, League Cup, Champions League, Europa League and Conference League fixtures over the prior 7 and 14 days. Use the same three-hour result lag as league fixtures. Unknown club mappings or calendar partitions leave the signal missing. [ESPN competition scores](https://www.espn.com/soccer/scoreboard/_/league/eng.league_cup)
- **Markets.** Use pre-deadline market prices to estimate home and away scoring rates and clean-sheet probabilities under an independent Poisson model. Historical requests stop a full aggregation interval before the FPL deadline, and returned timestamps are checked again. In older matches with match-result prices alone, infer both scoring rates from home, draw and away prices. [Price history API](https://docs.polymarket.com/api-reference/markets/get-prices-history)

All learned features use information available before the Gameweek deadline. Results also have to come from an earlier Gameweek. The pipeline uses the archived official deadline where it can verify one and otherwise estimates the deadline by applying the season’s FPL cutoff offset to the first scheduled kickoff. If a dated roster places a player at a different club from the actual fixture record, that fixture row is excluded from training and validation.

## Forecast and scoring

The forecast multiplies a player’s expected points conditional on appearing by their estimated probability of appearing. The availability classifier also estimates the chance of starting and reaching 60 minutes. A model trained with timestamped FPL availability learns how to interpret the flag and news age. Where availability inputs are unavailable to the fitted model, the live forecast uses the official flag as a multiplier. Each fixture is projected independently and Gameweek totals add the scheduled fixtures. A blank contributes zero.

The historical comparison uses expanding-season training and three full-season holdouts. All model families use the same evaluation rows and eligible feature columns in each fold. The cumulative feature comparison adds one data group at a time using histogram gradient boosting. A feature group needs varying training data and observed test data in at least one fold to enter the comparison. Market coverage must be read per fold because the pre-deadline price archive starts in 2024/25.

Lineup evaluation draws up to 20 deterministic 15-player squads per round, using prior appearances and the timestamped roster where available. Each candidate sees the same squads. Realized lineup points use the starting XI plus a second copy of the captain’s points. Captain points are also reported separately. This replay measures lineup choices on reconstructed squads, so its scores estimate the effect on selection for that player pool.

## November 17, 2026 results

The three holdouts cover 2023/24, 2024/25 and 2025/26, with 84,984 distinct player-Gameweeks and 2,220 reconstructed squad decisions across 111 rounds. The three ML models used the same extended data, including markets where the fold could learn from them. The recent-form baseline averages the previous six fixtures.

| Model | MAE | RMSE | Mean realized XI + captain | Mean captain points |
| --- | ---: | ---: | ---: | ---: |
| Recent-form baseline | 1.060 | 2.103 | 36.88 | 4.43 |
| Histogram gradient boosting | 0.935 | 1.902 | 39.23 | 5.27 |
| Regularized logistic + ridge | 0.942 | 1.909 | 38.99 | 5.23 |
| CatBoost | 0.939 | 1.903 | 39.08 | 5.19 |

The deployed model keeps histogram gradient boosting and includes team xG, learned FPL availability, set-piece duties and club congestion. This configuration scored **39.233 points per replayed lineup and 5.295 per captain pick**. Its MAE was **0.934** and RMSE was **1.901**. The old feature set with its original FPL availability multiplier scored **39.132**, **5.228**, **0.900 MAE** and **1.905 RMSE** on the same data. The updated default prioritizes lineup and captain decisions at the cost of higher average absolute error.

| Cumulative histogram model | MAE | Mean XI + captain | Mean captain points |
| --- | ---: | ---: | ---: |
| Original features with availability multiplier | 0.900 | 39.132 | 5.228 |
| Add team xG | 0.902 | 39.038 | 5.132 |
| Learn FPL availability from flags and age | 0.935 | 39.108 | 5.219 |
| Add set-piece duties | 0.933 | 39.182 | 5.247 |
| Add club congestion | 0.934 | 39.233 | 5.295 |
| Add market data | 0.935 | 39.235 | 5.270 |

The timestamp checks retained 672 historical fixture quotes across 2024/25 and 2025/26, plus all 10 upcoming Gameweek 11 fixtures. Only the 2025/26 holdout had prior-season market data to learn from. With such a small lineup difference and lower average captain points, market features remain experimental.

In a separate 2025/26 ablation with market and team inputs held fixed, individual cup workload improved MAE from **0.935 to 0.933** and changed lineup points from **39.83 to 39.74** across 740 replayed squads. It also remains experimental. The holdouts inform the choice of default configuration. Unseen future seasons will provide a separate test of that choice.

Reproduce the main comparison and the individual workload ablation with the recorded data sources enabled in the benchmark cache.

```console
python -m fpl benchmark --exclude-group player_cup_workload --output benchmark.json
python -m fpl benchmark --family hist_gradient_boosting --seasons 2025 --focus-group player_cup_workload --output workload.json
```

Machine-readable reports are in [the benchmark directory](benchmarks/).

## Optional inputs

**Market prices.** To retrieve the public archive and capture the upcoming round, run this from the app directory. Historical requests are cached.

```console
python -m fpl.polymarket --season 2024 --season 2025 --live
```

The benchmark reads `market.csv` in its cache directory. Set `FPL_ENABLE_MARKET=1` to use it in the dashboard model, or set `FPL_MARKET_CSV` to choose and enable a different file. Include `season`, `fixture` and `as_of`, with an explicit UTC offset, and goal estimates such as `home_expected_goals` and `away_expected_goals`. Optional `home_clean_sheet_probability` and `away_clean_sheet_probability` can be supplied directly. The loader also accepts `home_win_odds`, `draw_odds`, `away_win_odds`, `over_25_odds` and `under_25_odds` as decimal prices. Rerun the capture command to refresh upcoming quotes.

**Individual cup workload.** Run `python -m fpl warm-workload` to cache completed match lineups and substitutions from the same cup and European competitions. Enable the feature in the dashboard with `FPL_ENABLE_PLAYER_WORKLOAD=1`. It uses matched FPL players’ appearances, starts and estimated minutes over the prior seven days, with a four-hour lag after kickoff. Unmatched records and minutes with incomplete substitution evidence stay missing. [ESPN match data example](https://site.api.espn.com/apis/site/v2/sports/soccer/eng.league_cup/summary?event=727880)

**Independent predicted minutes.** Set `FPL_MINUTES_CSV` or save `predicted-minutes.csv` in the cache. Include `season`, `gameweek`, the stable FPL code in `player`, and the timezone-aware `context_asof` timestamp. Supply `expected_minutes` as minutes per fixture, `predicted_start_probability` as a value from 0 to 1, or both. The app uses the latest eligible record before the deadline. The same feed needs historical observations for the model to learn its predictive value. The published benchmark had no timestamped independent minutes feed.
