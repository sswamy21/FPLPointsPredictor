# FPL Lineup Optimizer

Forecast the upcoming Gameweek for a 15-player Fantasy Premier League squad. The dashboard shows the optimal formation, starters, captain, vice-captain, substitutes, and projected points. Select a player for their individual fixtures and estimated chances of appearing, starting, and reaching 60 minutes.

## Run the app

On Windows, double-click `Start FPL.cmd`. It uses Python 3.11 or newer and opens `http://127.0.0.1:8765/` in your browser. Keep its terminal open while you use the dashboard. On the first run, downloading player histories and fitting the model may take several minutes. Use **Refresh data** to fetch official updates before the deadline.

From the application directory, the terminal commands are:

```console
python -m venv .venv
```

Activate `.venv` with `.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on macOS or Linux. Then run:

```console
python -m pip install -r requirements.txt
python -m fpl serve
```

The server binds to loopback. The workspace launcher uses the prepared cache in `work/fpl-cache`. A standalone copy uses `.cache` in its application directory. Set `FPL_CACHE_DIR` or pass `--cache` before the command to use a different location. The raw historical CSVs and live responses have independent caches. The fitted model is reused when the training data and model code match.

## Import your squad

- **Manager ID** imports the latest eligible 15-player squad from your public FPL entry. Find the ID after `/entry/` in your FPL team URL. Transfers made after the published snapshot are private until the next deadline. An expired Free Hit is skipped to recover the underlying squad.
- **Upload file** accepts a UTF-8 CSV with an `id` or `name` column, or a JSON array with 15 FPL IDs or names. The JSON can also be an object with a `players` array. Use current-season FPL player IDs when a name is ambiguous. The app can generate a sample CSV after it has loaded the current season.

The same workflow is available from the terminal. The `optimize` command prints JSON.

```console
python -m fpl optimize --squad squad.csv
python -m fpl optimize --manager 123456 --refresh
```

## Find transfer targets

Load a squad, then open **Transfer targets** below the bench. Enter your cash in the bank and remaining free transfers. The app compares the next five Gameweeks by default, or all remaining rounds if fewer are scheduled. You can choose a shorter horizon or select someone to sell. For exact affordability, copy your players’ selling prices from the FPL Transfers page. Missing values use current market prices as estimates and the affected recommendations are marked.

Each result is an alternative single transfer. The app tests same-position replacements within your budget and the three-per-club limit, then re-optimizes the XI and captain in each selected round. It ranks the moves by projected points, including legal autosubs and vice-captain cover, after the four-point hit when you have zero free transfers. When no swap has a positive net gain, the app recommends holding. [Official transfer and selling-price rules](https://www.premierleague.com/en/news/2174907)

Later rounds use today’s player form and the scheduled opposition. Upcoming availability flags only apply to their own Gameweek, so check team news for injuries and suspensions that could extend further. The recommendations use your current bank and prices and leave the decision and FPL submission to you.

## Forecast and scoring

The model learns from the [historical FPL archive](https://github.com/vaastav/Fantasy-Premier-League) and the official [bootstrap](https://fantasy.premierleague.com/api/bootstrap-static/), [fixtures](https://fantasy.premierleague.com/api/fixtures/), and [element summary](https://fantasy.premierleague.com/api/element-summary/1/) endpoints. Stable player codes link season-specific IDs. Club form uses stable club codes and falls back to league priors for clubs without recent Premier League results. Missing xG, xA and xGC remain missing. Where starts are unavailable, 45 or more minutes serves as a proxy.

Features come from completed prior Gameweeks. Results become eligible three hours after kickoff. The model uses 3-, 6-, and 20-fixture player form, venue, rest, and a difficulty estimate based on previous team and opponent results. Archived FPL snapshots recover official deadlines, availability and set-piece roles where available. The app can also use team xG, cup and European workload, and deadline-stamped market or minutes estimates. [Forecast methodology and validation](METHODOLOGY.md)

In doubles, fixture points and expected minutes are summed. In blanks, both are zero. The player details display expected minutes, the chances of playing and reaching 60 minutes, and the official upcoming fixture difficulty.

The optimizer checks every legal formation and selects the XI with the highest sum of starter points and captain bonus. The two highest-projected starters are captain and vice-captain. The projected total then includes expected cover from the selected bench and vice-captain, broken out under the pitch. Bench players are shown in points order, with the reserve goalkeeper labeled separately from the three outfield substitution priorities. [Official formation and captaincy rules](https://www.premierleague.com/en/news/2174899/1000)

## Verification

```console
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests check deadline leakage, chronological backtesting, double and blank Gameweeks, lineup legality, and exact optimality against exhaustive combinations. To also run the live smoke test, start the dashboard and set `FPL_TEST_URL` to `http://127.0.0.1:8765`. The test uses public Manager ID `1` by default. Set `FPL_TEST_MANAGER` to select a different public ID.

Install the optional benchmark dependencies to rerun the comparison across model families.

```console
python -m pip install -r requirements-benchmark.txt
python -m fpl benchmark --output benchmark.json
```

The benchmark reports MAE, RMSE, and points from historically replayed lineup and captain decisions. See the methodology for results, coverage and optional data feeds. [Forecast methodology and validation](METHODOLOGY.md)
