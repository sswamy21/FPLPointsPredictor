"""Loopback-only dashboard and shared forecast lifecycle."""

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock, Thread

import joblib
import pandas as pd
import sklearn
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from . import congestion, data, markets, model, player_workload, snapshots
from .lineup import POSITION_IDS, SquadError, demo_squad, optimize, resolve_upload
from .transfers import recommend_transfers

LOGGER = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent


def default_cache():
    workspace_work = PROJECT.parent.parent / "work"
    if PROJECT.parent.name == "outputs" and workspace_work.is_dir():
        return workspace_work / "fpl-cache"
    return PROJECT / ".cache"


def season_label(year):
    year = int(year)
    return f"{year}/{(year + 1) % 100:02d}"


def _model_key(history, training_fixtures, context, exclude_groups):
    digest = hashlib.sha256()
    digest.update(sklearn.__version__.encode())
    digest.update(pd.__version__.encode())
    digest.update(repr(exclude_groups).encode())
    for module in (data, model, snapshots, congestion, player_workload, markets):
        digest.update(Path(module.__file__).read_bytes())
    for frame in (history, training_fixtures, *context.values()):
        for column, dtype in frame.dtypes.items():
            digest.update(f"{column}:{dtype};".encode())
        digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
        digest.update(repr(frame.attrs).encode())
    return digest.hexdigest()


def _cached_model(cache, history, training_fixtures, deadline, context):
    exclude_groups = []
    if os.environ.get("FPL_ENABLE_PLAYER_WORKLOAD") != "1":
        exclude_groups.append("player_cup_workload")
        context = dict(context, player_workload=context["player_workload"].iloc[:0])
    if os.environ.get("FPL_ENABLE_MARKET") != "1" and not os.environ.get("FPL_MARKET_CSV"):
        exclude_groups.append("market")
        context = dict(context, market_context=context["market_context"].iloc[:0])
    key = _model_key(history, training_fixtures, context, exclude_groups)
    cache.mkdir(parents=True, exist_ok=True)
    artifact = cache / "predictor.joblib"
    if artifact.is_file():
        try:
            saved_key, trained = joblib.load(artifact)
            if saved_key == key and trained.version == model.MODEL_VERSION:
                return trained
        except Exception:
            LOGGER.warning("Could not read the local model cache. Refitting it.", exc_info=True)

    trained = model.fit_model(history, training_fixtures, validate=True, as_of=deadline,
                              exclude_groups=exclude_groups, **context)
    with NamedTemporaryFile(dir=cache, suffix=".joblib", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        joblib.dump((key, trained), temporary, compress=3)
        temporary.replace(artifact)
    finally:
        temporary.unlink(missing_ok=True)
    return trained


@dataclass(frozen=True)
class Snapshot:
    live: object
    projections_by_week: dict
    coverage: dict
    metrics: dict

    @property
    def projections(self):
        return self.projections_by_week[self.live.gameweek]


class ForecastService:
    def __init__(self, cache):
        self.cache = Path(cache).expanduser().resolve()
        self.store = data.DataStore(self.cache)
        self.lock = Lock()
        self.snapshot = None
        self.loading = False
        self.phase = None
        self.error = None

    def _set_phase(self, phase):
        with self.lock:
            self.phase = phase

    def _build(self, refresh):
        try:
            self._set_phase("Fetching official FPL fixtures and player histories…")
            live = self.store.live(refresh=refresh)
            self._set_phase("Loading historical results and pre-deadline fixture context…")
            history, training_fixtures, full_fixtures, context = load_inputs(self.store, live, refresh=refresh)
            self._set_phase("Training or loading the model and projecting the upcoming Gameweek…")
            trained = _cached_model(
                self.cache, history, training_fixtures, live.deadline, training_context(context, training_fixtures)
            )
            predictions = model.forecast_gameweeks(
                trained, history, full_fixtures, live.bootstrap, live.season, live.gameweek, live.deadline, **context
            )
            expected_ids = {
                player["id"] for player in live.bootstrap["elements"]
                if player["element_type"] in POSITION_IDS
            }
            missing = expected_ids - predictions[live.gameweek].keys()
            if missing:
                raise RuntimeError(f"The predictor omitted {len(missing)} official FPL players.")
            years = sorted(int(year) for year in history["season"].unique() if year < live.season)
            if len(years) < 10:
                raise RuntimeError(f"The archive contains {len(years)} seasons. At least 10 are required.")
            current_rows = int(history["season"].eq(live.season).sum())
            coverage = {
                "seasons": len(years), "first": season_label(years[0]), "last": season_label(years[-1]),
                "rows": len(history) - current_rows, "live_rows": current_rows,
                "predeadline_player_snapshots": len(context["player_context"]),
                "predicted_minutes_snapshots": len(context["minutes_context"]),
                "market_fixture_snapshots": len(context["market_context"]),
                "cup_team_matches": len(context["other_fixtures"]),
                "cup_player_matches": len(context["player_workload"]),
                "active_feature_groups": [
                    name for name, features in model.FEATURE_GROUPS.items()
                    if set(features).intersection(trained.feature_columns)
                ],
            }
            with self.lock:
                self.snapshot = Snapshot(live, predictions, coverage, trained.metrics)
        except Exception as error:
            LOGGER.exception("Forecast build failed")
            with self.lock:
                self.error = str(error) or type(error).__name__
        finally:
            with self.lock:
                self.loading = False
                self.phase = None

    def start(self, refresh=False, background=True):
        with self.lock:
            if self.loading:
                return
            self.loading = True
            self.error = None
        if background:
            Thread(target=self._build, args=(refresh,), daemon=True, name="FPL forecast").start()
        else:
            self._build(refresh)

    def status(self):
        with self.lock:
            snapshot, loading, phase, error = self.snapshot, self.loading, self.phase, self.error
        details = {
            "state": "loading" if loading else "ready" if snapshot else "error",
            "message": phase if loading else error if snapshot is None else None,
            "season": None, "gameweek": None, "deadline": None, "updated_at": None,
            "coverage": None, "metrics": None, "warning": None,
        }
        if snapshot is not None:
            details.update(
                snapshot_details(snapshot), coverage=snapshot.coverage, metrics=snapshot.metrics
            )
            if error:
                details["warning"] = f"Refresh failed. Showing the last successful forecast. {error}"
        return details

    def current(self):
        with self.lock:
            if self.loading or self.snapshot is None:
                return None
            snapshot = self.snapshot
        if snapshot.live.deadline <= datetime.now(timezone.utc):
            self.start(refresh=True)
            return None
        return snapshot


def snapshot_details(snapshot):
    live = snapshot.live
    return {
        "season": season_label(live.season), "gameweek": live.gameweek,
        "deadline": live.deadline.isoformat(), "updated_at": live.refreshed_at.isoformat(),
        "warning": getattr(live, "warning", None),
    }


def combine(archived, archived_fixtures, live):
    """Keep completed prior rounds in fit and the full schedule in forecast."""
    completed = live.fixtures.loc[
        live.fixtures["finished"] & (live.fixtures["gameweek"] < live.gameweek)
        & (live.fixtures["kickoff"] < live.deadline)
    ]
    realized = live.history.loc[live.history["fixture"].isin(completed["fixture"])]
    history = pd.concat([archived, realized], ignore_index=True)
    training_fixtures = pd.concat([archived_fixtures, completed], ignore_index=True)
    full_fixtures = pd.concat([archived_fixtures, live.fixtures], ignore_index=True)
    return history, training_fixtures, full_fixtures


def _official_deadlines(fixtures, player_context):
    official = player_context.groupby(["season", "gameweek"])["context_deadline"].first()
    joined = fixtures.join(official.rename("official_deadline"), on=["season", "gameweek"])
    joined["deadline"] = joined.pop("official_deadline").combine_first(joined["deadline"])
    return joined


def _context_path(cache, env, filename):
    configured = os.environ.get(env)
    if configured:
        return Path(configured).expanduser()
    path = cache / filename
    return path if path.is_file() else None


def load_inputs(store, live, *, refresh=False):
    archived, archived_fixtures = store.history(through=live.season - 1)
    history, training, full = combine(archived, archived_fixtures, live)
    cache = store.cache_dir
    players = snapshots.load_player_context(cache, full, live)
    full = _official_deadlines(full, players)
    training = _official_deadlines(training, players)
    market_path = _context_path(cache, "FPL_MARKET_CSV", "market.csv")
    minutes_path = _context_path(cache, "FPL_MINUTES_CSV", "predicted-minutes.csv")
    other_fixtures = congestion.load_other_fixtures(cache, refresh=refresh)
    context = {
        "player_context": players,
        "market_context": markets.load_market_context(market_path, full),
        "minutes_context": snapshots.load_minutes_context(minutes_path, full),
        "other_fixtures": other_fixtures,
        "player_workload": player_workload.load_player_workload(cache, other_fixtures, live),
    }
    return history, training, full, context


def training_context(context, fixtures):
    event_keys = fixtures[["season", "gameweek"]].drop_duplicates()
    fixture_keys = fixtures[["season", "fixture"]].drop_duplicates()
    other = context["other_fixtures"]
    workload = context["player_workload"]
    latest_deadline = fixtures["deadline"].max()
    return {
        "player_context": context["player_context"].merge(event_keys, on=["season", "gameweek"]),
        "minutes_context": context["minutes_context"].merge(event_keys, on=["season", "gameweek"]),
        "market_context": context["market_context"].merge(fixture_keys, on=["season", "fixture"]),
        "other_fixtures": other.loc[other["kickoff"] < latest_deadline],
        "player_workload": workload.loc[pd.to_datetime(workload["observed_at"], utc=True) < latest_deadline],
    }


def _money(value, label):
    try:
        amount = Decimal(str(value))
        if amount.is_finite() and 0 <= amount < 1000:
            tenths = amount * 10
            if tenths == tenths.to_integral_value():
                return int(tenths)
    except InvalidOperation:
        pass
    raise SquadError(f"Enter {label} in £m with at most one decimal place.")


def create_app(cache=None, *, warm=True):
    app = Flask(__name__, static_folder="static", static_url_path="/assets")
    app.config.update(MAX_CONTENT_LENGTH=256 * 1024, TRUSTED_HOSTS=["127.0.0.1", "localhost"])
    service = ForecastService(cache or os.environ.get("FPL_CACHE_DIR") or default_cache())
    app.extensions["fpl"] = service
    if warm:
        service.start()

    @app.errorhandler(HTTPException)
    def http_error(error):
        if request.path.startswith("/api/"):
            return jsonify(error=error.description), error.code
        return error

    @app.errorhandler(SquadError)
    def squad_error(error):
        return jsonify(error=str(error)), 400

    @app.after_request
    def response_headers(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/")
    def dashboard():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/status")
    def status():
        return jsonify(service.status())

    @app.post("/api/refresh")
    def refresh():
        service.start(refresh=True)
        return jsonify(service.status()), 202

    @app.get("/api/players")
    def players():
        snapshot = service.current()
        if snapshot is None:
            return jsonify(error=service.status()["message"] or "Forecast is loading."), 409
        return jsonify(players=sorted(snapshot.projections.values(), key=lambda player: player["id"]))

    @app.get("/sample.csv")
    def sample():
        snapshot = service.current()
        if snapshot is None:
            return jsonify(error=service.status()["message"] or "Forecast is loading."), 409
        contents = "id\n" + "\n".join(map(str, demo_squad(snapshot.projections))) + "\n"
        return app.response_class(
            contents, mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=fpl-squad.csv"}
        )

    @app.post("/api/optimize")
    def recommend():
        payload = request.get_json()
        if not isinstance(payload, dict):
            raise SquadError("Send a Manager ID, a squad file, or a demo request.")
        snapshot = service.current()
        if snapshot is None:
            return jsonify(error=service.status()["message"] or "Forecast is loading."), 409

        live = snapshot.live
        pick_gameweek = None
        if "manager_id" in payload:
            raw_id = payload["manager_id"]
            if isinstance(raw_id, bool) or not isinstance(raw_id, (int, str)):
                raise SquadError("Enter a positive FPL Manager ID.")
            raw_id = str(raw_id).strip()
            if not raw_id.isascii() or not raw_id.isdecimal() or len(raw_id) > 12 or int(raw_id) == 0:
                raise SquadError("Enter a positive FPL Manager ID.")
            manager_id = int(raw_id)
            try:
                ids, pick_gameweek = service.store.public_picks(manager_id, live)
            except (data.PublicPicksUnavailable, ValueError, LookupError) as error:
                raise SquadError(str(error)) from error
            except data.DataError as error:
                LOGGER.warning("FPL manager picks lookup failed", exc_info=True)
                return jsonify(error=str(error) or "Could not reach the FPL picks endpoint."), 502
            kind, label = "manager", f"Manager {manager_id:,}"
        elif payload.get("demo") is True:
            ids = demo_squad(snapshot.projections)
            kind, label = "demo", "Example squad"
        else:
            filename = payload.get("filename")
            contents = payload.get("contents")
            if not isinstance(filename, str) or not filename.strip() or not isinstance(contents, str):
                raise SquadError("Choose a CSV or JSON squad file.")
            ids = resolve_upload(filename, contents, live.bootstrap["elements"])
            kind, label = "upload", filename[:180]

        result = optimize(ids, snapshot.projections)
        result.update(snapshot_details(snapshot), source={"type": kind, "label": label, "gameweek": pick_gameweek})
        result["transfer_gameweeks"] = list(snapshot.projections_by_week)
        if kind == "manager":
            manager_warning = f"FPL publishes squads after each deadline. Imported the GW{pick_gameweek} squad. Upload a file to include transfers made since then."
            result["warning"] = " ".join(filter(None, (result["warning"], manager_warning)))
        return jsonify(result)

    @app.post("/api/transfers")
    def transfer_targets():
        payload = request.get_json()
        if not isinstance(payload, dict):
            raise SquadError("Send the current squad and transfer budget.")
        snapshot = service.current()
        if snapshot is None:
            return jsonify(error=service.status()["message"] or "Forecast is loading."), 409
        season, gameweek = season_label(snapshot.live.season), snapshot.live.gameweek
        if payload.get("season") != season or payload.get("gameweek") != gameweek:
            return jsonify(error="Re-optimize your squad for the current Gameweek, then search for transfers."), 409

        ids = payload.get("squad_ids")
        if not isinstance(ids, list) or len(ids) != 15 or any(type(player) is not int for player in ids):
            raise SquadError("Supply the 15 FPL player IDs in your current squad.")
        horizon = payload.get("horizon", len(snapshot.projections_by_week))
        if type(horizon) is not int or not 1 <= horizon <= len(snapshot.projections_by_week):
            raise SquadError("Choose an available Gameweek horizon.")
        free_transfers = payload.get("free_transfers")
        if type(free_transfers) is not int or not 0 <= free_transfers <= 5:
            raise SquadError("Enter the number of free transfers available, from 0 to 5.")
        outgoing = payload.get("outgoing_id")
        if outgoing is not None and (type(outgoing) is not int or outgoing not in ids):
            raise SquadError("Choose a player from your squad to sell.")

        raw_sales = payload.get("selling_prices", {})
        if not isinstance(raw_sales, dict) or raw_sales.keys() - {str(player) for player in ids}:
            raise SquadError("Selling prices must correspond to players in your squad.")
        selling_prices = {int(player): _money(price, "each selling price") for player, price in raw_sales.items()}
        bank = _money(payload.get("bank"), "your cash in the bank")
        weeks = dict(list(snapshot.projections_by_week.items())[:horizon])
        result = recommend_transfers(ids, weeks, snapshot.live.bootstrap["elements"], bank=bank,
                                     free_transfers=free_transfers, selling_prices=selling_prices, outgoing_id=outgoing)
        result.update(season=season, gameweek=gameweek)
        return jsonify(result)

    return app
