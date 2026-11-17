import argparse
import json
import logging
import os
import webbrowser
from pathlib import Path
from threading import Timer

from werkzeug.serving import make_server

from .app import create_app, default_cache, load_inputs
from .benchmark import FAMILIES
from .data import DataStore
from .model import FEATURE_GROUPS


def main():
    parser = argparse.ArgumentParser(description="Forecast FPL points and optimize a 15-player squad.")
    parser.add_argument("--cache", type=Path, help="Directory for raw data and the fitted model.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Run the local dashboard.")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-browser", action="store_true")
    cli = commands.add_parser("optimize", help="Print a lineup as JSON.")
    choice = cli.add_mutually_exclusive_group(required=True)
    choice.add_argument("--squad", type=Path)
    choice.add_argument("--manager")
    choice.add_argument("--demo", action="store_true")
    cli.add_argument("--refresh", action="store_true", help="Fetch a fresh live snapshot first.")
    benchmark = commands.add_parser("benchmark", help="Compare forecast models on historical holdouts.")
    benchmark.add_argument("--seasons", type=int, nargs="+", help="Validation seasons by starting year.")
    benchmark.add_argument("--output", type=Path, help="Write the JSON results to this file.")
    benchmark.add_argument("--family", action="append", choices=FAMILIES, help="Restrict the model families compared.")
    ablation = benchmark.add_mutually_exclusive_group()
    ablation.add_argument("--exclude-group", action="append", choices=FEATURE_GROUPS, help="Omit a feature group.")
    ablation.add_argument("--focus-group", choices=FEATURE_GROUPS, help="Test one group with all other inputs fixed.")
    commands.add_parser("warm-workload", help="Cache past cup and European player appearances.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "benchmark":
        from .benchmark import run_benchmark
        from .model import FeatureHistory, _observations

        store = DataStore(args.cache or os.environ.get("FPL_CACHE_DIR") or default_cache())
        live = store.live()
        history, _, fixtures, context = load_inputs(store, live)
        logging.info("Building point-in-time feature snapshots for %s player fixtures.", f"{len(history):,}")
        observations = _observations(FeatureHistory(history, fixtures, **context), as_of=live.deadline)
        groups = FEATURE_GROUPS.copy()
        for group in dict.fromkeys(args.exclude_group or ()):
            observations = observations.drop(columns=list(groups.pop(group)))
        if args.focus_group:
            groups = {args.focus_group: groups[args.focus_group]}
        report = run_benchmark(
            observations, history=history, player_context=context["player_context"],
            feature_groups=groups, families=args.family or FAMILIES, seasons=args.seasons, progress=logging.info,
        )
        output = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8", newline="\n")
        else:
            print(output, end="")
        return

    if args.command == "warm-workload":
        from .congestion import load_other_fixtures
        from .player_workload import load_player_workload

        store = DataStore(args.cache or os.environ.get("FPL_CACHE_DIR") or default_cache())
        live = store.live()
        fixtures = load_other_fixtures(store.cache_dir, refresh=True)
        logging.info("Caching player workload from %s completed cup and European matches.", fixtures["event_id"].nunique())
        workload = load_player_workload(store.cache_dir, fixtures, live, download=True)
        print(f"Cached {len(workload):,} matched FPL player appearances and bench records.")
        return

    if args.command == "serve":
        app = create_app(args.cache, warm=False)
        server = make_server("127.0.0.1", args.port, app, threaded=True)
        app.extensions["fpl"].start()
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"FPL dashboard at {url}", flush=True)
        if not args.no_browser:
            Timer(0.25, webbrowser.open, args=(url,)).start()
        server.serve_forever()
        return

    if args.squad:
        try:
            contents = args.squad.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            parser.exit(1, f"Could not read squad file. {error}\n")
        payload = {"filename": args.squad.name, "contents": contents}
    elif args.manager:
        payload = {"manager_id": args.manager}
    else:
        payload = {"demo": True}
    app = create_app(args.cache, warm=False)
    service = app.extensions["fpl"]
    service.start(refresh=args.refresh, background=False)
    status = service.status()
    if status["state"] != "ready":
        parser.exit(1, f'{status["message"]}\n')
    response = app.test_client().post("/api/optimize", json=payload)
    result = response.get_json()
    if response.status_code != 200:
        parser.exit(1, f'{result["error"]}\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
