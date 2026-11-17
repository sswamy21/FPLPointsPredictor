import argparse
import json
import logging
import webbrowser
from pathlib import Path
from threading import Timer

from werkzeug.serving import make_server

from .app import create_app


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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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
