# -*- coding: utf-8 -*-
"""
CLI entry point for the YangtzeDelta Proposal Analyser.

Provides two modes:
  - pipeline: Run the full analysis pipeline on a proposal
  - server:   Start the FastAPI web server
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="YangtzeDelta Proposal Analyser CLI"
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    sub.add_parser("pipeline", help="Run the full analysis pipeline")

    srv = sub.add_parser("server", help="Start the FastAPI web server")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "pipeline":
        from src.tools.run_pipeline import run_full_pipeline
        run_full_pipeline()
    elif args.command == "server":
        import uvicorn
        uvicorn.run("src.api.server:app", host=args.host, port=args.port, reload=False)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
