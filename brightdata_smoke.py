#!/usr/bin/env python3
"""Simple Bright Data smoke test.

This script replicates the Bright Data Google Trends request used by the
application. Provide keywords and optional zone/timeframe/geo arguments.
The Bright Data API token is read from the ``BRIGHTDATA_TOKEN`` environment
variable by default.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from urllib.parse import urlencode

import requests

BRIGHTDATA_ENDPOINT = "https://api.brightdata.com/request"


def build_trends_url(keywords: list[str], *, timeframe: str, geo: str | None) -> str:
    """Create the Google Trends explore URL Bright Data expects."""
    params = {
        "q": ",".join(keywords),
        "hl": "en",
        "date": timeframe,
        "brd_json": "1",
        # Request only timeseries data to avoid heavy parsing
        "brd_trends": "timeseries",
    }
    if geo:
        params["geo"] = geo
    return f"https://trends.google.com/trends/explore?{urlencode(params)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Bright Data smoke test")
    parser.add_argument("keywords", nargs="+", help="Keywords to query")
    parser.add_argument("--zone", default="serp_api6", help="Bright Data SERP zone")
    parser.add_argument("--timeframe", default="today 5-y", help="Google Trends timeframe")
    parser.add_argument("--geo", default="", help="Google Trends geo code")
    parser.add_argument("--token", default=os.getenv("BRIGHTDATA_TOKEN"), help="Bright Data API token")
    args = parser.parse_args()

    if not args.token:
        parser.error("Bright Data token not provided. Set BRIGHTDATA_TOKEN or use --token")

    url = build_trends_url(args.keywords, timeframe=args.timeframe, geo=args.geo)
    payload = {"zone": args.zone, "url": url, "format": "raw"}
    headers = {
        "Authorization": f"Bearer {args.token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    start = time.time()
    try:
        resp = requests.post(BRIGHTDATA_ENDPOINT, json=payload, headers=headers, timeout=180)
    except Exception as exc:  # pragma: no cover - network errors
        print(f"Request failed: {exc}")
        return 1
    elapsed = time.time() - start
    print(f"Status: {resp.status_code} in {elapsed:.1f}s")
    print(resp.text[:500])
    return 0 if resp.ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
