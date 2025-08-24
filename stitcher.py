import hashlib
import json
import math
import os
import time
import itertools
import datetime as dt
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import requests
import logging

logger = logging.getLogger(__name__)
logger.propagate = True


# -----------------------
# Helpers / Caching
# -----------------------

def _mk_cache_path(cache_dir: str, params: Dict) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]
    return os.path.join(cache_dir, f"{h}.json")


def _load_cache(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# -----------------------
# Fetcher
# -----------------------

class TrendsFetcher:
    """Fetches Google Trends interest-over-time via SerpAPI."""

    def __init__(
        self,
        serpapi_key: str,
        geo: str = "",
        timeframe: str = "today 12-m",
        cache_dir: str = ".cache",
        sleep_ms: int = 250,
        use_cache: bool = True,
        debug: bool = False,
    ) -> None:
        self.key = serpapi_key
        self.geo = geo
        self.timeframe = timeframe
        self.cache_dir = cache_dir
        self.sleep_ms = sleep_ms
        self.use_cache = use_cache
        self.debug = debug

    def _log_debug(self, *args, **kwargs) -> None:
        if not self.debug:
            return
        try:  # pragma: no cover - best effort
            import streamlit as st

            st.write(*args, **kwargs)
        except Exception:
            print(*args, **kwargs)

    def fetch_batch(self, terms: List[str]) -> pd.DataFrame:
        if len(terms) > 5:
            raise ValueError("Max 5 terms per Trends batch.")
        self._log_debug(f"Fetching batch: {terms}")
        q = ",".join(terms)
        params = {
            "engine": "google_trends",
            "q": q,
            "data_type": "TIMESERIES",
            "time": self.timeframe,
            "api_key": self.key,
        }
        if self.geo:
            params["geo"] = self.geo

        cache_path = _mk_cache_path(self.cache_dir, params)
        self._log_debug("Cache path", cache_path)
        data = _load_cache(cache_path) if self.use_cache else None
        if data is None:
            self._log_debug("Request params", params)
            r = requests.get("https://serpapi.com/search", params=params, timeout=60)
            status = r.status_code
            body = getattr(r, "text", "") or ""
            self._log_debug("HTTP status", status)
            rate = r.headers.get("X-RateLimit-Remaining")
            if rate is not None:
                self._log_debug("X-RateLimit-Remaining", rate)
            try:
                r.raise_for_status()
                data = r.json()
                err = data.get("error") or data.get("error_message")
                if err:
                    raise RuntimeError(f"SerpAPI error: {err}")
            except Exception as e:
                msg = f"{e}; status={status}"
                if rate is not None:
                    msg += f"; X-RateLimit-Remaining={rate}"
                msg += f"; body={body[:200]}"
                raise RuntimeError(msg)
            self._log_debug("Response sample", json.dumps(data)[:200])
            if self.use_cache:
                _save_cache(cache_path, data)
            if self.sleep_ms > 0:
                time.sleep(self.sleep_ms / 1000.0)
        else:
            self._log_debug("Using cached data", cache_path)

        return self._parse_timeseries(data, terms)

    @staticmethod
    def _parse_timeseries(payload: Dict, terms: List[str]) -> pd.DataFrame:
        """
        Parse SerpAPI payload into long form [date, term, value].
        Handles a couple of common shapes.
        """
        # Try common paths
        timeline = None
        if isinstance(payload, dict):
            io = payload.get("interest_over_time", {})
            if isinstance(io, dict):
                timeline = io.get("timeline_data")
            if timeline is None:
                timeline = payload.get("timeline_data")

        if not timeline:
            keys = list(payload.keys()) if isinstance(payload, dict) else []
            raise RuntimeError(f"No timeline_data returned. Raw keys: {keys}")

        rows = []
        for point in timeline:
            # timestamp handling
            ts = point.get("time") or point.get("timestamp") or point.get("date") or point.get("formattedTime")
            if ts is None:
                continue
            date = TrendsFetcher._coerce_date(ts)

            # values can be list-of-dicts or list aligned to input terms
            vals = point.get("values") or point.get("value")
            if isinstance(vals, list) and len(vals) > 0:
                if isinstance(vals[0], dict) and "value" in vals[0]:
                    for v in vals:
                        q = v.get("query")
                        val = v.get("value")
                        if q is None or val is None:
                            continue
                        try:
                            rows.append({"date": date, "term": str(q), "value": float(val)})
                        except Exception:
                            continue
                else:
                    # assume aligned to provided "terms"
                    for t, v in itertools.zip_longest(terms, vals):
                        if t is None or v is None:
                            continue
                        try:
                            rows.append({"date": date, "term": str(t), "value": float(v)})
                        except Exception:
                            continue

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("Empty dataframe from Trends batch parse.")
        return df

    @staticmethod
    def _coerce_date(ts) -> dt.date:
        # Try Unix seconds
        try:
            if isinstance(ts, (int, float)) and ts > 1000000000:
                return dt.datetime.utcfromtimestamp(int(ts)).date()
        except Exception:
            pass
        # Try ISO date
        try:
            return pd.to_datetime(ts).date()
        except Exception:
            # last resort: today
            return dt.date.today()


# -----------------------
# Batching and math
# -----------------------

def make_batches(terms: List[str], group_size: int = 5) -> List[List[str]]:
    """
    Create overlapping groups up to size 5 so every term co-occurs with others.
    Sliding window with step=(group_size-1).
    """
    uniq_terms = list(dict.fromkeys([t.strip() for t in terms if t.strip()]))
    n = len(uniq_terms)
    if n <= group_size:
        return [uniq_terms]

    step = max(1, group_size - 1)
    batches = []
    for i in range(0, n, step):
        chunk = uniq_terms[i:i + group_size]
        if chunk:
            batches.append(chunk)
        if i + group_size >= n:
            break

    # Ensure last terms have overlap if missed
    covered = set(itertools.chain.from_iterable(batches))
    missing = [t for t in uniq_terms if t not in covered]
    if missing:
        mid = uniq_terms[len(uniq_terms) // 2]
        extra = list(dict.fromkeys([mid] + missing))[:group_size]
        batches.append(extra)

    return batches


def robust_ratio(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    """
    Robust ratio median with MAD and count; ignores zeros/infs.
    """
    mask = (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return (np.nan, np.nan, 0)
    r = a[mask] / b[mask]
    r = r[np.isfinite(r)]
    if r.size == 0:
        return (np.nan, np.nan, 0)
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med))) if r.size > 1 else 0.0
    return (med, mad, int(r.size))


def pairwise_ratios(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    For each (batch_id, date), compute pairwise ratios between co-occurring terms.
    Aggregate across dates by median.
    """
    out = []
    for (bid, date), g in df_long.groupby(["batch_id", "date"], sort=False):
        # single-date wide
        w = g.pivot_table(index="date", columns="term", values="value", aggfunc="mean")
        if w.empty:
            continue
        terms = list(w.columns)
        vals = w.iloc[0].to_numpy(dtype=float)
        for i, j in itertools.combinations(range(len(terms)), 2):
            vi = np.array([vals[i]])
            vj = np.array([vals[j]])
            med_ij, mad_ij, n_ij = robust_ratio(vi, vj)
            med_ji, mad_ji, n_ji = robust_ratio(vj, vi)
            if n_ij > 0 and math.isfinite(med_ij):
                out.append({"term_i": terms[i], "term_j": terms[j], "ratio_med": med_ij, "ratio_mad": mad_ij, "n": n_ij})
            if n_ji > 0 and math.isfinite(med_ji):
                out.append({"term_i": terms[j], "term_j": terms[i], "ratio_med": med_ji, "ratio_mad": mad_ji, "n": n_ji})

    if not out:
        return pd.DataFrame(columns=["term_i", "term_j", "ratio_med", "ratio_mad", "n"])

    df = (
        pd.DataFrame(out)
        .groupby(["term_i", "term_j"], as_index=False)
        .agg({"ratio_med": "median", "ratio_mad": "median", "n": "sum"})
    )
    return df


def consensus_scale(pairwise: pd.DataFrame, terms: List[str]) -> pd.Series:
    """
    Solve for global log-scales x_i such that exp(x_i - x_j) ~ ratio_ij.
    Weighted least squares on edges with weight n / (1 + MAD).
    Returns positive scales normalized so max(scale)=1.0 (we’ll scale to 100 later).
    """
    terms = list(dict.fromkeys(terms))
    if pairwise.empty:
        return pd.Series({t: 1.0 for t in terms})

    idx = {t: k for k, t in enumerate(terms)}
    m = len(terms)
    rows = []
    weights = []
    rhs = []

    for _, row in pairwise.iterrows():
        ti, tj = row["term_i"], row["term_j"]
        if ti not in idx or tj not in idx:
            continue
        r = row["ratio_med"]
        if not (r > 0 and math.isfinite(r)):
            continue
        w = row["n"] / (1.0 + (row["ratio_mad"] if math.isfinite(row["ratio_mad"]) else 0.0))
        if w <= 0:
            continue
        e = np.zeros(m)
        e[idx[ti]] = 1.0
        e[idx[tj]] = -1.0
        rows.append(e)
        weights.append(w)
        rhs.append(math.log(r))

    if not rows:
        return pd.Series({t: 1.0 for t in terms})

    A = np.vstack(rows)
    W = np.diag(np.array(weights, dtype=float))
    y = np.array(rhs, dtype=float)

    ATA = A.T @ W @ A + 1e-6 * np.eye(m)
    ATy = A.T @ W @ y
    x = np.linalg.solve(ATA, ATy)

    # Center (gauge fix)
    x = x - np.mean(x)
    s = np.exp(x)
    s = s / np.max(s)
    return pd.Series({t: float(s[idx[t]]) for t in terms})


def score_pivots(df_long: pd.DataFrame, terms: List[str]) -> pd.DataFrame:
    """
    Score candidate pivots for transparency (not needed for scaling).
    Higher score is better (coverage, connectivity high; stability low).
    """
    recs = []
    for t in terms:
        g = df_long[df_long["term"] == t]
        vals = g["value"].to_numpy(dtype=float)
        total = vals.size
        nonzero = np.count_nonzero(vals > 0)
        coverage = nonzero / total if total else 0.0

        if total > 2:
            diffs = np.abs(np.diff(vals))
            med = np.median(vals[vals > 0]) if np.any(vals > 0) else np.nan
            stability = np.median(diffs) / med if med and med > 0 else np.inf
        else:
            stability = np.inf

        partners = df_long[df_long["term"] != t].groupby(["batch_id", "date"])["term"].nunique().sum()
        recs.append({"term": t, "coverage": coverage, "stability": stability, "connectivity": partners})

    df = pd.DataFrame(recs)
    df["score"] = (
        df["coverage"].rank(ascending=False)
        + df["connectivity"].rank(ascending=False)
        + df["stability"].replace([np.inf, np.nan], df["stability"].max() + 1).rank(ascending=True)
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def stitch_terms(
    serpapi_key: str,
    terms: List[str],
    geo: str = "",
    timeframe: str = "today 12-m",
    group_size: int = 5,
    cache_dir: str = ".cache",
    sleep_ms: int = 250,
    verbose: bool = True,
    use_cache: bool = True,
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Main pipeline:
      - batch fetch
      - compute pairwise ratios
      - solve consensus scaling
      - return wide timeseries (date x terms) on a comparable scale (global max=100),
        pivot_scores, and per-term scale factors.
    """
    log = logger.info if verbose else logger.debug
    log("[stitch_terms] Starting with %d terms", len(terms))

    fetcher = TrendsFetcher(
        serpapi_key,
        geo=geo,
        timeframe=timeframe,
        cache_dir=cache_dir,
        sleep_ms=sleep_ms,
        use_cache=use_cache,
        debug=debug,
    )
    batches = make_batches(terms, group_size=group_size)
    log("[stitch_terms] Created %d batches", len(batches))

    frames = []
    for i, batch in enumerate(batches, start=1):
        log("[stitch_terms] Fetching batch %d/%d: %s", i, len(batches), batch)
        df = fetcher.fetch_batch(batch)
        df["batch_id"] = f"batch_{i}"
        frames.append(df)

    df_long = pd.concat(frames, ignore_index=True)
    df_long["date"] = pd.to_datetime(df_long["date"]).dt.date
    log("[stitch_terms] Fetched %d rows", len(df_long))

    # Ratios & scales
    log("[stitch_terms] Computing pairwise ratios")
    pw = pairwise_ratios(df_long)
    if pw.empty:
        msg = "No overlapping data found; default scale of 1 used."
        try:  # pragma: no cover - streamlit optional
            import streamlit as st

            st.warning(msg)
        except Exception:
            logger.warning("[stitch_terms] %s", msg)
    else:
        log("[stitch_terms] %d pairwise ratios computed", len(pw))
    log("[stitch_terms] Computing consensus scale")
    scales = consensus_scale(pw, terms)  # max(scale)=1.0

    # Build wide (average across any duplicate points from overlapping batches)
    log("[stitch_terms] Building wide dataframe")
    all_dates = sorted(df_long["date"].unique())
    wide = (
        df_long.pivot_table(index="date", columns="term", values="value", aggfunc="mean")
        .reindex(index=all_dates, columns=terms)
        .astype(float)
    )

    # Apply per-term scale (no per-term normalization here!)
    for t in terms:
        wide[t] = wide[t] * scales.get(t, 1.0)

    # Normalize so global max across all terms/time = 100
    globmax = np.nanmax(wide.to_numpy(dtype=float))
    if globmax and globmax > 0:
        wide = wide * (100.0 / globmax)

    log("[stitch_terms] Computing pivot scores")
    pivot_scores = score_pivots(
        df_long.assign(date=pd.to_datetime(df_long["date"]))
    , terms)

    log("[stitch_terms] Done")
    return wide.reset_index().rename(columns={"index": "date"}), pivot_scores, scales
