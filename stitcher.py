        if len(timeline) > 0:
            logger.debug(f"First timeline point: {timeline[0]}")

        rows = []
        for i, point in enumerate(timeline):
            # timestamp handling - try multiple formats
            ts = None
            for ts_key in ['time', 'timestamp', 'date', 'formattedTime', 'formatted_time']:
                ts = point.get(ts_key)
                if ts is not None:
                    break
            
            if ts is None:
                logger.warning(f"No timestamp found in point {i}: {point}")
                continue
                
            date = TrendsFetcher._coerce_date(ts)

            # Try multiple value extraction methods
            vals = None
            
            # Method 1: values or value field
            vals = point.get("values") or point.get("value")
            
            # Method 2: Check if point itself contains term-value pairs
            if vals is None:
                # Look for term names in the point keys
                term_values = {}
                for key, value in point.items():
                    if key not in ['time', 'timestamp', 'date', 'formattedTime', 'formatted_time']:
                        try:
                            term_values[key] = float(value)
                        except (ValueError, TypeError):
                            continue
                if term_values:
                    for term, value in term_values.items():
                        rows.append({"date": date, "term": str(term), "value": value})
                    continue
            
            # Method 3: Check for interest_over_time structure within point
            if vals is None:
                io_point = point.get("interest_over_time", {})
                if isinstance(io_point, dict):
                    for term, value in io_point.items():
                        if term not in ['time', 'timestamp', 'date', 'formattedTime', 'formatted_time']:
                            try:
                                rows.append({"date": date, "term": str(term), "value": float(value)})
                            except (ValueError, TypeError):
                                continue
                    continue

            # Process vals if found
            if isinstance(vals, list) and len(vals) > 0:
                if isinstance(vals[0], dict) and "value" in vals[0]:
                    # List of dicts with query/value
                    for v in vals:
                        q = v.get("query") or v.get("term")
                        val = v.get("value")
                        if q is None or val is None:
                            continue
                        try:
                            rows.append({"date": date, "term": str(q), "value": float(val)})
                        except (ValueError, TypeError):
                            continue
                else:
                    # List of values aligned to input terms
                    for t, v in itertools.zip_longest(terms, vals):
                        if t is None or v is None:
                            continue
                        try:
                            rows.append({"date": date, "term": str(t), "value": float(v)})
                        except (ValueError, TypeError):
                            continue
            elif isinstance(vals, dict):
                # Direct dict mapping terms to values
                for term, value in vals.items():
                    if term not in ['time', 'timestamp', 'date', 'formattedTime', 'formatted_time']:
                        try:
                            rows.append({"date": date, "term": str(term), "value": float(value)})
                        except (ValueError, TypeError):
                            continue

        df = pd.DataFrame(rows)
        logger.debug(f"Parsed {len(df)} rows from timeline")
        if not df.empty:
            logger.debug(f"Sample parsed data:\n{df.head()}")
            logger.debug(f"Terms found: {df['term'].unique()}")
            logger.debug(f"Value range: {df['value'].min()} to {df['value'].max()}")
        
        if df.empty:
            logger.error("No valid data parsed from timeline")
            logger.error(f"Timeline sample: {timeline[:2] if timeline else 'Empty timeline'}")
            raise RuntimeError("Empty dataframe from Trends batch parse.")
        return df

    @staticmethod
    def _coerce_date(ts) -> dt.date:
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        
        # Handle string representation of datetime.date objects
        if isinstance(ts, str):
            # Check for "datetime.date(year, month, day)" format
            if ts.startswith("datetime.date(") and ts.endswith(")"):
                try:
                    # Extract the date components from "datetime.date(2025, 8, 24)"
                    date_str = ts[13:-1]  # Remove "datetime.date(" and ")"
                    # Parse the components
                    import ast
                    date_tuple = ast.literal_eval(date_str)
                    if len(date_tuple) == 3:
                        year, month, day = date_tuple
                        result = dt.date(year, month, day)
                        logger.debug(f"Successfully parsed datetime.date string: {ts} -> {result}")
                        return result
                except Exception as e:
                    logger.debug(f"Failed to parse datetime.date string {ts}: {e}")
                    pass
            
            # Try to parse as ISO date string
            try:
                result = pd.to_datetime(ts).date()
                logger.debug(f"Successfully parsed ISO date string: {ts} -> {result}")
                return result
            except Exception as e:
                logger.debug(f"Failed to parse ISO date string {ts}: {e}")
                pass
        
        # Try Unix seconds (handle both int/float and string representations)
        try:
            # Convert to float if it's a string
            if isinstance(ts, str):
                ts_float = float(ts)
            else:
                ts_float = float(ts)
            
            # Check if it's a reasonable Unix timestamp (after 2000, before 2100)
            if 946684800 <= ts_float <= 4102444800:  # 2000-01-01 to 2100-01-01
                result = dt.datetime.utcfromtimestamp(int(ts_float)).date()
                logger.debug(f"Successfully parsed Unix timestamp: {ts} -> {result}")
                return result
        except Exception as e:
            logger.debug(f"Failed to parse Unix timestamp {ts}: {e}")
            pass
        
        # Try pandas datetime parsing
        try:
            result = pd.to_datetime(ts).date()
            logger.debug(f"Successfully parsed with pandas: {ts} -> {result}")
            return result
        except Exception as e:
            logger.debug(f"Failed to parse with pandas {ts}: {e}")
            pass
            
        # last resort: today - only warn if debug is enabled
        logger.debug(f"Could not parse date {ts}, using today's date")
        return dt.date.today()

    def _map_timeframe_to_dfs_range(self, timeframe: str) -> str:
        t = (timeframe or "").strip().lower()
        mapping = {
            "today 1-m": "past_30_days",
            "today 3-m": "past_90_days",
            "today 12-m": "past_12_months",
            "today 5-y": "past_5_years",
            "today 10-y": "past_10_years",
            "today 15-y": "past_15_years",
            "today 20-y": "past_20_years",
            "all": "all",
        }
        return mapping.get(t, "all")

    def _normalize_dataforseo_payload(self, payload: Dict) -> Dict:
        """
        Normalize DataForSEO explore/live response to a structure with interest_over_time
        and/or timeline_data so that _parse_timeseries can consume it.
        """
        if not isinstance(payload, dict):
            return payload
        
        # Debug logging
        self._log_debug(f"DataForSEO payload keys: {list(payload.keys())}")
        
        # Handle the new DataForSEO response format with 'items' array
        if "items" in payload and isinstance(payload["items"], list) and payload["items"]:
            self._log_debug(f"Found items array with {len(payload['items'])} items")
            items = payload["items"]
            for i, item in enumerate(items):
                self._log_debug(f"Item {i}: {list(item.keys()) if isinstance(item, dict) else type(item)}")
                if isinstance(item, dict) and "data" in item:
                    data = item["data"]
                    self._log_debug(f"Found data array with {len(data) if isinstance(data, list) else 'not a list'} items")
                    if isinstance(data, list) and data:
                        # Convert DataForSEO format to timeline_data format
                        timeline_data = []
                        for point in data:
                            if isinstance(point, dict):
                                # Convert DataForSEO format to expected format
                                # Log all available keys in the data point for debugging
                                self._log_debug(f"Data point keys: {list(point.keys())}")
                                
                                # Try different possible field names for the value
                                value = (point.get("value") or 
                                        point.get("values") or 
                                        point.get("interest_value") or 
                                        point.get("interest") or 
                                        point.get("score") or 
                                        0)
                                self._log_debug(f"Found value: {value}")
                                if not isinstance(value, list):
                                    value = [value]  # Convert single value to array
                                
                                timeline_point = {
                                    "time": point.get("date_from", ""),
                                    "timestamp": point.get("timestamp", 0),
                                    "value": value
                                }
                                timeline_data.append(timeline_point)
                        self._log_debug(f"Created timeline_data with {len(timeline_data)} points")
                        return {"timeline_data": timeline_data}
        
        # Handle the old DataForSEO response format with 'tasks' array
        tasks = payload.get("tasks")
        if isinstance(tasks, list) and tasks:
            task0 = tasks[0] or {}
            result = task0.get("result")
            if isinstance(result, list) and result:
                core = result[0] or {}
                if isinstance(core, dict):
                    if "interest_over_time" in core or "timeline_data" in core:
                        return core
                    iot = core.get("interest_over_time")
                    if isinstance(iot, dict) and ("timeline_data" in iot or isinstance(iot, list)):
                        return {"interest_over_time": iot}
                    # Data may store series as 'timeline' or similar
                    for k, v in core.items():
                        if isinstance(v, list) and v and isinstance(v[0], dict) and any(
                            kk in v[0] for kk in ["time", "timestamp", "date", "formattedTime", "formatted_time"]
                        ):
                            return {"timeline_data": v}
                return core
        return payload


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


def pairwise_ratio_samples(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Return ratios for each pair aggregated by batch (not by date).
    One row per batch observation: term_i, term_j, ratio, batch_id
    Only measures batch-to-batch instability, not temporal volatility.
    """
    rows = []
    
    # First, aggregate data by batch (average across all dates in each batch)
    batch_aggregated = df_long.groupby(["batch_id", "term"], as_index=False)["value"].mean()
    
    # Then calculate ratios between terms within each batch
    for bid in batch_aggregated["batch_id"].unique():
        batch_data = batch_aggregated[batch_aggregated["batch_id"] == bid]
        if len(batch_data) < 2:
            continue
            
        # Create a simple lookup for this batch
        term_values = dict(zip(batch_data["term"], batch_data["value"]))
        terms = list(term_values.keys())
        
        for i, j in itertools.combinations(range(len(terms)), 2):
            ti, tj = terms[i], terms[j]
            vi, vj = term_values[ti], term_values[tj]
            if vi > 0 and vj > 0 and np.isfinite(vi) and np.isfinite(vj):
                rows.append({"term_i": ti, "term_j": tj, "ratio": vi / vj, "batch_id": bid})
                rows.append({"term_i": tj, "term_j": ti, "ratio": vj / vi, "batch_id": bid})
    
    return pd.DataFrame(rows)


def instability_metrics(samples: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarise batch-to-batch instability per pair and per term.
    Pairwise metrics: median, MAD, IQR, min, max, count, CV on log-ratio.
    Per-term score: average of pairwise CVs involving the term (lower=more stable).
    Only measures variation across different batches, not temporal variation.
    """
    if samples.empty:
        return (pd.DataFrame(columns=[
            "term_i", "term_j", "median", "mad", "iqr", "min", "max", "count", "cv_log"
        ]),
        pd.DataFrame(columns=["term", "instability_score", "pairs_count"]))

    def _iqr(x): 
        q = np.quantile(x, [0.25, 0.75])
        return float(q[1] - q[0])

    def _cv_log(x):
        lx = np.log(x[(x > 0) & np.isfinite(x)])
        if lx.size < 2:
            return np.nan
        return float(np.std(lx, ddof=1) / max(1e-12, abs(np.mean(lx))))

    pair = (samples
            .groupby(["term_i","term_j"], as_index=False)
            .agg(median=("ratio","median"),
                 mad=("ratio", lambda v: float(np.median(np.abs(v - np.median(v))) if len(v)>1 else 0.0)),
                 iqr=("ratio", _iqr),
                 min=("ratio","min"),
                 max=("ratio","max"),
                 count=("ratio","size"),
                 cv_log=("ratio", _cv_log))
            )

    # per-term instability score = mean cv_log across pairs (ignore NaNs)
    per_term = (pair
                .groupby("term_i", as_index=False)
                .agg(instability_score=("cv_log", lambda v: float(np.nanmean(v))),
                     pairs_count=("cv_log","size"))
                .rename(columns={"term_i":"term"}))

    return pair, per_term


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
    provider: str = "serpapi",
    geo: str = "",
    timeframe: str = "today 12-m",
    group_size: int = 5,
    cache_dir: str = ".cache",
    sleep_ms: int = 250,
    verbose: bool = True,
    use_cache: bool = True,
    debug: bool = False,
    collect_raw_responses: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    Main pipeline:
      - batch fetch
      - compute pairwise ratios
      - solve consensus scaling
      - return wide timeseries (date x terms) on a comparable scale (global max=100),
        pivot_scores, per-term scale factors, pairwise metrics, term instability scores,
        and ratio samples for diagnostics.
    """
    log = logger.info if verbose else logger.debug
    print(f"DEBUG: stitch_terms called with provider: {provider}")
    log("[stitch_terms] Starting with %d terms", len(terms))

    fetcher = TrendsFetcher(
        serpapi_key,
        provider=provider,
        geo=geo,
        timeframe=timeframe,
        cache_dir=cache_dir,
        sleep_ms=sleep_ms,
        use_cache=use_cache,
        debug=debug,
        collect_raw_responses=collect_raw_responses,
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
    
    # Debug: Check the data before date conversion
    if debug:
        log("[stitch_terms] DEBUG: Data before date conversion")
        log(f"[stitch_terms] DEBUG: df_long shape: {df_long.shape}")
        log(f"[stitch_terms] DEBUG: df_long columns: {list(df_long.columns)}")
        log(f"[stitch_terms] DEBUG: Sample data before date conversion:")
        log(f"[stitch_terms] DEBUG: {df_long.head()}")
        log(f"[stitch_terms] DEBUG: Date column type: {df_long['date'].dtype}")
        log(f"[stitch_terms] DEBUG: Value column type: {df_long['value'].dtype}")
        log(f"[stitch_terms] DEBUG: Value range: {df_long['value'].min()} to {df_long['value'].max()}")
        log(f"[stitch_terms] DEBUG: NaN count in values: {df_long['value'].isna().sum()}")
    
    df_long["date"] = pd.to_datetime(df_long["date"]).dt.date
    
    # Debug: Check the data after date conversion
    if debug:
        log("[stitch_terms] DEBUG: Data after date conversion")
        log(f"[stitch_terms] DEBUG: df_long shape: {df_long.shape}")
        log(f"[stitch_terms] DEBUG: Date column type: {df_long['date'].dtype}")
        log(f"[stitch_terms] DEBUG: Value range: {df_long['value'].min()} to {df_long['value'].max()}")
        log(f"[stitch_terms] DEBUG: NaN count in values: {df_long['value'].isna().sum()}")
        log(f"[stitch_terms] DEBUG: Sample data after date conversion:")
        log(f"[stitch_terms] DEBUG: {df_long.head()}")
    
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
    
    if debug:
        log(f"[stitch_terms] DEBUG: All dates: {all_dates[:10]}... (total: {len(all_dates)})")
        log(f"[stitch_terms] DEBUG: Terms: {terms}")
        log(f"[stitch_terms] DEBUG: df_long term counts: {df_long['term'].value_counts().to_dict()}")
    
    wide = (
        df_long.pivot_table(index="date", columns="term", values="value", aggfunc="mean")
        .reindex(index=all_dates, columns=terms)
        .astype(float)
    )
    
    if debug:
        log(f"[stitch_terms] DEBUG: Wide dataframe shape: {wide.shape}")
        log(f"[stitch_terms] DEBUG: Wide dataframe columns: {list(wide.columns)}")
        log(f"[stitch_terms] DEBUG: Wide dataframe sample:")
        log(f"[stitch_terms] DEBUG: {wide.head()}")
        log(f"[stitch_terms] DEBUG: NaN count in wide: {wide.isna().sum().sum()}")
        log(f"[stitch_terms] DEBUG: Value range in wide: {wide.min().min()} to {wide.max().max()}")

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

    # Collect samples + metrics for diagnostics
    samples = pairwise_ratio_samples(df_long)
    pair_metrics, term_instability = instability_metrics(samples)

    log("[stitch_terms] Done")
    # Return two extra frames for diagnostics plus raw responses
    return (wide.reset_index().rename(columns={"index": "date"}),
            pivot_scores,
            scales,
            pair_metrics,
            term_instability,
            samples,
            fetcher.raw_responses)
