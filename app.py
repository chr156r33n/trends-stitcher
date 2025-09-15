from datetime import date
import traceback
from typing import Tuple
import logging
import io
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Ensure the local stitcher module is imported even if a similarly named
# package is installed in the environment.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from stitcher import stitch_terms

# Custom logging handler for Streamlit
class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_buffer = io.StringIO()
    
    def emit(self, record):
        log_entry = self.format(record)
        self.log_buffer.write(log_entry + '\n')
        # Store in session state for display
        if 'debug_logs' not in st.session_state:
            st.session_state.debug_logs = []
        st.session_state.debug_logs.append(log_entry)
        # Keep only last 50 log entries
        if len(st.session_state.debug_logs) > 50:
            st.session_state.debug_logs = st.session_state.debug_logs[-50:]

# Set up custom logging
def setup_debug_logging():
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Add our custom handler
    handler = StreamlitLogHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add to root logger
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Also add to stitcher logger
    stitcher_logger = logging.getLogger('stitcher')
    stitcher_logger.addHandler(handler)
    stitcher_logger.setLevel(logging.DEBUG)

# Function to display debug logs in sidebar
def show_debug_logs():
    if 'debug_logs' in st.session_state and st.session_state.debug_logs:
        with st.expander("🐛 Debug Logs", expanded=False):
            for log_entry in st.session_state.debug_logs[-20:]:  # Show last 20 entries
                st.text(log_entry)

def explore_autocomplete_options(terms: list, api_key: str):
    """Explore autocomplete options for each term to find better entity-based searches"""
    import requests
    
    st.subheader("Autocomplete Suggestions")
    st.caption("Below are the autocomplete suggestions for each of your terms. Entity-based searches (marked as 'Topic' or specific entity types) often provide better trend data than simple keyword searches.")
    
    all_suggestions = {}
    
    for term in terms:
        st.write(f"**{term}**")
        
        try:
            params = {
                "api_key": api_key,
                "engine": "google_trends_autocomplete",
                "q": term
            }
            
            response = requests.get("https://serpapi.com/search", params=params, timeout=30)
            if response.status_code == 200:
                results = response.json()
                
                if "suggestions" in results and results["suggestions"]:
                    suggestions = results["suggestions"]
                    all_suggestions[term] = suggestions
                    
                    # Create a DataFrame for better display
                    suggestion_data = []
                    for i, suggestion in enumerate(suggestions):
                        suggestion_data.append({
                            "Option": i + 1,
                            "Query": suggestion.get("q", ""),
                            "Title": suggestion.get("title", ""),
                            "Type": suggestion.get("type", ""),
                            "Link": suggestion.get("link", "")
                        })
                    
                    df = pd.DataFrame(suggestion_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Add download button for this term's suggestions
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        f"Download {term} suggestions CSV",
                        csv_data.encode("utf-8"),
                        file_name=f"autocomplete_{term.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key=f"autocomplete_{term.replace(' ', '_')}_download"
                    )
                    
                else:
                    st.info(f"No autocomplete suggestions found for '{term}'")
            else:
                st.error(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            st.error(f"Error fetching autocomplete for '{term}': {str(e)}")
        
        st.write("---")
    
    # Create combined download for all suggestions
    if all_suggestions:
        st.subheader("Download All Suggestions")
        combined_data = []
        for term, suggestions in all_suggestions.items():
            for suggestion in suggestions:
                combined_data.append({
                    "Original Term": term,
                    "Query": suggestion.get("q", ""),
                    "Title": suggestion.get("title", ""),
                    "Type": suggestion.get("type", ""),
                    "Link": suggestion.get("link", "")
                })
        
        combined_df = pd.DataFrame(combined_data)
        st.download_button(
            "Download All Autocomplete Suggestions CSV",
            combined_df.to_csv(index=False).encode("utf-8"),
            file_name="all_autocomplete_suggestions.csv",
            mime="text/csv",
            key="all_autocomplete_download"
        )
        
        # Show summary statistics
        st.subheader("Summary")
        type_counts = combined_df["Type"].value_counts()
        st.write("**Suggestion types found:**")
        for suggestion_type, count in type_counts.items():
            st.write(f"- {suggestion_type}: {count} suggestions")

# If this script is executed with `python app.py` instead of
# `streamlit run app.py`, re-launch it via the Streamlit CLI so the
# interactive app works as expected.
try:  # pragma: no cover - best effort safeguard
    from streamlit.runtime.scriptrunner_utils import script_run_context
    if script_run_context.get_script_run_ctx() is None:  # not running via `streamlit run`
        import sys
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
except Exception:
    pass

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_scaled' not in st.session_state:
    st.session_state.df_scaled = None
if 'pivot_scores' not in st.session_state:
    st.session_state.pivot_scores = None
if 'scales' not in st.session_state:
    st.session_state.scales = None
if 'terms' not in st.session_state:
    st.session_state.terms = None
if 'current_params' not in st.session_state:
    st.session_state.current_params = None

if 'selected_chart_terms' not in st.session_state:
    st.session_state.selected_chart_terms = None
if 'show_small_multiples' not in st.session_state:
    st.session_state.show_small_multiples = False
if 'show_debug_chart' not in st.session_state:
    st.session_state.show_debug_chart = False
if 'start_date' not in st.session_state:
    st.session_state.start_date = None
if 'end_date' not in st.session_state:
    st.session_state.end_date = None
if 'timeframe' not in st.session_state:
    st.session_state.timeframe = "all"

# Initialize stability diagnostics session state
if 'pair_metrics' not in st.session_state:
    st.session_state.pair_metrics = None
if 'term_instability' not in st.session_state:
    st.session_state.term_instability = None
if 'ratio_samples' not in st.session_state:
    st.session_state.ratio_samples = None
if 'raw_responses' not in st.session_state:
    st.session_state.raw_responses = None

# Initialize toggle states
if 'show_debug' not in st.session_state:
    st.session_state.show_debug = False
if 'verbose_logs' not in st.session_state:
    st.session_state.verbose_logs = False
if 'show_stability' not in st.session_state:
    st.session_state.show_stability = False
if 'show_ribbons' not in st.session_state:
    st.session_state.show_ribbons = False
if 'show_all_data' not in st.session_state:
    st.session_state.show_all_data = False
if 'use_cache' not in st.session_state:
    st.session_state.use_cache = True
if 'collect_raw_responses' not in st.session_state:
    st.session_state.collect_raw_responses = True

st.set_page_config(page_title="Google Trend Stitcher", layout="wide")
st.title("Google Trend Stitcher")

# Initialize logging to Streamlit UI
setup_debug_logging()

# Show cache status
if st.session_state.data_loaded:
    st.success(f"Data loaded: {len(st.session_state.terms) if st.session_state.terms else 0} terms, {st.session_state.df_scaled.shape[0] if st.session_state.df_scaled is not None else 0} data points")
    if 'pair_metrics' in st.session_state and st.session_state.pair_metrics is not None and not st.session_state.pair_metrics.empty:
        st.info("📊 **Stability diagnostics available!** Expand the 'Stability Diagnostics' section below to analyze data consistency and variability.")
else:
    st.info("No data loaded. Enter parameters and click 'Run' to fetch data.")

def get_suggested_timeframe(start_date, end_date):
    """Suggest the best timeframe based on date filters"""
    if not start_date and not end_date:
        return "all"  # Default to maximum data range
    
    # Calculate the date range
    if start_date and end_date:
        date_range = (end_date - start_date).days
    elif start_date:
        date_range = (date.today() - start_date).days
    elif end_date:
        date_range = (end_date - date.today()).days
    else:
        return "all"
    
    # Suggest timeframe based on range
    if date_range <= 30:
        return "today 1-m"
    elif date_range <= 90:
        return "today 3-m"
    elif date_range <= 365:
        return "today 12-m"
    elif date_range <= 1825:  # 5 years
        return "today 5-y"
    elif date_range <= 3650:  # 10 years
        return "today 10-y"
    else:
        return "all"  # Use 'all' for very long ranges

def get_current_params(serpapi_key, terms_text, geo, timeframe, group_size, cache_dir, sleep_ms, use_cache, provider):
    """Get current parameters as a hashable tuple for comparison"""
    terms = tuple(sorted([t.strip() for t in terms_text.splitlines() if t.strip()]))
    return (provider, serpapi_key, terms, geo, timeframe, group_size, cache_dir, sleep_ms, use_cache)

def should_reload_data(current_params):
    """Check if we need to reload data based on parameter changes"""
    if not st.session_state.data_loaded:
        return True
    if st.session_state.current_params != current_params:
        return True
    return False

# Sidebar
with st.sidebar:
    st.subheader("Data")
    provider_label_map = {
        "serpapi": "SerpAPI API Key",
        "dataforseo": "DataForSEO API Key (Basic token or login:password)",
        "brightdata": "Bright Data API Token",
    }
    provider_human = st.selectbox("Provider", ["SerpAPI", "DataForSEO", "Bright Data"], index=0)
    provider = provider_human.lower().replace(" ", "")
    serpapi_key = st.text_input(provider_label_map.get(provider, "API Key"), type="password", value="")
    
    # Add Bright Data zone field when Bright Data is selected
    brightdata_zone = ""
    if provider == "brightdata":
        brightdata_zone = st.text_input("Bright Data Zone", value="", help="Enter your Bright Data SERP API zone name (e.g., 'serp_zone_1')")
    
    terms_text = st.text_area("Terms (one per line)", "nike\nadidas\npuma\nnew balance\nasics\nOn Running\nSolomon")
    geo = st.text_input("Geo (e.g. GB, US — optional)", value="")
    
    # Use selectbox for timeframe with better options
    timeframe_options = [
        "all",           # Maximum data range (recommended)
        "today 1-m",     # 1 month
        "today 3-m",     # 3 months  
        "today 12-m",    # 12 months
        "today 5-y",     # 5 years
        "today 10-y",    # 10 years
        "today 15-y",    # 15 years
        "today 20-y",    # 20 years
    ]
    
    # Auto-select longer timeframe if date filters are set
    if st.session_state.start_date or st.session_state.end_date:
        # If user has set date filters, suggest a longer timeframe
        suggested_timeframe = get_suggested_timeframe(st.session_state.start_date, st.session_state.end_date)
        st.info(f"Date filters detected. Using longer timeframe ({suggested_timeframe}) to get more data for filtering.")
        current_timeframe = st.session_state.timeframe if st.session_state.timeframe in timeframe_options else suggested_timeframe
        timeframe = st.selectbox("API Timeframe", timeframe_options, index=timeframe_options.index(current_timeframe))
    else:
        current_timeframe = st.session_state.timeframe if st.session_state.timeframe in timeframe_options else "today 5-y"
        timeframe = st.selectbox("API Timeframe", timeframe_options, index=timeframe_options.index(current_timeframe))
    
    # Update session state
    st.session_state.timeframe = timeframe
    
    group_size = st.slider("Batch size (max 5)", 2, 5, 3)
    st.caption("3: Most robust stitching (more overlap, slower) | 5: Most efficient (fewer API calls, less overlap)")
    
    # Move buttons above autocomplete explorer
    run = st.button("Run")
    
    # Add cache management buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Cache"):
            st.session_state.data_loaded = False
            st.session_state.df_scaled = None
            st.session_state.pivot_scores = None
            st.session_state.scales = None
            st.session_state.terms = None
            st.session_state.current_params = None

            st.session_state.selected_chart_terms = None
            st.session_state.show_small_multiples = False
            st.session_state.show_debug_chart = False
            st.session_state.start_date = None
            st.session_state.end_date = None
            st.session_state.timeframe = "all"
            
            # Clear stability diagnostics
            st.session_state.pair_metrics = None
            st.session_state.term_instability = None
            st.session_state.ratio_samples = None
            st.session_state.raw_responses = None
            
            # Reset toggle states
            st.session_state.show_debug = False
            st.session_state.verbose_logs = False
            st.session_state.show_stability = False
            st.session_state.show_ribbons = False
            st.session_state.show_all_data = False
            st.session_state.use_cache = True
            st.session_state.collect_raw_responses = True
            
            st.success("Cache cleared! Click 'Run' to reload data.")
            st.rerun()
    
    with col2:
        if st.button("Force Reload"):
            st.session_state.data_loaded = False
            st.success("Forcing reload on next run...")

    st.markdown("---")
    st.subheader("Autocomplete Explorer")
    st.caption("Discover better search terms and entities")
    
    if provider != "serpapi":
        st.button("Explore Autocomplete Options", key="sidebar_autocomplete", disabled=True)
        st.info("Autocomplete is only available with SerpAPI. Select SerpAPI to enable.")
    else:
        if st.button("Explore Autocomplete Options", key="sidebar_autocomplete"):
            if serpapi_key:
                with st.spinner("Fetching autocomplete suggestions..."):
                    # Get terms from session state if available
                    if hasattr(st.session_state, 'terms') and st.session_state.terms:
                        explore_autocomplete_options(st.session_state.terms, serpapi_key)
                    else:
                        st.error("No terms available. Please run the analysis first.")
            else:
                st.error("Please enter your SerpAPI key first.")
    

    st.markdown("---")
    st.subheader("Date Range")
    start_date = st.date_input("Start date (optional)", value=st.session_state.start_date)
    st.session_state.start_date = start_date
    end_date = st.date_input("End date (optional)", value=st.session_state.end_date)
    st.session_state.end_date = end_date
    
    # Add warning about timeframe vs date filtering
    if start_date or end_date:
        st.info("**Note**: Date filtering works on the data returned by the API. To get more data for filtering, consider using a longer timeframe (e.g., 'today 5-y' instead of 'today 12-m').")
    
    # Add button to set maximum timeframe for date filtering
    if start_date or end_date:
        if st.button("Set Maximum Timeframe for Date Filtering"):
            st.session_state.timeframe = "all"
            st.success("Set timeframe to maximum ('all') for better date filtering!")
            st.rerun()
    
    st.markdown("---")
    with st.expander("Advanced Options", expanded=False):
        st.subheader("Advanced")
        # Use temp directory for cloud environments
        import tempfile
        import os
        default_cache = tempfile.gettempdir() if os.environ.get('STREAMLIT_SERVER_RUN_ON_SAVE') else ".cache"
        cache_dir = st.text_input("Cache directory", value=default_cache)
        sleep_ms = st.number_input("Request sleep (ms)", min_value=0, value=250)
        use_cache = st.checkbox("Use cache", value=st.session_state.get('use_cache', True), key='use_cache_checkbox')
        show_debug = st.checkbox("Show debug logs", value=st.session_state.get('show_debug', False), key='show_debug_checkbox')
        verbose_logs = st.checkbox("Verbose logging", value=st.session_state.get('verbose_logs', False), key='verbose_logs_checkbox')
        show_stability = st.checkbox("Show stability diagnostics (overlap variance)", value=st.session_state.get('show_stability', False), key='show_stability_checkbox')
        show_ribbons = st.checkbox("Show fan-out (instability) bands", value=st.session_state.get('show_ribbons', False), key='show_ribbons_checkbox')
        collect_raw_responses = st.checkbox("Collect raw API responses", value=st.session_state.get('collect_raw_responses', True), key='collect_raw_responses_checkbox')
        if show_ribbons:
            st.caption("💡 Shows uncertainty bands on the main chart based on pairwise ratio variability")

    # Add link to blog post at bottom of sidebar
    st.markdown("---")
    st.markdown("[Learn more about Trend Stitching](https://www.chris-green.net/post/trends-stitcher)")

def filter_date_range(df_scaled: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    df = df_scaled.copy()
    df["date"] = pd.to_datetime(df["date"])
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    return df

@st.cache_data
def cached_filter_date_range(df_scaled: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Cached version of date range filtering"""
    return filter_date_range(df_scaled, start_date, end_date)

def melt_long(df_scaled: pd.DataFrame) -> pd.DataFrame:
    long = df_scaled.melt(id_vars=["date"], var_name="term", value_name="value")
    long["date"] = pd.to_datetime(long["date"])
    return long

@st.cache_data
def cached_melt_long(df_scaled: pd.DataFrame) -> pd.DataFrame:
    """Cached version of melt operation"""
    return melt_long(df_scaled)

def infer_cadence(dates):
    """Detect data cadence from date series"""
    s = pd.to_datetime(dates).sort_values().drop_duplicates()
    if len(s) < 3:
        return "unknown"
    med_gap = s.diff().dropna().dt.days.median()
    if med_gap <= 2: return "daily"
    if 3 <= med_gap <= 10: return "weekly"
    if 25 <= med_gap <= 35: return "monthly"
    return "unknown"

def yoy_table(long_df: pd.DataFrame, term: str) -> pd.DataFrame:
    """
    Calculate year-over-year comparison for a given term.
    Uses 365-day lag with nearest merge for weekly data, tolerance-based matching for others.
    """
    # Filter data for the specific term and sort by date
    g = long_df[long_df["term"] == term].sort_values("date").copy()
    
    if g.empty:
        return pd.DataFrame(columns=["date", "current", "previous_year", "abs_diff", "pct_diff"])
    
    # Convert dates to datetime if they aren't already
    g["date"] = pd.to_datetime(g["date"])
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"YoY calculation for term: {term}")
    logger.debug(f"Data range: {g['date'].min()} to {g['date'].max()}")
    logger.debug(f"Total data points: {len(g)}")
    
    # Detect data cadence
    cadence = infer_cadence(g["date"])
    logger.debug(f"Detected data cadence: {cadence}")
    
    if cadence == "weekly":
        # Use 365-day lag with nearest merge for weekly data
        logger.debug("Using 365-day lag approach for weekly data")

        # Create prior year data by shifting forward 365 days (1 year)
        prior_data = g.copy()
        prior_data["date"] = prior_data["date"] + pd.Timedelta(days=365)
        prior_data = prior_data.rename(columns={"value": "prior_value"})
        prior_data = prior_data[["date", "prior_value"]]

        # Merge using merge_asof with tolerance
        g = g.sort_values("date")
        prior_data = prior_data.sort_values("date")

        result = pd.merge_asof(
            g,
            prior_data,
            on="date",
            direction="nearest",
            tolerance=pd.Timedelta(days=4)
        )

        logger.debug(f"365-day lag merge: {len(result)} rows, {result['prior_value'].notna().sum()} matches")

    elif cadence == "monthly":
        # Month/Year merge for monthly data
        logger.debug("Using year/month merge for monthly data")
        g["year"] = g["date"].dt.year
        g["month"] = g["date"].dt.month

        prior = g[["year", "month", "value"]].copy()
        prior["year"] = prior["year"] + 1
        prior = prior.rename(columns={"value": "prior_value"})

        result = pd.merge(g, prior, on=["year", "month"], how="left")

    else:
        # Use tolerance-based matching for daily/irregular data
        tolerance_days = 3 if cadence == "daily" else 7
        logger.debug(f"Using tolerance-based matching with {tolerance_days} day tolerance")

        g["prev_year_date"] = g["date"] - pd.DateOffset(years=1)

        prior_values = []
        for _, row in g.iterrows():
            target_date = row["prev_year_date"]
            current_date = row["date"]

            prev_year_data = g[g["date"].dt.year == target_date.year]
            if not prev_year_data.empty:
                date_diffs = abs((prev_year_data["date"] - target_date).dt.days)
                within_tolerance = date_diffs <= tolerance_days

                if within_tolerance.any():
                    closest_idx = date_diffs[within_tolerance].idxmin()
                    prior_value = prev_year_data.loc[closest_idx, "value"]
                    logger.debug(
                        f"  {current_date} -> {target_date} (tolerance: ±{tolerance_days}d) -> {prev_year_data.loc[closest_idx, 'date']} = {prior_value}"
                    )
                else:
                    prior_value = np.nan
                    logger.debug(
                        f"  {current_date} -> {target_date} (tolerance: ±{tolerance_days}d) -> No match in previous year"
                    )
            else:
                prior_value = np.nan
                logger.debug(
                    f"  {current_date} -> {target_date} (tolerance: ±{tolerance_days}d) -> No previous year data"
                )

            prior_values.append(prior_value)

        result = g.copy()
        result["prior_value"] = prior_values
    
    # Calculate differences
    result["abs_diff"] = result["value"] - result["prior_value"]
    result["pct_diff"] = np.where(
        result["prior_value"] > 0,
        (result["abs_diff"] / result["prior_value"]) * 100.0,
        np.nan
    )
    
    # Debug: Show summary statistics
    valid_prev_year = result["prior_value"].notna().sum()
    logger.debug(f"Valid previous year matches: {valid_prev_year} out of {len(result)}")
    
    # Return the result with proper column names
    final_result = result[["date", "value", "prior_value", "abs_diff", "pct_diff"]].rename(
        columns={"value": "current", "prior_value": "previous_year"}
    )
    
    return final_result

def line_chart_multi(long_df: pd.DataFrame, selected_terms: list, title: str):
    data = long_df[long_df["term"].isin(selected_terms)]
    if data.empty:
        st.info("No data to plot for the selected terms.")
        return
    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Comparable Index (max=100 overall)"),
            color=alt.Color("term:N", legend=alt.Legend(title="Term")),
            tooltip=["date:T", "term:N", alt.Tooltip("value:Q", format=".2f")]
        )
        .properties(title=title, height=380)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

@st.cache_data
def create_line_chart(data, selected_terms, title):
    """Cached version of line chart creation with interactive filtering"""
    if data.empty:
        return None
    
    # Create a selection for interactive filtering
    selection = alt.selection_multi(fields=['term'], bind='legend')
    
    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Comparable Index (max=100 overall)"),
            color=alt.Color("term:N", legend=alt.Legend(title="Term")),
            tooltip=["date:T", "term:N", alt.Tooltip("value:Q", format=".2f")],
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
        )
        .add_selection(selection)
        .properties(title=title, height=380)
        .interactive()
    )
    return chart

def small_multiples(long_df: pd.DataFrame, all_terms: list):
    data = long_df[long_df["term"].isin(all_terms)]
    if data.empty:
        st.info("No data to plot.")
        return
    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("date:T", title=""),
            y=alt.Y("value:Q", title="Index"),
            facet=alt.Facet("term:N", columns=4),
            tooltip=["date:T", "term:N", alt.Tooltip("value:Q", format=".2f")]
        )
        .resolve_scale(y="independent")
        .properties(title="Small multiples (per term)", height=150)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

@st.cache_data
def create_small_multiples(data, all_terms):
    """Cached version of small multiples chart creation"""
    if data.empty:
        return None
    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("date:T", title=""),
            y=alt.Y("value:Q", title="Index"),
            facet=alt.Facet("term:N", columns=4),
            tooltip=["date:T", "term:N", alt.Tooltip("value:Q", format=".2f")]
        )
        .resolve_scale(y="independent")
        .properties(title="Small multiples (per term)", height=150)
        .interactive()
    )
    return chart

@st.cache_data
def create_yoy_monthly_chart(long_df: pd.DataFrame, term: str) -> alt.Chart:
    """
    Create a YoY chart with month on X-axis and lines for each year.
    Defaults to showing past 3 years of data.
    """
    # Get YoY data for the term
    yt = yoy_table(long_df, term)
    
    if yt.empty:
        return None
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"YoY chart for {term}:")
    print(f"DEBUG: YoY chart for {term}:")
    print(f"DEBUG:   Total rows: {len(yt)}")
    print(f"DEBUG:   Date range: {yt['date'].min()} to {yt['date'].max()}")
    print(f"DEBUG:   Rows with pct_diff: {yt['pct_diff'].notna().sum()}")
    logger.debug(f"  Total rows: {len(yt)}")
    logger.debug(f"  Date range: {yt['date'].min()} to {yt['date'].max()}")
    logger.debug(f"  Rows with pct_diff: {yt['pct_diff'].notna().sum()}")
    logger.debug(f"  Rows with abs_diff: {yt['abs_diff'].notna().sum()}")
    
    # Filter to only rows with valid YoY data
    yt_valid = yt.dropna(subset=['pct_diff']).copy()
    
    if yt_valid.empty:
        logger.debug(f"  No valid pct_diff data found")
        return None
    print(f"DEBUG: No valid pct_diff data found for {term}")
    
    logger.debug(f"  Valid rows: {len(yt_valid)}")
    logger.debug(f"  Valid date range: {yt_valid['date'].min()} to {yt_valid['date'].max()}")
    
    # Filter to past 3 years (or less if not enough data)
    # Use a more robust method that handles leap years properly
    latest_date = yt_valid['date'].max()
    latest_year = latest_date.year
    three_years_ago_year = latest_year - 3
    
    # Get the actual earliest date in the data
    earliest_date = yt_valid['date'].min()
    earliest_year = earliest_date.year
    
    # Use the later of: 3 years ago or earliest available data
    start_year = max(three_years_ago_year, earliest_year)
    
    # Filter to the selected year range
    yt_filtered = yt_valid[yt_valid['date'].dt.year >= start_year].copy()
    
    logger.debug(f"  Latest date: {latest_date} (year: {latest_year})")
    logger.debug(f"  Three years ago year: {three_years_ago_year}")
    logger.debug(f"  Earliest year: {earliest_year}")
    logger.debug(f"  Start year (filter): {start_year}")
    
    if yt_filtered.empty:
        logger.debug(f"  No data after filtering")
        return None
    
    logger.debug(f"  Filtered rows: {len(yt_filtered)}")
    logger.debug(f"  Filtered date range: {yt_filtered['date'].min()} to {yt_filtered['date'].max()}")
    
    # Extract month and year for visualization
    yt_filtered['month'] = yt_filtered['date'].dt.month
    yt_filtered['month_name'] = yt_filtered['date'].dt.strftime('%b')  # Jan, Feb, etc.
    yt_filtered['year'] = yt_filtered['date'].dt.year
    
    # Simple approach: just use month_name and year for grouping
    # Create the line chart with zero line
    line_chart = (
        alt.Chart(yt_filtered)
        .mark_line(point=True)
        .encode(
            x=alt.X('month_name:N', title='Month', sort=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
            y=alt.Y('pct_diff:Q', title='YoY % Difference'),
            color=alt.Color('year:N', title='Year'),
            tooltip=[
                alt.Tooltip('year:N', title='Year'),
                alt.Tooltip('month_name:N', title='Month'),
                alt.Tooltip('pct_diff:Q', title='YoY % Diff', format='.2f'),
                alt.Tooltip('current:Q', title='Current Value', format='.2f'),
                alt.Tooltip('previous_year:Q', title='Previous Year', format='.2f')
            ]
        )
        .properties(
            title=f'{term} - YoY % Difference by Month ({start_year}-{latest_year})',
            height=300,
            width=400
        )
        .interactive()
    )
    
    # Create a zero line
    zero_line = (
        alt.Chart(pd.DataFrame({'y': [0]}))
        .mark_rule(color='red', strokeDash=[5, 5], strokeWidth=2)
        .encode(y='y:Q')
        .properties(height=300, width=400)
    )
    
    # Layer the charts
    chart = alt.layer(line_chart, zero_line)
    
    return chart





@st.cache_data
def create_yoy_absolute_chart(long_df: pd.DataFrame, term: str) -> alt.Chart:
    """
    Create a YoY chart showing absolute differences with month on X-axis.
    Defaults to showing past 3 years of data.
    """
    # Get YoY data for the term
    yt = yoy_table(long_df, term)
    
    if yt.empty or yt['abs_diff'].isna().all():
        return None
    
    # Filter to only rows with valid YoY data
    yt_valid = yt.dropna(subset=['abs_diff']).copy()
    
    if yt_valid.empty:
        return None
    
    # Filter to past 3 years (or less if not enough data)
    # Use a more robust method that handles leap years properly
    latest_date = yt_valid['date'].max()
    latest_year = latest_date.year
    three_years_ago_year = latest_year - 3
    
    # Get the actual earliest date in the data
    earliest_date = yt_valid['date'].min()
    earliest_year = earliest_date.year
    
    # Use the later of: 3 years ago or earliest available data
    start_year = max(three_years_ago_year, earliest_year)
    
    # Filter to the selected year range
    yt_filtered = yt_valid[yt_valid['date'].dt.year >= start_year].copy()
    
    if yt_filtered.empty:
        return None
    
    # Extract month and year for visualization
    yt_filtered['month'] = yt_filtered['date'].dt.month
    yt_filtered['month_name'] = yt_filtered['date'].dt.strftime('%b')  # Jan, Feb, etc.
    yt_filtered['year'] = yt_filtered['date'].dt.year
    
    # Simple approach: just use month_name and year for grouping
    # Create the line chart with zero line
    line_chart = (
        alt.Chart(yt_filtered)
        .mark_line(point=True)
        .encode(
            x=alt.X('month_name:N', title='Month', sort=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
            y=alt.Y('abs_diff:Q', title='YoY Absolute Difference'),
            color=alt.Color('year:N', title='Year'),
            tooltip=[
                alt.Tooltip('year:N', title='Year'),
                alt.Tooltip('month_name:N', title='Month'),
                alt.Tooltip('abs_diff:Q', title='YoY Abs Diff', format='.2f'),
                alt.Tooltip('current:Q', title='Current Value', format='.2f'),
                alt.Tooltip('previous_year:Q', title='Previous Year', format='.2f')
            ]
        )
        .properties(
            title=f'{term} - YoY Absolute Difference by Month ({start_year}-{latest_year})',
            height=300,
            width=400
        )
        .interactive()
    )
    
    # Create a zero line
    zero_line = (
        alt.Chart(pd.DataFrame({'y': [0]}))
        .mark_rule(color='red', strokeDash=[5, 5], strokeWidth=2)
        .encode(y='y:Q')
        .properties(height=300, width=400)
    )
    
    # Layer the charts
    chart = alt.layer(line_chart, zero_line)
    
    return chart

@st.cache_data
def create_yoy_monthly_chart_all_data(long_df: pd.DataFrame, term: str) -> alt.Chart:
    """
    Create a YoY chart with ALL data (no 5-year filter) for debugging.
    """
    # Get YoY data for the term
    yt = yoy_table(long_df, term)
    
    if yt.empty:
        return None
    
    # Filter to only rows with valid YoY data
    yt_valid = yt.dropna(subset=['pct_diff']).copy()
    
    if yt_valid.empty:
        return None
    
    # Extract month and year for visualization
    yt_valid['month'] = yt_valid['date'].dt.month
    yt_valid['month_name'] = yt_valid['date'].dt.strftime('%b')  # Jan, Feb, etc.
    yt_valid['year'] = yt_valid['date'].dt.year
    
    yt_valid['year_month'] = yt_valid['date'].dt.to_period('M').dt.to_timestamp()
    # Create the chart
    chart = (
        alt.Chart(yt_valid)
        .mark_line(point=True)
        .encode(
            x=alt.X('month_name:N', title='Month', sort=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
            y=alt.Y('pct_diff:Q', title='YoY % Difference'),
            color=alt.Color('year:N', title='Year'),
            tooltip=[
                alt.Tooltip('year:N', title='Year'),
                alt.Tooltip('month_name:N', title='Month'),
                alt.Tooltip('pct_diff:Q', title='YoY % Diff', format='.2f'),
                alt.Tooltip('current:Q', title='Current Value', format='.2f'),
                alt.Tooltip('previous_year:Q', title='Previous Year', format='.2f')
            ]
        )
        .properties(
            title=f'{term} - YoY % Difference by Month (ALL DATA)',
            height=300,
            width=400
        )
        .interactive()
    )
    
    return chart

def ribbon_bounds_from_pairs(term: str, pm: pd.DataFrame, q: float = 0.84) -> Tuple[float,float]:
    """
    Approximate a global multiplier band per term using median and dispersion
    of its pairwise ratios.
    Returns (lo_mult, hi_mult). q=0.84 ~ ~1 SD equivalent for normal.
    """
    rel = pm[pm["term_i"] == term]
    if rel.empty:
        return (1.0, 1.0)
    # construct a symmetric relative uncertainty ~ exp(±sd_log)
    # use cv_log as proxy: sd_log ≈ cv_log * |mean_log| (rough)
    # keep it conservative if NaN
    cv = np.nanmean(rel["cv_log"].values)
    if not np.isfinite(cv) or cv <= 0:
        return (1.0, 1.0)
    
    # Cap the coefficient of variation to prevent extreme bounds
    # This prevents ribbons from going too far beyond the 0-100 scale
    cv_capped = min(cv, 1.5)  # Cap at 1.5 to keep bounds reasonable
    
    sd = float(cv_capped)  # proxy on log space
    lo = np.exp(-sd)
    hi = np.exp(+sd)
    
    # Additional safety cap on the multipliers
    lo = max(lo, 0.1)  # Don't go below 10% of the value
    hi = min(hi, 3.0)  # Don't go above 300% of the value
    
    return (lo, hi)


@st.cache_data
def create_yoy_monthly_chart_debug(long_df: pd.DataFrame, term: str) -> alt.Chart:
    """
    Create a YoY chart showing all data points, including those without YoY calculations.
    """
    # Get YoY data for the term
    yt = yoy_table(long_df, term)
    
    if yt.empty:
        return None
    
    # Don't filter out NaN values - show all data
    yt_all = yt.copy()
    
    # Extract month and year for visualization
    yt_all['month'] = yt_all['date'].dt.month
    yt_all['month_name'] = yt_all['date'].dt.strftime('%b')  # Jan, Feb, etc.
    yt_all['year'] = yt_all['date'].dt.year
    
    yt_all['year_month'] = yt_all['date'].dt.to_period('M').dt.to_timestamp()
    # Add a flag for data points with YoY calculations
    yt_all['has_yoy'] = yt_all['pct_diff'].notna()
    
    # Create the chart
    chart = (
        alt.Chart(yt_all)
        .mark_line(point=True)
        .encode(
            x=alt.X('month_name:N', title='Month', sort=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
            y=alt.Y('pct_diff:Q', title='YoY % Difference'),
            color=alt.Color('year:N', title='Year'),
            opacity=alt.condition(alt.datum.has_yoy, alt.value(1), alt.value(0.3)),
            tooltip=[
                alt.Tooltip('year:N', title='Year'),
                alt.Tooltip('month_name:N', title='Month'),
                alt.Tooltip('pct_diff:Q', title='YoY % Diff', format='.2f'),
                alt.Tooltip('current:Q', title='Current Value', format='.2f'),
                alt.Tooltip('previous_year:Q', title='Previous Year', format='.2f'),
                alt.Tooltip('has_yoy:N', title='Has YoY Data')
            ]
        )
        .properties(
            title=f'{term} - YoY % Difference by Month (All Data Points)',
            height=300,
            width=400
        )
        .interactive()
    )
    
    return chart

if run:
    if not serpapi_key:
        st.error("Please enter your API key.")
        st.stop()



    terms = [t.strip() for t in terms_text.splitlines() if t.strip()]
    if len(terms) < 2:
        st.error("Enter at least two terms.")
        st.stop()

    current_params = get_current_params(serpapi_key, terms_text, geo, timeframe, group_size, cache_dir, sleep_ms, use_cache, provider)
    if should_reload_data(current_params):
        with st.spinner("Fetching, stitching, and preparing views..."):
            try:
                # Live progress UI
                progress_bar = st.progress(0)
                status_placeholder = st.empty()

                def on_progress(evt: dict):
                    stage = evt.get("stage", "")
                    msg = evt.get("message", "")
                    total = evt.get("total_batches") or 0
                    current = evt.get("current_batch") or 0
                    # Map stages to rough progress percentages
                    stage_weights = {
                        "start": 0,
                        "batching": 5,
                        "fetch": 5 + int(70 * (current / max(1, total))) if total else 20,
                        "fetched": 75,
                        "pairwise": 82,
                        "scaling": 88,
                        "wide": 93,
                        "done": 100,
                    }
                    pct = stage_weights.get(stage, min(99, 5 + int(70 * (current / max(1, total))) if total else 25))
                    progress_bar.progress(min(100, max(0, pct)))
                    status_placeholder.info(f"{msg}")

                # Add debug info
                if show_debug:
                    st.info(f"Starting with {len(terms)} terms: {terms}")
                    st.info(f"Cache dir: {cache_dir}")
                    st.info(f"Using cache: {use_cache}")
                    st.info(f"API timeframe: {timeframe}")
                    st.info(f"Date filters: {start_date} to {end_date}")
                    st.info(f"Note: API timeframe ({timeframe}) may limit available data regardless of date filters")
                
                result = stitch_terms(
                    serpapi_key=serpapi_key,
                    terms=terms,
                    provider=provider,
                    geo=geo,
                    timeframe=timeframe,
                    group_size=group_size,
                    cache_dir=cache_dir,
                    sleep_ms=int(sleep_ms),
                    verbose=verbose_logs,
                    use_cache=use_cache,
                    debug=show_debug,
                    collect_raw_responses=collect_raw_responses,
                    brightdata_zone=brightdata_zone,  # Add this line
                    progress_callback=on_progress,
                )

                if collect_raw_responses:
                    (
                        df_scaled,
                        pivot_scores,
                        scales,
                        pair_metrics,
                        term_instability,
                        ratio_samples,
                        raw_responses,
                    ) = result
                else:
                    (
                        df_scaled,
                        pivot_scores,
                        scales,
                        pair_metrics,
                        term_instability,
                        ratio_samples,
                    ) = result
                    raw_responses = []

                # Ensure progress shows completion
                try:
                    progress_bar.progress(100)
                    status_placeholder.success("Completed stitching")
                except Exception:
                    pass

                if show_debug:
                    st.success(f"Successfully fetched data: {df_scaled.shape}")
                    st.info(f"Data columns: {list(df_scaled.columns)}")
                    st.info(f"Date range: {df_scaled['date'].min()} to {df_scaled['date'].max()}")
                    
                    # Show raw API response for debugging
                    st.subheader("Raw API Response Debug")
                    try:
                        from stitcher import TrendsFetcher
                        fetcher = TrendsFetcher(
                            serpapi_key=serpapi_key,
                            provider=provider,
                            geo=geo,
                            timeframe=timeframe,
                            cache_dir=cache_dir,
                            sleep_ms=int(sleep_ms),
                            use_cache=False,  # Force fresh request
                            debug=True,
                        )
                        test_df = fetcher.fetch_batch(terms[:2])  # Test with first 2 terms
                        st.success(f"Test API call successful: {test_df.shape}")
                        st.json(test_df.head(10).to_dict('records'))
                        
                        # Test individual term API calls
                        st.write("**Individual Term API Tests:**")
                        for term in terms[:3]:  # Test first 3 terms
                            try:
                                single_term_df = fetcher.fetch_batch([term])
                                st.write(f"  {term}: {single_term_df.shape} data points")
                                if not single_term_df.empty:
                                    st.write(f"    - Date range: {single_term_df['date'].min()} to {single_term_df['date'].max()}")
                                    st.write(f"    - Value range: {single_term_df[term].min():.2f} to {single_term_df[term].max():.2f}")
                                    st.write(f"    - Sample: {single_term_df[term].head(3).tolist()}")
                                else:
                                    st.error(f"    - No data returned for {term}")
                            except Exception as e:
                                st.error(f"    - Error fetching {term}: {str(e)}")
                                
                    except Exception as debug_e:
                        st.error(f"Debug API call failed: {debug_e}")
                        st.code(traceback.format_exc())
                    
                    # Enhanced raw data debugging
                    st.subheader("Comprehensive Raw Data Debug")
                    
                    # Show raw data for each term
                    st.write("**Raw data for each term:**")
                    for term in terms:
                        if term in df_scaled.columns:
                            term_data = df_scaled[['date', term]].dropna()
                            st.write(f"**{term}**: {len(term_data)} data points")
                            if not term_data.empty:
                                st.write(f"  - Date range: {term_data['date'].min()} to {term_data['date'].max()}")
                                st.write(f"  - Value range: {term_data[term].min():.2f} to {term_data[term].max():.2f}")
                                st.write(f"  - Non-zero values: {(term_data[term] > 0).sum()}")
                                st.write(f"  - Zero values: {(term_data[term] == 0).sum()}")
                                st.write(f"  - NaN values: {term_data[term].isna().sum()}")
                                
                                # Show sample data
                                st.write(f"  - Sample data (first 5 rows):")
                                st.dataframe(term_data.head())
                            else:
                                st.error(f"  - No data for {term}")
                        else:
                            st.error(f"**{term}**: Column not found in data")
                    
                    # Show data quality summary
                    st.write("**Data Quality Summary:**")
                    st.write(f"- Total terms requested: {len(terms)}")
                    st.write(f"- Terms with data: {sum(1 for term in terms if term in df_scaled.columns)}")
                    st.write(f"- Terms with non-zero data: {sum(1 for term in terms if term in df_scaled.columns and df_scaled[term].max() > 0)}")
                    
                    # Show magnitude differences and scaling analysis
                    st.write("**Magnitude Analysis (Before Scaling):**")
                    term_max_values = {}
                    for term in terms:
                        if term in df_scaled.columns:
                            max_val = df_scaled[term].max()
                            term_max_values[term] = max_val
                            st.write(f"  {term}: max value = {max_val:.2f}")
                    
                    if term_max_values:
                        max_term = max(term_max_values, key=term_max_values.get)
                        min_term = min(term_max_values, key=term_max_values.get)
                        max_val = term_max_values[max_term]
                        min_val = term_max_values[min_term]
                        ratio = max_val / min_val if min_val > 0 else float('inf')
                        
                        st.write(f"**Magnitude Differences:**")
                        st.write(f"- Most popular term: {max_term} (max = {max_val:.2f})")
                        st.write(f"- Least popular term: {min_term} (max = {min_val:.2f})")
                        st.write(f"- Magnitude ratio: {ratio:.1f}:1")
                        
                        if ratio > 100:
                            st.warning(f"**Large magnitude difference detected!** The most popular term is {ratio:.0f}x more popular than the least popular term. This may cause scaling issues.")
                        elif ratio > 50:
                            st.info(f"**Moderate magnitude difference:** The most popular term is {ratio:.0f}x more popular than the least popular term.")
                        else:
                            st.success(f"**Good magnitude balance:** Terms have similar popularity levels.")
                    
                    # Show raw data shape and info
                    st.write("**Raw DataFrame Info:**")
                    st.write(f"- Shape: {df_scaled.shape}")
                    st.write(f"- Columns: {list(df_scaled.columns)}")
                    st.write(f"- Memory usage: {df_scaled.memory_usage(deep=True).sum() / 1024:.2f} KB")
                    
                    # Show sample of raw data
                    st.write("**Sample of Raw Data (first 10 rows):**")
                    st.dataframe(df_scaled.head(10))
                    
                    # Show data types
                    st.write("**Data Types:**")
                    st.write(df_scaled.dtypes.to_dict())
                    
                    # Show missing data
                    st.write("**Missing Data Summary:**")
                    missing_data = df_scaled.isnull().sum()
                    st.write(missing_data[missing_data > 0].to_dict() if missing_data.sum() > 0 else "No missing data")

                # Store in session state
                st.session_state.df_scaled = df_scaled
                st.session_state.pivot_scores = pivot_scores
                st.session_state.scales = scales
                st.session_state.terms = terms
                st.session_state.data_loaded = True
                st.session_state.current_params = current_params
                st.session_state.pair_metrics = pair_metrics
                st.session_state.term_instability = term_instability
                st.session_state.ratio_samples = ratio_samples
                st.session_state.raw_responses = raw_responses
                
                # Show scaling analysis in debug mode
                if show_debug:
                    st.subheader("Scaling Analysis Debug")
                    st.write("**Scale Factors Applied:**")
                    for term in terms:
                        if term in scales.index:
                            scale_factor = scales[term]
                            original_max = df_scaled[term].max() if term in df_scaled.columns else 0
                            scaled_max = original_max * scale_factor
                            st.write(f"  {term}:")
                            st.write(f"    - Original max: {original_max:.2f}")
                            st.write(f"    - Scale factor: {scale_factor:.4f}")
                            st.write(f"    - Scaled max: {scaled_max:.2f}")
                            if scale_factor < 0.1:
                                st.warning(f"    - **Very small scale factor!** This term will appear near-zero in charts.")
                            elif scale_factor < 0.5:
                                st.info(f"    - **Small scale factor** - term will appear reduced in charts.")
                    
                    st.write("**Scaling Summary:**")
                    st.write(f"- Reference term: {scales.idxmax()} (scale = {scales.max():.4f})")
                    st.write(f"- Most scaled down: {scales.idxmin()} (scale = {scales.min():.4f})")
                    st.write(f"- Scale factor range: {scales.min():.4f} to {scales.max():.4f}")
                    
                    scale_ratio = scales.max() / scales.min() if scales.min() > 0 else float('inf')
                    st.write(f"- Scale factor ratio: {scale_ratio:.1f}:1")
                    
                    if scale_ratio > 100:
                        st.error("**Extreme scaling detected!** Some terms will be nearly invisible in charts due to very small scale factors.")
                        st.info("**Suggestion:** Try using the 'Autocomplete Explorer' to find more comparable terms, or group terms with similar popularity levels.")
                    elif scale_ratio > 50:
                        st.warning("**Large scaling detected!** Some terms may be hard to see in charts.")
                        st.info("**Suggestion:** Consider using the 'Autocomplete Explorer' to find more comparable terms.")
                    else:
                        st.success("**Reasonable scaling** - all terms should be visible in charts.")

            except Exception as e:
                # Friendlier JSON/network error hints
                msg = str(e)
                if "JSON parse failed" in msg or "Expecting value" in msg:
                    st.error("The API response was not valid JSON.")
                    st.info("Tips: Verify your API key, endpoint, and select the correct provider. For Bright Data, ensure the zone is correct and the endpoint returns JSON.")
                    st.code(msg)
                else:
                    st.error(f"Error during data fetching: {msg}")
                if show_debug:
                    st.write("Full traceback:")
                    st.code(traceback.format_exc())
                st.stop()
    else:
        st.info("Using cached data. Change parameters to reload.")
        df_scaled = st.session_state.df_scaled
        pivot_scores = st.session_state.pivot_scores
        scales = st.session_state.scales
        terms = st.session_state.terms
        pair_metrics = st.session_state.pair_metrics
        term_instability = st.session_state.term_instability
        ratio_samples = st.session_state.ratio_samples
        raw_responses = st.session_state.raw_responses

    # Apply filters to the data
    df_scaled = cached_filter_date_range(df_scaled, start_date, end_date)
    
    long_df = cached_melt_long(df_scaled)
    
    if long_df.empty:
        st.error("No data available for charting. Please check your API key and terms.")
        st.stop()
    
    if long_df['value'].isna().all():
        st.error("All values are NaN. This might indicate an API response parsing issue.")
        st.stop()

    if show_debug:
        st.info(f"Chart data shape: {long_df.shape}")
        st.info(f"Available terms: {long_df['term'].unique()}")
        st.info(f"Value range: {long_df['value'].min()} to {long_df['value'].max()}")

    st.caption("Tip: Click on terms in the legend to show/hide them. Click multiple times to select multiple terms.")
    
    # Show helpful message if ribbons are enabled but no stability data
    if show_ribbons and ('pair_metrics' not in st.session_state or st.session_state.pair_metrics is None or st.session_state.pair_metrics.empty):
        st.info("💡 **Fan-out bands require stability data.** Run the analysis first to enable instability bands on the chart.")
    
    # Show ribbon bounds info when enabled
    if show_ribbons and 'pair_metrics' in st.session_state and st.session_state.pair_metrics is not None and not st.session_state.pair_metrics.empty:
        pm = st.session_state.pair_metrics
        ribbon_info = []
        for t in terms:
            lo, hi = ribbon_bounds_from_pairs(t, pm)
            if np.isfinite(lo) and np.isfinite(hi):
                ribbon_info.append(f"{t}: {lo:.2f}x to {hi:.2f}x")
        if ribbon_info:
            st.caption(f"💡 **Ribbon bounds:** {', '.join(ribbon_info)}. Values show the uncertainty range multipliers applied to each term.")
        
        # Add explanation for 2-term case
        if len(terms) == 2:
            st.caption("📊 **Note:** With 2 terms, instability measures batch-to-batch variation. If terms appear in multiple batches, this shows how consistent their relationship is across different contexts.")
    
    # Create the main chart
    chart_title = "Google Trends: Comparable Time Series"
    
    chart = create_line_chart(long_df, terms, chart_title)
    if chart:
        if show_ribbons and 'pair_metrics' in st.session_state and not st.session_state.pair_metrics.empty:
            pm = st.session_state.pair_metrics
            # Build bands for selected_terms (or all terms)
            bands = []
            for t in terms:
                lo, hi = ribbon_bounds_from_pairs(t, pm)
                if not np.isfinite(lo) or not np.isfinite(hi):
                    continue
                tmp = long_df[long_df["term"]==t][["date","value"]].copy()
                tmp["term"] = t
                tmp["lo"] = tmp["value"] * lo
                tmp["hi"] = tmp["value"] * hi
                bands.append(tmp)
            if bands:
                band_df = pd.concat(bands, ignore_index=True)
                band = (alt.Chart(band_df)
                        .mark_area(opacity=0.2)
                        .encode(
                            x="date:T",
                            y="lo:Q",
                            y2="hi:Q",
                            color=alt.Color("term:N", legend=None)
                        ))
                # Layer the charts and ensure the main chart's legend is preserved
                layered_chart = alt.layer(band, chart).resolve_scale(color='independent')
                st.altair_chart(layered_chart, use_container_width=True)
            else:
                st.altair_chart(chart, use_container_width=True)
        else:
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No data to plot.")
    
    # Show popularity analysis for all users (not just debug mode)
    if 'df_scaled' in locals() and df_scaled is not None and not df_scaled.empty:
        # Calculate popularity differences from original data
        term_max_values = {}
        for term in terms:
            if term in df_scaled.columns:
                max_val = df_scaled[term].max()
                term_max_values[term] = max_val
        
        if term_max_values:
            max_term = max(term_max_values, key=term_max_values.get)
            min_term = min(term_max_values, key=term_max_values.get)
            max_val = term_max_values[max_term]
            min_val = term_max_values[min_term]
            
            # Handle cases where min_val is 0 (no data for that term)
            if min_val == 0:
                st.warning(f"**Data Notice:** '{min_term}' has no search data in Google Trends (all values are 0). This term will not appear in the chart above.")
                st.info("**Tip:** Try the 'Autocomplete Explorer' in the sidebar to find alternative search terms, or check if the term is spelled correctly.")
            else:
                ratio = max_val / min_val
                
                # Show user-friendly popularity analysis
                if ratio > 100:
                    st.warning(f"**Popularity Notice:** '{max_term}' is {ratio:.0f}x more popular than '{min_term}' in Google Trends. The less popular terms may appear near-zero in the chart above due to scaling.")
                    st.info("**Tip:** Try the 'Autocomplete Explorer' in the sidebar to find more comparable terms, or group terms with similar popularity levels.")
                elif ratio > 50:
                    st.info(f"**Popularity Notice:** '{max_term}' is {ratio:.0f}x more popular than '{min_term}'. Some terms may be hard to see in the chart due to scaling differences.")
                elif ratio > 10:
                    st.info(f"**Popularity Notice:** '{max_term}' is {ratio:.0f}x more popular than '{min_term}'. All terms should be visible in the chart.")
                else:
                    st.success(f"**Good balance:** '{max_term}' is only {ratio:.1f}x more popular than '{min_term}'. All terms should be clearly visible in the chart.")
    
    # Display the comparable time series table below the chart
    st.subheader("Comparable Time Series (max=100 across ALL terms)")
    st.caption("This table shows the trend data where all terms are scaled to the same range (0-100) so you can easily compare their relative popularity over time.")
    st.dataframe(df_scaled.head(20))
    st.download_button(
        "Download full timeseries CSV",
        df_scaled.to_csv(index=False).encode("utf-8"),
        file_name="trends_stitched_scaled.csv",
        mime="text/csv",
        key="full_timeseries_download"
    )
    
    st.markdown("---")
    st.subheader("Year-on-Year (YoY) Analysis")

    # Show YoY data availability info for all terms
    yoy_data_available = {}
    for term in terms:
        yt = yoy_table(long_df, term)
        if not yt.empty:
            first_with_prev = yt[yt['previous_year'].notna()]['date'].min()
            if pd.notna(first_with_prev):
                yoy_data_available[term] = first_with_prev
            else:
                yoy_data_available[term] = None
        else:
            yoy_data_available[term] = None
    
    # Display availability info
    available_terms = [term for term, date in yoy_data_available.items() if date is not None]
    if available_terms:
        earliest_date = min([yoy_data_available[term] for term in available_terms])
        st.info(f"YoY Data Available: {len(available_terms)}/{len(terms)} terms have YoY data from {earliest_date} onwards")
    else:
        st.warning("No YoY data available - insufficient historical data for year-over-year comparison")

    # Debug YoY data if requested
    if show_debug:
        st.subheader("YoY Debug Info")
        for term in terms:
            yt = yoy_table(long_df, term)
            st.write(f"**{term}**: {yt.shape[0]} rows, {yt['pct_diff'].notna().sum()} with YoY data")
            
            # Show detailed YoY data analysis
            if not yt.empty:
                st.write(f"  - Date range: {yt['date'].min()} to {yt['date'].max()}")
                st.write(f"  - Years with data: {sorted(yt['date'].dt.year.unique())}")
                
                # Show recent data specifically
                recent_data = yt[yt['date'] >= pd.Timestamp('2024-01-01')]
                if not recent_data.empty:
                    st.write(f"  - 2024+ data: {len(recent_data)} rows")
                    st.write(f"  - 2024+ with pct_diff: {recent_data['pct_diff'].notna().sum()} rows")
                    
                    # Show which months have YoY data
                    recent_with_yoy = recent_data[recent_data['pct_diff'].notna()]
                    if not recent_with_yoy.empty:
                        months_with_yoy = sorted(recent_with_yoy['date'].dt.strftime('%Y-%m').unique())
                        st.write(f"  - 2024+ months with YoY data: {months_with_yoy}")
                    
                    # Show which months are missing YoY data
                    recent_without_yoy = recent_data[recent_data['pct_diff'].isna()]
                    if not recent_without_yoy.empty:
                        months_without_yoy = sorted(recent_without_yoy['date'].dt.strftime('%Y-%m').unique())
                        st.write(f"  - 2024+ months WITHOUT YoY data: {months_without_yoy}")
                else:
                    st.write(f"  - No 2024+ data found")
                
                # Show sample of recent data
                if not recent_data.empty:
                    st.write("  - Recent data sample:")
                    st.dataframe(recent_data[['date', 'current', 'previous_year', 'pct_diff']].head(5))

    # Create YoY charts for all terms
    st.subheader("YoY % Difference by Month")

    
    # Add toggle for showing all data vs filtered data
    show_all_data = st.checkbox("Show all data (not just 3 years)", value=st.session_state.get('show_all_data', False), key='show_all_data_checkbox')
    show_debug_chart = st.checkbox("Show debug chart (all data points)", value=st.session_state.get('show_debug_chart', False), key='show_debug_chart_checkbox')
    

    
    # Calculate how many charts per row based on number of terms
    charts_per_row = min(2, len(terms))
    
    st.caption("These charts show how each term's popularity changed compared to the same month in previous years. The red line at zero shows no change - values above zero mean higher popularity than the previous year, below zero means lower popularity.")
    for i in range(0, len(terms), charts_per_row):
        cols = st.columns(charts_per_row)
        for j, term in enumerate(terms[i:i+charts_per_row]):
            with cols[j]:
                if show_debug_chart:
                    chart = create_yoy_monthly_chart_debug(long_df, term)
                elif show_all_data:
                    chart = create_yoy_monthly_chart_all_data(long_df, term)
                else:
                    chart = create_yoy_monthly_chart(long_df, term)
                if chart:
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info(f"No YoY data for {term}")

    # Create YoY absolute difference charts for all terms
    st.subheader("YoY Absolute Difference by Month")
    
    st.caption("These charts show the actual difference in popularity values (not percentages). The red line at zero shows no change - positive values mean higher popularity than the previous year, negative values mean lower popularity.")
    for i in range(0, len(terms), charts_per_row):
        cols = st.columns(charts_per_row)
        for j, term in enumerate(terms[i:i+charts_per_row]):
            with cols[j]:
                chart = create_yoy_absolute_chart(long_df, term)
                if chart:
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info(f"No YoY data for {term}")

    # Download all YoY data
    st.subheader("Download YoY Data")
    all_yoy_data = {}
    all_yoy_data_filtered = {}
    
    for term in terms:
        yt = yoy_table(long_df, term)
        if not yt.empty:
            all_yoy_data[term] = yt
            
            # Create filtered version (past 3 years)
            yt_valid = yt.dropna(subset=['pct_diff']).copy()
            if not yt_valid.empty:
                latest_date = yt_valid['date'].max()
                latest_year = latest_date.year
                three_years_ago_year = latest_year - 3
                earliest_date = yt_valid['date'].min()
                earliest_year = earliest_date.year
                start_year = max(three_years_ago_year, earliest_year)
                yt_filtered = yt_valid[yt_valid['date'].dt.year >= start_year].copy()
                if not yt_filtered.empty:
                    all_yoy_data_filtered[term] = yt_filtered
    
    if all_yoy_data:
        # Create combined CSV for all data
        combined_data = []
        for term, yt in all_yoy_data.items():
            yt_copy = yt.copy()
            yt_copy['term'] = term
            combined_data.append(yt_copy)
        
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            st.download_button(
                "Download All YoY Data (Full History)",
                combined_df.to_csv(index=False).encode("utf-8"),
                file_name="all_yoy_data_full.csv",
                mime="text/csv",
                key="all_yoy_data_full_download"
            )
            


    
    if all_yoy_data_filtered:
        # Create combined CSV for filtered data (past 5 years)
        combined_filtered_data = []
        for term, yt in all_yoy_data_filtered.items():
            yt_copy = yt.copy()
            yt_copy['term'] = term
            combined_filtered_data.append(yt_copy)
        
        if combined_filtered_data:
            combined_filtered_df = pd.concat(combined_filtered_data, ignore_index=True)
            
            # Show date range info
            if not combined_filtered_df.empty:
                date_range_start = combined_filtered_df['date'].min()
                date_range_end = combined_filtered_df['date'].max()
                st.info(f"3-Year YoY Data Range: {date_range_start.strftime('%Y-%m-%d')} to {date_range_end.strftime('%Y-%m-%d')}")
            
            st.download_button(
                "Download YoY Data (Past 3 Years)",
                combined_filtered_df.to_csv(index=False).encode("utf-8"),
                file_name="yoy_data_past_3_years.csv",
                mime="text/csv",
                key="yoy_data_3_years_download"
            )
            


    st.markdown("---")
    with st.expander("Raw API Responses", expanded=False):
        st.subheader("Raw API Response Data")
        st.caption("Download the raw JSON responses from the selected provider for offline validation and debugging.")
        
        if 'raw_responses' in st.session_state and st.session_state.raw_responses is not None and len(st.session_state.raw_responses) > 0:
            # Show summary of responses
            st.write(f"**API Calls Made:** {len(st.session_state.raw_responses)}")
            
            # Show details for each response
            for i, response in enumerate(st.session_state.raw_responses):
                with st.expander(f"Batch {i+1}: {', '.join(response['batch_terms'])}", expanded=False):
                    st.write(f"**Terms:** {', '.join(response['batch_terms'])}")
                    st.write(f"**Request Parameters:**")
                    st.json(response['request_params'])
                    
                    # Network diagnostics
                    st.write("**Network Diagnostics:**")
                    diag_cols = st.columns(2)
                    with diag_cols[0]:
                        st.write(f"Provider: {response.get('provider', 'unknown')}")
                        st.write(f"HTTP Status: {response.get('http_status', 'unknown')}")
                        if response.get('request_endpoint'):
                            st.write(f"Endpoint: {response.get('request_endpoint')}")
                    with diag_cols[1]:
                        hdrs = response.get('response_headers') or {}
                        ctype = response.get('content_type') or hdrs.get('Content-Type')
                        st.write(f"Content-Type: {ctype}")
                        if 'X-RateLimit-Remaining' in (hdrs or {}):
                            st.write(f"X-RateLimit-Remaining: {hdrs.get('X-RateLimit-Remaining')}")
                    
                    # Response structure and sample
                    st.write(f"**Response Structure:**")
                    st.json({k: type(v).__name__ for k, v in response.get('response_data', {}).items()} if isinstance(response.get('response_data'), dict) else str(type(response.get('response_data'))))
                    
                    sample = response.get('response_text_sample')
                    if sample:
                        st.write("**Response Body (first 500 chars):**")
                        st.code(sample)
                    st.write(f"**Response Structure:**")
                    st.json({k: type(v).__name__ for k, v in response['response_data'].items()})
            
            # Download all responses as JSON
            import json
            all_responses_json = json.dumps(st.session_state.raw_responses, indent=2, default=str)
            st.download_button(
                "Download All Raw API Responses (JSON)",
                all_responses_json.encode("utf-8"),
                file_name="raw_api_responses.json",
                mime="application/json",
                key="raw_responses_download"
            )
            
            # Download individual response files
            st.write("**Download Individual Responses:**")
            for i, response in enumerate(st.session_state.raw_responses):
                response_json = json.dumps(response, indent=2, default=str)
                st.download_button(
                    f"Download Batch {i+1} Response",
                    response_json.encode("utf-8"),
                    file_name=f"batch_{i+1}_{'_'.join(response['batch_terms'])}.json",
                    mime="application/json",
                    key=f"batch_{i+1}_download"
                )
        else:
            st.info("No raw API responses available. Run the analysis to collect response data.")
    
    with st.expander("Detailed Analysis & Validation", expanded=False):
        st.subheader("Explainability")
        st.caption("This table shows how the scaling algorithm adjusted each term's maximum value to make all terms comparable. It helps verify that the data normalization worked correctly.")
        st.dataframe(pivot_scores)
        st.download_button(
            "Download pivot scores CSV",
            pivot_scores.to_csv(index=False).encode("utf-8"),
            file_name="pivot_scores.csv",
            mime="text/csv",
            key="pivot_scores_download"
        )

        st.markdown("---")
        st.subheader("Per-Term Consensus Scale Factors")
        st.caption("This table shows the scaling factors used to normalize each term's data. Higher scale factors mean the term was scaled up more (had lower original popularity), while lower scale factors mean the term was scaled down more (had higher original popularity).")
        scales_df = pd.DataFrame({"term": scales.index, "scale": scales.values}).sort_values("scale", ascending=False)
        st.dataframe(scales_df)
        st.download_button(
            "Download scale factors CSV",
            scales_df.to_csv(index=False).encode("utf-8"),
            file_name="scale_factors.csv",
            mime="text/csv",
            key="scale_factors_download"
        )
        
        # Add algorithm validation table
        st.subheader("Algorithm Validation - Scaling Details")
        st.caption("This table shows the original maximum values and how they were scaled to make terms comparable. The reference term has a scale of 1.0 and its original maximum becomes 100 in the normalized data.")
        
        # Get original maximum values from the data
        original_max_values = {}
        for term in terms:
            if term in df_scaled.columns:
                original_max_values[term] = df_scaled[term].max()
            else:
                original_max_values[term] = 0
        
        # Create validation table
        validation_data = []
        for term in terms:
            original_max = original_max_values.get(term, 0)
            scale = scales.get(term, 1.0)
            normalized_max = original_max * scale
            
            validation_data.append({
                "Term": term,
                "Original Max": round(original_max, 2),
                "Scale Factor": round(scale, 4),
                "Normalized Max": round(normalized_max, 2),
                "Reference Term": "Yes" if scale == 1.0 else "No"
            })
        
        validation_df = pd.DataFrame(validation_data)
        validation_df = validation_df.sort_values("Original Max", ascending=False)
        st.dataframe(validation_df, use_container_width=True)
        
        # Add download button for validation data
        st.download_button(
            "Download algorithm validation CSV",
            validation_df.to_csv(index=False).encode("utf-8"),
            file_name="algorithm_validation.csv",
            mime="text/csv",
            key="algorithm_validation_download"
        )
        
        # Add summary statistics
        st.subheader("Scaling Summary")
        reference_term = validation_df[validation_df['Reference Term'] == 'Yes']['Term'].iloc[0] if not validation_df[validation_df['Reference Term'] == 'Yes'].empty else "Unknown"
        max_original = validation_df['Original Max'].max()
        min_original = validation_df['Original Max'].min()
        
        st.write(f"**Reference Term**: {reference_term}")
        st.write(f"**Original Popularity Range**: {min_original:.2f} to {max_original:.2f}")
        st.write(f"**Scaling Ratio**: {max_original/min_original:.1f}:1 (most to least popular)")
        
        if max_original/min_original > 10:
            st.info("Large Popularity Gap: The most popular term is significantly more popular than the least popular term. Scaling makes them comparable but the original data had very different popularity levels.")
        elif max_original/min_original > 5:
            st.warning("Moderate Popularity Gap: There's a notable difference in original popularity between terms.")
        else:
            st.success("Good Popularity Balance: Terms have relatively similar original popularity levels.")
    
    # Stability Diagnostics Section - Always available in accordion
    with st.expander("Stability Diagnostics", expanded=show_stability):
        st.subheader("Stability Diagnostics")
        st.caption("These diagnostics help identify inconsistencies in Google Trends data across different batches during the stitching process. Lower values indicate more stable/consistent comparisons between terms across batches.")
        
        # Add explanation for different term counts
        if len(terms) == 2:
            st.info("📊 **2 Terms Analysis:** Instability measures batch-to-batch variation in the ratio between these 2 terms. High instability means the API returns different relative values when the terms are fetched in different contexts.")
        elif len(terms) == 3:
            st.info("📊 **3 Terms Analysis:** Instability measures consistency of ratios across different batches. Each term pair may appear in multiple batches with different context terms.")
        else:
            st.info(f"📊 **{len(terms)} Terms Analysis:** Instability measures how consistently term relationships are preserved across different batches during the stitching process.")

        if 'pair_metrics' in st.session_state and not st.session_state.pair_metrics.empty:
            pm = st.session_state.pair_metrics.copy()
            st.caption("Pairwise instability (lower CV on log-ratio = more stable).")
            st.dataframe(pm.sort_values("cv_log", na_position="last").head(20))

            # Heatmap matrix of cv_log
            st.markdown("**Pairwise instability heatmap (cv_log)**")
            # build square matrix
            all_terms = sorted(set(pm["term_i"]).union(pm["term_j"]))
            mat = pd.DataFrame(index=all_terms, columns=all_terms, dtype=float)
            for _, r in pm.iterrows():
                mat.loc[r["term_i"], r["term_j"]] = r["cv_log"]
            mat = mat.replace([np.inf, -np.inf], np.nan)

            hm_data = (mat
                       .reset_index()
                       .melt(id_vars="index", var_name="term_j", value_name="cv_log")
                       .rename(columns={"index":"term_i"}))

            heat = (alt.Chart(hm_data)
                    .mark_rect()
                    .encode(
                        x=alt.X("term_j:N", title=""),
                        y=alt.Y("term_i:N", title=""),
                        color=alt.Color("cv_log:Q", title="cv(log ratio)", scale=alt.Scale(scheme="reds")),
                        tooltip=["term_i:N","term_j:N", alt.Tooltip("cv_log:Q", format=".3f")]
                    )
                    .properties(height=400))
            st.altair_chart(heat, use_container_width=True)

            # Box/violin of ratio variability for selected pairs
            st.markdown("**Distribution of observed ratios (selected pairs)**")
            rs = st.session_state.ratio_samples
            pair_opts = sorted({(a,b) for a,b in zip(rs["term_i"], rs["term_j"]) if a!=b})
            default_pairs = [p for p in pair_opts if p[0]==terms[0]][:3] or pair_opts[:3]
            pick = st.multiselect("Pairs", options=[f"{a} :: {b}" for a,b in pair_opts],
                                  default=[f"{a} :: {b}" for a,b in default_pairs])
            if pick:
                sel = []
                for p in pick:
                    a,b = [x.strip() for x in p.split("::")]
                    sel.append(rs[(rs["term_i"]==a) & (rs["term_j"]==b)].assign(pair=p))
                dist = pd.concat(sel, ignore_index=True) if sel else pd.DataFrame(columns=rs.columns)
                if not dist.empty:
                    box = (alt.Chart(dist)
                           .mark_boxplot()
                           .encode(x=alt.X("pair:N", title="Pair"),
                                   y=alt.Y("ratio:Q", title="Observed ratio"),
                                   tooltip=["pair:N", alt.Tooltip("ratio:Q", format=".3f")])
                           .properties(height=300))
                    st.altair_chart(box, use_container_width=True)

            # Per-term "instability score"
            st.markdown("**Per-term instability score**")
            ti = st.session_state.term_instability.sort_values("instability_score")
            st.dataframe(ti)
            
            # Download stability diagnostics data
            st.markdown("**Download Stability Diagnostics Data**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    "Download Pair Metrics CSV",
                    pm.to_csv(index=False).encode("utf-8"),
                    file_name="pairwise_instability_metrics.csv",
                    mime="text/csv",
                    key="pair_metrics_download"
                )
            
            with col2:
                st.download_button(
                    "Download Term Instability CSV",
                    ti.to_csv(index=False).encode("utf-8"),
                    file_name="term_instability_scores.csv",
                    mime="text/csv",
                    key="term_instability_download"
                )
            
            with col3:
                st.download_button(
                    "Download Ratio Samples CSV",
                    st.session_state.ratio_samples.to_csv(index=False).encode("utf-8"),
                    file_name="ratio_samples.csv",
                    mime="text/csv",
                    key="ratio_samples_download"
                )

    # Show debug logs
    show_debug_logs()