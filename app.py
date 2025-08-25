from datetime import date
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from stitcher import stitch_terms

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
if 'smoothing_days' not in st.session_state:
    st.session_state.smoothing_days = "7"
if 'start_date' not in st.session_state:
    st.session_state.start_date = None
if 'end_date' not in st.session_state:
    st.session_state.end_date = None
if 'timeframe' not in st.session_state:
    st.session_state.timeframe = "all"

st.set_page_config(page_title="Trends Stitcher", layout="wide")
st.title("Google Trends: Auto-Stitched Comparable Scale")

# Show cache status
if st.session_state.data_loaded:
    st.success(f"✅ Data loaded: {len(st.session_state.terms) if st.session_state.terms else 0} terms, {st.session_state.df_scaled.shape[0] if st.session_state.df_scaled is not None else 0} data points")
else:
    st.info("📊 No data loaded. Enter parameters and click 'Run' to fetch data.")

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

def get_current_params(serpapi_key, terms_text, geo, timeframe, group_size, cache_dir, sleep_ms, use_cache):
    """Get current parameters as a hashable tuple for comparison"""
    terms = tuple(sorted([t.strip() for t in terms_text.splitlines() if t.strip()]))
    return (serpapi_key, terms, geo, timeframe, group_size, cache_dir, sleep_ms, use_cache)

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
    serpapi_key = st.text_input("SerpAPI API Key", type="password", value="")
    terms_text = st.text_area("Terms (one per line)", "nike\nadidas\npuma\nnew balance\nasics")
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
        st.info(f"📅 Date filters detected. Using longer timeframe ({suggested_timeframe}) to get more data for filtering.")
        current_timeframe = st.session_state.timeframe if st.session_state.timeframe in timeframe_options else suggested_timeframe
        timeframe = st.selectbox("API Timeframe", timeframe_options, index=timeframe_options.index(current_timeframe))
    else:
        current_timeframe = st.session_state.timeframe if st.session_state.timeframe in timeframe_options else "today 5-y"
        timeframe = st.selectbox("API Timeframe", timeframe_options, index=timeframe_options.index(current_timeframe))
    
    # Update session state
    st.session_state.timeframe = timeframe
    
    group_size = st.slider("Batch size (max 5)", 2, 5, 5)

    st.markdown("---")
    st.subheader("Smoothing & Range")
    smoothing_options = ["None", "3", "7", "30", "90", "365"]
    smoothing_index = smoothing_options.index(st.session_state.smoothing_days) if st.session_state.smoothing_days in smoothing_options else 2
    smoothing_days = st.selectbox("Smoothing window", smoothing_options, index=smoothing_index)
    st.session_state.smoothing_days = smoothing_days
    
    start_date = st.date_input("Start date (optional)", value=st.session_state.start_date)
    st.session_state.start_date = start_date
    end_date = st.date_input("End date (optional)", value=st.session_state.end_date)
    st.session_state.end_date = end_date
    
    # Add warning about timeframe vs date filtering
    if start_date or end_date:
        st.info("💡 **Note**: Date filtering works on the data returned by the API. To get more data for filtering, consider using a longer timeframe (e.g., 'today 5-y' instead of 'today 12-m').")
    
    # Add general guidance about timeframe
    st.info("📊 **Timeframe Guide**:\n"
            "• **'all'**: Maximum data range (recommended for date filtering)\n"
            "• **1-m to 12-m**: Good for recent trends\n"
            "• **5-y to 20-y**: Good for historical analysis\n"
            "• **Note**: Use 'all' for maximum flexibility with date filtering")
    
    # Add button to set maximum timeframe for date filtering
    if start_date or end_date:
        if st.button("🔄 Set Maximum Timeframe for Date Filtering"):
            st.session_state.timeframe = "all"
            st.success("Set timeframe to maximum ('all') for better date filtering!")
            st.rerun()
    
    # Add note about SerpAPI limitations
    st.info("⚠️ **SerpAPI Limitation**: If you're still getting limited date ranges, SerpAPI may be overriding your timeframe. Use the '🔍 Test SerpAPI Timeframes' button to check what's actually supported.")

    st.markdown("---")
    st.subheader("Chart options")
    show_small_multiples = st.checkbox("Small multiples (one chart per term)", value=st.session_state.show_small_multiples)
    # Update session state
    st.session_state.show_small_multiples = show_small_multiples

    st.markdown("---")
    st.subheader("Advanced")
    # Use temp directory for cloud environments
    import tempfile
    import os
    default_cache = tempfile.gettempdir() if os.environ.get('STREAMLIT_SERVER_RUN_ON_SAVE') else ".cache"
    cache_dir = st.text_input("Cache directory", value=default_cache)
    sleep_ms = st.number_input("Request sleep (ms)", min_value=0, value=250)
    use_cache = st.checkbox("Use cache", value=True)
    show_debug = st.checkbox("Show debug logs", value=False)
    verbose_logs = st.checkbox("Verbose logging", value=False)

    run = st.button("Run")
    
    # Add cache management buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Cache"):
            st.session_state.data_loaded = False
            st.session_state.df_scaled = None
            st.session_state.pivot_scores = None
            st.session_state.scales = None
            st.session_state.terms = None
            st.session_state.current_params = None

            st.session_state.selected_chart_terms = None
            st.session_state.show_small_multiples = False
            st.session_state.smoothing_days = "7"
            st.session_state.start_date = None
            st.session_state.end_date = None
            st.session_state.timeframe = "all"
            st.success("Cache cleared! Click 'Run' to reload data.")
            st.rerun()
    
    with col2:
        if st.button("🔄 Force Reload"):
            st.session_state.data_loaded = False
            st.success("Forcing reload on next run...")
    
    # Add test button
    if st.button("🧪 Test Data Parsing"):
        st.subheader("Data Parsing Test")
        test_data_parsing()
    
    # Add YoY test button
    if st.button("📊 Test YoY Calculation"):
        st.subheader("YoY Calculation Test")
        test_yoy_calculation()
    
    # Add YoY chart data test button
    if st.button("📈 Test YoY Chart Data"):
        st.subheader("YoY Chart Data Test")
        if st.session_state.df_scaled is not None and st.session_state.terms:
            long_df = cached_melt_long(st.session_state.df_scaled)
            for term in st.session_state.terms[:3]:  # Test first 3 terms
                st.write(f"**Testing {term}:**")
                yt = yoy_table(long_df, term)
                if not yt.empty:
                    st.write(f"  Total YoY data: {len(yt)} rows")
                    st.write(f"  Date range: {yt['date'].min()} to {yt['date'].max()}")
                    
                    # Test the chart filtering
                    yt_valid = yt.dropna(subset=['pct_diff']).copy()
                    st.write(f"  Valid pct_diff data: {len(yt_valid)} rows")
                    
                    if not yt_valid.empty:
                        latest_date = yt_valid['date'].max()
                        five_years_ago = latest_date - pd.Timedelta(days=5*365)
                        earliest_date = yt_valid['date'].min()
                        start_date = max(five_years_ago, earliest_date)
                        
                        st.write(f"  Latest date: {latest_date}")
                        st.write(f"  5 years ago: {five_years_ago}")
                        st.write(f"  Filter start date: {start_date}")
                        
                        yt_filtered = yt_valid[yt_valid['date'] >= start_date].copy()
                        st.write(f"  After 5-year filter: {len(yt_filtered)} rows")
                        
                        if not yt_filtered.empty:
                            st.write(f"  Filtered date range: {yt_filtered['date'].min()} to {yt_filtered['date'].max()}")
                            
                            # Show recent data specifically
                            recent_filtered = yt_filtered[yt_filtered['date'] >= pd.Timestamp('2024-01-01')]
                            st.write(f"  2024+ in filtered data: {len(recent_filtered)} rows")
                            
                            if not recent_filtered.empty:
                                st.write("  Recent filtered data sample:")
                                st.dataframe(recent_filtered[['date', 'current', 'previous_year', 'pct_diff']].head(3))
                        else:
                            st.error("  No data after filtering!")
                else:
                    st.write("  No YoY data found")
                st.write("---")
        else:
            st.error("No data loaded. Please run the analysis first.")
    
    # Add raw YoY data viewer button
    if st.button("📋 View Raw YoY Data"):
        st.subheader("Raw YoY Data Viewer")
        if st.session_state.df_scaled is not None and st.session_state.terms:
            long_df = cached_melt_long(st.session_state.df_scaled)
            term_to_view = st.selectbox("Select term to view:", st.session_state.terms)
            
            if term_to_view:
                yt = yoy_table(long_df, term_to_view)
                if not yt.empty:
                    st.write(f"**Raw YoY data for {term_to_view}:**")
                    st.write(f"Total rows: {len(yt)}")
                    st.write(f"Date range: {yt['date'].min()} to {yt['date'].max()}")
                    
                    # Show recent data
                    recent_data = yt[yt['date'] >= pd.Timestamp('2024-01-01')]
                    if not recent_data.empty:
                        st.write(f"**2024+ data ({len(recent_data)} rows):**")
                        st.dataframe(recent_data)
                    else:
                        st.write("No 2024+ data found")
                    
                    # Show all data
                    st.write("**All YoY data:**")
                    st.dataframe(yt)
                else:
                    st.write("No YoY data found for this term")
        else:
            st.error("No data loaded. Please run the analysis first.")
    
    # Add SerpAPI timeframe test button
    if st.button("🔍 Test SerpAPI Timeframes"):
        if serpapi_key:
            test_serpapi_timeframes(serpapi_key)
        else:
            st.error("Please enter your SerpAPI key first.")
    
    # Add import test button
    if st.button("🔧 Test Imports"):
        st.subheader("Import Test")
        try:
            import sys
            st.write(f"Python path: {sys.path}")
            st.write(f"Current directory: {os.getcwd()}")
            
            try:
                import stitcher
                st.success("✅ Successfully imported stitcher module")
                st.write(f"stitcher module location: {stitcher.__file__}")
            except Exception as e:
                st.error(f"❌ Failed to import stitcher: {e}")
                
            try:
                from stitcher import stitch_terms
                st.success("✅ Successfully imported stitch_terms function")
            except Exception as e:
                st.error(f"❌ Failed to import stitch_terms: {e}")
                
            try:
                from stitcher import TrendsFetcher
                st.success("✅ Successfully imported TrendsFetcher class")
            except Exception as e:
                st.error(f"❌ Failed to import TrendsFetcher: {e}")
                
        except Exception as e:
            st.error(f"❌ Import test failed: {e}")
            st.code(traceback.format_exc())

def test_serpapi_timeframes(api_key: str):
    """Test different SerpAPI timeframes to see what data ranges they actually return"""
    import requests
    
    test_timeframes = [
        "all",           # Maximum data range
        "today 1-m",
        "today 3-m", 
        "today 12-m",
        "today 5-y",
        "today 10-y",
        "today 20-y"
    ]
    
    st.subheader("🔍 SerpAPI Timeframe Test")
    st.write("Testing what data ranges different timeframes actually return...")
    
    for tf in test_timeframes:
        try:
            params = {
                "engine": "google_trends",
                "q": "nike",  # Single term for testing
                "data_type": "TIMESERIES",
                "date": tf,
                "api_key": api_key,
            }
            
            r = requests.get("https://serpapi.com/search", params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                
                # Check what SerpAPI actually used
                actual_timeframe = "Unknown"
                if 'search_parameters' in data and 'date' in data['search_parameters']:
                    actual_timeframe = data['search_parameters']['date']
                
                # Get date range from data
                date_range = "No data"
                if 'interest_over_time' in data:
                    io_data = data['interest_over_time']
                    if isinstance(io_data, dict) and 'timeline_data' in io_data:
                        timeline = io_data['timeline_data']
                        if timeline and len(timeline) > 0:
                            first_date = timeline[0].get('date', 'Unknown')
                            last_date = timeline[-1].get('date', 'Unknown')
                            date_range = f"{first_date} to {last_date}"
                
                st.write(f"**{tf}**: SerpAPI used '{actual_timeframe}', Data: {date_range}")
                
            else:
                st.write(f"**{tf}**: HTTP {r.status_code}")
                
        except Exception as e:
            st.write(f"**{tf}**: Error - {str(e)}")

def test_yoy_calculation():
    """Test function to verify YoY calculation"""
    import pandas as pd
    from datetime import date, timedelta
    
    # Create test data with known YoY relationships
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    test_data = []
    
    for i, d in enumerate(dates):
        # Create some test values with a known pattern
        base_value = 50 + 10 * np.sin(i / 30)  # Seasonal pattern
        test_data.append({
            "date": d,
            "term": "test_term",
            "value": base_value
        })
    
    df = pd.DataFrame(test_data)
    st.write("Test data sample:")
    st.write(df.head(10))
    
    # Test YoY calculation
    yt = yoy_table(df, "test_term")
    st.write("YoY calculation result:")
    st.write(yt.head(10))
    
    if not yt.empty:
        st.write(f"YoY data shape: {yt.shape}")
        st.write(f"Non-null previous year values: {yt['previous_year'].notna().sum()}")
        st.write(f"Non-null pct_diff values: {yt['pct_diff'].notna().sum()}")
    else:
        st.error("YoY calculation returned empty result")

def test_data_parsing():
    """Test function to verify data parsing"""
    try:
        import pandas as pd
        from datetime import date
        
        # Test data similar to what you're getting from the API
        test_data = [
            {"date": "datetime.date(2025, 8, 24)", "term": "nike", "value": 74},
            {"date": "datetime.date(2025, 8, 24)", "term": "adidas", "value": 45},
            {"date": "datetime.date(2025, 8, 25)", "term": "nike", "value": 71},
            {"date": "datetime.date(2025, 8, 25)", "term": "adidas", "value": 43},
        ]
        
        # Test Unix timestamp data (like what you're getting now)
        unix_test_data = [
            {"date": "1723939200", "term": "nike", "value": 74},
            {"date": "1723939200", "term": "adidas", "value": 45},
            {"date": "1724544000", "term": "nike", "value": 71},
            {"date": "1724544000", "term": "adidas", "value": 43},
        ]
        
        # Test the date parsing
        try:
            from stitcher import TrendsFetcher
            st.success("✅ Successfully imported TrendsFetcher")
            
            st.subheader("Testing datetime.date string parsing:")
            for item in test_data:
                parsed_date = TrendsFetcher._coerce_date(item["date"])
                st.write(f"Original: {item['date']} -> Parsed: {parsed_date}")
            
            st.subheader("Testing Unix timestamp parsing:")
            for item in unix_test_data:
                parsed_date = TrendsFetcher._coerce_date(item["date"])
                st.write(f"Original: {item['date']} -> Parsed: {parsed_date}")
                
        except Exception as e:
            st.error(f"❌ Failed to import or use TrendsFetcher: {e}")
            st.code(traceback.format_exc())
            return None, None
        
        # Test DataFrame creation with Unix timestamps
        st.subheader("Testing DataFrame with Unix timestamps:")
        df = pd.DataFrame(unix_test_data)
        st.write("Original DataFrame:")
        st.write(df)
        
        # Test date conversion
        df["date"] = pd.to_datetime(df["date"]).dt.date
        st.write("After date conversion:")
        st.write(df)
        
        # Test pivot
        wide = df.pivot_table(index="date", columns="term", values="value", aggfunc="mean")
        st.write("Pivoted DataFrame:")
        st.write(wide)
        
        return df, wide
        
    except Exception as e:
        st.error(f"❌ Test failed: {e}")
        st.code(traceback.format_exc())
        return None, None

def test_api_connection(api_key: str) -> bool:
    """Simple test to verify API connectivity"""
    try:
        import requests
        test_params = {
            "engine": "google_trends",
            "q": "test",
            "data_type": "TIMESERIES",
            "time": "today 1-m",
            "api_key": api_key,
        }
        r = requests.get("https://serpapi.com/search", params=test_params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return not bool(data.get("error") or data.get("error_message"))
        return False
    except Exception:
        return False

def infer_step_days(dates: pd.Series) -> float:
    d = pd.to_datetime(dates).sort_values().drop_duplicates()
    if len(d) < 2:
        return 1.0
    diffs = d.diff().dropna().dt.days.to_numpy()
    return float(np.median(diffs)) if len(diffs) else 1.0

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

def apply_smoothing(df_scaled: pd.DataFrame, smoothing_days: str) -> pd.DataFrame:
    if smoothing_days == "None":
        return df_scaled
    
    # Add debugging info
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Applying smoothing with window: {smoothing_days} days")
    logger.debug(f"Input data shape: {df_scaled.shape}")
    logger.debug(f"Input date range: {df_scaled['date'].min()} to {df_scaled['date'].max()}")
    
    k_days = int(smoothing_days)
    out = df_scaled.copy()
    out["date"] = pd.to_datetime(out["date"])
    step = infer_step_days(out["date"])
    periods = max(1, int(round(k_days / step)))
    
    logger.debug(f"Calculated step: {step} days, periods: {periods}")
    
    wide = out.set_index("date").sort_index()
    
    # Improved rolling window with better edge handling
    for col in [c for c in wide.columns if c != "date"]:
        # Use center=True for symmetric windows and handle NaN values properly
        # Require at least 50% of the window to have data for a valid calculation
        min_periods = max(1, int(periods * 0.5))  # At least 50% of window must have data
        original_values = wide[col].copy()
        wide[col] = wide[col].rolling(
            window=periods, 
            min_periods=min_periods,
            center=True  # Center the window for symmetric smoothing
        ).mean()
        
        # Log smoothing statistics
        nan_before = original_values.isna().sum()
        nan_after = wide[col].isna().sum()
        logger.debug(f"Column {col}: NaN before={nan_before}, after={nan_after}, periods={periods}, min_periods={min_periods}")
    
    result = wide.reset_index()
    logger.debug(f"Smoothed data shape: {result.shape}")
    return result

@st.cache_data
def cached_apply_smoothing(df_scaled: pd.DataFrame, smoothing_days: str) -> pd.DataFrame:
    """Cached version of smoothing"""
    return apply_smoothing(df_scaled, smoothing_days)

def melt_long(df_scaled: pd.DataFrame) -> pd.DataFrame:
    long = df_scaled.melt(id_vars=["date"], var_name="term", value_name="value")
    long["date"] = pd.to_datetime(long["date"])
    return long

@st.cache_data
def cached_melt_long(df_scaled: pd.DataFrame) -> pd.DataFrame:
    """Cached version of melt operation"""
    return melt_long(df_scaled)

def yoy_table(long_df: pd.DataFrame, term: str) -> pd.DataFrame:
    """
    Calculate year-over-year comparison for a given term.
    Only matches exact previous year dates (365 days earlier).
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
    
    # Create a lookup table for previous year data
    # For each date in the data, calculate what the previous year date should be
    g["prev_year_date"] = g["date"] - pd.Timedelta(days=365)
    
    # Create a mapping from current dates to values (for previous year lookup)
    date_value_map = g.set_index("date")["value"].to_dict()
    
    # For each row, look up the value from exactly 365 days ago
    g["prior_value"] = g["prev_year_date"].map(date_value_map)
    
    # Debug: Show some examples of the mapping
    logger.debug(f"Sample date mappings:")
    for i, row in g.head(5).iterrows():
        logger.debug(f"  {row['date']} -> prev_year_date: {row['prev_year_date']} -> value: {row['prior_value']}")
    
    # Calculate differences
    g["abs_diff"] = g["value"] - g["prior_value"]
    g["pct_diff"] = np.where(
        g["prior_value"] > 0,
        (g["abs_diff"] / g["prior_value"]) * 100.0,
        np.nan
    )
    
    # Debug: Show summary statistics
    valid_prev_year = g["prior_value"].notna().sum()
    logger.debug(f"Valid previous year matches: {valid_prev_year} out of {len(g)}")
    
    # Return the result with proper column names
    result = g[["date", "value", "prior_value", "abs_diff", "pct_diff"]].rename(
        columns={"value": "current", "prior_value": "previous_year"}
    )
    
    return result

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
    """Cached version of line chart creation"""
    if data.empty:
        return None
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
    Defaults to showing past 5 years of data.
    """
    # Get YoY data for the term
    yt = yoy_table(long_df, term)
    
    if yt.empty:
        return None
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"YoY chart for {term}:")
    logger.debug(f"  Total rows: {len(yt)}")
    logger.debug(f"  Date range: {yt['date'].min()} to {yt['date'].max()}")
    logger.debug(f"  Rows with pct_diff: {yt['pct_diff'].notna().sum()}")
    logger.debug(f"  Rows with abs_diff: {yt['abs_diff'].notna().sum()}")
    
    # Filter to only rows with valid YoY data
    yt_valid = yt.dropna(subset=['pct_diff']).copy()
    
    if yt_valid.empty:
        logger.debug(f"  No valid pct_diff data found")
        return None
    
    logger.debug(f"  Valid rows: {len(yt_valid)}")
    logger.debug(f"  Valid date range: {yt_valid['date'].min()} to {yt_valid['date'].max()}")
    
    # Filter to past 5 years (or less if not enough data)
    latest_date = yt_valid['date'].max()
    five_years_ago = latest_date - pd.Timedelta(days=5*365)
    
    # Get the actual earliest date in the data
    earliest_date = yt_valid['date'].min()
    
    # Use the later of: 5 years ago or earliest available data
    start_date = max(five_years_ago, earliest_date)
    
    logger.debug(f"  Latest date: {latest_date}")
    logger.debug(f"  Five years ago: {five_years_ago}")
    logger.debug(f"  Earliest available: {earliest_date}")
    logger.debug(f"  Start date (filter): {start_date}")
    
    # Filter to the selected date range
    yt_filtered = yt_valid[yt_valid['date'] >= start_date].copy()
    
    if yt_filtered.empty:
        logger.debug(f"  No data after filtering")
        return None
    
    logger.debug(f"  Filtered rows: {len(yt_filtered)}")
    logger.debug(f"  Filtered date range: {yt_filtered['date'].min()} to {yt_filtered['date'].max()}")
    
    # Extract month and year for visualization
    yt_filtered['month'] = yt_filtered['date'].dt.month
    yt_filtered['month_name'] = yt_filtered['date'].dt.strftime('%b')  # Jan, Feb, etc.
    yt_filtered['year'] = yt_filtered['date'].dt.year
    
    # Create the chart
    chart = (
        alt.Chart(yt_filtered)
        .mark_line(point=True)
        .encode(
            x=alt.X('month_name:N', title='Month', sort=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
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
            title=f'{term} - YoY % Difference by Month ({start_date.strftime("%Y")}-{latest_date.strftime("%Y")})',
            height=300,
            width=400
        )
        .interactive()
    )
    
    return chart

@st.cache_data
def create_yoy_absolute_chart(long_df: pd.DataFrame, term: str) -> alt.Chart:
    """
    Create a YoY chart showing absolute differences with month on X-axis.
    Defaults to showing past 5 years of data.
    """
    # Get YoY data for the term
    yt = yoy_table(long_df, term)
    
    if yt.empty or yt['abs_diff'].isna().all():
        return None
    
    # Filter to only rows with valid YoY data
    yt_valid = yt.dropna(subset=['abs_diff']).copy()
    
    if yt_valid.empty:
        return None
    
    # Filter to past 5 years (or less if not enough data)
    latest_date = yt_valid['date'].max()
    five_years_ago = latest_date - pd.Timedelta(days=5*365)
    
    # Get the actual earliest date in the data
    earliest_date = yt_valid['date'].min()
    
    # Use the later of: 5 years ago or earliest available data
    start_date = max(five_years_ago, earliest_date)
    
    # Filter to the selected date range
    yt_filtered = yt_valid[yt_valid['date'] >= start_date].copy()
    
    if yt_filtered.empty:
        return None
    
    # Extract month and year for visualization
    yt_filtered['month'] = yt_filtered['date'].dt.month
    yt_filtered['month_name'] = yt_filtered['date'].dt.strftime('%b')  # Jan, Feb, etc.
    yt_filtered['year'] = yt_filtered['date'].dt.year
    
    # Create the chart
    chart = (
        alt.Chart(yt_filtered)
        .mark_line(point=True)
        .encode(
            x=alt.X('month_name:N', title='Month', sort=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
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
            title=f'{term} - YoY Absolute Difference by Month ({start_date.strftime("%Y")}-{latest_date.strftime("%Y")})',
            height=300,
            width=400
        )
        .interactive()
    )
    
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
    
    # Create the chart
    chart = (
        alt.Chart(yt_valid)
        .mark_line(point=True)
        .encode(
            x=alt.X('month_name:N', title='Month', sort=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
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

if run:
    if not serpapi_key:
        st.error("Please enter your SerpAPI key.")
        st.stop()

    # Test API connection first
    with st.spinner("Testing API connection..."):
        if not test_api_connection(serpapi_key):
            st.error("❌ API connection test failed. Please check your SerpAPI key and internet connection.")
            st.stop()
        else:
            st.success("✅ API connection successful!")

    terms = [t.strip() for t in terms_text.splitlines() if t.strip()]
    if len(terms) < 2:
        st.error("Enter at least two terms.")
        st.stop()

    current_params = get_current_params(serpapi_key, terms_text, geo, timeframe, group_size, cache_dir, sleep_ms, use_cache)
    if should_reload_data(current_params):
        with st.spinner("Fetching, stitching, and preparing views..."):
            try:
                # Add debug info
                if show_debug:
                    st.info(f"Starting with {len(terms)} terms: {terms}")
                    st.info(f"Cache dir: {cache_dir}")
                    st.info(f"Using cache: {use_cache}")
                    st.info(f"API timeframe: {timeframe}")
                    st.info(f"Date filters: {start_date} to {end_date}")
                    st.info(f"Note: API timeframe ({timeframe}) may limit available data regardless of date filters")
                
                df_scaled, pivot_scores, scales = stitch_terms(
                    serpapi_key=serpapi_key,
                    terms=terms,
                    geo=geo,
                    timeframe=timeframe,
                    group_size=group_size,
                    cache_dir=cache_dir,
                    sleep_ms=int(sleep_ms),
                    verbose=verbose_logs,
                    use_cache=use_cache,
                    debug=show_debug,
                )

                if show_debug:
                    st.success(f"Successfully fetched data: {df_scaled.shape}")
                    st.info(f"Data columns: {list(df_scaled.columns)}")
                    st.info(f"Date range: {df_scaled['date'].min()} to {df_scaled['date'].max()}")
                    
                    # Show raw API response for debugging
                    st.subheader("🔍 Raw API Response Debug")
                    try:
                        from stitcher import TrendsFetcher
                        fetcher = TrendsFetcher(
                            serpapi_key=serpapi_key,
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
                    except Exception as debug_e:
                        st.error(f"Debug API call failed: {debug_e}")
                        st.code(traceback.format_exc())

                # Store in session state
                st.session_state.df_scaled = df_scaled
                st.session_state.pivot_scores = pivot_scores
                st.session_state.scales = scales
                st.session_state.terms = terms
                st.session_state.data_loaded = True
                st.session_state.current_params = current_params
                
            except Exception as e:
                st.error(f"Error during data fetching: {str(e)}")
                if show_debug:
                    st.write("Full traceback:")
                    st.code(traceback.format_exc())
                st.stop()
    else:
        st.info("✅ Using cached data. Change parameters to reload.")
        df_scaled = st.session_state.df_scaled
        pivot_scores = st.session_state.pivot_scores
        scales = st.session_state.scales
        terms = st.session_state.terms

    # Apply filters to the data
    if show_debug:
        st.subheader("🔍 Date Filtering Debug")
        st.write(f"Original data shape: {df_scaled.shape}")
        st.write(f"Original date range: {df_scaled['date'].min()} to {df_scaled['date'].max()}")
        st.write(f"Start date filter: {start_date}")
        st.write(f"End date filter: {end_date}")
    
    df_scaled = cached_filter_date_range(df_scaled, start_date, end_date)
    
    if show_debug:
        st.write(f"After date filtering shape: {df_scaled.shape}")
        if not df_scaled.empty:
            st.write(f"After date filtering range: {df_scaled['date'].min()} to {df_scaled['date'].max()}")
    
    df_scaled = cached_apply_smoothing(df_scaled, smoothing_days)

    st.subheader("Comparable Time Series (max=100 across ALL terms)")
    st.caption("Consensus scaling + optional smoothing + date filtering.")
    st.dataframe(df_scaled.head(20))
    st.download_button(
        "Download full timeseries CSV",
        df_scaled.to_csv(index=False).encode("utf-8"),
        file_name="trends_stitched_scaled.csv",
        mime="text/csv"
    )

    long_df = cached_melt_long(df_scaled)

    # Validate data before charting
    if long_df.empty:
        st.error("No data available for charting. Please check your API key and terms.")
        st.stop()
    
    if long_df['value'].isna().all():
        st.error("All values are NaN. This might indicate an API response parsing issue.")
        st.stop()

    st.markdown("### Chart selection")
    default_terms = terms[:min(5, len(terms))]
    
    # Use session state for selected terms, with fallback to defaults
    if st.session_state.selected_chart_terms is None or not all(term in terms for term in st.session_state.selected_chart_terms):
        st.session_state.selected_chart_terms = default_terms
    
    selected_terms = st.multiselect("Terms to chart", options=terms, default=st.session_state.selected_chart_terms)
    if not selected_terms:
        selected_terms = default_terms
    
    # Update session state with current selection
    st.session_state.selected_chart_terms = selected_terms
    


    if show_debug:
        st.info(f"Chart data shape: {long_df.shape}")
        st.info(f"Available terms: {long_df['term'].unique()}")
        st.info(f"Value range: {long_df['value'].min()} to {long_df['value'].max()}")

    if show_small_multiples:
        chart = create_small_multiples(long_df, terms)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data to plot.")
    else:
        chart = create_line_chart(long_df, selected_terms, "All Terms (selected)")
        if chart:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data to plot for the selected terms.")

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
        st.info(f"📊 **YoY Data Available**: {len(available_terms)}/{len(terms)} terms have YoY data from {earliest_date} onwards")
    else:
        st.warning("⚠️ **No YoY data available** - insufficient historical data for year-over-year comparison")

    # Debug YoY data if requested
    if show_debug:
        st.subheader("🔍 YoY Debug Info")
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
                else:
                    st.write(f"  - No 2024+ data found")
                
                # Show sample of recent data
                if not recent_data.empty:
                    st.write("  - Recent data sample:")
                    st.dataframe(recent_data[['date', 'current', 'previous_year', 'pct_diff']].head(5))

    # Create YoY charts for all terms
    st.subheader("YoY % Difference by Month")
    st.caption("Month on X-axis, YoY % difference on Y-axis. Each line represents a different year. Defaults to past 5 years of data.")
    
    # Add toggle for showing all data vs filtered data
    show_all_data = st.checkbox("Show all data (not just 5 years)", value=False)
    
    # Calculate how many charts per row based on number of terms
    charts_per_row = min(3, len(terms))
    
    for i in range(0, len(terms), charts_per_row):
        cols = st.columns(charts_per_row)
        for j, term in enumerate(terms[i:i+charts_per_row]):
            with cols[j]:
                if show_all_data:
                    chart = create_yoy_monthly_chart_all_data(long_df, term)
                else:
                    chart = create_yoy_monthly_chart(long_df, term)
                if chart:
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info(f"No YoY data for {term}")

    # Create YoY absolute difference charts for all terms
    st.subheader("YoY Absolute Difference by Month")
    st.caption("Month on X-axis, YoY absolute difference on Y-axis. Each line represents a different year. Defaults to past 5 years of data.")
    
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
            
            # Create filtered version (past 5 years)
            yt_valid = yt.dropna(subset=['pct_diff']).copy()
            if not yt_valid.empty:
                latest_date = yt_valid['date'].max()
                five_years_ago = latest_date - pd.Timedelta(days=5*365)
                earliest_date = yt_valid['date'].min()
                start_date = max(five_years_ago, earliest_date)
                yt_filtered = yt_valid[yt_valid['date'] >= start_date].copy()
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
                mime="text/csv"
            )
            
            # Show sample of combined data
            st.write("**Full YoY Data Sample:**")
            st.dataframe(combined_df.head(20))
    
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
                st.info(f"📅 **5-Year YoY Data Range**: {date_range_start.strftime('%Y-%m-%d')} to {date_range_end.strftime('%Y-%m-%d')}")
            
            st.download_button(
                "Download YoY Data (Past 5 Years)",
                combined_filtered_df.to_csv(index=False).encode("utf-8"),
                file_name="yoy_data_past_5_years.csv",
                mime="text/csv"
            )
            
            # Show sample of filtered data
            st.write("**5-Year YoY Data Sample:**")
            st.dataframe(combined_filtered_df.head(20))

    st.markdown("---")
    st.subheader("Explainability")
    st.caption("Pivot scores are *not* used for scaling; they help sanity-check anchors if you need one.")
    st.dataframe(pivot_scores)
    st.download_button(
        "Download pivot scores CSV",
        pivot_scores.to_csv(index=False).encode("utf-8"),
        file_name="pivot_scores.csv",
        mime="text/csv"
    )

    st.subheader("Per-Term Consensus Scale Factors")
    scales_df = pd.DataFrame({"term": scales.index, "scale": scales.values}).sort_values("scale", ascending=False)
    st.dataframe(scales_df)
    st.download_button(
        "Download scale factors CSV",
        scales_df.to_csv(index=False).encode("utf-8"),
        file_name="scale_factors.csv",
        mime="text/csv"
    )
