from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from stitcher import stitch_terms

st.set_page_config(page_title="Trends Stitcher", layout="wide")
st.title("Google Trends: Auto-Stitched Comparable Scale")

# Sidebar
with st.sidebar:
    st.subheader("Data")
    serpapi_key = st.text_input("SerpAPI API Key", type="password", value="")
    terms_text = st.text_area("Terms (one per line)", "nike\nadidas\npuma\nnew balance\nasics")
    geo = st.text_input("Geo (e.g. GB, US — optional)", value="")
    timeframe = st.text_input("Timeframe", value="today 5-y")
    group_size = st.slider("Batch size (max 5)", 2, 5, 5)

    st.markdown("---")
    st.subheader("Smoothing & Range")
    smoothing_days = st.selectbox("Smoothing window", ["None", "3", "7", "30", "90", "365"], index=2)
    start_date = st.date_input("Start date (optional)", value=None)
    end_date = st.date_input("End date (optional)", value=None)

    st.markdown("---")
    st.subheader("Chart options")
    show_small_multiples = st.checkbox("Small multiples (one chart per term)", value=False)

    st.markdown("---")
    st.subheader("Advanced")
    cache_dir = st.text_input("Cache directory", value=".cache")
    sleep_ms = st.number_input("Request sleep (ms)", min_value=0, value=250)

    run = st.button("Run")

def infer_step_days(dates: pd.Series) -> float:
    d = pd.to_datetime(dates).sort_values().drop_duplicates()
    if len(d) < 2:
        return 1.0
    diffs = d.diff().dropna().dt.days.to_numpy()
    return float(np.median(diffs)) if len(diffs) else 1.0

def apply_smoothing(df_scaled: pd.DataFrame, smoothing_days: str) -> pd.DataFrame:
    if smoothing_days == "None":
        return df_scaled
    k_days = int(smoothing_days)
    out = df_scaled.copy()
    out["date"] = pd.to_datetime(out["date"])
    step = infer_step_days(out["date"])
    periods = max(1, int(round(k_days / step)))
    wide = out.set_index("date").sort_index()
    for col in [c for c in wide.columns if c != "date"]:
        wide[col] = wide[col].rolling(window=periods, min_periods=max(1, periods // 2)).mean()
    return wide.reset_index()

def filter_date_range(df_scaled: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    df = df_scaled.copy()
    df["date"] = pd.to_datetime(df["date"])
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    return df

def melt_long(df_scaled: pd.DataFrame) -> pd.DataFrame:
    long = df_scaled.melt(id_vars=["date"], var_name="term", value_name="value")
    long["date"] = pd.to_datetime(long["date"])
    return long

def yoy_table(long_df: pd.DataFrame, term: str) -> pd.DataFrame:
    g = long_df[long_df["term"] == term].sort_values("date").copy()
    g["prior_date"] = g["date"] - pd.Timedelta(days=365)
    prior = g[["date", "value"]].copy()
    prior.columns = ["prior_date", "prior_value"]
    merged = pd.merge(g, prior, on="prior_date", how="left")
    merged["abs_diff"] = merged["value"] - merged["prior_value"]
    merged["pct_diff"] = np.where(
        merged["prior_value"] > 0,
        (merged["abs_diff"] / merged["prior_value"]) * 100.0,
        np.nan
    )
    return merged[["date", "value", "prior_value", "abs_diff", "pct_diff"]].rename(
        columns={"value": "current", "prior_value": "previous_year"}
    )

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

if run:
    if not serpapi_key:
        st.error("Please enter your SerpAPI key.")
        st.stop()

    terms = [t.strip() for t in terms_text.splitlines() if t.strip()]
    if len(terms) < 2:
        st.error("Enter at least two terms.")
        st.stop()

    with st.spinner("Fetching, stitching, and preparing views..."):
        df_scaled, pivot_scores, scales = stitch_terms(
            serpapi_key=serpapi_key,
            terms=terms,
            geo=geo,
            timeframe=timeframe,
            group_size=group_size,
            cache_dir=cache_dir,
            sleep_ms=int(sleep_ms),
        )

        df_scaled = filter_date_range(df_scaled, start_date, end_date)
        df_scaled = apply_smoothing(df_scaled, smoothing_days)

    st.subheader("Comparable Time Series (max=100 across ALL terms)")
    st.caption("Consensus scaling + optional smoothing + date filtering.")
    st.dataframe(df_scaled.head(20))
    st.download_button(
        "Download full timeseries CSV",
        df_scaled.to_csv(index=False).encode("utf-8"),
        file_name="trends_stitched_scaled.csv",
        mime="text/csv"
    )

    long_df = melt_long(df_scaled)

    st.markdown("### Chart selection")
    default_terms = terms[:min(5, len(terms))]
    selected_terms = st.multiselect("Terms to chart", options=terms, default=default_terms)
    if not selected_terms:
        selected_terms = default_terms

    if show_small_multiples:
        small_multiples(long_df, terms)
    else:
        line_chart_multi(long_df, selected_terms, "All Terms (selected)")

    st.markdown("---")
    st.subheader("Year-on-Year (YoY)")

    yoy_term = st.selectbox("Pick a term for YoY analysis", options=terms, index=0)
    yt = yoy_table(long_df, yoy_term)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{yoy_term} — Current vs Prior Year**")
        plot_df = yt.dropna(subset=["current", "previous_year"]).copy()
        curr_df = plot_df[["date", "current"]].assign(series="Current").rename(columns={"current":"value"})
        prev_df = plot_df[["date", "previous_year"]].assign(series="Previous Year").rename(columns={"previous_year":"value"})
        comb = pd.concat([curr_df, prev_df], ignore_index=True)
        ch = (
            alt.Chart(comb)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value:Q", title="Index"),
                color=alt.Color("series:N", title=""),
                tooltip=["date:T", "series:N", alt.Tooltip("value:Q", format=".2f")]
            )
            .properties(height=360)
            .interactive()
        )
        st.altair_chart(ch, use_container_width=True)

    with col2:
        st.markdown(f"**{yoy_term} — % Diff vs Prior Year**")
        pct = yt.dropna(subset=["pct_diff"]).copy()
        ch2 = (
            alt.Chart(pct)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("pct_diff:Q", title="% Diff"),
                tooltip=["date:T", alt.Tooltip("pct_diff:Q", format=".2f")]
            )
            .properties(height=360)
            .interactive()
        )
        st.altair_chart(ch2, use_container_width=True)

    st.markdown("**YoY Table**")
    st.dataframe(yt.head(50))
    st.download_button(
        f"Download YoY CSV ({yoy_term})",
        yt.to_csv(index=False).encode("utf-8"),
        file_name=f"yoy_{yoy_term}.csv",
        mime="text/csv"
    )

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
