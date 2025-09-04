"""
A Streamlit app for visualizing **weekly** YouTube brand metrics (mentions, API hits, views, likes, comments).
Works with the combined weekly snapshot files:
  - brand_weekly_for_streamlit.csv (from pivot_brand_counts.py)
  - youtube_brand_channel_weekly_snapshot_ALL.csv (weekly per-channel breakdown)
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_CANDIDATES = [
    Path("../snapshot_data"),
    Path("../data"),
    Path("data"),
]
BRAND_WEEKLY_FILE = "brand_weekly_for_streamlit.csv"
CHANNEL_WEEKLY_FILE = "youtube_brand_channel_weekly_snapshot_ALL.csv"

def find_file(fname: str) -> Path:
    for base in DATA_CANDIDATES:
        p = base / fname
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {fname} in any of: {[str(b) for b in DATA_CANDIDATES]}")

brand_path = find_file(BRAND_WEEKLY_FILE)
df = pd.read_csv(brand_path, encoding='utf-8')
if 'week_start' not in df.columns or 'keyword' not in df.columns:
    st.stop()
df['week_start'] = pd.to_datetime(df['week_start'])
if 'week_end' in df.columns:
    df['week_end'] = pd.to_datetime(df['week_end'])
available_metrics = [c for c in [
    'weekly_video_mentions', 'weekly_api_hits', 'weekly_views', 'weekly_likes', 'weekly_comments'
] if c in df.columns]
if not available_metrics:
    st.stop()

min_d, max_d = df['week_start'].min().date(), df['week_start'].max().date()
st.set_page_config(page_title="YouTube Brands – Weekly Snapshot", layout='wide')
st.title("YouTube Brands – Weekly Snapshot (Weekly granularity)")
metric = st.selectbox("Metric", available_metrics, index=0)
wk_range = st.date_input("Week range (by week_start)", value=(min_d, max_d))
if isinstance(wk_range, tuple):
    start_date, end_date = wk_range
else:
    start_date, end_date = min_d, wk_range
mask = (df['week_start'].dt.date >= start_date) & (df['week_start'].dt.date <= end_date)
week = df.loc[mask].copy()

brand_cols = sorted(week['keyword'].unique())
totals = (week.groupby('keyword')[metric].sum().sort_values(ascending=False))
default_brands = list(totals.head(5).index)
selected = st.multiselect("Pick brands to plot", options=brand_cols, default=default_brands)
if selected:
    long_df = week[week['keyword'].isin(selected)][['week_start','keyword',metric]].rename(columns={'keyword':'brand','week_start':'x','%s'%metric:'y'})
    fig = px.line(long_df, x='x', y='y', color='brand', markers=True,
                  title=f'Weekly {metric} for selected brands')
    fig.update_layout(xaxis_title='Week start (UTC)', yaxis_title=metric.replace('_',' ').title(), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Show data table"):
        st.dataframe(long_df.sort_values(["brand","x"]).reset_index(drop=True))
else:
    st.info("Select at least one brand to display the plot.")

if {'weekly_api_hits','weekly_video_mentions'}.issubset(set(week.columns)) and selected:
    diff = (week[week['keyword'].isin(selected)]
              .pivot_table(index='week_start', columns='keyword', values='weekly_api_hits', aggfunc='sum') -
            week[week['keyword'].isin(selected)]
              .pivot_table(index='week_start', columns='keyword', values='weekly_video_mentions', aggfunc='sum'))
    st.subheader("API Hits minus Text-Confirmed (diagnostic)")
    st.line_chart(diff.fillna(0))

try:
    ch_path = find_file(CHANNEL_WEEKLY_FILE)
    chw = pd.read_csv(ch_path, encoding='utf-8')
except Exception as e:
    chw = None

if selected and chw is not None:
    need_cols = {"week_start", "keyword", "channel", "subscribers", "views", "likeCount", "commentCount"}
    if need_cols.issubset(set(chw.columns)):
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        chw["week_start"] = pd.to_datetime(chw["week_start"])
        if "week_end" in chw.columns:
            chw["week_end"] = pd.to_datetime(chw["week_end"])
        else:
            chw["week_end"] = chw["week_start"] + pd.Timedelta(days=6)

        overlaps = (chw["week_start"] <= end_dt) & (chw["week_end"] >= start_dt)
        mask = overlaps & (chw["keyword"].isin(selected))

        sub = chw.loc[mask, [
            "keyword", "channel", "subscribers", "views", "likeCount", "commentCount"
        ]].copy()

        for col in ["subscribers", "views", "likeCount", "commentCount"]:
            sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0).astype(int)

        if not sub.empty:
            agg = (sub.groupby(["keyword", "channel"], as_index=False)
                     .agg({
                         "subscribers": "max",
                         "views": "sum",
                         "likeCount": "sum",
                         "commentCount": "sum",
                     }))
            agg["engagement"] = agg["views"].fillna(0) + agg["likeCount"].fillna(0) + agg["commentCount"].fillna(0)

            top_reach = (agg.sort_values(["keyword", "subscribers"], ascending=[True, False])
                            .groupby("keyword").head(3).reset_index(drop=True))
            top_reach = top_reach.rename(columns={
                "subscribers": "reach (subscribers)",
                "views": "total views",
                "likeCount": "total likes",
                "commentCount": "total comments",
            })
            st.subheader("Top channels for selected brands (by reach: subscribers)")
            st.dataframe(top_reach, use_container_width=True)

            top_eng = (agg.sort_values(["keyword", "engagement"], ascending=[True, False])
                         .groupby("keyword").head(3).reset_index(drop=True))
            top_eng = top_eng.rename(columns={
                "engagement": "total engagement (views+likes+comments)",
                "views": "total views",
                "likeCount": "total likes",
                "commentCount": "total comments",
            })
            st.subheader("Top channels for selected brands (by engagement: views + likes + comments)")
            st.dataframe(top_eng[["keyword", "channel", "total engagement (views+likes+comments)", "total views", "total likes", "total comments"]], use_container_width=True)
        else:
            try:
                min_w = chw["week_start"].min()
                max_w = chw["week_end"].max() if "week_end" in chw.columns else (chw["week_start"].max() + pd.Timedelta(days=6))
                brands_in_file = ", ".join(sorted(map(str, set(chw["keyword"].unique()))))
                st.info(f"No weekly channel data after filtering. Available weeks: {min_w.date()} → {max_w.date()}. Brands present: {brands_in_file}.")
            except Exception:
                st.info("No weekly channel data available for the selected window/brands.")
    else:
        st.info("Weekly channel summary missing required columns.")
elif selected:
    st.caption("Tip: Run your YouTube collector so it produces 'youtube_brand_channel_weekly_snapshot_ALL.csv' to show Reach & Engagement tables.")

st.markdown("---")
show_top10 = st.checkbox("Show Top 10 brands (by total over selected weeks)", value=False)
if show_top10 and not week.empty:
    totals_sel = (week.groupby('keyword')[metric].sum().sort_values(ascending=False).head(10))
    top10 = list(totals_sel.index)
    top_long = week[week['keyword'].isin(top10)][['week_start','keyword',metric]].rename(columns={'keyword':'brand'})
    fig2 = px.line(top_long, x='week_start', y=metric, color='brand', markers=True,
                   title=f'Top 10 brands over selected weeks – {metric}')
    fig2.update_layout(xaxis_title='Week start (UTC)', yaxis_title=metric.replace('_',' ').title())
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(totals_sel.rename('total').reset_index().rename(columns={'keyword':'brand'}), use_container_width=True)
