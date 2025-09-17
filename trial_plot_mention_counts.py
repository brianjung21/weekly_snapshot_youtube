"""
A Streamlit app for visualizing **weekly** YouTube brand metrics (mentions, API hits, views, likes, comments).
Works with the combined weekly snapshot files:
  - youtube_brand_weekly_snapshot_ALL.csv (weekly brand metrics)
  - youtube_brand_channel_weekly_snapshot_ALL.csv (weekly per-channel breakdown)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_CANDIDATES = [
    Path("processes/snapshot_data"),
]
BRAND_WEEKLY_FILE = "youtube_brand_weekly_snapshot_ALL.csv"
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
# include averages if present in newer snapshots
available_metrics = [c for c in [
    'weekly_video_mentions', 'weekly_api_hits', 'weekly_views', 'weekly_likes', 'weekly_comments',
    'avg_views_per_video', 'avg_likes_per_video', 'avg_comments_per_video'
] if c in df.columns]
if not available_metrics:
    st.stop()

min_d, max_d = df['week_start'].min().date(), df['week_start'].max().date()
st.set_page_config(page_title="YouTube Brands – Weekly Snapshot", layout='wide')
st.title("YouTube Brands – Weekly Snapshot")
metric = st.selectbox("Metric", available_metrics, index=0)
wk_range = st.date_input("Week range (by week_start)", value=(min_d, max_d))
if isinstance(wk_range, tuple):
    start_date, end_date = wk_range
else:
    start_date, end_date = min_d, wk_range
mask = (df['week_start'].dt.date >= start_date) & (df['week_start'].dt.date <= end_date)
week = df.loc[mask].copy()

# Freshness banner: when were details fetched for rows in view?
if 'as_of_details_fetched_utc' in week.columns:
    s = pd.to_datetime(week['as_of_details_fetched_utc'], errors='coerce')
    if not s.dropna().empty:
        st.caption(f"as-of window for engagement fetch: {s.min()} → {s.max()} (UTC)")

brand_cols = sorted(week['keyword'].unique())
totals = (week.groupby('keyword')[metric].sum().sort_values(ascending=False))
default_brands = list(totals.head(5).index)
selected = st.multiselect("Pick brands to plot", options=brand_cols, default=default_brands)
if selected:
    long_df = week[week['keyword'].isin(selected)][['week_start','keyword',metric]].rename(columns={'keyword':'brand','week_start':'x','%s'%metric:'y'})
    fig = px.line(long_df, x='x', y='y', color='brand', markers=True,
                  title=f'Weekly {metric} for selected brands')
    # Peaks & events highlighter: z-score >= 2 per brand
    try:
        tmp = week[week['keyword'].isin(selected)][['week_start','keyword',metric]].copy()
        tmp = tmp.rename(columns={metric:'val'})
        tmp = tmp.sort_values(['keyword','week_start'])
        g = tmp.groupby('keyword')
        mu = g['val'].transform('mean')
        sd = g['val'].transform('std').replace(0, np.nan)
        z = (tmp['val'] - mu) / sd
        peaks = tmp.loc[z >= 2].rename(columns={'keyword':'brand','week_start':'x','val':'y'})
        if not peaks.empty:
            fig.add_trace(go.Scatter(x=peaks['x'], y=peaks['y'], mode='markers',
                                     marker=dict(size=10, symbol='star'),
                                     name='peaks (≥2σ)',
                                     hovertext=peaks['brand']))
    except Exception:
        pass
    fig.update_layout(xaxis_title='Week start (UTC)', yaxis_title=metric.replace('_',' ').title(), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Show data table"):
        st.dataframe(long_df.sort_values(["brand","x"]).reset_index(drop=True))
else:
    st.info("Select at least one brand to display the plot.")

# === Analyst-facing views ===
if selected:
    tabs = st.tabs(["Share of Voice", "Engagement per Mention", "Momentum", "Top N channels per brand", "Concentration (HHI)", "Rank movement", "Engagement mix", "Engagement Premium"])

    # 1) Share of Voice (mentions share per week)
    with tabs[0]:
        if 'weekly_video_mentions' in week.columns:
            sov = week.copy()
            sov['total_mentions_week'] = sov.groupby('week_start')['weekly_video_mentions'].transform('sum')
            sov['sov'] = (sov['weekly_video_mentions'] / sov['total_mentions_week'].replace(0, pd.NA)).fillna(0.0)
            sov_sel = sov[sov['keyword'].isin(selected)][['week_start','keyword','sov']]
            fig_sov = px.area(sov_sel, x='week_start', y='sov', color='keyword',
                              title='Share of Voice (share of weekly mentions)')
            fig_sov.update_layout(xaxis_title='Week start (UTC)', yaxis_title='Share of voice')
            fig_sov.update_yaxes(tickformat='.0%', range=[0,1])
            st.plotly_chart(fig_sov, use_container_width=True)
            with st.expander("Show SoV table"):
                st.dataframe(sov_sel.pivot_table(index='week_start', columns='keyword', values='sov').fillna(0).round(3))
        else:
            st.info("`weekly_video_mentions` not available in file.")

    # 2) Engagement per Mention
    with tabs[1]:
        need_cols = {'weekly_views','weekly_likes','weekly_comments','weekly_video_mentions'}
        if need_cols.issubset(set(week.columns)):
            epm = week.copy()
            # make sure numeric with zeros for missing
            views = pd.to_numeric(epm['weekly_views'], errors='coerce').fillna(0)
            likes = pd.to_numeric(epm['weekly_likes'], errors='coerce').fillna(0)
            comments = pd.to_numeric(epm['weekly_comments'], errors='coerce').fillna(0)
            denom = pd.to_numeric(epm['weekly_video_mentions'], errors='coerce').fillna(0)
            epm['total_eng'] = views + likes + comments
            # safe division: 0 when denom <= 0
            epm['engagement_per_mention'] = np.where(denom > 0, epm['total_eng'] / denom, 0.0)
            epm_sel = epm[epm['keyword'].isin(selected)][['week_start','keyword','engagement_per_mention']]
            fig_epm = px.line(epm_sel, x='week_start', y='engagement_per_mention', color='keyword', markers=True,
                              title='Engagement per mention (views+likes+comments per video)')
            fig_epm.update_layout(xaxis_title='Week start (UTC)', yaxis_title='Engagement / mention')
            st.plotly_chart(fig_epm, use_container_width=True)
            with st.expander("Show EPM table"):
                st.dataframe(epm_sel.pivot_table(index='week_start', columns='keyword', values='engagement_per_mention').fillna(0).round(1))
        else:
            st.info("Required columns for engagement per mention are missing.")

    # 3) Momentum (WoW % change)
    with tabs[2]:
        metric_for_mom = st.selectbox("Momentum basis", [m for m in ['weekly_video_mentions','weekly_views','weekly_likes','weekly_comments'] if m in week.columns], index=0)
        mom = week[['week_start','keyword',metric_for_mom]].copy().rename(columns={metric_for_mom:'val'})
        mom = mom.sort_values(['keyword','week_start'])
        mom['pct_change'] = mom.groupby('keyword')['val'].pct_change().replace([pd.NA, pd.NaT], 0)
        mom_sel = mom[mom['keyword'].isin(selected)]
        fig_mom = px.line(mom_sel, x='week_start', y='pct_change', color='keyword', markers=True,
                          title=f"Momentum (WoW % change of {metric_for_mom})")
        fig_mom.update_yaxes(tickformat='.0%')
        fig_mom.update_layout(xaxis_title='Week start (UTC)', yaxis_title='% change WoW')
        st.plotly_chart(fig_mom, use_container_width=True)
        with st.expander("Show momentum table"):
            st.dataframe(mom_sel.pivot_table(index='week_start', columns='keyword', values='pct_change').fillna(0).round(3))

    # 4) Top N channels per brand (by engagement) using channel weekly snapshot
    with tabs[3]:
        try:
            ch_path = find_file(CHANNEL_WEEKLY_FILE)
            chw = pd.read_csv(ch_path, encoding='utf-8')
        except Exception:
            chw = None
        if chw is None:
            st.info("Channel weekly file not found.")
        else:
            need_cols = {"week_start", "keyword", "channel", "views", "likeCount", "commentCount"}
            if need_cols.issubset(set(chw.columns)):
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                chw = chw.copy()
                chw["week_start"] = pd.to_datetime(chw["week_start"])
                if "week_end" in chw.columns:
                    chw["week_end"] = pd.to_datetime(chw["week_end"])
                else:
                    chw["week_end"] = chw["week_start"] + pd.Timedelta(days=6)
                overlaps = (chw["week_start"] <= end_dt) & (chw["week_end"] >= start_dt)
                mask_ch = overlaps & (chw["keyword"].isin(selected))
                sub = chw.loc[mask_ch, ["keyword","channel","views","likeCount","commentCount"]].copy()
                for col in ["views","likeCount","commentCount"]:
                    sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0)
                if sub.empty:
                    st.info("No weekly channel data for the selected window/brands.")
                else:
                    agg = (sub.groupby(["keyword","channel"], as_index=False)
                           .agg({"views":"sum","likeCount":"sum","commentCount":"sum"}))
                    agg["engagement"] = agg["views"] + agg["likeCount"] + agg["commentCount"]
                    top_n = st.slider("Top N per brand (by engagement)", 3, 20, 5)
                    topn = (agg.sort_values(["keyword","engagement"], ascending=[True, False])
                              .groupby("keyword").head(top_n).reset_index(drop=True))
                    fig_top = px.bar(topn, x='channel', y='engagement', color='keyword', barmode='group',
                                     title='Top N channels per brand (by engagement)')
                    fig_top.update_layout(xaxis_title='Channel', yaxis_title='Engagement (views+likes+comments)')
                    st.plotly_chart(fig_top, use_container_width=True)
                    with st.expander("Show Top N table"):
                        st.dataframe(topn.sort_values(["keyword","engagement"], ascending=[True, False]), use_container_width=True)
            else:
                st.info("Channel weekly summary missing required columns.")

    # 5) Concentration (HHI) — based on channel engagement shares per brand-week
    with tabs[4]:
        try:
            ch_path = find_file(CHANNEL_WEEKLY_FILE)
            chw = pd.read_csv(ch_path, encoding='utf-8')
        except Exception:
            chw = None
        if chw is None:
            st.info("Channel weekly file not found.")
        else:
            need_cols = {"week_start","keyword","channel","views","likeCount","commentCount"}
            if need_cols.issubset(set(chw.columns)):
                chw = chw.copy()
                chw["week_start"] = pd.to_datetime(chw["week_start"])
                if "week_end" in chw.columns:
                    chw["week_end"] = pd.to_datetime(chw["week_end"])
                else:
                    chw["week_end"] = chw["week_start"] + pd.Timedelta(days=6)
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                overlaps = (chw["week_start"] <= end_dt) & (chw["week_end"] >= start_dt)
                sub = chw.loc[overlaps & (chw["keyword"].isin(selected)), ["week_start","keyword","channel","views","likeCount","commentCount"]].copy()
                for col in ["views","likeCount","commentCount"]:
                    sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0)
                if sub.empty:
                    st.info("No channel data in selected window.")
                else:
                    sub["eng"] = sub["views"] + sub["likeCount"] + sub["commentCount"]
                    totals = sub.groupby(["week_start","keyword"], as_index=False)["eng"].sum().rename(columns={"eng":"tot"})
                    m = sub.merge(totals, on=["week_start","keyword"], how="left")
                    m["share"] = np.where(m["tot"] > 0, m["eng"] / m["tot"], 0.0)
                    hhi = (m.assign(sq=lambda x: x["share"]**2)
                             .groupby(["week_start","keyword"], as_index=False)["sq"].sum()
                             .rename(columns={"sq":"hhi"}))
                    fig_hhi = px.line(hhi, x="week_start", y="hhi", color="keyword", markers=True,
                                      title="Concentration of exposure (HHI)")
                    fig_hhi.update_layout(xaxis_title='Week start (UTC)', yaxis_title='HHI (0–1, higher = more concentrated)')
                    st.plotly_chart(fig_hhi, use_container_width=True)
                    with st.expander("Show HHI table"):
                        st.dataframe(hhi.pivot_table(index='week_start', columns='keyword', values='hhi').round(3))
            else:
                st.info("Channel weekly summary missing required columns.")

    # 6) Rank movement (by chosen metric): lower rank is better
    with tabs[5]:
        rank_metric = st.selectbox("Rank basis", [m for m in ['weekly_video_mentions','weekly_views','weekly_likes','weekly_comments'] if m in week.columns], index=0, key='rank_metric')
        rk = week[["week_start","keyword",rank_metric]].copy().rename(columns={rank_metric:"val"})
        rk = rk.sort_values(["week_start","val"], ascending=[True, False])
        rk["rank"] = rk.groupby("week_start")["val"].rank(method="min", ascending=False)
        rk = rk.sort_values(["keyword","week_start"])
        rk["rank_change"] = rk.groupby("keyword")["rank"].diff().fillna(0) * -1  # positive = improved rank
        rk_sel = rk[rk["keyword"].isin(selected)]
        fig_rk = px.line(rk_sel, x="week_start", y="rank", color="keyword", markers=True,
                         title=f"Rank movement (lower is better) – {rank_metric}")
        fig_rk.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_rk, use_container_width=True)
        with st.expander("Show rank table"):
            st.dataframe(rk_sel.sort_values(["week_start","rank"]).reset_index(drop=True))

    # 7) Engagement mix (views vs likes vs comments)
    with tabs[6]:
        need_cols = {'weekly_views','weekly_likes','weekly_comments'}
        if need_cols.issubset(set(week.columns)):
            mix = (week[['week_start','keyword','weekly_views','weekly_likes','weekly_comments']]
                   .melt(id_vars=['week_start','keyword'], var_name='kind', value_name='val'))
            mix = mix[mix['keyword'].isin(selected)]
            fig_mix = px.area(mix, x='week_start', y='val', color='kind', facet_row='keyword',
                              title='Engagement mix by brand')
            fig_mix.update_layout(yaxis_title='Count', xaxis_title='Week start (UTC)')
            st.plotly_chart(fig_mix, use_container_width=True)
            with st.expander("Show mix table"):
                st.dataframe(mix.pivot_table(index=['week_start','keyword'], columns='kind', values='val').fillna(0).round(0))
        else:
            st.info("Required columns for engagement mix are missing.")

    # 8) Engagement Premium Index (EPM / weekly median EPM)
    with tabs[7]:
        need_cols = {'weekly_views','weekly_likes','weekly_comments','weekly_video_mentions'}
        if need_cols.issubset(set(week.columns)):
            base = week.copy()
            # compute EPM for all brands; use NaN where mentions==0 to avoid skewing the weekly median
            v = pd.to_numeric(base['weekly_views'], errors='coerce')
            l = pd.to_numeric(base['weekly_likes'], errors='coerce')
            c = pd.to_numeric(base['weekly_comments'], errors='coerce')
            n = pd.to_numeric(base['weekly_video_mentions'], errors='coerce')
            epm_all = (v.fillna(0) + l.fillna(0) + c.fillna(0)) / n.replace(0, np.nan)
            base['epm'] = epm_all

            use_selected_baseline = st.toggle("Baseline on selected brands only", value=False, help="If on, the weekly median is computed only over the currently selected brands; otherwise over all brands available in that week.")
            if use_selected_baseline:
                mask_sel = base['keyword'].isin(selected)
                # compute per-week median using only selected brands; fall back to global if none selected that week
                med = (base[mask_sel].groupby('week_start')['epm'].median()
                                  .rename('med'))
                base = base.merge(med, on='week_start', how='left')
                # if med is NaN (no selected brands that week), recompute global median for that week
                med_global = base.groupby('week_start')['epm'].median().rename('med_global')
                base = base.merge(med_global, on='week_start', how='left')
                base['baseline'] = base['med'].fillna(base['med_global'])
            else:
                base['baseline'] = base.groupby('week_start')['epm'].transform('median')

            base['epi'] = base['epm'] / base['baseline'].replace(0, np.nan)
            epi_sel = base[base['keyword'].isin(selected)][['week_start','keyword','epi']]
            fig_epi = px.line(epi_sel, x='week_start', y='epi', color='keyword', markers=True,
                              title='Engagement Premium Index (EPM / weekly median EPM)')
            fig_epi.update_layout(xaxis_title='Week start (UTC)', yaxis_title='Index (1.0 = weekly median)', hovermode='x unified')
            st.plotly_chart(fig_epi, use_container_width=True)
            with st.expander("Show EPI table"):
                st.dataframe(epi_sel.pivot_table(index='week_start', columns='keyword', values='epi').round(2))
            st.caption("Interpretation: values > 1.0 mean above-median engagement-per-mention for that week; < 1.0 below-median.")
        else:
            st.info("Required columns for Engagement Premium are missing.")
else:
    st.info("Select at least one brand to show Share of Voice, Engagement per Mention, Momentum, and Top N views.")

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
