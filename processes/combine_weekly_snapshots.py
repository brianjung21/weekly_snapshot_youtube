from pathlib import Path
import pandas as pd

DATA_DIR = Path("snapshot_data")

brand_files = sorted(DATA_DIR.glob("youtube_brand_weekly_snapshot*.csv"))
chan_files = sorted(DATA_DIR.glob("youtube_brand_channel_weekly_snapshot*.csv"))


def combine(files, key_cols):
    if not files:
        raise SystemExit("No files found.")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=key_cols, keep='last')
    try:
        wk = out[['week_start', 'week_end']].drop_duplicates()
        wk['week_start'] = pd.to_datetime(wk['week_start'])
        wk['week_end'] = pd.to_datetime(wk['week_end'])
        bad = wk[(wk['week_end'] - wk['week_start']).dt.days != 6]
        if len(bad):
            print("[warn] Non 7-day rows detected:\n", bad)
    except Exception:
        pass
    return out


brand_all = combine(brand_files, ['week_start', 'week_end', "keyword"])
chan_all = combine(chan_files, ['week_start', 'week_end', 'keyword', 'channel_id'])

brand_all = brand_all.sort_values(['week_start', 'keyword']).reset_index(drop=True)
chan_all = chan_all.sort_values(['week_start', 'keyword', 'channel']).reset_index(drop=True)

brand_all.to_csv(DATA_DIR / "youtube_brand_weekly_snapshot_ALL.csv", index=False)
chan_all.to_csv(DATA_DIR / "youtube_brand_channel_weekly_snapshot_ALL.csv", index=False)

print("wrote: ",
      DATA_DIR / "youtube_brand_weekly_snapshot_ALL.csv",
      DATA_DIR / "youtube_brand_channel_weekly_snapshot_ALL.csv")
