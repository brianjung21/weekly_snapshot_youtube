"""
YouTube brand mention counter - SNAPSHOT (fixed 7-day week).
- Reads keywords_with_aliases.csv with columns: Keyword, Aliases (pipe-separated)
- Searches YouTube Data API v3 within a fixed UTC 7-day week [start, end], inclusive of start and end,
  e.g., Aug 1–7, Aug 8–14, etc. (non-overlapping weekly series).
- Alias-aware local regex match on title+description (+tags).
- Dedupes by (keyword, video_id) for the week.
- Aggregates ONE weekly row per brand (zero-fills brands with no hits).
- Also writes a per-channel breakdown for the same week.
- If the specified week_end is not exactly 6 days after week_start,
  the script auto-corrects week_end to week_start + 6 days and prints a notice.

Outputs:
1) youtube_brand_weekly_snapshot1.csv
    columns: week_start, week_end, keyword, weekly_api_hits, weekly_video_mentions, weekly_views, weekly_likes, weekly_comments, top_channels_week
    weekly_api_hits = unique videos returned by API; weekly_video_mentions = text-confirmed matches.
2) youtube_brand_channel_weekly_snapshot1.csv
    per (keyword, channel) weekly breakdown with engagement + channel stats
"""

import re
import unicodedata
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Iterable
import requests
import pandas as pd

# Support multiple API keys with automatic rotation on quota/429/403
API_KEYS = [
 # add more keys here, e.g., "KEY_2", "KEY_3", ...
]

class KeyRing:
    def __init__(self, keys):
        ks = [k for k in (keys or []) if k]
        if not ks:
            raise ValueError("No API keys provided in API_KEYS")
        self.keys = ks
        self.i = 0

    @property
    def current(self):
        return self.keys[self.i]

    def advance(self):
        self.i = (self.i + 1) % len(self.keys)
        return self.current

    def inject(self, params: dict) -> dict:
        # return a shallow-copied params dict with 'key' set to the current key
        p = dict(params or {})
        p["key"] = self.current
        return p

KEYRING = KeyRing(API_KEYS)

# ===== Range mode (optional): run many consecutive weeks and resume from checkpoint =====
RUN_RANGE = True  # set True to iterate weeks; False keeps single-week mode
RANGE_START_UTC = '2025-08-29'  # inclusive week starts from here
RANGE_END_UTC   = '2025-09-16'  # target horizon; weeks will be stepped by 7 days up to this date

from pathlib import Path
KEYWORDS_PATH = Path("fnb_keywords_sample.csv")

CHECKPOINT_PATH = Path('snapshot_data/weekly_snapshot_checkpoint.json')

# Point outputs to the consolidated "ALL" files so history accumulates in one place
WEEKLY_BRAND_OUT = Path('snapshot_data/youtube_brand_weekly_snapshot_ALL.csv')
WEEKLY_CHANNEL_OUT = Path('snapshot_data/youtube_brand_channel_weekly_snapshot_ALL.csv')

WEEK_START_UTC = '2025-08-22'
WEEK_END_UTC = '2025-08-28'
import json

def save_ckpt(week_start: str, week_end: str, brand_idx: int):
    try:
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CHECKPOINT_PATH.open('w', encoding='utf-8') as f:
            json.dump({
                'week_start': week_start,
                'week_end': week_end,
                'brand_idx': int(brand_idx),
                'ts_utc': datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
            }, f)
    except Exception as e:
        print(f"[ckpt] failed to save checkpoint: {e}")

def load_ckpt():
    try:
        if CHECKPOINT_PATH.exists():
            with CHECKPOINT_PATH.open('r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[ckpt] failed to load checkpoint: {e}")
    return None

def clear_ckpt():
    try:
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
    except Exception as e:
        print(f"[ckpt] failed to clear checkpoint: {e}")


def ensure_seven_day_window(start_str: str, end_str: str):
    start_dt = datetime.fromisoformat(start_str).date()
    end_dt = datetime.fromisoformat(end_str).date()
    delta = (end_dt - start_dt).days
    if delta != 6:
        corrected_end = start_dt + timedelta(days=6)
        print(f"Note: Adjusted week_end from {end_dt.isoformat()} to {corrected_end.isoformat()} to enforce 7-day snapshot window.")
        end_dt = corrected_end
    return start_dt.isoformat(), end_dt.isoformat()


PAGE_SLEEP_SEC = 1.0
BRAND_SLEEP_SEC = 1.0

BASE = 'https://www.googleapis.com/youtube/v3'

REGION_CODE = None
RELEVANCE_LANGUAGE = None

INCLUDE_TAGS = True
MAX_RETRIES = 5
BACKOFF_BASE = 0.5
BACKOFF_MAX = 8.0

# If True, brand matching uses a Hangul‑friendly boundary: \w → [0-9A-Za-z가-힣]
# This reduces accidental substring matches in mixed Korean/ASCII text.
USE_KO_BOUNDARY = False


def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "")).lower()


def brand_to_pattern(kw: str) -> re.Pattern:
    tokens = [re.escape(t) for t in kw.split() if t]
    if not tokens:
        return re.compile(r"$^")
    sep = r"[-_\.\s]*"  # allow hyphen/underscore/dot/space between tokens
    core = sep.join(tokens)
    if USE_KO_BOUNDARY:
        # Negative lookaround against Korean+ASCII alphanumerics to emulate word boundaries in mixed text
        left  = r"(?<![0-9A-Za-z가-힣])"
        right = r"(?![0-9A-Za-z가-힣])"
    else:
        left  = r"(?<!\w)"
        right = r"(?!\w)"
    pat = rf"{left}{core}{right}"
    return re.compile(pat, re.IGNORECASE)


def load_keywords(csv_path: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "Keyword" not in df.columns:
        raise ValueError("Expected a header column named 'Keyword'")
    if "Aliases" not in df.columns:
        df["Aliases"] = ""
    out = {}
    for _, r in df.iterrows():
        main = str(r["Keyword"]).strip().lower()
        if not main:
            continue
        alts = [a.strip().lower() for a in str(r["Aliases"]).split("|") if a.strip()]
        out[main] = [main] + alts
    return out


def iso8601(dt: datetime) -> str:
    return dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _sleep_backoff(attempt: int):
    delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** attempt))
    time.sleep(delay)


def get_json(url: str, params: dict) -> dict:
    """GET JSON with retries/backoff on 5xx/429 and network errors."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < MAX_RETRIES - 1:
                    _sleep_backoff(attempt)
                    continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                _sleep_backoff(attempt)
                continue
            raise

def get_json_with_keys(url: str, params: dict) -> dict:
    """GET JSON with exponential backoff, auto-rotating API keys on 403/429.
    Tries each key in KEYRING at most once per attempt horizon, backing off between rounds.
    """
    last_exc = None
    # Try up to MAX_RETRIES rounds; within each round we may cycle keys
    for attempt in range(MAX_RETRIES):
        # try as many keys as we have before backing off
        for _ in range(len(KEYRING.keys)):
            try:
                p = KEYRING.inject(params)
                resp = requests.get(url, params=p, timeout=30)
                if resp.status_code in (429, 403, 500, 502, 503, 504):
                    # rotate key on 429/403 immediately; for 5xx we'll backoff as well
                    last_exc = requests.HTTPError(f"status={resp.status_code} body={resp.text[:200]}")
                    if resp.status_code in (429, 403):
                        KEYRING.advance()
                        continue
                    # for 5xx, fall through to backoff after key cycle
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_exc = e
                # rotate key and try next key
                KEYRING.advance()
                continue
        # after trying all keys once, back off and retry
        _sleep_backoff(attempt)
    # exhausted all retries/keys
    if last_exc:
        raise last_exc
    raise RuntimeError("get_json_with_keys failed with no exception context")


def yt_videos_details(video_ids: List[str]) -> dict:
    out = {}
    if not video_ids:
        return out
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        params = {
            "part": "snippet,statistics",
            "id": ",".join(chunk),
            "maxResults": 50,
        }
        data = get_json_with_keys(f"{BASE}/videos", params)
        for it in data.get("items", []):
            vid = it.get("id")
            sn = it.get("snippet", {}) or {}
            st = it.get("statistics", {}) or {}
            out[vid] = {
                "tags": sn.get("tags", []) or [],
                "likeCount": int(st.get("likeCount", 0) or 0),
                "commentCount": int(st.get("commentCount", 0) or 0),
                "viewCount": int(st.get('viewCount', 0) or 0),
                "channelId": sn.get("channelId", "")
            }
        time.sleep(PAGE_SLEEP_SEC)
    return out


def yt_channels_stats(channel_ids: List[str]) -> dict:
    out = {}
    if not channel_ids:
        return out
    for i in range(0, len(channel_ids), 50):
        chunk = channel_ids[i:i+50]
        params = {
            "part": "statistics",
            "id": ",".join(chunk),
            "maxResults": 50,
        }
        data = get_json_with_keys(f"{BASE}/channels", params)
        for it in data.get("items", []):
            cid = it.get("id")
            st = it.get("statistics", {}) or {}
            subs = int(st.get("subscriberCount", 0) or 0)
            vids = int(st.get("videoCount", 0) or 0)
            out[cid] = {"subscribers": subs, "videoCount": vids}
        time.sleep(PAGE_SLEEP_SEC)
    return out


def yt_search_pages(q: str, published_after: str, published_before: str) -> Iterable[dict]:
    params = {
        "part": "snippet",
        "q": q,
        "type": "video",
        "order": "date",
        "maxResults": 50,
        "publishedAfter": published_after,
        "publishedBefore": published_before,
    }
    if REGION_CODE:
        params["regionCode"] = REGION_CODE
    if RELEVANCE_LANGUAGE:
        params["relevanceLanguage"] = RELEVANCE_LANGUAGE

    token = None
    while True:
        if token:
            params['pageToken'] = token
        data = get_json_with_keys(f"{BASE}/search", params)
        for it in data.get("items", []):
            yield it
        token = data.get("nextPageToken")
        if not token:
            break
        time.sleep(PAGE_SLEEP_SEC)


def weekly_window_utc(week_start: str, week_end: str):
    start_dt = datetime.fromisoformat(week_start).replace(tzinfo=timezone.utc)
    end_dt_inclusive = datetime.fromisoformat(week_end).replace(tzinfo=timezone.utc)
    after = iso8601(start_dt)
    before = iso8601(end_dt_inclusive + timedelta(days=1))
    return after, before, start_dt.date().isoformat(), end_dt_inclusive.date().isoformat()



def run_one_week(ws: str, we: str, resume_brand_idx: int = 0):
    kw_map = load_keywords(KEYWORDS_PATH)
    if not kw_map:
        print("No keywords found.")
        return

    ws, we = ensure_seven_day_window(ws, we)
    published_after, published_before, week_start, week_end = weekly_window_utc(ws, we)
    print(f"[RUN] Week {week_start} → {week_end} (UTC)")

    patterns = {k: [brand_to_pattern(v) for v in vs] for k, vs in kw_map.items()}
    brand_api_ids = {k: set() for k in kw_map.keys()}

    rows = []
    keys_list = list(kw_map.keys())
    for bi, main_kw in enumerate(keys_list):
        if bi < resume_brand_idx:
            continue
        print(f"[YouTube SNAPSHOT] {main_kw} | {week_start}–{week_end}")
        pats = patterns[main_kw]
        matched = 0

        video_ids = []
        hits = []
        try:
            for item in yt_search_pages(main_kw, published_after, published_before):
                vid = item['id']['videoId']
                snip = item.get('snippet', {})
                hits.append({
                    'video_id': vid,
                    'title': snip.get('title', ''),
                    'description': snip.get('description', ''),
                    'channel': snip.get('channelTitle', ''),
                    'published_at': snip.get('publishedAt', ''),
                })
                video_ids.append(vid)
                brand_api_ids[main_kw].add(vid)
        except Exception as e:
            print(f"[error] search failed for {main_kw}: {e}")
            save_ckpt(week_start, week_end, bi)
            raise

        details_fetched_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        try:
            id2det = yt_videos_details(video_ids) if video_ids else {}
        except Exception as e:
            print(f"[error] videos.details failed for {main_kw}: {e}")
            save_ckpt(week_start, week_end, bi)
            raise

        for h in hits:
            det = id2det.get(h['video_id'], {})
            tags = det.get("tags", []) if INCLUDE_TAGS else []
            text_norm = normalize_text(f"{h['title']}\n{h['description']}\n{' '.join(tags)}")
            if any(p.search(text_norm) for p in pats):
                rows.append({
                    "keyword": main_kw,
                    "channel": h["channel"],
                    "channel_id": det.get("channelId", ""),
                    "video_id": h["video_id"],
                    "likeCount": det.get("likeCount", 0),
                    "commentCount": det.get("commentCount", 0),
                    "viewCount": det.get("viewCount", 0),
                })
                matched += 1
        print(f"    -> {matched} videos matched")

        # Save checkpoint to resume at next brand on any subsequent run
        save_ckpt(week_start, week_end, bi + 1)
        time.sleep(BRAND_SLEEP_SEC)

    # --- aggregation & writes (same as before) ---
    if rows:
        df = pd.DataFrame(rows).drop_duplicates(subset=['keyword', 'video_id'])
        per_ch_week = (df.groupby(['keyword', 'channel', 'channel_id'], as_index=False)
                       .agg({
            "video_id": "count",
            "viewCount": "sum",
            "likeCount": "sum",
            "commentCount": "sum"
        })
        .rename(columns={"video_id": "matched_videos", "viewCount": "views"}))

        unique_channel_ids = sorted(c for c in per_ch_week['channel_id'].unique().tolist() if c)
        cid2stats = yt_channels_stats(unique_channel_ids)
        per_ch_week['subscribers'] = per_ch_week['channel_id'].map(lambda c: cid2stats.get(c, {}).get("subscribers", 0))
        per_ch_week['channel_video_count'] = per_ch_week['channel_id'].map(lambda c: cid2stats.get(c, {}).get('videoCount', 0))

        brand_week = (per_ch_week.groupby(['keyword'], as_index=False)
                      .agg({
            "matched_videos": "sum",
            "views": "sum",
            "likeCount": "sum",
            "commentCount": "sum",
        })
        .rename(columns={
            "matched_videos": "weekly_video_mentions",
            "views": "weekly_views",
            "likeCount": "weekly_likes",
            "commentCount": "weekly_comments"
        }))

        api_counts = pd.DataFrame({
            'keyword': list(brand_api_ids.keys()),
            'weekly_api_hits': [len(v) for v in brand_api_ids.values()],
        })
        brand_week = brand_week.merge(api_counts, on='keyword', how='right').fillna({
            'weekly_video_mentions': 0,
            'weekly_views': 0,
            'weekly_likes': 0,
            'weekly_comments': 0,
        })

        # Averages
        bw = brand_week.copy()
        n = (bw["weekly_video_mentions"].replace({0: pd.NA})).astype("Int64")
        bw["avg_views_per_video"] = (bw["weekly_views"] / n).astype(float).fillna(0.0)
        bw["avg_likes_per_video"] = (bw["weekly_likes"] / n).astype(float).fillna(0.0)
        bw["avg_comments_per_video"] = (bw["weekly_comments"] / n).astype(float).fillna(0.0)
        brand_week = bw

        def pick_top_channels(g: pd.DataFrame) -> str:
            tmp = g.copy()
            for col in ["subscribers", "channel_video_count", "likeCount", "commentCount"]:
                tmp[col] = tmp[col].fillna(0)
                tmp[f"r_{col}"] = tmp[col].rank(method="min", ascending=False)
            tmp["rank_sum"] = tmp[["r_subscribers", "r_channel_video_count", "r_likeCount", "r_commentCount"]].sum(axis=1)
            tmp = tmp.sort_values(["rank_sum", "subscribers", "likeCount"], ascending=[True, False, False])
            return ";".join(tmp["channel"].head(3).tolist())

        tops = (per_ch_week.groupby(["keyword"], as_index=False)
                .apply(lambda g: pd.Series({"top_channels_week": pick_top_channels(g)}))
                .reset_index(drop=True))
        out = brand_week.merge(tops, on='keyword', how='left')

        details_fetched_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        out["as_of_details_fetched_utc"] = details_fetched_ts
        per_ch_week["as_of_details_fetched_utc"] = details_fetched_ts
    else:
        out = (pd.DataFrame({'keyword': sorted(load_keywords(KEYWORDS_PATH).keys())})
               .assign(weekly_api_hits=0,
                       weekly_video_mentions=0,
                       weekly_views=0,
                       weekly_likes=0,
                       weekly_comments=0,
                       avg_views_per_video=0.0,
                       avg_likes_per_video=0.0,
                       avg_comments_per_video=0.0,
                       top_channels_week=""))
        per_ch_week = pd.DataFrame(columns=[
            "keyword", "channel", "channel_id", "matched_videos", "views",
            "likeCount", "commentCount", "subscribers", "channel_video_count"
        ])
        details_fetched_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        out["as_of_details_fetched_utc"] = details_fetched_ts
        per_ch_week["as_of_details_fetched_utc"] = details_fetched_ts

    out.insert(0, "week_start", week_start)
    out.insert(1, "week_end", week_end)
    per_ch_week.insert(0, "week_start", week_start)
    per_ch_week.insert(1, "week_end", week_end)
    per_ch_week["engagement"] = per_ch_week["likeCount"].astype(int) + per_ch_week["commentCount"].astype(int)

    WEEKLY_BRAND_OUT.parent.mkdir(parents=True, exist_ok=True)
    WEEKLY_CHANNEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    def append_dedup(df_new: pd.DataFrame, path: Path, key_cols: List[str]):
        if path.exists():
            df_old = pd.read_csv(path, encoding="utf-8")
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all = df_all.drop_duplicates(subset=key_cols, keep="last")
        else:
            df_all = df_new
        df_all.to_csv(path, index=False, encoding="utf-8")
        return df_all

    append_dedup(out, WEEKLY_BRAND_OUT, ["week_start", "week_end", "keyword"])
    append_dedup(per_ch_week, WEEKLY_CHANNEL_OUT, ["week_start", "week_end", "keyword", "channel_id"])

    print(f"Wrote weekly brand snapshot -> {WEEKLY_BRAND_OUT.resolve()}")
    print(f"Wrote weekly channel snapshot -> {WEEKLY_CHANNEL_OUT.resolve()}")

    # clear checkpoint after successful write for this week
    clear_ckpt()


def main():
    if RUN_RANGE:
        # Try resume from checkpoint first
        ck = load_ckpt()
        if ck:
            print(f"[ckpt] Resuming week {ck['week_start']} → {ck['week_end']} at brand_idx={ck['brand_idx']}")
            try:
                run_one_week(ck['week_start'], ck['week_end'], resume_brand_idx=int(ck.get('brand_idx', 0)))
                # advance to next week after successful resume
                start_dt = datetime.fromisoformat(ck['week_start']).date() + timedelta(days=7)
            except Exception:
                return  # leave checkpoint for the next run
        else:
            start_dt = datetime.fromisoformat(RANGE_START_UTC).date()

        end_horizon = datetime.fromisoformat(RANGE_END_UTC).date()
        while start_dt <= end_horizon:
            ws = start_dt.isoformat()
            we = (start_dt + timedelta(days=6)).isoformat()
            save_ckpt(ws, we, 0)  # set checkpoint at the start of the week
            try:
                run_one_week(ws, we, resume_brand_idx=0)
            except Exception as e:
                print(f"[halt] Stopping due to error/quota. Checkpoint saved for {ws} → {we}. Error: {e}")
                return
            start_dt = start_dt + timedelta(days=7)
        print("[done] Range complete.")
    else:
        run_one_week(WEEK_START_UTC, WEEK_END_UTC, resume_brand_idx=0)


if __name__ == "__main__":
    main()
