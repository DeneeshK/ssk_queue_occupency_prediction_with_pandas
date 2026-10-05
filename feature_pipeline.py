"""
feature_pipeline.py
===================
Single source of truth for turning the raw `InOut` Excel workbook into model input.

Training, retraining and inference ALL call the functions in this file. There is no
second implementation of any feature anywhere else.

The logic is a direct port of the validated Phase-2 notebook (Experiments.ipynb).
Nothing about the notebook's feature definitions has been changed.

Pipeline
--------
raw Excel -> validate_and_clean -> build_occupancy_grid -> add_reliability
          -> add_features (+ targets) -> `wide` frame
`wide` frame -> build_training_set()      (long format, one row per timestamp x horizon)
`wide` frame -> latest_inference_frame()  (the single latest valid state x 6 horizons)

Timestamp semantics (preserved from the notebook)
-------------------------------------------------
Each row is labelled with the START of its 15-minute bucket. `active_vehicles` at row t
is the cumulative sum of net flow through the END of bucket t (it includes bucket t's own
events). Flow / rolling features at t use only the buckets BEFORE t (shift(1)).
Consequence: a row is only meaningful once its bucket has fully closed.

`active_vehicles` is an in-system occupancy / congestion proxy, NOT a physical queue length.

Reliability
-----------
Aug 22, Aug 23 and Aug 24 00:00-11:45 are unreliable (incomplete entry coverage plus a
12-hour post-gap washout). The 720-minute washout is an ENGINEERING ASSUMPTION based on the
observed maximum dwell (~720 min); it is not a confirmed operational limit.
Underlying values are never modified or zero-filled. Any feature whose source interval(s) are
unreliable is set to NaN, any target in an unreliable interval is NaN, and rows with any NaN
are dropped from supervised data.
"""

from __future__ import annotations

import pandas as pd


# --------------------------------------------------------------------------------------
# Constants (all taken from the notebook / spec)
# --------------------------------------------------------------------------------------
RAW_SHEET = "InOut"
RAW_COLUMNS = [
    "Country", "Plate Prefix", "Plate Number", "Entry Time", "Exit Time",
    "Risk Category", "Duration in Min", "Time Taken",
]
KEEP_COLUMNS = ["Entry Time", "Exit Time", "Risk Category", "Duration in Min"]

INTERVAL = "15min"
INTERVAL_MINUTES = 15

# Known unreliable observation periods (notebook section 4).
UNRELIABLE_DATES = ["2026-08-22", "2026-08-23"]
WASHOUT_START = pd.Timestamp("2026-08-24 00:00")
WASHOUT_END = pd.Timestamp("2026-08-24 12:00")  # exclusive -> last bad bucket is 11:45

# Lags in 15-minute steps.
LAGS = {
    "lag_15m": 1, "lag_30m": 2, "lag_1h": 4, "lag_2h": 8,
    "lag_3h": 12, "lag_4h": 16, "lag_6h": 24, "lag_24h": 96,
}

# Trailing flow windows in 15-minute steps.
FLOW_WINDOWS = {"15m": 1, "30m": 2, "1h": 4}
FLOW_SOURCES = ["arrivals", "exits", "net_flow"]
ROLLING_STEPS = 4  # 1 hour

# Forecast horizons: minutes -> 15-minute steps.
HORIZON_STEPS = {15: 1, 30: 2, 60: 4, 120: 8, 180: 12, 360: 24}
HORIZONS = list(HORIZON_STEPS)

# The 28 per-timestamp features, in notebook order.
BASE_FEATURES = [
    "active_vehicles",
    "lag_15m", "lag_30m", "lag_1h", "lag_2h", "lag_3h", "lag_4h", "lag_6h", "lag_24h",
    "arrivals_15m", "arrivals_30m", "arrivals_1h",
    "exits_15m", "exits_30m", "exits_1h",
    "net_flow_15m", "net_flow_30m", "net_flow_1h",
    "occupancy_change_15m", "occupancy_change_1h", "occupancy_change_2h",
    "rolling_mean_occupancy_1h", "rolling_mean_arrivals_1h",
    "rolling_mean_exits_1h", "rolling_mean_net_flow_1h",
    "hour", "day_of_week", "is_weekend",
]
# Exactly what XGBoost receives (29 columns, fixed order). timestamp / target are NEVER included.
MODEL_FEATURES = BASE_FEATURES + ["horizon_minutes"]
LONG_COLUMNS = ["timestamp"] + MODEL_FEATURES + ["target"]

# Chronological split boundaries (notebook section 16.1).
TRAIN_END = pd.Timestamp("2026-09-05 04:45:00")        # train:      timestamp <  TRAIN_END
VALIDATION_END = pd.Timestamp("2026-09-11 18:00:00")   # validation: TRAIN_END <= timestamp < VALIDATION_END
                                                       # test:       timestamp >= VALIDATION_END


class DataValidationError(ValueError):
    """The raw workbook does not meet the expected schema or content rules."""


class InsufficientHistoryError(ValueError):
    """Not enough reliable history to build the features for the latest interval."""


def target_column(minutes: int) -> str:
    return f"target_{minutes}m"


# --------------------------------------------------------------------------------------
# 1. Raw input
# --------------------------------------------------------------------------------------
def load_raw_excel(path) -> pd.DataFrame:
    """Read the `InOut` sheet exactly as the notebook does."""
    try:
        return pd.read_excel(path, sheet_name=RAW_SHEET)
    except ValueError as e:  # e.g. sheet not found
        raise DataValidationError(f"Could not read sheet '{RAW_SHEET}' from {path}: {e}") from e


def validate_and_clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw schema and return the 4 retained columns.

    Keeps Entry Time, Exit Time, Risk Category, Duration in Min. Drops everything else.
    Missing Risk Category -> 'UNKNOWN'. The supplied Duration in Min is kept as-is.
    """
    missing = [c for c in KEEP_COLUMNS if c not in raw.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s) {missing}. Expected raw columns: {RAW_COLUMNS}"
        )
    if len(raw) == 0:
        raise DataValidationError("Raw workbook contains no rows.")

    df = raw[KEEP_COLUMNS].copy()

    for col in ("Entry Time", "Exit Time"):
        try:
            df[col] = pd.to_datetime(df[col], errors="raise")
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"Column '{col}' contains unparseable timestamps: {e}") from e
        n_missing = int(df[col].isna().sum())
        if n_missing:
            # Open visits (no exit yet) are not handled by the validated notebook logic.
            raise DataValidationError(
                f"Column '{col}' has {n_missing} missing value(s). Every row must have both "
                f"an Entry Time and an Exit Time."
            )

    n_bad = int((df["Exit Time"] < df["Entry Time"]).sum())
    if n_bad:
        raise DataValidationError(f"{n_bad} row(s) have Exit Time earlier than Entry Time.")

    df["Duration in Min"] = pd.to_numeric(df["Duration in Min"], errors="coerce")
    if df["Duration in Min"].isna().any():
        raise DataValidationError("Column 'Duration in Min' contains missing or non-numeric values.")

    df["Risk Category"] = df["Risk Category"].fillna("UNKNOWN")
    return df


# --------------------------------------------------------------------------------------
# 2. Canonical 15-minute grid + occupancy
# --------------------------------------------------------------------------------------
def build_occupancy_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Complete 15-min grid with arrivals, exits, net_flow, active_vehicles (notebook sec. 2-3)."""
    entry_bucket = df["Entry Time"].dt.floor(INTERVAL)
    exit_bucket = df["Exit Time"].dt.floor(INTERVAL)

    arrivals = df.groupby(entry_bucket).size().rename("arrivals")
    exits = df.groupby(exit_bucket).size().rename("exits")
    events = pd.concat([arrivals, exits], axis=1, sort=False).fillna(0)

    full_index = pd.date_range(
        start=df["Entry Time"].min().floor(INTERVAL),
        end=df["Exit Time"].max().floor(INTERVAL),
        freq=INTERVAL,
    )
    grid = events.reindex(full_index, fill_value=0)
    grid.index.name = "timestamp"

    grid["net_flow"] = grid["arrivals"] - grid["exits"]            # Entry = +1, Exit = -1
    grid["active_vehicles"] = grid["net_flow"].cumsum()           # A_t = A_{t-1} + arrivals_t - exits_t
    return grid


# --------------------------------------------------------------------------------------
# 3. Reliability
# --------------------------------------------------------------------------------------
def add_reliability(grid: pd.DataFrame) -> pd.DataFrame:
    """Add boolean `is_reliable`. Values are NOT altered, only flagged (notebook sec. 4)."""
    out = grid.copy()
    unreliable = out.index.normalize().isin(pd.to_datetime(UNRELIABLE_DATES))
    unreliable |= (out.index >= WASHOUT_START) & (out.index < WASHOUT_END)
    out["is_reliable"] = ~unreliable
    return out


# --------------------------------------------------------------------------------------
# 4. Features and targets
# --------------------------------------------------------------------------------------
def add_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Add the 28 base features and the 6 targets to a reliability-flagged grid.

    Every feature is NaN unless ALL source intervals it depends on are reliable.
    """
    g = grid.copy()
    reliable = g["is_reliable"]

    # Occupancy lags: valid only if the interval the lag is read from is reliable.
    for name, steps in LAGS.items():
        source_ok = reliable.shift(steps, fill_value=False)
        g[name] = g["active_vehicles"].shift(steps).where(source_ok)

    # Trailing flow sums over completed intervals before t; all window intervals must be reliable.
    for source in FLOW_SOURCES:
        for suffix, window in FLOW_WINDOWS.items():
            window_ok = (
                reliable.shift(1, fill_value=False).astype(int).rolling(window).sum().eq(window)
            )
            g[f"{source}_{suffix}"] = g[source].shift(1).rolling(window).sum().where(window_ok)

    # Occupancy changes (inherit NaN from the already-masked lags).
    g["occupancy_change_15m"] = g["active_vehicles"] - g["lag_15m"]
    g["occupancy_change_1h"] = g["active_vehicles"] - g["lag_1h"]
    g["occupancy_change_2h"] = g["active_vehicles"] - g["lag_2h"]

    # Backward-looking 1h rolling means (t-1 ... t-4); all four intervals must be reliable.
    rolling_ok = (
        reliable.shift(1, fill_value=False).astype(int).rolling(ROLLING_STEPS).sum().eq(ROLLING_STEPS)
    )
    rolling_sources = {
        "rolling_mean_occupancy_1h": "active_vehicles",
        "rolling_mean_arrivals_1h": "arrivals",
        "rolling_mean_exits_1h": "exits",
        "rolling_mean_net_flow_1h": "net_flow",
    }
    for name, source in rolling_sources.items():
        g[name] = g[source].shift(1).rolling(ROLLING_STEPS).mean().where(rolling_ok)

    # Calendar features (from the row timestamp only).
    g["hour"] = g.index.hour
    g["day_of_week"] = g.index.dayofweek
    g["is_weekend"] = g["day_of_week"].isin([5, 6]).astype(int)

    # Direct multi-horizon targets: active_vehicles(t + H), NaN if t + H is unreliable or off-grid.
    for minutes, steps in HORIZON_STEPS.items():
        future_ok = reliable.shift(-steps, fill_value=False)
        g[target_column(minutes)] = g["active_vehicles"].shift(-steps).where(future_ok)

    return g


def build_wide_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """THE shared entry point: raw InOut dataframe -> wide feature/target frame (index = timestamp)."""
    df = validate_and_clean(raw)
    grid = build_occupancy_grid(df)
    grid = add_reliability(grid)
    return add_features(grid)


def build_wide_frame_from_excel(path) -> pd.DataFrame:
    return build_wide_frame(load_raw_excel(path))


# --------------------------------------------------------------------------------------
# 5. Training view: long format, one row per (timestamp, horizon)
# --------------------------------------------------------------------------------------
def build_training_set(wide: pd.DataFrame) -> pd.DataFrame:
    """Long-format supervised dataset (notebook sections 13 + 18).

    A row is kept only if the current interval is reliable and every feature and the target are
    available. No fillna. Columns: timestamp, 28 features, horizon_minutes, target.
    """
    parts = []
    for minutes in HORIZONS:
        tcol = target_column(minutes)
        part = wide.loc[wide["is_reliable"], BASE_FEATURES + [tcol]].dropna()
        part = part.rename(columns={tcol: "target"})
        part["horizon_minutes"] = minutes
        parts.append(part.reset_index())
    return pd.concat(parts, ignore_index=True)[LONG_COLUMNS]


def split_chronological(long_df: pd.DataFrame):
    """Chronological train / validation / test split on `timestamp` (never shuffled)."""
    ts = long_df["timestamp"]
    train = long_df[ts < TRAIN_END]
    validation = long_df[(ts >= TRAIN_END) & (ts < VALIDATION_END)]
    test = long_df[ts >= VALIDATION_END]
    return train, validation, test


# --------------------------------------------------------------------------------------
# 6. Inference views
# --------------------------------------------------------------------------------------
def _is_unreliable_timestamp(ts: pd.Timestamp) -> bool:
    """Return True when an exact timestamp falls in a known unreliable period."""
    ts = pd.Timestamp(ts)
    if ts.normalize() in pd.to_datetime(UNRELIABLE_DATES):
        return True
    return WASHOUT_START <= ts < WASHOUT_END


def _interval_is_reliable(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Return True when the open interval (start, end] does not overlap bad periods."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end <= start:
        return True

    # Any overlap with either complete bad date makes the interval unreliable.
    for date_text in UNRELIABLE_DATES:
        day_start = pd.Timestamp(date_text)
        day_end = day_start + pd.Timedelta(days=1)
        if start < day_end and end > day_start:
            return False

    # Washout interval is [WASHOUT_START, WASHOUT_END).
    if start < WASHOUT_END and end > WASHOUT_START:
        return False

    return True


def _active_vehicles_at(df: pd.DataFrame, ts: pd.Timestamp) -> int:
    """Count recorded vehicles inside the system at an exact timestamp."""
    return int(((df["Entry Time"] <= ts) & (df["Exit Time"] > ts)).sum())


def _window_counts(
    df: pd.DataFrame,
    end: pd.Timestamp,
    minutes: int,
) -> tuple[int, int, int]:
    """Count entries/exits in the interval (end-minutes, end]."""
    start = end - pd.Timedelta(minutes=minutes)
    if not _interval_is_reliable(start, end):
        raise InsufficientHistoryError(
            f"Requested {minutes}-minute history ending at {end} overlaps an unreliable period."
        )

    arrivals = int(((df["Entry Time"] > start) & (df["Entry Time"] <= end)).sum())
    exits = int(((df["Exit Time"] > start) & (df["Exit Time"] <= end)).sum())
    return arrivals, exits, arrivals - exits


def inference_frame_at(raw: pd.DataFrame, prediction_time) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Build model features anchored to an exact prediction timestamp.

    Unlike the historical 15-minute training grid, inference can use an arbitrary time such as
    10:17. The current state is measured exactly at that time, and recent windows are relative to
    that exact timestamp. A prediction at 10:17 therefore uses lag_15m from 10:02 and forecasts
    10:32 / 10:47 / 11:17 / ... for the requested horizons.

    The returned frame has one row per horizon and exactly MODEL_FEATURES in fixed order.
    """
    df = validate_and_clean(raw)
    ts = pd.Timestamp(prediction_time)

    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    if ts < df["Entry Time"].min():
        raise InsufficientHistoryError(
            f"Prediction time {ts} is earlier than the first observed entry {df['Entry Time'].min()}."
        )

    if _is_unreliable_timestamp(ts):
        raise InsufficientHistoryError(
            f"Prediction time {ts} falls in a known unreliable period. Cannot predict from it."
        )

    # The exact current state is known from the original vehicle intervals.
    current_active = _active_vehicles_at(df, ts)

    # Exact historical occupancy lags relative to prediction_time.
    lag_minutes = {
        "lag_15m": 15,
        "lag_30m": 30,
        "lag_1h": 60,
        "lag_2h": 120,
        "lag_3h": 180,
        "lag_4h": 240,
        "lag_6h": 360,
        "lag_24h": 1440,
    }
    lag_values: dict[str, int] = {}
    for name, minutes in lag_minutes.items():
        lag_ts = ts - pd.Timedelta(minutes=minutes)
        if _is_unreliable_timestamp(lag_ts):
            raise InsufficientHistoryError(
                f"Cannot build {name}: source timestamp {lag_ts} is unreliable."
            )
        lag_values[name] = _active_vehicles_at(df, lag_ts)

    # Exact recent flow windows ending at prediction_time.
    flow_values: dict[str, int] = {}
    for minutes in (15, 30, 60):
        arrivals, exits, net_flow = _window_counts(df, ts, minutes)
        suffix = {15: "15m", 30: "30m", 60: "1h"}[minutes]
        flow_values[f"arrivals_{suffix}"] = arrivals
        flow_values[f"exits_{suffix}"] = exits
        flow_values[f"net_flow_{suffix}"] = net_flow

    # Occupancy changes relative to exact lag timestamps.
    occupancy_change = {
        "occupancy_change_15m": current_active - lag_values["lag_15m"],
        "occupancy_change_1h": current_active - lag_values["lag_1h"],
        "occupancy_change_2h": current_active - lag_values["lag_2h"],
    }

    # Historical 1-hour rolling context: use the four completed 15-minute snapshots before t.
    past_times = [ts - pd.Timedelta(minutes=15 * k) for k in (1, 2, 3, 4)]
    if not all(not _is_unreliable_timestamp(x) for x in past_times):
        raise InsufficientHistoryError(
            f"Cannot build 1-hour rolling features for {ts}: reliable history is unavailable."
        )

    past_occupancy = [_active_vehicles_at(df, x) for x in past_times]
    past_flow = [_window_counts(df, x, 15) for x in past_times]
    rolling_mean_occupancy = sum(past_occupancy) / 4.0
    rolling_mean_arrivals = sum(v[0] for v in past_flow) / 4.0
    rolling_mean_exits = sum(v[1] for v in past_flow) / 4.0
    rolling_mean_net_flow = sum(v[2] for v in past_flow) / 4.0

    base = {
        "active_vehicles": current_active,
        **lag_values,
        **flow_values,
        **occupancy_change,
        "rolling_mean_occupancy_1h": rolling_mean_occupancy,
        "rolling_mean_arrivals_1h": rolling_mean_arrivals,
        "rolling_mean_exits_1h": rolling_mean_exits,
        "rolling_mean_net_flow_1h": rolling_mean_net_flow,
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "is_weekend": int(ts.dayofweek in [5, 6]),
    }

    rows = []
    for horizon in HORIZONS:
        row = dict(base)
        row["horizon_minutes"] = horizon
        rows.append(row)

    X = pd.DataFrame(rows, columns=MODEL_FEATURES)
    return ts, X


def inference_frame_from_excel(path, prediction_time=None) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Load a raw InOut workbook and build exact-time inference features.

    If `prediction_time` is omitted, the latest observed Exit/Entry time is used.
    For a live workflow, pass the actual current timestamp explicitly.
    """
    raw = load_raw_excel(path)
    clean = validate_and_clean(raw)
    if prediction_time is None:
        prediction_time = max(clean["Entry Time"].max(), clean["Exit Time"].max())
    return inference_frame_at(raw, prediction_time)


def latest_inference_frame(wide: pd.DataFrame):
    """Backward-compatible inference view for the validated 15-minute training grid.

    Prefer `inference_frame_at()` or `inference_frame_from_excel()` for production inference,
    because those functions support an arbitrary prediction timestamp such as 10:17.
    """
    ts = wide.index[-1]
    row = wide.iloc[-1]

    if not bool(row["is_reliable"]):
        raise InsufficientHistoryError(
            f"The latest interval {ts} falls in a known unreliable period "
            f"(Aug 22-23, or Aug 24 00:00-11:45 washout). Cannot predict from it."
        )

    missing = [f for f in BASE_FEATURES if pd.isna(row[f])]
    if missing:
        raise InsufficientHistoryError(
            f"Cannot build features for {ts}: unavailable feature(s) {missing}. "
            f"These need reliable history before this interval (up to 24h for lag_24h; 1h for "
            f"flow/rolling features). "
            f"The workbook has {len(wide)} 15-minute intervals starting {wide.index[0]}."
        )

    X = pd.DataFrame([row[BASE_FEATURES].to_dict() for _ in HORIZONS])
    X["horizon_minutes"] = HORIZONS
    return ts, X[MODEL_FEATURES]
