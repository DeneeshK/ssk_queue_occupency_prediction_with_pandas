import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import feature_pipeline as fp  # noqa: E402

SPEC_FEATURES = [
    "active_vehicles", "lag_15m", "lag_30m", "lag_1h", "lag_2h", "lag_3h", "lag_4h", "lag_6h",
    "lag_24h", "arrivals_15m", "arrivals_30m", "arrivals_1h", "exits_15m", "exits_30m",
    "exits_1h", "net_flow_15m", "net_flow_30m", "net_flow_1h", "occupancy_change_15m",
    "occupancy_change_1h", "occupancy_change_2h", "rolling_mean_occupancy_1h",
    "rolling_mean_arrivals_1h", "rolling_mean_exits_1h", "rolling_mean_net_flow_1h",
    "hour", "day_of_week", "is_weekend", "horizon_minutes",
]


# ----------------------------------------------------------------------------- helpers
def make_raw(start="2026-08-01", end="2026-09-16", seed=0, per_hour=6):
    """Synthetic raw `InOut` frame: no entries on Aug 22-23 (mimics the real gap)."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, end, freq="1h")
    entries = []
    for h in hours:
        if h.strftime("%Y-%m-%d") in ("2026-08-22", "2026-08-23"):
            continue
        n = rng.poisson(per_hour)
        entries += [h + pd.Timedelta(seconds=float(s)) for s in rng.uniform(0, 3600, n)]
    entry = pd.Series(sorted(entries))
    dur = rng.uniform(20, 720, len(entry))
    exit_ = entry + pd.to_timedelta(dur, unit="m")
    return pd.DataFrame({
        "Country": np.nan, "Plate Prefix": np.nan, "Plate Number": np.nan,
        "Entry Time": entry, "Exit Time": exit_,
        "Risk Category": rng.choice(["GREEN", "RED", None], len(entry)),
        "Duration in Min": dur.round(2), "Time Taken": "x",
    })


@pytest.fixture(scope="module")
def raw():
    return make_raw()


@pytest.fixture(scope="module")
def wide(raw):
    return fp.build_wide_frame(raw)


def tiny_raw(rows):
    """rows: list of (entry, exit) strings."""
    return pd.DataFrame({
        "Entry Time": pd.to_datetime([r[0] for r in rows]),
        "Exit Time": pd.to_datetime([r[1] for r in rows]),
        "Risk Category": "GREEN", "Duration in Min": 10.0,
    })


# ----------------------------------------------------------------------------- input
def test_raw_columns_constant():
    assert fp.RAW_COLUMNS == ["Country", "Plate Prefix", "Plate Number", "Entry Time",
                              "Exit Time", "Risk Category", "Duration in Min", "Time Taken"]


def test_validation_drops_columns_and_keeps_supplied_duration(raw):
    df = fp.validate_and_clean(raw)
    assert list(df.columns) == fp.KEEP_COLUMNS
    assert (df["Duration in Min"].values == raw["Duration in Min"].values).all()


@pytest.mark.parametrize("col", fp.KEEP_COLUMNS)
def test_missing_required_column_raises(raw, col):
    with pytest.raises(fp.DataValidationError, match=col):
        fp.validate_and_clean(raw.drop(columns=[col]))


def test_missing_risk_becomes_unknown(raw):
    assert raw["Risk Category"].isna().any()
    df = fp.validate_and_clean(raw)
    assert not df["Risk Category"].isna().any()
    assert "UNKNOWN" in set(df["Risk Category"])


def test_exit_before_entry_and_missing_exit_raise(raw):
    bad = raw.copy()
    bad.loc[0, "Exit Time"] = bad.loc[0, "Entry Time"] - pd.Timedelta(minutes=1)
    with pytest.raises(fp.DataValidationError, match="earlier"):
        fp.validate_and_clean(bad)
    bad = raw.copy()
    bad.loc[0, "Exit Time"] = pd.NaT
    with pytest.raises(fp.DataValidationError, match="missing"):
        fp.validate_and_clean(bad)


# ----------------------------------------------------------------------------- time series
def test_grid_is_complete_and_15_minutes(wide):
    gaps = wide.index.to_series().diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(minutes=15)]


def test_entry_exit_and_occupancy_hand_example():
    # A: 00:05 -> 00:35 ; B: 00:10 -> 00:20 ; empty buckets must still exist
    g = fp.build_occupancy_grid(fp.validate_and_clean(
        tiny_raw([("2026-08-01 00:05", "2026-08-01 00:35"), ("2026-08-01 00:10", "2026-08-01 00:20")])))
    assert list(g.index.strftime("%H:%M")) == ["00:00", "00:15", "00:30"]
    assert g["arrivals"].tolist() == [2, 0, 0]
    assert g["exits"].tolist() == [0, 1, 1]
    assert g["net_flow"].tolist() == [2, -1, -1]
    assert g["active_vehicles"].tolist() == [2, 1, 0]


def test_event_totals_balance(wide):
    assert wide["arrivals"].sum() == wide["exits"].sum()
    assert wide["active_vehicles"].iloc[-1] == 0


def test_reliability_flags(wide):
    assert (~wide["is_reliable"]).sum() == 96 + 96 + 48
    assert not wide.loc["2026-08-22":"2026-08-23", "is_reliable"].any()
    assert not wide.loc["2026-08-24 00:00":"2026-08-24 11:45", "is_reliable"].any()
    assert wide.loc["2026-08-24 12:00", "is_reliable"]
    assert wide.loc["2026-08-21 23:45", "is_reliable"]


# ----------------------------------------------------------------------------- features
def test_feature_list_exact_order_and_no_leak_columns():
    assert fp.MODEL_FEATURES == SPEC_FEATURES and len(fp.MODEL_FEATURES) == 29
    assert "timestamp" not in fp.MODEL_FEATURES and "target" not in fp.MODEL_FEATURES
    assert fp.HORIZONS == [15, 30, 60, 120, 180, 360]


def test_lag_and_rolling_semantics():
    n = 120  # 30 hours, fully reliable (after Sep 1)
    idx = pd.date_range("2026-09-01", periods=n, freq="15min", name="timestamp")
    g = pd.DataFrame({"arrivals": np.arange(n, dtype=float), "exits": np.arange(n, dtype=float) * 0.5},
                     index=idx)
    g["net_flow"] = g["arrivals"] - g["exits"]
    g["active_vehicles"] = g["net_flow"].cumsum()
    g["is_reliable"] = True
    f = fp.add_features(g)
    t = 100
    assert f["lag_15m"].iloc[t] == g["active_vehicles"].iloc[t - 1]
    assert f["lag_24h"].iloc[t] == g["active_vehicles"].iloc[t - 96]
    assert f["arrivals_15m"].iloc[t] == g["arrivals"].iloc[t - 1]            # previous bucket only
    assert f["arrivals_1h"].iloc[t] == g["arrivals"].iloc[t - 4:t].sum()      # excludes bucket t
    assert f["rolling_mean_occupancy_1h"].iloc[t] == g["active_vehicles"].iloc[t - 4:t].mean()
    assert f["occupancy_change_1h"].iloc[t] == g["active_vehicles"].iloc[t] - g["active_vehicles"].iloc[t - 4]
    assert np.isnan(f["lag_24h"].iloc[95]) and not np.isnan(f["lag_24h"].iloc[96])


def test_no_future_leakage(wide):
    """Features at t must be identical when everything after t is removed or altered."""
    base = wide[["arrivals", "exits", "net_flow", "active_vehicles", "is_reliable"]]
    full = fp.add_features(base)
    for t in [pd.Timestamp("2026-08-10 09:00"), pd.Timestamp("2026-08-30 17:45"), pd.Timestamp("2026-09-10 03:15")]:
        truncated = fp.add_features(base.loc[:t])
        pd.testing.assert_series_equal(truncated.loc[t, fp.BASE_FEATURES], full.loc[t, fp.BASE_FEATURES],
                                       check_names=False)
        altered = base.copy()
        future = altered.index > t
        altered.loc[future, ["arrivals", "exits"]] = 999.0
        altered["net_flow"] = altered["arrivals"] - altered["exits"]
        altered.loc[future, "active_vehicles"] = 12345.0
        pd.testing.assert_series_equal(fp.add_features(altered).loc[t, fp.BASE_FEATURES],
                                       full.loc[t, fp.BASE_FEATURES], check_names=False)


def test_reliability_propagation(wide):
    isna = lambda ts, c: pd.isna(wide.loc[ts, c])
    # lag_24h reads t-24h: invalid until the source is past the washout
    assert isna("2026-08-25 11:45", "lag_24h") and not isna("2026-08-25 12:00", "lag_24h")
    assert isna("2026-08-24 17:45", "lag_6h") and not isna("2026-08-24 18:00", "lag_6h")
    # 1h flow / rolling windows need t-1..t-4 reliable
    for c in ("arrivals_1h", "exits_1h", "net_flow_1h", "rolling_mean_occupancy_1h", "rolling_mean_arrivals_1h"):
        assert isna("2026-08-24 12:45", c) and not isna("2026-08-24 13:00", c)
    assert isna("2026-08-24 12:00", "arrivals_15m") and not isna("2026-08-24 12:15", "arrivals_15m")
    # derived features inherit NaN; nothing is zero-filled inside the gap
    assert isna("2026-08-24 12:45", "occupancy_change_1h")
    gap = wide.loc["2026-08-22 00:15":"2026-08-24 11:45"]
    assert gap[["lag_15m", "arrivals_15m", "exits_15m", "arrivals_1h", "rolling_mean_occupancy_1h"]].isna().all().all()
    # the first gap interval may legitimately read a reliable source (Aug 21 23:45)
    assert not isna("2026-08-22 00:00", "lag_15m")


# ----------------------------------------------------------------------------- targets / training set
def test_target_alignment_and_reliability(wide):
    train = fp.build_training_set(wide)
    assert set(train["horizon_minutes"]) == set(fp.HORIZONS)
    for minutes in fp.HORIZONS:
        part = train[train["horizon_minutes"] == minutes]
        future = wide["active_vehicles"].reindex(part["timestamp"] + pd.Timedelta(minutes=minutes)).values
        assert (part["target"].values == future).all()
        # neither the current nor the target interval is unreliable
        assert wide["is_reliable"].reindex(part["timestamp"]).all()
        assert wide["is_reliable"].reindex(part["timestamp"] + pd.Timedelta(minutes=minutes)).all()
    assert not train.isna().any().any()
    assert list(train.columns) == fp.LONG_COLUMNS


# ----------------------------------------------------------------------------- splits
def test_chronological_split(wide):
    train, val, test = fp.split_chronological(fp.build_training_set(wide))
    assert len(train) and len(val) and len(test)
    assert train["timestamp"].max() < fp.TRAIN_END <= val["timestamp"].min()
    assert val["timestamp"].max() < fp.VALIDATION_END <= test["timestamp"].min()
    assert len(train) + len(val) + len(test) == len(fp.build_training_set(wide))


# ----------------------------------------------------------------------------- inference view
def test_latest_inference_frame(wide):
    ts, X = fp.latest_inference_frame(wide)
    assert ts == wide.index[-1]
    assert list(X.columns) == fp.MODEL_FEATURES and X["horizon_minutes"].tolist() == fp.HORIZONS
    assert not X.isna().any().any()


def test_inference_fails_without_history():
    short = fp.build_wide_frame(make_raw("2026-09-01", "2026-09-01 06:00", per_hour=6))
    with pytest.raises(fp.InsufficientHistoryError, match="lag_24h"):
        fp.latest_inference_frame(short)


def test_inference_fails_in_unreliable_period():
    w = fp.build_wide_frame(make_raw("2026-08-20", "2026-08-23 20:00"))
    with pytest.raises(fp.InsufficientHistoryError, match="unreliable"):
        fp.latest_inference_frame(w)


# ----------------------------------------------------------------------------- notebook equivalence
def notebook_reference(raw):
    """Literal port of the notebook cells (sections 1-18), used only to verify the shared pipeline."""
    df = raw[["Entry Time", "Exit Time", "Risk Category", "Duration in Min"]].copy()
    df["Risk Category"] = df["Risk Category"].fillna("UNKNOWN")
    df["Entry Bucket"] = df["Entry Time"].dt.floor("15min")
    df["Exit Bucket"] = df["Exit Time"].dt.floor("15min")
    arrivals = df.groupby("Entry Bucket").size().rename("arrivals")
    exits = df.groupby("Exit Bucket").size().rename("exits")
    ev = pd.concat([arrivals, exits], axis=1, sort=False).fillna(0)
    full_index = pd.date_range(df["Entry Time"].min().floor("15min"), df["Exit Time"].max().floor("15min"), freq="15min")
    ev = ev.reindex(full_index, fill_value=0)
    ev.index.name = "timestamp"
    ev["net_flow"] = ev["arrivals"] - ev["exits"]
    ev["active_vehicles"] = ev["net_flow"].cumsum()
    ml_base = ev.copy()

    candidate_lags = {"lag_15m": 1, "lag_30m": 2, "lag_1h": 4, "lag_2h": 8, "lag_3h": 12, "lag_4h": 16,
                      "lag_6h": 24, "lag_24h": 96}
    for n, s in candidate_lags.items():
        ml_base[n] = ml_base["active_vehicles"].shift(s)
    ml_base["is_reliable"] = True
    incomplete_dates = pd.to_datetime(["2026-08-22", "2026-08-23"]).date
    date_mask = pd.Series(ml_base.index.date, index=ml_base.index).isin(incomplete_dates)
    ml_base.loc[date_mask, "is_reliable"] = False
    wm = (ml_base.index >= pd.Timestamp("2026-08-24 00:00")) & (ml_base.index < pd.Timestamp("2026-08-24 12:00"))
    ml_base.loc[wm, "is_reliable"] = False
    for n, s in candidate_lags.items():
        src = ml_base["is_reliable"].shift(s, fill_value=False).astype("boolean")
        ml_base[n] = ml_base[n].mask(~src)
    for p in ("arrivals", "exits", "net_flow"):
        ml_base[f"{p}_15m"] = ml_base[p].shift(1)
        ml_base[f"{p}_30m"] = ml_base[p].shift(1).rolling(2).sum()
        ml_base[f"{p}_1h"] = ml_base[p].shift(1).rolling(4).sum()
    ml_base["occupancy_change_15m"] = ml_base["active_vehicles"] - ml_base["lag_15m"]
    ml_base["occupancy_change_1h"] = ml_base["active_vehicles"] - ml_base["lag_1h"]
    ml_base["occupancy_change_2h"] = ml_base["active_vehicles"] - ml_base["lag_2h"]
    ml_base["hour"] = ml_base.index.hour
    ml_base["day_of_week"] = ml_base.index.dayofweek
    ml_base["is_weekend"] = ml_base["day_of_week"].isin([5, 6]).astype(int)
    fh = {"target_15m": 1, "target_30m": 2, "target_1h": 4, "target_2h": 8, "target_3h": 12, "target_6h": 24}
    for n, s in fh.items():
        ml_base[n] = ml_base["active_vehicles"].shift(-s)
    for n, s in fh.items():
        ml_base[n] = ml_base[n].mask(~ml_base["is_reliable"].shift(-s, fill_value=False))
    flow = {"arrivals_15m": 1, "arrivals_30m": 2, "arrivals_1h": 4, "exits_15m": 1, "exits_30m": 2,
            "exits_1h": 4, "net_flow_15m": 1, "net_flow_30m": 2, "net_flow_1h": 4}
    for n, w in flow.items():
        ok = ml_base["is_reliable"].shift(1, fill_value=False).rolling(w).sum().eq(w)
        ml_base[n] = ml_base[n].mask(~ok)
    rs = {"rolling_mean_occupancy_1h": "active_vehicles", "rolling_mean_arrivals_1h": "arrivals",
          "rolling_mean_exits_1h": "exits", "rolling_mean_net_flow_1h": "net_flow"}
    for n, c in rs.items():
        ml_base[n] = ml_base[c].shift(1).rolling(4).mean()
    rok = ml_base["is_reliable"].shift(1, fill_value=False).rolling(4).sum().eq(4)
    for n in rs:
        ml_base[n] = ml_base[n].mask(~rok)

    feature_columns = SPEC_FEATURES[:-1]
    hm = {"target_15m": 15, "target_30m": 30, "target_1h": 60, "target_2h": 120, "target_3h": 180, "target_6h": 360}
    parts = []
    for n, m in hm.items():
        t = ml_base.loc[ml_base["is_reliable"], feature_columns + [n]].dropna().rename(columns={n: "target"})
        t["horizon_minutes"] = m
        parts.append(t.reset_index())
    return pd.concat(parts, ignore_index=True)[["timestamp", *feature_columns, "horizon_minutes", "target"]], ml_base


def test_matches_notebook_exactly(raw, wide):
    expected, expected_wide = notebook_reference(raw)
    actual = fp.build_training_set(wide)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=0, atol=0)
    # also the full wide frame (NaN pattern included) for the shared columns
    cols = fp.BASE_FEATURES
    pd.testing.assert_frame_equal(wide[cols].astype(float), expected_wide[cols].astype(float), check_dtype=False)


def test_exact_time_inference_is_anchored_to_requested_timestamp():
    rows = [
        ("2026-09-01 09:55", "2026-09-01 10:40"),
        ("2026-09-01 10:05", "2026-09-01 10:35"),
        ("2026-09-01 10:10", "2026-09-01 10:20"),
        ("2026-09-01 10:12", "2026-09-01 10:50"),
    ]
    raw = tiny_raw(rows)
    ts, X = fp.inference_frame_at(raw, "2026-09-01 10:17")

    assert ts == pd.Timestamp("2026-09-01 10:17")
    assert list(X.columns) == fp.MODEL_FEATURES
    assert X["horizon_minutes"].tolist() == fp.HORIZONS

    # Four recorded vehicles are inside at 10:17; one 15-minute lag is at 10:02.
    assert X.loc[X["horizon_minutes"] == 15, "active_vehicles"].iloc[0] == 4
    assert X.loc[X["horizon_minutes"] == 15, "lag_15m"].iloc[0] == 1

    # Recent 15-minute window is (10:02, 10:17]: three arrivals, no exits.
    assert X["arrivals_15m"].iloc[0] == 3
    assert X["exits_15m"].iloc[0] == 0
    assert X["net_flow_15m"].iloc[0] == 3


def test_exact_time_inference_rejects_unreliable_history():
    rows = [
        ("2026-08-21 23:35", "2026-08-22 12:00"),
        ("2026-09-01 00:00", "2026-09-01 02:00"),
    ]
    raw = tiny_raw(rows)
    with pytest.raises(fp.InsufficientHistoryError, match="unreliable"):
        fp.inference_frame_at(raw, "2026-08-24 12:05")
