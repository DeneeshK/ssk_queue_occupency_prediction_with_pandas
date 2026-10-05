from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

import feature_pipeline as fp

DEFAULT_RAW = Path("data/raw/Vehicle_Data_ DwellTime _Tra.xlsx")
DEFAULT_MODELS_DIR = Path("models")


def load_active_version(models_dir: Path) -> str:
    path = models_dir / "active_version.json"
    if not path.exists():
        raise FileNotFoundError(f"Active model registry not found: {path}")
    version = json.loads(path.read_text()).get("active_version")
    if not version:
        raise ValueError("No active model version is registered.")
    return version


def validate_schema(models_dir: Path, version: str) -> list[str]:
    schema_path = models_dir / version / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}")
    features = json.loads(schema_path.read_text()).get("feature_names")
    if features != fp.MODEL_FEATURES:
        raise ValueError("Saved model feature schema does not match feature_pipeline.MODEL_FEATURES.")
    return features


def predict(raw_path: Path, models_dir: Path, prediction_time=None) -> dict:
    version = load_active_version(models_dir)
    features = validate_schema(models_dir, version)
    version_dir = models_dir / version

    ts, X = fp.inference_frame_from_excel(raw_path, prediction_time=prediction_time)

    point = joblib.load(version_dir / "xgb_multihorizon_point.joblib")
    p50 = joblib.load(version_dir / "xgb_multihorizon_p50.joblib")
    p90 = joblib.load(version_dir / "xgb_multihorizon_p90.joblib")

    if list(X.columns) != features:
        raise ValueError("Inference feature order does not match the saved model schema.")

    point_pred = point.predict(X[features])
    p50_pred = p50.predict(X[features])
    p90_pred = p90.predict(X[features])

    results = []
    for i, horizon in enumerate(fp.HORIZONS):
        if p90_pred[i] < p50_pred[i]:
            raise ValueError(f"P90 prediction is below P50 at horizon {horizon} minutes.")
        results.append({
            "horizon_minutes": horizon,
            "point_prediction": float(point_pred[i]),
            "p50_prediction": float(p50_pred[i]),
            "p90_prediction": float(p90_pred[i]),
        })

    current_active = int(X["active_vehicles"].iloc[0])
    return {
        "model_version": version,
        "prediction_time": pd.Timestamp(ts).isoformat(),
        "current_active_vehicles": current_active,
        "forecast": results,
    }


def predict_peak(raw_path: Path, models_dir: Path, prediction_time=None) -> dict:
    """Return the highest forecast among the configured horizons.

    Peak selection is based on the point forecast. P50 and P90 from the same
    horizon are included for context. No capacity threshold is assumed.
    """
    result = predict(raw_path, models_dir, prediction_time)
    peak = max(result["forecast"], key=lambda item: item["point_prediction"])
    peak_time = pd.Timestamp(result["prediction_time"]) + pd.Timedelta(
        minutes=int(peak["horizon_minutes"])
    )

    return {
        "model_version": result["model_version"],
        "prediction_time": result["prediction_time"],
        "current_active_vehicles": result["current_active_vehicles"],
        "peak_forecast": {
            "horizon_minutes": peak["horizon_minutes"],
            "forecast_timestamp": peak_time.isoformat(),
            "point_prediction": peak["point_prediction"],
            "p50_prediction": peak["p50_prediction"],
            "p90_prediction": peak["p90_prediction"],
        },
        "all_forecasts": result["forecast"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SSK occupancy predictions.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--prediction-time", type=str, default=None,
                        help="Optional exact prediction timestamp. Defaults to latest observed raw timestamp.")
    args = parser.parse_args()

    result = predict(args.raw, args.models_dir, args.prediction_time)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
