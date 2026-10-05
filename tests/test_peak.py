from pathlib import Path

import predict as pred


def test_predict_peak_selects_highest_point(monkeypatch):
    fake = {
        "model_version": "v001",
        "prediction_time": "2026-09-17T10:00:00",
        "current_active_vehicles": 180,
        "forecast": [
            {"horizon_minutes": 15, "point_prediction": 181.0, "p50_prediction": 180.0, "p90_prediction": 190.0},
            {"horizon_minutes": 60, "point_prediction": 210.0, "p50_prediction": 208.0, "p90_prediction": 230.0},
            {"horizon_minutes": 360, "point_prediction": 205.0, "p50_prediction": 207.0, "p90_prediction": 250.0},
        ],
    }
    monkeypatch.setattr(pred, "predict", lambda raw_path, models_dir, prediction_time=None: fake)

    result = pred.predict_peak(Path("input.xlsx"), Path("models"))

    assert result["peak_forecast"]["horizon_minutes"] == 60
    assert result["peak_forecast"]["point_prediction"] == 210.0
    assert result["peak_forecast"]["forecast_timestamp"] == "2026-09-17T11:00:00"
    assert result["all_forecasts"] == fake["forecast"]
