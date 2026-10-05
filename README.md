# SSK Vehicle Occupancy Forecasting

Simple production-oriented pipeline for the finalized multi-horizon XGBoost occupancy model.

## Layout

```text
data/raw/Vehicle_Data_ DwellTime _Tra.xlsx   # input Excel
feature_pipeline.py                           # single feature source of truth
train.py                                      # training + evaluation + versioning
predict.py                                    # offline inference
retrain.py                                    # same retraining flow
app.py                                        # FastAPI
models/vXXX/                                  # immutable model versions
models/active_version.json                    # active model pointer
```

## 1. Train

Place the original Excel workbook in `data/raw/` and run:

```bash
python train.py
```

This creates the deterministic supervised dataset:

```text
data/processed/supervised_multihorizon.csv
```

and a version such as:

```text
models/v001/
    xgb_multihorizon_point.joblib
    xgb_multihorizon_p50.joblib
    xgb_multihorizon_p90.joblib
    lightgbm_multihorizon_point.joblib
    feature_schema.json
    metrics.json
    training_metadata.json
```

The first candidate becomes active automatically. Later candidates are promoted only when validation mean MAE **and** validation mean MSE are both lower than the active model. The test set is recorded but not used for promotion.

## 2. Offline prediction

By default, use the latest observed timestamp from the raw workbook:

```bash
python predict.py
```

You can also provide an exact prediction timestamp:

```bash
python predict.py --prediction-time "2026-09-17 23:59:41"
```

The output contains current active vehicles plus point, P50 and P90 forecasts for 15m, 30m, 1h, 2h, 3h and 6h.

## 3. FastAPI

Start the API from the project root:

```bash
uvicorn app:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Postman — upload an Excel file

Request:

```text
POST http://127.0.0.1:8000/predict
```

Body → `form-data`:

```text
file              File     <select the Excel workbook>
prediction_time   Text     2026-09-17 23:59:41   # optional
```

If `prediction_time` is omitted, the latest observed timestamp in the uploaded workbook is used.

No file is required when testing the local default workbook; `/predict` will then use `data/raw/Vehicle_Data_ DwellTime _Tra.xlsx`.

## 4. Tests

```bash
pytest -q
```

## Notes

- `active_vehicles` is an in-system occupancy/congestion proxy, not a direct physical queue measurement.
- The feature pipeline is shared by training and inference.
- The August 22–23 unreliable period and Aug 24 washout are not fabricated or zero-filled.
- P50/P90 are XGBoost quantile models; LightGBM is a point-forecast benchmark only.
