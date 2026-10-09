from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from predict import predict, predict_peak

app = FastAPI(title="SSK Vehicle Occupancy Forecast API", version="1.0.0")
RAW_DEFAULT = Path("data/raw/Vehicle_Data_ DwellTime _Tra.xlsx")
MODELS_DIR = Path("models")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def forecast(
    file: UploadFile | None = File(default=None),
    prediction_time: str | None = Form(default=None),
) -> dict:
    """Predict occupancy from an optional uploaded raw InOut Excel workbook.

    If no file is uploaded, the default workbook under data/raw/ is used.
    """
    if file is None:
        raw_path = RAW_DEFAULT
        try:
            return predict(raw_path, MODELS_DIR, prediction_time)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    suffix = Path(file.filename or "input.xlsx").suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Please upload an Excel .xlsx or .xls file.")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            tmp.write(await file.read())
        return predict(temp_path, MODELS_DIR, prediction_time)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

@app.post("/predict/peak")
async def forecast_peak(
    file: UploadFile | None = File(default=None),
    prediction_time: str | None = Form(default=None),
) -> dict:
    """Return the highest predicted occupancy across all configured horizons."""
    if file is None:
        raw_path = RAW_DEFAULT
        try:
            return predict_peak(raw_path, MODELS_DIR, prediction_time)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    suffix = Path(file.filename or "input.xlsx").suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Please upload an Excel .xlsx or .xls file.")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            tmp.write(await file.read())
        return predict_peak(temp_path, MODELS_DIR, prediction_time)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)