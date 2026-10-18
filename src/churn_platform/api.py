
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from scripts.export_frozen_model import sha256_file
from scripts.final_heldout_evaluation import load_config
from scripts.train_baseline import FEATURES


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "final_random_forest.joblib"
META_PATH = (
    ROOT / "models" / "final_random_forest_metadata.json"
)


def load_trusted_model():
    if not MODEL_PATH.is_file() or not META_PATH.is_file():
        raise RuntimeError(
            "Model artifact missing. Run export_frozen_model first."
        )

    metadata = json.loads(
        META_PATH.read_text(encoding="utf-8")
    )

    config = load_config()

    if sha256_file(MODEL_PATH) != metadata["artifact_sha256"]:
        raise RuntimeError("Model artifact hash mismatch")

    if metadata["sklearn_version"] != sklearn.__version__:
        raise RuntimeError("scikit-learn version mismatch")

    if metadata["features"] != FEATURES:
        raise RuntimeError("Model feature mismatch")

    if metadata["parameters"] != config["model"]["parameters"]:
        raise RuntimeError("Frozen parameters mismatch")

    if metadata["training_rows"] != config["splits"][
        "expected_final_training_rows"
    ]:
        raise RuntimeError("Training row count mismatch")

    if metadata["held_out_test_used_for_training"]:
        raise RuntimeError("Invalid model training provenance")

    # Only load our locally generated, trusted artifact.
    # Never load arbitrary user-uploaded joblib files.
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]

    if bundle["features"] != FEATURES:
        raise RuntimeError("Artifact features mismatch")

    if bundle["dataset_sha256"] != metadata["dataset_sha256"]:
        raise RuntimeError("Dataset checksum mismatch")

    if list(model.classes_) != [0, 1]:
        raise RuntimeError("Invalid model classes")

    if list(model.feature_names_in_) != FEATURES:
        raise RuntimeError("Model column order mismatch")

    for name, value in config["model"]["parameters"].items():
        if model.get_params()[name] != value:
            raise RuntimeError("Model parameter mismatch")

    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_trusted_model()
    yield
    del app.state.bundle


app = FastAPI(
    title="Customer Churn Intelligence API",
    description="90-day customer purchase inactivity risk prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frequency_180d: float = Field(ge=0, allow_inf_nan=False)
    monetary_180d: float = Field(ge=0, allow_inf_nan=False)
    quantity_180d: float = Field(ge=0, allow_inf_nan=False)
    active_days_180d: float = Field(ge=0, allow_inf_nan=False)
    unique_products_180d: float = Field(ge=0, allow_inf_nan=False)
    recency_days: float = Field(ge=0, allow_inf_nan=False)
    tenure_180d: float = Field(ge=0, allow_inf_nan=False)
    avg_order_value_180d: float = Field(ge=0, allow_inf_nan=False)
    orders_30d: float = Field(ge=0, allow_inf_nan=False)
    spend_30d: float = Field(ge=0, allow_inf_nan=False)
    orders_90d: float = Field(ge=0, allow_inf_nan=False)
    spend_90d: float = Field(ge=0, allow_inf_nan=False)
    spend_share_30d: float = Field(ge=0, allow_inf_nan=False)
    has_repeat_purchase_day_180d: float = Field(
        ge=0, le=1, allow_inf_nan=False
    )
    avg_repeat_gap_days_180d: float = Field(
        ge=0, allow_inf_nan=False
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.get("/model-info")
def model_info():
    bundle = app.state.bundle

    return {
        "model": "frozen_random_forest",
        "target": "90_day_purchase_inactivity",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "training_snapshots": bundle["training_rows"],
        "threshold": bundle["threshold"],
        "probabilities_calibrated": False,
    }


@app.post("/predict")
def predict(payload: CustomerFeatures):
    bundle = app.state.bundle

    row = pd.DataFrame(
        [payload.model_dump()],
        columns=FEATURES,
    )

    model = bundle["model"]

    probability = float(
        model.predict_proba(row)[0, 1]
    )

    threshold = float(bundle["threshold"])

    return {
        "inactivity_risk_score": round(probability, 6),
        "predicted_inactive": probability >= threshold,
        "threshold": threshold,
        "prediction_horizon_days": 90,
        "note": (
            "Model risk score is not a guaranteed "
            "or calibrated real-world probability."
        ),
    }
