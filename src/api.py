"""
Scoring service: given a customer's feature vector, return the predicted
uplift from sending them a promo/discount, plus a recommendation.

Run with:
    uvicorn src.api:app --reload

The model artifact is produced by scripts/train.py — run that first.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "uplift_model.pkl"

app = FastAPI(
    title="Discount Targeting Uplift API",
    description="Scores customers on predicted incremental effect of a promo, "
    "not just predicted conversion probability.",
)

_model_bundle = None


def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="No trained model found. Run `python scripts/train.py --synthetic` first.",
            )
        with open(MODEL_PATH, "rb") as f:
            _model_bundle = pickle.load(f)
    return _model_bundle


class CustomerFeatures(BaseModel):
    features: list[float] = Field(..., min_length=12, max_length=12, description="f0..f11")


class UpliftResponse(BaseModel):
    predicted_uplift: float
    recommend_treatment: bool
    model_used: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=UpliftResponse)
def score(customer: CustomerFeatures):
    bundle = get_model_bundle()
    model = bundle["model"]
    uplift = float(model.predict_uplift([customer.features])[0])
    return UpliftResponse(
        predicted_uplift=uplift,
        # threshold at 0: only recommend the discount if it's predicted to
        # help. In a real deployment this threshold should account for
        # discount cost/margin, not just sign of the effect.
        recommend_treatment=uplift > 0,
        model_used=bundle["model_name"],
    )
