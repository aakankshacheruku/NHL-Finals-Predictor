from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any

from src.models.elo import win_prob_from_elo
from src.models.series import series_win_prob

from src.app.predictor_model import (
    model_info,
    predict_proba_from_features,
    predict_series_winner_prob,
)

app = FastAPI(title="NHL Finals Predictor", version="0.2.0")

@app.get("/ping")
def ping():
    return {"status": "ok"}

class EloSeriesRequest(BaseModel):
    elo_a: float
    elo_b: float
    home: str | None = Field(default="A")
    best_of: int = Field(default=7)

@app.post("/predict/elo")
def predict_elo(req: EloSeriesRequest):
    p_game = win_prob_from_elo(req.elo_a, req.elo_b, home=req.home)
    p_series = series_win_prob(p_game, best_of=req.best_of)
    return {
        "p_game_team_a": p_game,
        "p_game_team_b": 1 - p_game,
        "p_series_team_a": p_series,
        "p_series_team_b": 1 - p_series,
        "inputs": req.model_dump(),
    }

class ModelRawRequest(BaseModel):
    features: Dict[str, Any]

class ModelSeriesRequest(BaseModel):
    team_a: Dict[str, Any]
    team_b: Dict[str, Any]

@app.get("/model/info")
def get_model_info():
    try:
        return model_info()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/predict/model/raw")
def predict_model_raw(req: ModelRawRequest):
    try:
        p = predict_proba_from_features(req.features)
        return {"prob_team_a_wins": p, "prob_team_b_wins": 1 - p, "echo": {"features": req.features}}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model inference failed: {e}")

@app.post("/predict/model/series")
def predict_model_series(req: ModelSeriesRequest):
    try:
        p, meta = predict_series_winner_prob(req.team_a, req.team_b)
        return {"prob_team_a_wins": p, "prob_team_b_wins": 1 - p, **meta}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Series prediction failed: {e}")    
