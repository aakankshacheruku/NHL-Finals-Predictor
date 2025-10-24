from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import pandas as pd

# ----------------------------------------------------------------------
# Locate and load your saved model
# ----------------------------------------------------------------------
def _find_model_path() -> Path:
    """
    Looks for clf.joblib in common locations (new and old project structure).
    """
    candidate_paths = [
        Path(__file__).resolve().parents[1] / "data" / "outputs" / "model" / "clf.joblib",  # new app
        Path(__file__).resolve().parents[2] / "data" / "outputs" / "model" / "clf.joblib",  # old project
        Path("data/outputs/model/clf.joblib").resolve(),  # relative fallback
    ]
    for p in candidate_paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find clf.joblib. Checked:\n" + "\n".join(str(p) for p in candidate_paths)
    )

MODEL_PATH: Path = _find_model_path()
PIPELINE = joblib.load(MODEL_PATH)
ESTIMATOR = getattr(PIPELINE, "steps", [("model", PIPELINE)])[-1][1]


# ----------------------------------------------------------------------
# Model info
# ----------------------------------------------------------------------
def model_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "pipeline_type": type(PIPELINE).__name__,
        "estimator_type": type(ESTIMATOR).__name__,
        "model_path": str(MODEL_PATH),
        "has_predict_proba": hasattr(PIPELINE, "predict_proba"),
    }
    if hasattr(PIPELINE, "feature_names_in_"):
        info["feature_names_in_"] = list(PIPELINE.feature_names_in_)
    elif hasattr(ESTIMATOR, "feature_names_in_"):
        info["feature_names_in_"] = list(ESTIMATOR.feature_names_in_)
    return info


# ----------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------
def _required_feature_names() -> Optional[List[str]]:
    if hasattr(PIPELINE, "feature_names_in_"):
        return list(PIPELINE.feature_names_in_)
    if hasattr(ESTIMATOR, "feature_names_in_"):
        return list(ESTIMATOR.feature_names_in_)
    return None


def predict_proba_from_features(features: Dict[str, Any]) -> float:
    """
    Feed one feature dict into the pipeline and return the probability
    that Team A wins.
    """
    req = _required_feature_names()
    if req:
        # Fill missing expected columns with 0 or placeholder
        row = {name: features.get(name, 0.0) for name in req}
        X = pd.DataFrame([row], columns=req)
    else:
        X = pd.DataFrame([features])

    proba = PIPELINE.predict_proba(X)[0]
    return float(proba[1])


def predict_series_winner_prob(team_a: Dict[str, Any], team_b: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Predict the probability that Team A wins the series using the trained model.

    The model expects features:
    ['home', 'rest_days', 'back_to_back', 'win_pct_10',
     'goal_diff_pg_10', 'team', 'opp']
    """
    combined = {
        "home": team_a.get("home", 1),
        "rest_days": team_a.get("rest_days", 2),
        "back_to_back": team_a.get("back_to_back", 0),
        "win_pct_10": team_a.get("win_pct_10", 0.6),
        "goal_diff_pg_10": team_a.get("goal_diff_pg_10", 0.5),
        "team": team_a.get("team", "Panthers"),
        "opp": team_b.get("team", "Oilers"),
    }

    p = predict_proba_from_features(combined)
    return p, {"features_used": combined, "expected": _required_feature_names()}

