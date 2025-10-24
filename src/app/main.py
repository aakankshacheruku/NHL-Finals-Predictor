# src/app/main.py

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="NHL Finals Predictor", version="1.0")

# ---------- ROOT & HEALTH ENDPOINTS ----------

@app.get("/")
def root():
    """Redirects the user to the interactive API docs."""
    return RedirectResponse(url="/docs")

@app.get("/healthz")
def healthz():
    """Health check endpoint for Render and monitoring."""
    return {"ok": True}

# ---------- SAMPLE PREDICTOR ENDPOINT ----------

@app.get("/predict")
def predict(team_a_goals: int, team_b_goals: int):
    """
    Example endpoint showing how prediction might work.
    Replace this with your actual model logic.
    """
    # Dummy model logic (replace with your trained model)
    data = pd.DataFrame([[team_a_goals, team_b_goals]], columns=["team_a_goals", "team_b_goals"])
    model = LogisticRegression()
    model.fit([[0, 0], [1, 1]], [0, 1])
    prediction = model.predict(data)[0]
    winner = "Team A" if prediction == 1 else "Team B"
    return {"predicted_winner": winner}

