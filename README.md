# NHL Finals Predictor (Mac-ready)

A minimal FastAPI app with:
- Elo-based series predictor
- **Model-powered** series predictor endpoints wired for your `clf.joblib` Pipeline

## Quickstart (macOS + Terminal)

```bash
cd ~/Desktop/NHL-Finals-Predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn src.app.main:app --reload
# open http://127.0.0.1:8000/docs
```

## Where to put your trained model
If you have `NHL-Finals-2026/data/outputs/model/clf.joblib`, the app can find it there.
Or copy it here:
```
src/data/outputs/model/clf.joblib
```

## Endpoints
- `GET /ping`
- `GET /model/info` — see pipeline details & required feature names
- `POST /predict/model/raw` — feed a single-row feature dict directly to the pipeline
- `POST /predict/model/series` — pass Team A & Team B stats; we compute diffs & predict

You can adjust the feature mapping in `src/app/predictor_model.py`.
