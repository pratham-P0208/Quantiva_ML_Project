# Quantiva — Flask + Next.js migration

This project preserves the original `src/` modules, all `data/` CSV files, and the original `app.py` Streamlit interface. The new Flask API wraps the **same Python functions**; the Next.js frontend supplies a responsive, animated interface.

## Important: model artifact

The uploaded ZIP **does not contain `models/best_model.joblib`**. The original `src/selected_model.py` checks for that file and otherwise uses the original Random Forest fallback. This migration does not train, replace, or modify models. If you have a saved best-model artifact from another copy of your project, place it at `models/best_model.joblib` without modifying it. For portability, ensure it was saved with the same scikit-learn version used by the backend (`1.5.1`).

## Requirements

- Python 3.12
- Node.js 20+

## Start backend (terminal 1)

From the project root:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
python -m backend.server
```

Flask starts on `http://localhost:5000`. Check `http://localhost:5000/api/health`.

## Start frontend (terminal 2)

```bash
cd frontend
npm install
# Optional: copy .env.example to .env.local
npm run dev
```

Visit `http://localhost:3000`.

## Endpoints

- `GET /api/health` — backend status and saved-model availability
- `GET /api/dashboard` — historical growth, metadata, observation dates
- `POST /api/optimize` — `{"tickers":["...","..."],"objective":"Maximum Sharpe Ratio","max_weight":0.25,"risk_free_rate":0.065}`
- `POST /api/risk` — `{"tickers":["..."]}`

All optimization and risk computations use the original `src/optimizer.py` and `src/risk.py`. All features and predictions use the original `src/features.py`, `src/selected_model.py`, and `src/model.py`. No backtesting logic has been added or modified.

## Validation

Run `python -m pytest tests/test_core.py tests/test_api.py -q` after installing the backend requirements and `pytest`. The tests compare Flask endpoint outputs against the original functions. First optimization may take several minutes because the original Random Forest fallback trains a model for each stock.

## Deployment

The new architecture **cannot be hosted as a single Streamlit Community Cloud app**. Deploy Flask to a Python-capable hosting provider and Next.js to Vercel or a Node-capable host. Set `FRONTEND_ORIGINS=https://your-frontend-domain` on Flask and `NEXT_PUBLIC_API_URL=https://your-backend-domain` on Next.js. Configure CORS and keep the API available during requests. No credentials are needed for local CSV-based analysis.

## Notes

- The original Streamlit `app.py` remains for comparison.
- Dataset and source modules have not been edited.
- Financial figures are academic estimates and not investment advice.
