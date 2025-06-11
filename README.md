# 🌧️ Rain Predictor Dashboard

This AI-powered dashboard predicts rainfall probability using real-time weather data and a trained XGBoost model.

## 🌐 Live on Render (after deployment)

Visit your app: `https://your-app-name.onrender.com`

## 📁 Project Structure

- `final_dash.py` — Main Dash app
- `assets/logo.jpeg` — UI logo
- `rain_predictor_xgb.pkl` — Pre-trained model
- `Procfile` — Render startup command
- `requirements.txt` — Python packages

## 🚀 Local Setup

```bash
pip install -r requirements.txt
python final_dash.py
```

## ☁️ Deploy to Render

1. Push this project to GitHub.
2. Sign in at https://render.com.
3. Click "New Web Service".
4. Use:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn final_dash:app`

You're live!