# Exoplanet Classification Backend

This backend predicts whether an observed exoplanet signal is Confirmed, Candidate, or False Positive using data from NASA’s Kepler and TESS missions.
It is powered by a trained LightGBM machine learning model and served through a FastAPI web API.

Live API: https://nasa-exoplanet.onrender.com

## Project Overview

The goal is to help identify real exoplanets among thousands of detections by analyzing their orbital and stellar features.
The backend loads the trained model and scaler, applies preprocessing, and returns class predictions with probabilities.

## Files Included

1. exoplanet.ipynb: Notebook for training, preprocessing, and evaluating the LightGBM model
2. app.py: FastAPI backend that serves predictions
3. lgb_model.pkl: Trained LightGBM model
4. scaler.pkl: RobustScaler used for normalization
5. threshold.pkl: Custom decision threshold to improve recall for confirmed planets

## Tech Stack
1. Python 3.10+
2. LightGBM – model training
3. scikit-learn – scaling, SMOTE, and evaluation
4. FastAPI – web API for serving predictions
5. joblib – model and scaler saving/loading
6. numpy - numeric arrays and math
7. pandas - data loading and preprocessing
8. imbalanced-learn - SMOTE for class balancing

## How It Works
1. User Input: Planet and star features are sent to the API.
2. Scaling: Inputs are scaled using the same scaler as during training.
3. Model Prediction: The LightGBM model predicts probabilities for each class.
4. Custom Threshold: A tuned threshold (stored in threshold.pkl) adjusts the decision to improve recall for confirmed planets.
5. Output: The API returns both the predicted class and probabilities.

## Model Highlights
1. Model: LightGBM (Gradient Boosting Tree)
2. Accuracy: ~82.5% (default)
3. Recall for Confirmed Planets: 0.91 (after threshold tuning)
4. Balanced Classes: SMOTE used to handle few confirmed planets

## Running the API Locally

```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the API
uvicorn app:app --reload

# 3. Send a test request
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
          "orbital_period": 5.0,
          "transit_duration": 0.2,
          "transit_depth": 5000,
          "planet_radius": 3.0,
          "insolation_flux": 150.0,
          "equilibrium_temp": 600,
          "stellar_teff": 5500,
          "stellar_radius": 0.9,
          "stellar_mag": 12.0,
          "fpflag_nt": 0,
          "fpflag_ss": 0,
          "fpflag_co": 0,
          "fpflag_ec": 0
      }'
```
## Notes
1. Accuracy slightly decreases after threshold tuning, but recall for confirmed planets increases.
2. Recall was prioritized to ensure confirmed exoplanets are not missed.
3. More confirmed planet data would improve future versions

