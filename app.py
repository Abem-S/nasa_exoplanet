from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Load model & threshold
model = joblib.load("lgb_model.pkl")
threshold = joblib.load("threshold.pkl")

# Define expected features
feature_cols = [
    "orbital_period", "transit_duration", "transit_depth", 
    "planet_radius", "insolation_flux", "equilibrium_temp", 
    "stellar_teff", "stellar_radius", "stellar_mag", 
    "fpflag_nt", "fpflag_ss", "fpflag_co", "fpflag_ec"
]

# Create FastAPI app
app = FastAPI(title="Exoplanet Classifier API")

# root route for testing
@app.get("/")
def read_root():
    return {"status": "API is running"}

# Define expected input model
class PlanetData(BaseModel):
    orbital_period: float
    transit_duration: float
    transit_depth: float
    planet_radius: float
    insolation_flux: float
    equilibrium_temp: float
    stellar_teff: float
    stellar_radius: float
    stellar_mag: float
    fpflag_nt: int = 0
    fpflag_ss: int = 0
    fpflag_co: int = 0
    fpflag_ec: int = 0

# Prediction endpoint
@app.post("/predict")
def predict_planet(data: PlanetData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Check for missing features
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        return {"error": f"Missing columns: {missing_cols}"}

    # Reorder columns to match model input
    df = df[feature_cols]

    # Predict probabilities
    probs = model.predict_proba(df)[0]

    # CONFIRMED class index
    conf_idx = list(model.classes_).index("CONFIRMED")

    # Apply custom threshold for CONFIRMED
    if probs[conf_idx] >= threshold:
        prediction = "CONFIRMED"
    else:
        other_idx = [i for i in range(len(probs)) if i != conf_idx]
        prediction = model.classes_[other_idx[np.argmax(probs[other_idx])]]

    return {
        "prediction": prediction,
        "probabilities": dict(zip(model.classes_, probs))
    }

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
