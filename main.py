from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

import numpy as np
import onnxruntime as ort
import os
import json

# Load env
load_dotenv()

# Load ONNX model
onnx_session = ort.InferenceSession("./models/model.onnx")

# Load scaler parameters
with open("./models/scaler.json") as f:
    scaler_params = json.load(f)

scaler_mean = np.array(scaler_params["mean"])
scaler_scale = np.array(scaler_params["scale"])

def scale_features(features):
    return (features - scaler_mean) / scaler_scale

# FastAPI app
app = FastAPI()

class CHDInput(BaseModel):
    male: int
    age: float
    currentSmoker: int
    cigsPerDay: float
    BPMeds: int
    prevalentStroke: int
    prevalentHyp: int
    diabetes: int
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float

@app.post("/predict")
def predict_chd(input_data: CHDInput):
    # Buat array dari input user
    features = np.array([[ 
        input_data.male,
        input_data.age,
        input_data.currentSmoker,
        input_data.cigsPerDay,
        input_data.BPMeds,
        input_data.prevalentStroke,
        input_data.prevalentHyp,
        input_data.diabetes,
        input_data.totChol,
        input_data.sysBP,
        input_data.diaBP,
        input_data.BMI,
        input_data.heartRate,
        input_data.glucose
    ]], dtype=np.float32)

    # Scale
    features_scaled = scale_features(features).astype(np.float32)

    # Ambil nama input/output dari ONNX model
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    # Inference
    prediction = onnx_session.run([output_name], {input_name: features_scaled})[0][0][0]

    # Threshold
    risk = int(prediction > 0.5)

    return {
        "risk_score": round(float(prediction), 4),
        "risk_label": "At Risk" if risk == 1 else "No Risk"
    }
