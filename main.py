from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

import numpy as np
import tensorflow as tf
import google.generativeai as genai
import os
import json

load_dotenv()

chd_model = tf.keras.models.load_model("./models/chd_model.h5")
with open("./models/scaler.json") as f:
    scaler_params = json.load(f)

scaler_mean = np.array(scaler_params["mean"])
scaler_scale = np.array(scaler_params["scale"])

def scale_features(features):
    return (features-scaler_mean) / scaler_scale

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
    ]])

    features_scaled = scale_features(features)
    prediction = chd_model.predict(features_scaled)[0][0]
    risk = int(prediction > 0.5)

    return {
        "risk_score": round(float(prediction), 4),
        "risk_label": "At Risk" if risk == 1 else "No Risk"
    }
    
class GeminiInput(BaseModel):
    prompt: str

@app.post("/chat")
def gemini_response(input_data: GeminiInput):
    
    model = genai.GenerativeModel("gemini-2.0-flash",
    system_instruction=(
        "Kamu adalah asisten pribadi yang hanya menjawab pertanyaan tentang penyakit jantung koroner (Coronary Heart Disease / CHD). "
        "Jika ada pertanyaan di luar topik jantung koroner, jangan jawab dan katakan dengan sopan: "
        "'Maaf, saya hanya bisa membantu menjawab pertanyaan terkait penyakit jantung koroner.' "
        "Saat menjawab pertanyaan CHD, gunakan bahasa yang mudah dipahami, jelas, edukatif, "
        "dan sarankan konsultasi ke dokter jika ada keluhan serius."
    ))

    response = model.generate_content(input_data.prompt)
    return {"response": response.text}
