import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import uvicorn
import os

app = FastAPI()

# Load model
model = load_model("models/cough_model.keras")

# Prediction function
def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=16000)

    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

    max_len = 128
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc.reshape(1, 40, 128, 1)


@app.get("/")
def home():
    return {"message": "Cough Analysis API Running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as f:
        f.write(await file.read())

    features = extract_features(file_location)

    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)

    os.remove(file_location)

    if predicted_class == 0:
        result = "Healthy"
    else:
        result = "Respiratory Issue"

    return {
        "prediction": result,
        "confidence": float(np.max(prediction))
    }


if __name__ == "__main__":
    import uvicorn
    # Change 127.0.0.1 to 0.0.0.0
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)