import os
import librosa
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Respiratory Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once
MODEL_PATH = "models/cough_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def extract_api_features(audio_path):
    """Ensures output is ALWAYS (1, 64, 80, 3)"""
    y, sr = librosa.load(audio_path, sr=16000, duration=1.0)
    
    # Standardize audio length to 1 second
    if len(y) < 16000:
        y = np.pad(y, (0, 16000 - len(y)))
    else:
        y = y[:16000]
    
    # Extract Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=201)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Calculate Deltas
    delta = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)
    
    # Stack into 3 channels
    features = np.stack([mel_db, delta, delta2], axis=-1)
    
    # FORCE SHAPE to exactly 80 time steps
    if features.shape[1] > 80:
        features = features[:, :80, :]
    elif features.shape[1] < 80:
        pad_width = 80 - features.shape[1]
        features = np.pad(features, ((0,0), (0, pad_width), (0,0)), mode='constant')

    return features.reshape(1, 64, 80, 3)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Inference
        processed_input = extract_api_features(temp_path)
        prediction = model.predict(processed_input)
        
        prob = float(prediction[0][0])
        return {
            "prediction": "Symptomatic" if prob > 0.5 else "Healthy",
            "confidence": round(prob, 4),
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)