import os
import pickle
from typing import Dict, Any
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="subscription prediction")

model_name = os.getenv('MODEL_NAME', 'pipeline_v1.bin')

if not os.path.exists(model_name):
    raise FileNotFoundError(f"Model file: {model_name} not found")

with open(model_name,'rb') as f_in:
    pipeline = pickle.load(f_in)

@app.post('/predict')
def predict(client: Dict[str, Any]):
    proba = pipeline.predict_proba(client)[0, 1]

    return {
        "model_name": model_name,
        "subscription_probability": float(proba),
        "subscription": bool(proba >= 0.5)
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9696)