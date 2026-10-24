import sys
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pathlib import Path
from typing import List
from pydantic import BaseModel

from api.schemas import Customer, Prediction

app = FastAPI(title="Customer Persona Segmentation")

pipe = None
persona_map = {}

@app.on_event("startup")
def load_artifacts():
    global pipe, persona_map
    base_dir = Path(__file__).resolve().parent.parent
    
    # Add ml to sys path so joblib can find features.FeatureEngineer
    sys.path.append(str(base_dir / "ml"))
    
    artifacts_dir = base_dir / "artifacts"
    pipe_path = artifacts_dir / "segmenter.joblib"
    map_path = artifacts_dir / "persona_map.json"
    
    if pipe_path.exists():
        pipe = joblib.load(pipe_path)
    if map_path.exists():
        with open(map_path, "r") as f:
            persona_map = json.load(f)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipe is not None}

@app.get("/segments")
def get_segments():
    return persona_map

@app.post("/predict", response_model=Prediction)
def predict(c: Customer):
    X = pd.DataFrame([c.model_dump()])
    cid = int(pipe.predict(X)[0])
    dist = float(pipe.transform(X)[0][cid])
    p = persona_map.get(str(cid), {"name": "Unknown", "description": "No description available"})
    
    return Prediction(
        cluster_id=cid,
        persona=p["name"],
        description=p["description"],
        distance_to_centroid=dist,
    )

class BatchPrediction(BaseModel):
    predictions: List[Prediction]

@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(customers: List[Customer]):
    X = pd.DataFrame([c.model_dump() for c in customers])
    
    cluster_ids = pipe.predict(X)
    distances = pipe.transform(X)
    
    results = []
    for i, cid in enumerate(cluster_ids):
        cid = int(cid)
        dist = float(distances[i][cid])
        p = persona_map.get(str(cid), {"name": "Unknown", "description": "No description available"})
        
        results.append(Prediction(
            cluster_id=cid,
            persona=p["name"],
            description=p["description"],
            distance_to_centroid=dist,
        ))
        
    return BatchPrediction(predictions=results)

