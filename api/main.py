from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .preprocess import preprocess
from .predict import predict

app = FastAPI()

class PredictRequest(BaseModel):
    model_type: str
    data: List[List[int]]

@app.get('/predict')
def predict_endpoint(request: PredictRequest):
    data = preprocess(request.model_type, request.data)
    predicted = predict(request.model_type, data)
    return {'status': 'success', 'predicted': predicted}