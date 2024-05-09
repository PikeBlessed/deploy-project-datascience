from fastapi import FastAPI
from .app.models import PredictionRequest, PredictionResponse
from .app.views import get_prediction

app = FastAPI(docs_url='/')

@app.post('/v1/predictions')
def make_model_predictions(request: PredictionRequest):
    return PredictionResponse(reach=get_prediction(request))