from .models import PredictionRequest
from .api_utils import get_model, transform_to_data

model = get_model()

def get_prediction(request: PredictionRequest) -> int:
    data_to_predict = transform_to_data(request)
    prediction = model.predict(data_to_predict)[0]
    return max(0, prediction) #return 0 if the score model is negative