from pydantic import BaseModel

class PredictionRequest(BaseModel):
    format: int
    comments: int
    likes: int
    saved: int
    shares: int
    
class PredictionResponse(BaseModel):
    reach: int
