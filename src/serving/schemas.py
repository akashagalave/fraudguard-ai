from pydantic import BaseModel
from typing import Dict, Any, List, Optional


class PredictionRequest(BaseModel):
    features: Dict[str, Any]


class ShapReason(BaseModel):
    feature: str
    impact: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    top_reasons: Optional[List[ShapReason]] = None
