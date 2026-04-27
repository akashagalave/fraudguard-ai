from pydantic import BaseModel,EmailStr
from typing import Dict, Any, List, Optional



class PredictionRequest(BaseModel):
    transaction_id: str
    user_email: EmailStr
    features: Dict[str, Any]



class ShapReason(BaseModel):
    feature: str
    impact: float



class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_score: int            
    is_fraud: bool
    model_version: str          

    
    top_reasons: Optional[List[ShapReason]] = None
    human_explanations: Optional[List[str]] = None