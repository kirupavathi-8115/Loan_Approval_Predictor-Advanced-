from pydantic import BaseModel, Field

class Customer(BaseModel):
    age: int = Field(ge=18, le=100)
    income: float = Field(gt=0)
    total_spend: float = Field(ge=0)
    num_purchases: int = Field(ge=0)
    recency_days: int = Field(ge=0)
    web_visits_per_month: int = Field(ge=0)

class Prediction(BaseModel):
    cluster_id: int
    persona: str
    description: str
    distance_to_centroid: float
