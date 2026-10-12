from pydantic import BaseModel, Field
class ClassificationRequest(BaseModel):
    subject:str=Field(min_length=1)
    body:str=Field(min_length=1)

class ClassificationResponse(BaseModel):
    category:str
    priority:str
    needs_review: bool=False
    confidence:float=Field(ge=0.0, le=1.0)