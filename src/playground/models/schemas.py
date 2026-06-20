from pydantic import BaseModel


class ModelOut(BaseModel):
    id: str
    provider: str
    model_name: str
    display_name: str


class ModelsResponse(BaseModel):
    models: list[ModelOut]
