from pydantic import BaseModel

class ComentarioCreate(BaseModel):
    usuario_id: int
    establecimiento_id: int
    texto: str

class ComentarioResponse(BaseModel):
    id: int
    usuario_id: int
    establecimiento_id: int
    texto_original: str
    texto_censurado: str
    tfidf: float
    bert: float
    final_score: float
    clase: str

    class Config:
        orm_mode = True
