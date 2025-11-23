from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from datetime import datetime
from database import Base

class Comentario(Base):
    __tablename__ = "comentarios"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, nullable=False)
    establecimiento_id = Column(Integer, nullable=False)

    texto_original = Column(String(500))
    texto_censurado = Column(String(500))

    tfidf = Column(Float)
    bert = Column(Float)
    final_score = Column(Float)

    clase = Column(String(20))
    es_toxico = Column(Boolean, default=False)

    fecha = Column(DateTime, default=datetime.utcnow)
