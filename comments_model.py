from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base
from datetime import datetime

class ComentarioDB(Base):
    __tablename__ = "comentarios"

    id = Column(Integer, primary_key=True, autoincrement=True)

    usuario_id = Column(Integer, nullable=False)
    establecimiento_id = Column(Integer, nullable=False)

    texto_original = Column(String(500), nullable=False)
    texto_censurado = Column(String(500), nullable=False)

    tfidf = Column(Float)
    bert = Column(Float)
    final_score = Column(Float)
    clase = Column(String(20))

    fecha = Column(DateTime, default=datetime.utcnow)
