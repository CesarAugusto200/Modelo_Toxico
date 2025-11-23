from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
import requests

from database import SessionLocal, engine, Base
from comments_model import ComentarioDB
from schemas import ComentarioCreate, ComentarioResponse

from fastapi.middleware.cors import CORSMiddleware

# ==============================
# Configuración general
# ==============================

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear tablas
Base.metadata.create_all(bind=engine)


# ==============================
# Dependencia DB
# ==============================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==============================
# POST → Crear comentario
# ==============================
@app.post("/comentarios", response_model=ComentarioResponse)
def crear_comentario(data: ComentarioCreate, db: Session = Depends(get_db)):

    # ---- Llamada al modelo con protección ----
    try:
        respuesta = requests.post(
            "http://127.0.0.1:8001/clasificar",
            json={"texto": data.texto},
            timeout=10
        ).json()

    except Exception as e:
        return {"error": f"No se pudo conectar con el modelo: {str(e)}"}

    # ---- Validar respuesta del modelo ----
    if not isinstance(respuesta, dict) or "prob_final" not in respuesta:
        return {"error": "La API del modelo devolvió una respuesta inválida."}

    # ---- Crear registro ----
    nuevo = ComentarioDB(
        usuario_id=data.usuario_id,
        establecimiento_id=data.establecimiento_id,
        texto_original=respuesta["entrada"],
        texto_censurado=respuesta["texto_censurado"],
        tfidf=respuesta["prob_tfidf"],
        bert=respuesta["prob_bert"],
        final_score=respuesta["prob_final"],
        clase="tóxico" if respuesta["es_toxico"] else "normal"
    )

    db.add(nuevo)
    db.commit()
    db.refresh(nuevo)

    return nuevo


# ==============================
# GET → Comentarios por establecimiento
# ==============================
@app.get("/comentarios/establecimiento/{id}", response_model=list[ComentarioResponse])
def obtener_por_establecimiento(id: int, db: Session = Depends(get_db)):
    return db.query(ComentarioDB).filter(ComentarioDB.establecimiento_id == id).all()


# ==============================
# GET → Comentarios por usuario
# ==============================
@app.get("/comentarios/usuario/{id}", response_model=list[ComentarioResponse])
def obtener_por_usuario(id: int, db: Session = Depends(get_db)):
    return db.query(ComentarioDB).filter(ComentarioDB.usuario_id == id).all()


# ==============================
# DELETE → Borrar comentario
# ==============================
@app.delete("/comentarios/{id}")
def eliminar_comentario(id: int, db: Session = Depends(get_db)):
    comentario = db.query(ComentarioDB).filter(ComentarioDB.id == id).first()

    if not comentario:
        return {"error": "Comentario no encontrado"}

    db.delete(comentario)
    db.commit()

    return {"mensaje": "Comentario eliminado correctamente"}
