import re
import joblib
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# CARGA DE MODELOS
# -----------------------------
print("Cargando modelo TF-IDF clásico...")
clf = joblib.load("logreg_tfidf_toxic_es.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Cargando modelo BERT de toxicidad...")
model_name = "bgonzalezbustamante/bert-spanish-toxicity"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
torch.set_grad_enabled(False)

# -----------------------------
# FUNCIONES DE PROCESAMIENTO
# -----------------------------
def limpiar_texto(texto):
    texto = str(texto)
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"[^A-Za-zÁÉÍÓÚáéíóúñÑ0-9\s!?¿.,]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def normalizar_leet(texto):
    reemplazos = {
        '0': 'o', '1': 'i', '3': 'e', '4': 'a', '@': 'a',
        '$': 's', '5': 's', '7': 't', '8': 'b'
    }
    for k, v in reemplazos.items():
        texto = texto.replace(k, v)
    return texto

def detectar_toxicidad_tfidf(texto):
    X = vectorizer.transform([texto])
    return float(clf.predict_proba(X)[0][1])

def detectar_toxicidad_bert(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    return float(probs[1])

def fusion_toxicidad(prob_tfidf, prob_bert, w1=0.4, w2=0.6):
    return (prob_tfidf * w1) + (prob_bert * w2)

PATRONES_GROSERIAS = [
    r"pvt[a@]", r"put[ao]", r"mierd[a@]", r"vrg[a@]", r"verga",
    r"cabro[nm]", r"idiot[ao]", r"imbecil", r"pendej[o0x]", r"cabr[o0]n",
    r"chinga", r"cul[o0]", r"hdp", r"malparid[ao]", r"coñ[ao]",
    r"fck", r"wtf", r"bitch", r"assh", r"mamad[a@]", r"pito",
    r"m1erda", r"mi3rda", r"p3ndej", r"vrg", r"vtv", r"xd",
    r"tonto", r"payas", r"estupid[ao]", r"bastrad[ao]", r"asqueros[oa]"
]

def censurar_regex_total(texto):
    censurado = texto
    for patron in PATRONES_GROSERIAS:
        censurado = re.sub(
            patron,
            lambda m: "*" * len(m.group(0)),
            censurado,
            flags=re.IGNORECASE
        )
    return censurado


# -----------------------------
# API
# -----------------------------
class TextoEntrada(BaseModel):
    texto: str

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API de detección de toxicidad activa."}

@app.post("/clasificar")
def clasificar_texto(data: TextoEntrada):
    frase = data.texto.strip()

    frase_norm = normalizar_leet(frase)
    texto = limpiar_texto(frase_norm)

    prob_tfidf = detectar_toxicidad_tfidf(texto)
    prob_bert = detectar_toxicidad_bert(texto)
    prob_final = fusion_toxicidad(prob_tfidf, prob_bert)
    es_toxico = prob_final >= 0.5

    censurado = censurar_regex_total(frase)

    return {
        "entrada": frase,
        "prob_tfidf": prob_tfidf,
        "prob_bert": prob_bert,
        "prob_final": prob_final,
        "es_toxico": es_toxico,
        "texto_censurado": censurado
    }
