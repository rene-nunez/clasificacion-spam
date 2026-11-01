"""Carga de artefactos y registro central de modelos."""

import json
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# El orden define el orden mostrado por la API y la interfaz web.
MODELOS = {
    "Regresión Logística": "regresion_logistica.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
}

VECTORIZER_FILE = "vectorizer.pkl"
MANIFEST_FILE = "manifest.json"


def _comprobar_artefactos(archivos: list[str]) -> None:
    faltantes = [archivo for archivo in archivos if not (MODELS_DIR / archivo).is_file()]
    if faltantes:
        listado = ", ".join(faltantes)
        raise RuntimeError(
            f"Faltan artefactos del modelo en {MODELS_DIR}: {listado}. "
            "Ejecuta primero: python src/main.py"
        )


def cargar_vectorizer():
    _comprobar_artefactos([VECTORIZER_FILE])
    return joblib.load(MODELS_DIR / VECTORIZER_FILE)


def cargar_modelos() -> dict[str, object]:
    _comprobar_artefactos(list(MODELOS.values()))
    return {
        nombre: joblib.load(MODELS_DIR / archivo)
        for nombre, archivo in MODELOS.items()
    }


def cargar_manifest() -> dict | None:
    ruta = MODELS_DIR / MANIFEST_FILE
    if not ruta.is_file():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))
