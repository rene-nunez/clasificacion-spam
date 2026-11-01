"""Lógica de clasificación y validación de la API."""

from fastapi import HTTPException

from backend.config import modelos, vectorizer
from backend.schemas import ClasificacionResponse
from spamclf.inference import clasificar_texto


def clasificar(mensaje: str, modelo_nombre: str) -> ClasificacionResponse:
    if not mensaje.strip():
        raise HTTPException(400, "El mensaje no puede estar vacío")

    if len(mensaje) > 5000:
        raise HTTPException(400, "El mensaje es demasiado largo")

    modelo = modelos.get(modelo_nombre)

    if modelo is None:
        raise HTTPException(400, f"Modelo no válido: {modelo_nombre}")

    prediccion, confianza = clasificar_texto(mensaje, modelo, vectorizer)

    return ClasificacionResponse(
        prediccion=prediccion,
        etiqueta="spam" if prediccion == 1 else "ham",
        confianza=round(confianza * 100, 1),
    )
