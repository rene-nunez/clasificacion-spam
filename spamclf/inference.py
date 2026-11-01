"""Inferencia compartida por la CLI y la API."""

import math

from .preprocessing import limpiar_texto


def clasificar_texto(texto: str, modelo, vectorizer) -> tuple[int, float]:
    """Clasifica un mensaje y devuelve (predicción, confianza entre 0 y 1)."""
    limpio = limpiar_texto(texto)
    vector = vectorizer.transform([limpio])
    prediccion = int(modelo.predict(vector)[0])

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(vector)[0]
        indice = list(modelo.classes_).index(prediccion)
        confianza = float(probabilidades[indice])
    else:
        decision = float(modelo.decision_function(vector)[0])
        decision = max(min(decision, 100), -100)
        probabilidad = 1 / (1 + math.exp(-decision))
        confianza = probabilidad if prediccion == 1 else 1 - probabilidad

    return prediccion, confianza
