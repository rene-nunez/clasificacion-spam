"""Esquemas de entrada y salida de la API."""

from pydantic import BaseModel


class MensajeRequest(BaseModel):
    mensaje: str
    modelo: str = "Regresión Logística"  # Modelo por defecto

    model_config = {
        "json_schema_extra": {
            "example": {
                "mensaje": "Gana dinero rápido desde casa",
                "modelo": "SVM",
            },
        },
    }


class ClasificacionResponse(BaseModel):
    prediccion: int
    etiqueta: str
    confianza: float

    model_config = {
        "json_schema_extra": {
            "example": {"prediccion": 1, "etiqueta": "spam", "confianza": 95.3}
        }
    }
