"""Preprocesado de texto compartido por el entrenamiento y la inferencia."""

import re
from functools import lru_cache

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{6,}\d)(?!\d)")
MONEY_RE = re.compile(r"[$€£]|\b(?:usd|eur|dólares|dolares|euros)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-záéíóúüñ0-9_]+")

NEGACIONES = frozenset({"no", "ni", "nunca", "jamás", "jamas", "tampoco", "sin"})


@lru_cache(maxsize=1)
def cargar_stopwords() -> frozenset[str]:
    """Carga las stopwords en español sin descargarlas al importar el módulo.

    Las negaciones se conservan porque cambian el significado del mensaje y son
    relevantes para distinguir spam de ham.
    """
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("spanish")) - NEGACIONES
    except LookupError as exc:
        raise RuntimeError(
            "Faltan las stopwords de NLTK. Ejecuta: "
            "python -m nltk.downloader stopwords"
        ) from exc


def extraer_senales(texto: str) -> list[str]:
    """Devuelve tokens que representan señales típicas de spam.

    Se calculan sobre el texto original, antes de normalizar, para no perder
    mayúsculas, signos ni formatos especiales.
    """
    senales: list[str] = []

    letras = [caracter for caracter in texto if caracter.isalpha()]
    if len(letras) >= 10:
        proporcion_mayusculas = sum(c.isupper() for c in letras) / len(letras)
        if proporcion_mayusculas >= 0.3:
            senales.append("__mayusculas__")

    if texto.count("!") >= 2 or "?!" in texto or "!?" in texto:
        senales.append("__exclamaciones__")

    if "%" in texto:
        senales.append("__porcentaje__")

    if len(re.findall(r"\d", texto)) >= 7:
        senales.append("__numeros__")

    return senales


def limpiar_texto(texto: str) -> str:
    """Normaliza un mensaje y conserva señales útiles para el clasificador."""
    if not isinstance(texto, str):
        return ""

    original = texto

    normalizado = texto.lower()
    normalizado = URL_RE.sub(" __enlace__ ", normalizado)
    normalizado = EMAIL_RE.sub(" __correo__ ", normalizado)
    normalizado = PHONE_RE.sub(" __telefono__ ", normalizado)
    normalizado = MONEY_RE.sub(" __dinero__ ", normalizado)

    tokens = TOKEN_RE.findall(normalizado)
    stop_words = cargar_stopwords()
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]

    tokens.extend(extraer_senales(original))
    return " ".join(tokens)
