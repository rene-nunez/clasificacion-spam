"""Configuración y carga de los artefactos usados por la API."""

from spamclf.preprocessing import cargar_stopwords
from spamclf.registry import cargar_manifest, cargar_modelos, cargar_vectorizer

# Fallar al arrancar con un mensaje claro si faltan las stopwords de NLTK.
cargar_stopwords()

vectorizer = cargar_vectorizer()
modelos = cargar_modelos()
manifest = cargar_manifest()
