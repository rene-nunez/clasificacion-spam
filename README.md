# Clasificación de Spam

Proyecto de clasificación de mensajes de spam en español mediante aprendizaje automático. Entrena y evalúa modelos de Naive Bayes, regresión logística y SVM. Incluye un pipeline reproducible, una interfaz de línea de comandos (CLI) y una API web.

El entrenamiento utiliza el dataset [synthetic-spam-detection-dataset-spanish](https://huggingface.co/datasets/tanaos/synthetic-spam-detection-dataset-spanish), creado por Tanaos, con 15.016 mensajes.

## Instalación

Requiere Python 3.11 o superior.

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords # solo si usarás la CLI/API sin entrenar
npm --prefix frontend install
```

## Entrenamiento

```bash
python src/main.py
```

Genera `data/`, `models/` y `notebook/`. El pipeline descarga el dataset, limpia el texto, vectoriza con TF-IDF, ajusta hiperparámetros con validación cruzada y guarda las métricas en `models/manifest.json`. La semilla está fijada en 42.

## Instrucciones

### CLI

```bash
python src/prediccion.py
```

### Interfaz web (desarrollo)

```bash
# terminal 1
uvicorn backend.api:app --reload

# terminal 2
npm --prefix frontend run dev
```

## API

Documentación interactiva en `/docs`.

| Método | Ruta | Descripción |
| --- | --- | --- |
| GET | `/health` | Estado del servidor. |
| GET | `/api/modelos` | Modelos disponibles. |
| GET | `/api/metricas` | Métricas del entrenamiento. |
| POST | `/api/clasificar` | Clasificar un mensaje. |

CORS se configura con `CORS_ORIGINS`; host y puerto con `HOST` y `PORT`.

## Licencias

El dataset es de Tanaos y se distribuye bajo la licencia MIT. Ver [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) para más información.

Este proyecto se distribuye bajo la licencia Zlib. Ver [LICENSE](./LICENSE).
