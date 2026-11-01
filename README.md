# Clasificación de Spam

Clasificación de mensajes de spam en español con machine learning. Entrena modelos Naive Bayes, Regresión Logística y SVM, incluye un pipeline reproducible, una CLI y una API web.

Dataset de entrenamiento: [synthetic-spam-detection-dataset-spanish](https://huggingface.co/datasets/tanaos/synthetic-spam-detection-dataset-spanish) de Tanaos (15.016 mensajes).

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

Abrir http://localhost:3000.

### Interfaz web (producción)

```bash
npm --prefix frontend run build
python backend/api.py
```

Abrir http://localhost:8000.

## API

Documentación interactiva en `/docs`.

| Método | Ruta | Descripción |
| --- | --- | --- |
| GET | `/health` | Estado del servidor |
| GET | `/api/modelos` | Modelos disponibles |
| GET | `/api/metricas` | Métricas del entrenamiento |
| POST | `/api/clasificar` | Clasificar un mensaje |

CORS se configura con `CORS_ORIGINS`; host y puerto con `HOST` y `PORT`.

## Licencia

El dataset es de Tanaos y se distribuye bajo la licencia MIT. Ver [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) para más información.

Este proyecto se distribuye bajo la licencia Zlib. Ver [LICENSE](./LICENSE).
