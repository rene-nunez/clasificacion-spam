"""Aplicación FastAPI para la clasificación de spam."""

import logging
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Permite ejecutar tanto `python backend/api.py` como
# `uvicorn backend.api:app` desde la raíz del proyecto.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.clasificador import clasificar
from backend.config import manifest, modelos
from backend.schemas import ClasificacionResponse, MensajeRequest
from spamclf.registry import BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DIST_DIR = BASE_DIR / "frontend" / "dist"
ASSETS_DIR = DIST_DIR / "assets"
INDEX_FILE = DIST_DIR / "index.html"
RUTAS_RESERVADAS = {"api", "docs", "redoc", "openapi.json"}


def origenes_permitidos() -> list[str]:
    valor = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:8000",
    )
    return [origen.strip() for origen in valor.split(",") if origen.strip()]


tags_metadata = [
    {
        "name": "General",
        "description": "Endpoints de estado y monitoreo",
    },
    {
        "name": "Modelos",
        "description": "Listado de modelos ML disponibles para clasificación",
    },
    {
        "name": "Clasificación",
        "description": "Clasificación de mensajes como spam o ham",
    },
    {
        "name": "Frontend",
        "description": "Servir la aplicación web (solo en producción)",
    },
]

app = FastAPI(
    title="Clasificación de Spam",
    description=(
        "API REST para clasificar mensajes de texto como spam o ham usando "
        "modelos de Machine Learning entrenados en español."
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origenes_permitidos(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

if ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


@app.get("/", include_in_schema=False)
def root():
    if INDEX_FILE.is_file():
        return FileResponse(INDEX_FILE, headers={"Cache-Control": "no-cache"})
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    tags=["General"],
    summary="Health check",
)
def health():
    """Verifica que el servidor y los modelos estén operativos."""
    return {"status": "ok", "modelos": len(modelos)}


@app.get(
    "/api/modelos",
    tags=["Modelos"],
    summary="Listar modelos disponibles",
    response_model=list[str],
)
def listar_modelos():
    """Devuelve los nombres de los modelos ML cargados y listos para usar."""
    logger.info("Listado de modelos solicitado")
    return list(modelos.keys())


@app.get(
    "/api/metricas",
    tags=["Modelos"],
    summary="Consultar métricas del entrenamiento",
    response_model=dict,
)
def listar_metricas():
    """Devuelve el manifiesto con métricas y parámetros de los modelos."""
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail="No hay métricas disponibles. Ejecuta: python src/main.py",
        )
    return manifest


@app.post(
    "/api/clasificar",
    tags=["Clasificación"],
    summary="Clasificar un mensaje",
    response_model=ClasificacionResponse,
    responses={
        400: {"description": "Mensaje vacío, demasiado largo o modelo inválido"},
        422: {"description": "Error de validación (JSON mal formado)"},
    },
)
def clasificar_endpoint(req: MensajeRequest):
    logger.info("Clasificando mensaje con modelo: %s", req.modelo)
    return clasificar(req.mensaje, req.modelo)


@app.get("/{path:path}", tags=["Frontend"], include_in_schema=False)
def servir_frontend(path: str):
    """Sirve los archivos del frontend compilado y las rutas de la SPA."""
    if path.split("/", 1)[0] in RUTAS_RESERVADAS:
        raise HTTPException(status_code=404, detail="Not Found")

    if not INDEX_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail="Frontend no compilado. Ejecuta: cd frontend && npm run build",
        )

    dist_real = DIST_DIR.resolve()
    solicitado = (DIST_DIR / path).resolve()

    if path and solicitado.is_relative_to(dist_real) and solicitado.is_file():
        return FileResponse(solicitado)

    return FileResponse(INDEX_FILE, headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
