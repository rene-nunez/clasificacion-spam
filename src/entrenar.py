# Entrena y ajusta Naive Bayes, Regresión Logística y SVM

import json
import os
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from spamclf.registry import (
    MANIFEST_FILE,
    MODELS_DIR,
    MODELOS,
    VECTORIZER_FILE,
)


def crear_modelos() -> dict[str, tuple[object, dict[str, list]]]:
    """Devuelve cada modelo junto con su rejilla de hiperparámetros."""
    return {
        "Regresión Logística": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.1, 1.0, 10.0]},
        ),
        "Naive Bayes": (
            MultinomialNB(),
            {"alpha": [0.1, 0.5, 1.0]},
        ),
        "SVM": (
            CalibratedClassifierCV(
                estimator=LinearSVC(random_state=42),
                method="sigmoid",
                cv=3,
            ),
            {"estimator__C": [0.1, 1.0, 10.0]},
        ),
    }


def metricas(y_real, y_predicho) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_real, y_predicho)), 4),
        "precision": round(
            float(precision_score(y_real, y_predicho, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(y_real, y_predicho, zero_division=0)), 4),
        "f1": round(float(f1_score(y_real, y_predicho, zero_division=0)), 4),
    }


if __name__ == "__main__":
    train = pd.read_csv(os.path.join(BASE_DIR, "data", "train.csv"))
    test = pd.read_csv(os.path.join(BASE_DIR, "data", "test.csv"))

    train["mensaje_limpio"] = train["mensaje_limpio"].fillna("")
    test["mensaje_limpio"] = test["mensaje_limpio"].fillna("")

    vectorizer = joblib.load(os.path.join(MODELS_DIR, VECTORIZER_FILE))
    X_train = vectorizer.transform(train["mensaje_limpio"])
    X_test = vectorizer.transform(test["mensaje_limpio"])
    y_train = train["spam"]
    y_test = test["spam"]

    print(f"Datos cargados: {X_train.shape[0]} train, {X_test.shape[0]} test")

    validacion = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    manifest = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn": sklearn.__version__,
        "vectorizador": VECTORIZER_FILE,
        "dataset": {
            "train": int(len(train)),
            "test": int(len(test)),
            "spam_train": int(train["spam"].sum()),
            "spam_test": int(test["spam"].sum()),
        },
        "modelos": {},
    }

    resultados = []

    for nombre, (estimador, rejilla) in crear_modelos().items():
        print(f"\nAjustando {nombre}...")

        busqueda = GridSearchCV(
            estimador,
            rejilla,
            scoring="f1",
            cv=validacion,
            n_jobs=-1,
            refit=True,
        )
        busqueda.fit(X_train, y_train)

        modelo = busqueda.best_estimator_
        y_predicho = modelo.predict(X_test)
        metricas_test = metricas(y_test, y_predicho)
        matriz = confusion_matrix(y_test, y_predicho).tolist()

        archivo = MODELOS[nombre]
        joblib.dump(modelo, os.path.join(MODELS_DIR, archivo))

        manifest["modelos"][nombre] = {
            "archivo": archivo,
            "mejores_parametros": busqueda.best_params_,
            "f1_validacion_cruzada": round(float(busqueda.best_score_), 4),
            "test": metricas_test,
            "matriz_confusion": matriz,
        }

        resultados.append((nombre, busqueda.best_score_, metricas_test))
        print(f"Mejores parámetros: {busqueda.best_params_}")
        print(f"F1 en validación cruzada: {busqueda.best_score_:.4f}")
        print(f"Modelo guardado: models/{archivo}")

    ruta_manifest = os.path.join(MODELS_DIR, MANIFEST_FILE)
    with open(ruta_manifest, "w", encoding="utf-8") as archivo_manifest:
        json.dump(manifest, archivo_manifest, ensure_ascii=False, indent=2)
        archivo_manifest.write("\n")

    print("\n" + "-" * 58)
    print(f"{'Modelo':22s} {'CV F1':>8s} {'Test F1':>8s} {'Prec.':>8s} {'Recall':>8s}")
    print("-" * 58)
    for nombre, f1_cv, metricas_test in resultados:
        print(
            f"{nombre:22s} {f1_cv:8.4f} {metricas_test['f1']:8.4f} "
            f"{metricas_test['precision']:8.4f} {metricas_test['recall']:8.4f}"
        )

    print(f"\nManifiesto guardado: models/{MANIFEST_FILE}")
