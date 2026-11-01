# Evaluación de los modelos sobre el conjunto de prueba

import os
import sys
import unicodedata

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebook")
sys.path.insert(0, BASE_DIR)

from spamclf.registry import MODELOS, MODELS_DIR, VECTORIZER_FILE


def slug(nombre: str) -> str:
    """Convierte un nombre de modelo en un nombre de archivo sin acentos."""
    sin_acentos = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    return sin_acentos.lower().replace(" ", "_")


if __name__ == "__main__":
    os.makedirs(NOTEBOOK_DIR, exist_ok=True)

    test = pd.read_csv(os.path.join(BASE_DIR, "data", "test.csv"))
    test["mensaje_limpio"] = test["mensaje_limpio"].fillna("")

    vectorizer = joblib.load(os.path.join(MODELS_DIR, VECTORIZER_FILE))
    X_test = vectorizer.transform(test["mensaje_limpio"])
    y_test = test["spam"]

    predicciones = {}

    for nombre, archivo in MODELOS.items():
        print("-" * 30)
        print(f"{nombre}")
        print("-" * 30)

        modelo = joblib.load(os.path.join(MODELS_DIR, archivo))
        y_predicho = modelo.predict(X_test)
        predicciones[nombre] = y_predicho

        matriz = confusion_matrix(y_test, y_predicho)
        print("\nMatriz de confusión:")
        print("              Predicción")
        print("              ham  spam")
        print(f"Real  ham  {matriz[0, 0]:>4} {matriz[0, 1]:>4}")
        print(f"      spam {matriz[1, 0]:>4} {matriz[1, 1]:>4}")

        print("\nMétricas:")
        print(
            classification_report(
                y_test,
                y_predicho,
                target_names=["ham", "spam"],
                digits=4,
                zero_division=0,
            )
        )

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            matriz,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["ham", "spam"],
            yticklabels=["ham", "spam"],
        )
        plt.title(f"Matriz de confusión: {nombre}")
        plt.ylabel("Real")
        plt.xlabel("Predicción")
        plt.tight_layout()
        archivo_matriz = os.path.join(NOTEBOOK_DIR, f"matriz_{slug(nombre)}.png")
        plt.savefig(archivo_matriz)
        plt.close()
        print(f"\nGráfico guardado: {archivo_matriz}")
        print("")

    print("-" * 30)
    print("Análisis de errores")
    print("-" * 30)

    print("\nComparación de errores por modelo:")
    print(f"{'Modelo':25s} {'FP':>5s} {'FN':>5s} {'Total':>5s}")
    print("-" * 42)
    for nombre, y_predicho in predicciones.items():
        fp = int(((y_test == 0) & (y_predicho == 1)).sum())
        fn = int(((y_test == 1) & (y_predicho == 0)).sum())
        print(f"{nombre:25s} {fp:5d} {fn:5d} {fp + fn:5d}")

    test["fp_todos"] = y_test == 0
    test["fn_todos"] = y_test == 1

    for y_predicho in predicciones.values():
        test["fp_todos"] &= y_predicho == 1
        test["fn_todos"] &= y_predicho == 0

    print("")
    print("Casos donde todos los modelos fallaron:")
    print(
        "Falsos positivos (ham que todos dijeron spam): "
        f"{int(test['fp_todos'].sum())}"
    )
    print(
        "Falsos negativos (spam que todos dijeron ham): "
        f"{int(test['fn_todos'].sum())}"
    )

    if test["fp_todos"].sum() > 0 or test["fn_todos"].sum() > 0:
        print("\nEjemplos más difíciles:")
        for _, mensaje in test[test["fp_todos"] | test["fn_todos"]].head(6).iterrows():
            tipo = "FP" if mensaje["fp_todos"] else "FN"
            print(f"  [{tipo}] {mensaje['mensaje_limpio'][:80]}")
