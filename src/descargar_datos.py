# Script para descargar el dataset de spam en español desde Hugging Face

import os
from datasets import load_dataset
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Raíz

if __name__ == "__main__":
    dataset = load_dataset("tanaos/synthetic-spam-detection-dataset-spanish")

    df = pd.DataFrame(dataset["train"])

    df.rename(columns={"text": "mensaje", "labels": "spam"}, inplace=True)
    df["label"] = df["spam"].map({0: "ham", 1: "spam"})

    ruta_salida = os.path.join(BASE_DIR, "data", "spam_dataset.csv")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)

    print(f"Dataset guardado en {ruta_salida}")
    print(f"Total de mensajes: {len(df)}")
    print(f"Spam: {df['spam'].sum()} | Ham: {len(df) - df['spam'].sum()}")
