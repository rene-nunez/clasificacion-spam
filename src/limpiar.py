# Limpieza y normalización del texto

import os
import sys

import nltk
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from spamclf.preprocessing import limpiar_texto

if __name__ == "__main__":
    nltk.download("stopwords", quiet=True)

    ruta_csv = os.path.join(BASE_DIR, "data", "spam_dataset.csv")
    df = pd.read_csv(ruta_csv)

    print(f"Dataset cargado: {len(df)} mensajes")

    antes = len(df)
    df = df.drop_duplicates(subset=["mensaje"])
    despues = len(df)
    print(f"\nDuplicados eliminados: {antes - despues}")

    df["mensaje"] = df["mensaje"].fillna("")
    df["mensaje_limpio"] = df["mensaje"].apply(limpiar_texto)

    ruta_salida = os.path.join(BASE_DIR, "data", "spam_limpio.csv")
    df.to_csv(ruta_salida, index=False)
    print("\nDataset limpio guardado: data/spam_limpio.csv")

    print("\nComparacion antes/despues:")
    for i in range(min(3, len(df))):
        original = df["mensaje"].iloc[i]
        limpio = df["mensaje_limpio"].iloc[i]
        print(f"\nOriginal: {original}")
        print(f"Limpio: {limpio}")
