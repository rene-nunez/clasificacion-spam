# División del dataset en entrenamiento (80%) y prueba (20%)

import os

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    ruta_csv = os.path.join(BASE_DIR, "data", "spam_limpio.csv")
    df = pd.read_csv(ruta_csv)

    print(f"Dataset cargado: {len(df)} mensajes")

    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["spam"]
    )

    train.to_csv(os.path.join(BASE_DIR, "data", "train.csv"), index=False)
    test.to_csv(os.path.join(BASE_DIR, "data", "test.csv"), index=False)

    print(f"\nTrain: {len(train)} mensajes")
    print(f"spam: {train['spam'].sum()}, ham: {len(train) - train['spam'].sum()}")
    print("")
    print(f"Test: {len(test)} mensajes")
    print(f"spam: {test['spam'].sum()}, ham: {len(test) - test['spam'].sum()}")
    print("\nProporcion spam/ham mantenida en ambos")
