# Análisis exploratorio del dataset

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebook")

if __name__ == "__main__":
    os.makedirs(NOTEBOOK_DIR, exist_ok=True)

    ruta_csv = os.path.join(BASE_DIR, "data", "spam_dataset.csv")
    df = pd.read_csv(ruta_csv)
    df["mensaje"] = df["mensaje"].fillna("")

    print("Análisis exploratorio del dataset")

    print(f"\nDimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")

    print("-" * 30)

    print("\nDistribución de clases")
    print(df["label"].value_counts())
    print("\nPorcentajes:")
    print(df["label"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="label", hue="label", palette={"ham": "blue", "spam": "pink"}, legend=False)
    plt.title("Distribución de clases: Ham vs Spam")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOK_DIR, "distribucion_clases.png"))
    plt.close()
    print(f"\nGráfico creado: {os.path.join(NOTEBOOK_DIR, 'distribucion_clases.png')}")

    print("-" * 30)

    df["longitud"] = df["mensaje"].str.len()

    print("\nEstadísticas de longitud (caracteres)")
    print(df.groupby("label")["longitud"].describe().round(1))

    plt.figure(figsize=(10, 5))
    for clase, color in [("ham", "blue"), ("spam", "pink")]:
        subset = df[df["label"] == clase]["longitud"]
        plt.hist(subset, bins=50, alpha=0.5, label=clase, color=color)
    plt.title("Distribución de longitud de mensajes")
    plt.xlabel("Cantidad de caracteres")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOK_DIR, "histograma_longitud.png"))
    plt.close()
    print(f"Gráfico creado: {os.path.join(NOTEBOOK_DIR, 'histograma_longitud.png')}")

    print("-" * 30)

    print("\nNubes de palabras por clase")

    texto_ham = " ".join(df[df["label"] == "ham"]["mensaje"])
    texto_spam = " ".join(df[df["label"] == "spam"]["mensaje"])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    nube_ham = WordCloud(width=400, height=300, background_color="white", colormap="Blues").generate(texto_ham)
    plt.imshow(nube_ham, interpolation="bilinear")
    plt.axis("off")
    plt.title("Palabras ham más frecuentes")

    plt.subplot(1, 2, 2)
    nube_spam = WordCloud(width=400, height=300, background_color="white", colormap="Reds").generate(texto_spam)
    plt.imshow(nube_spam, interpolation="bilinear")
    plt.axis("off")
    plt.title("Palabras spam más frecuentes")

    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOK_DIR, "nubes_palabras.png"))
    plt.close()
    print(f"Gráfico creado: {os.path.join(NOTEBOOK_DIR, 'nubes_palabras.png')}")

    print("-" * 30)

    print("\nEjemplos de ham")
    print(df[df["label"] == "ham"]["mensaje"].iloc[:5].to_string(index=False))

    print("\nEjemplos de spam")
    print(df[df["label"] == "spam"]["mensaje"].iloc[:5].to_string(index=False))

    print("-" * 30)

    print("\nCalidad de datos")
    print(f"Valores nulos: {df.isnull().sum().sum()}")
    duplicados = df.duplicated(subset=["mensaje"]).sum()
    print(f"Mensajes duplicados: {duplicados} ({duplicados / len(df) * 100:.1f}%)")

    print("-" * 30)

    print("\nAnalisis completado. Revisa los graficos en la carpeta notebook/")
