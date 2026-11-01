# Clasificador interactivo por línea de comandos

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from spamclf.inference import clasificar_texto
from spamclf.registry import cargar_modelos, cargar_vectorizer


def main() -> None:
    vectorizer = cargar_vectorizer()
    modelos = cargar_modelos()
    nombres = list(modelos.keys())
    indice = 0

    print("Clasificación de spam")
    print(f"Modelo actual: {nombres[indice]} (cambia con: /modelo)")
    print("Escribe un mensaje para clasificarlo (o 'salir' para terminar)")
    print("")

    try:
        while True:
            entrada = input("> ")
            if entrada.lower() == "salir":
                break

            if entrada.startswith("/modelo"):
                indice = (indice + 1) % len(nombres)
                print(f"Modelo cambiado a: {nombres[indice]}")
                continue

            prediccion, confianza = clasificar_texto(
                entrada, modelos[nombres[indice]], vectorizer
            )
            etiqueta = "spam" if prediccion == 1 else "ham"
            print(f"{etiqueta} ({confianza:.1%} de confianza)")
            print("")
    except (EOFError, KeyboardInterrupt):
        print("")


if __name__ == "__main__":
    main()
