# Ejecuta el pipeline completo de entrenamiento y evaluación

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = [
    ("Descarga de datos", "descargar_datos.py"),
    ("Exploración de datos", "explorar.py"),
    ("Limpieza de texto", "limpiar.py"),
    ("División train/test", "dividir.py"),
    ("Vectorización TF-IDF", "vectorizar.py"),
    ("Entrenamiento de modelos", "entrenar.py"),
    ("Evaluación de modelos", "evaluar.py"),
]

if __name__ == "__main__":
    print("-" * 30)
    print("Clasificación de Spam")
    print("-" * 30)

    for paso, (nombre, script) in enumerate(SCRIPTS, 1):
        print(f"\nPaso {paso}/{len(SCRIPTS)}: {nombre}")
        ruta = os.path.join(BASE_DIR, "src", script)

        # Sin capture_output para que la salida se muestre en tiempo real.
        resultado = subprocess.run([sys.executable, "-u", ruta])

        if resultado.returncode != 0:
            print(f"Error en paso {paso} ({nombre})", file=sys.stderr)
            sys.exit(resultado.returncode)

    print("\n" + "-" * 30)
    print("Pipeline completado exitosamente.")
    print("-" * 30)
