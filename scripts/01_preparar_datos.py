from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. Rutas del proyecto
# ------------------------------------------------------------

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_ORIGINALES = RAIZ / "datos" / "originales"
CARPETA_PROCESADOS = RAIZ / "datos" / "procesados"

CARPETA_PROCESADOS.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Nombres de las variables
# ------------------------------------------------------------

COLUMNAS_PIXELES = [f"pixel_{numero:02d}" for numero in range(1, 65)]
COLUMNAS_DATOS = COLUMNAS_PIXELES + ["clase"]


# ------------------------------------------------------------
# 3. Función para cargar un archivo original
# ------------------------------------------------------------

def cargar_archivo(nombre_archivo: str, prefijo_id: str) -> pd.DataFrame:
    ruta = CARPETA_ORIGINALES / nombre_archivo

    datos = pd.read_csv(
        ruta,
        header=None,
        names=COLUMNAS_DATOS,
        dtype=int,
    )

    identificadores = [
        f"{prefijo_id}_{numero:04d}"
        for numero in range(1, len(datos) + 1)
    ]

    datos.insert(0, "id_original", identificadores)
    datos.insert(1, "archivo_origen", nombre_archivo)

    return datos


# ------------------------------------------------------------
# 4. Cargar los dos archivos
# ------------------------------------------------------------

datos_entrenamiento_original = cargar_archivo(
    "optdigits.tra",
    "TRA",
)

datos_prueba_oficial = cargar_archivo(
    "optdigits.tes",
    "TES",
)


# ------------------------------------------------------------
# 5. Verificaciones de integridad
# ------------------------------------------------------------

assert len(datos_entrenamiento_original) == 3823
assert len(datos_prueba_oficial) == 1797

for nombre, datos in [
    ("entrenamiento original", datos_entrenamiento_original),
    ("prueba oficial", datos_prueba_oficial),
]:
    assert not datos.isna().any().any(), (
        f"Se encontraron datos faltantes en {nombre}"
    )

    valores_pixeles = datos[COLUMNAS_PIXELES].to_numpy()

    assert valores_pixeles.min() >= 0
    assert valores_pixeles.max() <= 16

    assert datos["clase"].between(0, 9).all()


# ------------------------------------------------------------
# 6. División estratificada
# ------------------------------------------------------------

entrenamiento_interno, validacion = train_test_split(
    datos_entrenamiento_original,
    test_size=0.20,
    random_state=42,
    stratify=datos_entrenamiento_original["clase"],
)

# Ordenamos por ID solamente para facilitar la lectura y trazabilidad.
entrenamiento_interno = entrenamiento_interno.sort_values(
    "id_original"
).reset_index(drop=True)

validacion = validacion.sort_values(
    "id_original"
).reset_index(drop=True)

datos_prueba_oficial = datos_prueba_oficial.sort_values(
    "id_original"
).reset_index(drop=True)


# ------------------------------------------------------------
# 7. Registrar la partición de cada observación
# ------------------------------------------------------------

entrenamiento_interno.insert(2, "particion", "entrenamiento")
validacion.insert(2, "particion", "validacion")
datos_prueba_oficial.insert(2, "particion", "prueba_oficial")


# ------------------------------------------------------------
# 8. Comprobar que no exista superposición
# ------------------------------------------------------------

ids_entrenamiento = set(entrenamiento_interno["id_original"])
ids_validacion = set(validacion["id_original"])
ids_prueba = set(datos_prueba_oficial["id_original"])

assert ids_entrenamiento.isdisjoint(ids_validacion)
assert ids_entrenamiento.isdisjoint(ids_prueba)
assert ids_validacion.isdisjoint(ids_prueba)

assert len(entrenamiento_interno) == 3058
assert len(validacion) == 765
assert len(datos_prueba_oficial) == 1797


# ------------------------------------------------------------
# 9. Guardar los archivos procesados
# ------------------------------------------------------------

entrenamiento_interno.to_csv(
    CARPETA_PROCESADOS / "entrenamiento_interno.csv",
    index=False,
)

validacion.to_csv(
    CARPETA_PROCESADOS / "validacion.csv",
    index=False,
)

datos_prueba_oficial.to_csv(
    CARPETA_PROCESADOS / "prueba_oficial.csv",
    index=False,
)


# Archivo reducido que permite rastrear la ubicación de cada observación.
trazabilidad = pd.concat(
    [
        entrenamiento_interno,
        validacion,
        datos_prueba_oficial,
    ],
    ignore_index=True,
)[
    [
        "id_original",
        "archivo_origen",
        "particion",
        "clase",
    ]
]

trazabilidad.to_csv(
    CARPETA_PROCESADOS / "particion_datos.csv",
    index=False,
)


# ------------------------------------------------------------
# 10. Mostrar el resultado
# ------------------------------------------------------------

print("PREPARACIÓN COMPLETADA")
print("----------------------")
print(f"Entrenamiento original: {len(datos_entrenamiento_original)}")
print(f"Entrenamiento interno:  {len(entrenamiento_interno)}")
print(f"Validación:             {len(validacion)}")
print(f"Prueba oficial:         {len(datos_prueba_oficial)}")
print(
    f"Total del dataset:      "
    f"{len(datos_entrenamiento_original) + len(datos_prueba_oficial)}"
)

print("\nDistribución de clases:")

distribucion = pd.crosstab(
    trazabilidad["clase"],
    trazabilidad["particion"],
)

print(distribucion)

print("\nArchivos creados en:")
print(CARPETA_PROCESADOS)