from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


# ============================================================
# 1. Modo de ejecución
# ============================================================

# Primero se deja en True para comprobar datos y configuración
# sin entrenar ninguna red.
SOLO_VERIFICAR = False


# ============================================================
# 2. Rutas
# ============================================================

RAIZ = Path(__file__).resolve().parents[1]

CARPETA_PROCESADOS = RAIZ / "datos" / "procesados"
CARPETA_RESULTADOS = RAIZ / "resultados"

RUTA_ENTRENAMIENTO = (
    CARPETA_PROCESADOS / "entrenamiento_interno.csv"
)

RUTA_VALIDACION = (
    CARPETA_PROCESADOS / "validacion.csv"
)

RUTA_RESULTADOS = (
    CARPETA_RESULTADOS
    / "experimento_definitivo_375_ejecuciones.csv"
)

RUTA_CONFIGURACION = (
    CARPETA_RESULTADOS
    / "configuracion_experimento_definitivo.json"
)

RUTA_TEMPORAL = (
    CARPETA_RESULTADOS
    / "experimento_definitivo_temporal.csv"
)

CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Protocolo congelado
# ============================================================

ANCHOS = list(range(1, 26))
SEMILLAS = list(range(1, 16))

ACTIVACION = "tanh"
OPTIMIZADOR = "adam"

MAX_ITERACIONES = 1600
TASA_APRENDIZAJE = 0.001
TAMANIO_LOTE = 64
REGULARIZACION_ALPHA = 0.0001
TOLERANCIA = 0.0001
ITERACIONES_SIN_MEJORA = 10

COLUMNAS_PIXELES = [
    f"pixel_{numero:02d}"
    for numero in range(1, 65)
]

COLUMNAS_RESULTADOS = [
    "ancho",
    "semilla",
    "parametros",
    "precision_entrenamiento",
    "precision_validacion",
    "iteraciones",
    "perdida_final",
    "tiempo_segundos",
    "advertencia_convergencia",
    "fecha_hora_finalizacion",
]


# ============================================================
# 4. Función para calcular SHA-256
# ============================================================

def calcular_sha256(ruta: Path) -> str:
    calculador = sha256()

    with ruta.open("rb") as archivo:
        while bloque := archivo.read(1024 * 1024):
            calculador.update(bloque)

    return calculador.hexdigest().upper()


# ============================================================
# 5. Cargar los datos
# ============================================================

entrenamiento = pd.read_csv(RUTA_ENTRENAMIENTO)
validacion = pd.read_csv(RUTA_VALIDACION)

assert len(entrenamiento) == 3058
assert len(validacion) == 765

assert entrenamiento["id_original"].is_unique
assert validacion["id_original"].is_unique

assert set(entrenamiento["id_original"]).isdisjoint(
    set(validacion["id_original"])
)

assert set(entrenamiento["clase"].unique()) == set(range(10))
assert set(validacion["clase"].unique()) == set(range(10))


# ============================================================
# 6. Preparar entradas y respuestas
# ============================================================

X_entrenamiento = (
    entrenamiento[COLUMNAS_PIXELES].to_numpy(dtype=float)
    / 16.0
)

y_entrenamiento = entrenamiento["clase"].to_numpy()

X_validacion = (
    validacion[COLUMNAS_PIXELES].to_numpy(dtype=float)
    / 16.0
)

y_validacion = validacion["clase"].to_numpy()

assert X_entrenamiento.shape == (3058, 64)
assert X_validacion.shape == (765, 64)

assert X_entrenamiento.min() >= 0.0
assert X_entrenamiento.max() <= 1.0

assert X_validacion.min() >= 0.0
assert X_validacion.max() <= 1.0


# ============================================================
# 7. Registrar la configuración completa
# ============================================================

configuracion = {
    "dataset": "UCI Optical Recognition of Handwritten Digits",
    "archivo_entrenamiento": RUTA_ENTRENAMIENTO.name,
    "archivo_validacion": RUTA_VALIDACION.name,
    "sha256_entrenamiento": calcular_sha256(
        RUTA_ENTRENAMIENTO
    ),
    "sha256_validacion": calcular_sha256(
        RUTA_VALIDACION
    ),
    "observaciones_entrenamiento": len(entrenamiento),
    "observaciones_validacion": len(validacion),
    "cantidad_entradas": 64,
    "cantidad_salidas": 10,
    "anchos": ANCHOS,
    "semillas": SEMILLAS,
    "activacion": ACTIVACION,
    "optimizador": OPTIMIZADOR,
    "max_iteraciones": MAX_ITERACIONES,
    "tasa_aprendizaje": TASA_APRENDIZAJE,
    "tamanio_lote": TAMANIO_LOTE,
    "regularizacion_alpha": REGULARIZACION_ALPHA,
    "tolerancia": TOLERANCIA,
    "iteraciones_sin_mejora": ITERACIONES_SIN_MEJORA,
    "early_stopping": False,
    "normalizacion": "division de los pixeles por 16",
    "version_python": platform.python_version(),
    "version_numpy": np.__version__,
    "version_pandas": pd.__version__,
    "version_scikit_learn": sklearn.__version__,
}


# ============================================================
# 8. Verificación sin entrenamiento
# ============================================================

if SOLO_VERIFICAR:
    print("VERIFICACIÓN DEL EXPERIMENTO DEFINITIVO")
    print("---------------------------------------")
    print(f"Entrenamiento: {X_entrenamiento.shape}")
    print(f"Validación:    {X_validacion.shape}")
    print(f"Anchos:        {len(ANCHOS)}")
    print(f"Semillas:      {len(SEMILLAS)}")
    print(
        f"Ejecuciones:   "
        f"{len(ANCHOS) * len(SEMILLAS)}"
    )
    print(f"Activación:    {ACTIVACION}")
    print(f"Máximo:        {MAX_ITERACIONES}")
    print(
        f"Rango entrenamiento: "
        f"[{X_entrenamiento.min()}, "
        f"{X_entrenamiento.max()}]"
    )
    print(
        f"Rango validación:    "
        f"[{X_validacion.min()}, "
        f"{X_validacion.max()}]"
    )
    print("\nVerificación completada.")
    print("No se entrenó ninguna red.")

    sys.exit(0)


# ============================================================
# 9. Proteger la configuración
# ============================================================

if RUTA_RESULTADOS.exists() and not RUTA_CONFIGURACION.exists():
    raise RuntimeError(
        "Existe un archivo de resultados, pero no existe "
        "su configuración. No es seguro continuar."
    )

if RUTA_CONFIGURACION.exists():
    with RUTA_CONFIGURACION.open(
        "r",
        encoding="utf-8",
    ) as archivo:
        configuracion_anterior = json.load(archivo)

    if configuracion_anterior != configuracion:
        raise RuntimeError(
            "La configuración actual es diferente de la "
            "configuración con la que comenzó el experimento."
        )

else:
    with RUTA_CONFIGURACION.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            configuracion,
            archivo,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# 10. Recuperar ejecuciones anteriores
# ============================================================

if RUTA_RESULTADOS.exists():
    tabla_resultados = pd.read_csv(RUTA_RESULTADOS)

    columnas_faltantes = (
        set(COLUMNAS_RESULTADOS)
        - set(tabla_resultados.columns)
    )

    if columnas_faltantes:
        raise RuntimeError(
            "Faltan columnas en el archivo de resultados: "
            f"{columnas_faltantes}"
        )

    if tabla_resultados.duplicated(
        subset=["ancho", "semilla"]
    ).any():
        raise RuntimeError(
            "Existen combinaciones duplicadas de ancho y semilla."
        )

    resultados = tabla_resultados[
        COLUMNAS_RESULTADOS
    ].to_dict("records")

else:
    resultados = []


combinaciones_completadas = {
    (int(resultado["ancho"]), int(resultado["semilla"]))
    for resultado in resultados
}

combinaciones_esperadas = {
    (ancho, semilla)
    for ancho in ANCHOS
    for semilla in SEMILLAS
}

if not combinaciones_completadas.issubset(
    combinaciones_esperadas
):
    raise RuntimeError(
        "El archivo contiene combinaciones no contempladas "
        "en el protocolo definitivo."
    )


# ============================================================
# 11. Función para guardar sin perder resultados
# ============================================================

def guardar_resultados() -> None:
    tabla = pd.DataFrame(
        resultados,
        columns=COLUMNAS_RESULTADOS,
    )

    tabla = tabla.sort_values(
        ["ancho", "semilla"]
    ).reset_index(drop=True)

    tabla.to_csv(
        RUTA_TEMPORAL,
        index=False,
    )

    RUTA_TEMPORAL.replace(RUTA_RESULTADOS)


# ============================================================
# 12. Ejecutar las combinaciones pendientes
# ============================================================

TOTAL_EJECUCIONES = len(combinaciones_esperadas)

print("EXPERIMENTO DEFINITIVO")
print("----------------------")
print(
    f"Completadas anteriormente: "
    f"{len(combinaciones_completadas)}"
)
print(
    f"Pendientes: "
    f"{TOTAL_EJECUCIONES - len(combinaciones_completadas)}"
)

for ancho in ANCHOS:
    for semilla in SEMILLAS:
        combinacion = (ancho, semilla)

        if combinacion in combinaciones_completadas:
            continue

        numero_ejecucion = len(resultados) + 1

        print(
            f"[{numero_ejecucion:03d}/{TOTAL_EJECUCIONES}] "
            f"ancho={ancho:2d}, semilla={semilla:2d}...",
            end=" ",
            flush=True,
        )

        modelo = MLPClassifier(
            hidden_layer_sizes=(ancho,),
            activation=ACTIVACION,
            solver=OPTIMIZADOR,
            alpha=REGULARIZACION_ALPHA,
            batch_size=TAMANIO_LOTE,
            learning_rate_init=TASA_APRENDIZAJE,
            max_iter=MAX_ITERACIONES,
            shuffle=True,
            random_state=semilla,
            early_stopping=False,
            tol=TOLERANCIA,
            n_iter_no_change=ITERACIONES_SIN_MEJORA,
        )

        inicio = perf_counter()

        with warnings.catch_warnings(record=True) as advertencias:
            warnings.simplefilter(
                "always",
                ConvergenceWarning,
            )

            modelo.fit(
                X_entrenamiento,
                y_entrenamiento,
            )

        tiempo_segundos = perf_counter() - inicio

        hubo_advertencia = any(
            issubclass(
                advertencia.category,
                ConvergenceWarning,
            )
            for advertencia in advertencias
        )

        precision_entrenamiento = accuracy_score(
            y_entrenamiento,
            modelo.predict(X_entrenamiento),
        )

        precision_validacion = accuracy_score(
            y_validacion,
            modelo.predict(X_validacion),
        )

        cantidad_parametros = sum(
            matriz.size
            for matriz in modelo.coefs_
        ) + sum(
            vector.size
            for vector in modelo.intercepts_
        )

        parametros_teoricos = 75 * ancho + 10

        assert cantidad_parametros == parametros_teoricos

        resultado = {
            "ancho": ancho,
            "semilla": semilla,
            "parametros": cantidad_parametros,
            "precision_entrenamiento": (
                precision_entrenamiento
            ),
            "precision_validacion": precision_validacion,
            "iteraciones": modelo.n_iter_,
            "perdida_final": modelo.loss_,
            "tiempo_segundos": tiempo_segundos,
            "advertencia_convergencia": hubo_advertencia,
            "fecha_hora_finalizacion": (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            ),
        }

        resultados.append(resultado)
        combinaciones_completadas.add(combinacion)

        guardar_resultados()

        print(
            f"validación={precision_validacion:.4f}, "
            f"iteraciones={modelo.n_iter_:4d}, "
            f"tiempo={tiempo_segundos:.2f} s"
        )


# ============================================================
# 13. Verificación final
# ============================================================

tabla_final = pd.read_csv(RUTA_RESULTADOS)

cantidad_combinaciones = (
    tabla_final[
        ["ancho", "semilla"]
    ]
    .drop_duplicates()
    .shape[0]
)

assert len(tabla_final) == 375
assert cantidad_combinaciones == 375
assert not tabla_final.duplicated(
    subset=["ancho", "semilla"]
).any()

advertencias_totales = int(
    tabla_final["advertencia_convergencia"].sum()
)

tiempo_total = tabla_final["tiempo_segundos"].sum()


print("\nEXPERIMENTO COMPLETADO")
print("----------------------")
print(f"Filas:              {len(tabla_final)}")
print(f"Combinaciones:      {cantidad_combinaciones}")
print(f"Advertencias:       {advertencias_totales}")
print(f"Tiempo acumulado:   {tiempo_total / 60:.2f} minutos")
print(f"Archivo:            {RUTA_RESULTADOS}")