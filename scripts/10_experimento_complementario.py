from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


# ============================================================
# 1. Modo de ejecucion
# ============================================================

# La primera ejecucion debe mantenerse en True. En este modo se
# verifican datos y protocolo sin entrenar ninguna red ni crear
# los archivos del experimento.
SOLO_VERIFICAR = False


# ============================================================
# 2. Rutas separadas de los analisis anteriores
# ============================================================

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_PROCESADOS = RAIZ / "datos" / "procesados"
CARPETA_RESULTADOS = RAIZ / "resultados"

RUTA_ENTRENAMIENTO = CARPETA_PROCESADOS / "entrenamiento_interno.csv"
RUTA_VALIDACION = CARPETA_PROCESADOS / "validacion.csv"

RUTA_RESULTADOS = (
    CARPETA_RESULTADOS
    / "complementario_experimento_2250_mediciones.csv"
)
RUTA_CONFIGURACION = (
    CARPETA_RESULTADOS
    / "configuracion_complementario_experimento.json"
)
RUTA_TEMPORAL = (
    CARPETA_RESULTADOS
    / "complementario_experimento_2250_mediciones_temporal.csv"
)

CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Protocolo congelado
# ============================================================

ANCHOS = list(range(1, 26))
SEMILLAS = list(range(1, 16))
PUNTOS_CONTROL = [25, 50, 100, 200, 400, 800]

ACTIVACION = "tanh"
OPTIMIZADOR = "adam"
TASA_APRENDIZAJE = 0.001
TAMANIO_LOTE = 64
REGULARIZACION_ALPHA = 0.0001
TOLERANCIA = 0.0001
ITERACIONES_SIN_MEJORA = 10

CLASES = np.arange(10)
COLUMNAS_PIXELES = [
    f"pixel_{numero:02d}"
    for numero in range(1, 65)
]

COLUMNAS_RESULTADOS = [
    "ancho",
    "semilla",
    "iteracion",
    "parametros",
    "precision_entrenamiento",
    "precision_validacion",
    "perdida_entrenamiento",
    "tiempo_intervalo_segundos",
    "tiempo_acumulado_segundos",
    "tiempo_evaluacion_segundos",
    "fecha_hora_medicion",
]

HASH_ENTRENAMIENTO_ESPERADO = (
    "9746E6212B7D65FE41196B29508BFB52"
    "F9EA89A47AF2825EAFDE2C3BF56AB540"
)
HASH_VALIDACION_ESPERADO = (
    "65A5489166C35ED0343FE153AAC609998"
    "97E36DF3A28FB9763777D04C46A7F78"
)


# ============================================================
# 4. Funciones auxiliares
# ============================================================

def calcular_sha256(ruta: Path) -> str:
    calculador = sha256()
    with ruta.open("rb") as archivo:
        while bloque := archivo.read(1024 * 1024):
            calculador.update(bloque)
    return calculador.hexdigest().upper()


def crear_modelo(ancho: int, semilla: int) -> MLPClassifier:
    # Un unico RandomState persiste durante toda la trayectoria.
    # Esto evita reiniciar la secuencia aleatoria en cada epoca.
    generador = np.random.RandomState(semilla)

    return MLPClassifier(
        hidden_layer_sizes=(ancho,),
        activation=ACTIVACION,
        solver=OPTIMIZADOR,
        alpha=REGULARIZACION_ALPHA,
        batch_size=TAMANIO_LOTE,
        learning_rate_init=TASA_APRENDIZAJE,
        max_iter=1,
        shuffle=True,
        random_state=generador,
        early_stopping=False,
        tol=TOLERANCIA,
        n_iter_no_change=ITERACIONES_SIN_MEJORA,
    )


def guardar_resultados(resultados: list[dict]) -> None:
    tabla = pd.DataFrame(resultados, columns=COLUMNAS_RESULTADOS)
    tabla = tabla.sort_values(
        ["ancho", "semilla", "iteracion"]
    ).reset_index(drop=True)

    tabla.to_csv(RUTA_TEMPORAL, index=False)
    RUTA_TEMPORAL.replace(RUTA_RESULTADOS)


# ============================================================
# 5. Carga y verificacion de datos congelados
# ============================================================

hash_entrenamiento = calcular_sha256(RUTA_ENTRENAMIENTO)
hash_validacion = calcular_sha256(RUTA_VALIDACION)

assert hash_entrenamiento == HASH_ENTRENAMIENTO_ESPERADO
assert hash_validacion == HASH_VALIDACION_ESPERADO

entrenamiento = pd.read_csv(RUTA_ENTRENAMIENTO)
validacion = pd.read_csv(RUTA_VALIDACION)

assert len(entrenamiento) == 3058
assert len(validacion) == 765
assert entrenamiento["id_original"].is_unique
assert validacion["id_original"].is_unique
assert set(entrenamiento["id_original"]).isdisjoint(
    set(validacion["id_original"])
)
assert set(entrenamiento["clase"].unique()) == set(CLASES)
assert set(validacion["clase"].unique()) == set(CLASES)

X_entrenamiento = (
    entrenamiento[COLUMNAS_PIXELES].to_numpy(dtype=float) / 16.0
)
y_entrenamiento = entrenamiento["clase"].to_numpy()
X_validacion = (
    validacion[COLUMNAS_PIXELES].to_numpy(dtype=float) / 16.0
)
y_validacion = validacion["clase"].to_numpy()

assert X_entrenamiento.shape == (3058, 64)
assert X_validacion.shape == (765, 64)
assert np.isfinite(X_entrenamiento).all()
assert np.isfinite(X_validacion).all()
assert 0.0 <= X_entrenamiento.min() <= X_entrenamiento.max() <= 1.0
assert 0.0 <= X_validacion.min() <= X_validacion.max() <= 1.0


# ============================================================
# 6. Configuracion trazable
# ============================================================

TRAYECTORIAS_ESPERADAS = len(ANCHOS) * len(SEMILLAS)
FILAS_ESPERADAS = TRAYECTORIAS_ESPERADAS * len(PUNTOS_CONTROL)

configuracion = {
    "tipo_analisis": "experimento complementario de trayectorias",
    "dataset": "UCI Optical Recognition of Handwritten Digits",
    "archivo_entrenamiento": RUTA_ENTRENAMIENTO.name,
    "archivo_validacion": RUTA_VALIDACION.name,
    "prueba_oficial_utilizada": False,
    "sha256_entrenamiento": hash_entrenamiento,
    "sha256_validacion": hash_validacion,
    "observaciones_entrenamiento": len(entrenamiento),
    "observaciones_validacion": len(validacion),
    "cantidad_entradas": 64,
    "cantidad_salidas": 10,
    "anchos": ANCHOS,
    "semillas": SEMILLAS,
    "puntos_control": PUNTOS_CONTROL,
    "trayectorias_esperadas": TRAYECTORIAS_ESPERADAS,
    "filas_esperadas": FILAS_ESPERADAS,
    "activacion": ACTIVACION,
    "optimizador": OPTIMIZADOR,
    "metodo_incremental": "una llamada a partial_fit por epoca",
    "random_state": (
        "un objeto numpy RandomState persistente por trayectoria"
    ),
    "tasa_aprendizaje": TASA_APRENDIZAJE,
    "tamanio_lote": TAMANIO_LOTE,
    "regularizacion_alpha": REGULARIZACION_ALPHA,
    "tolerancia_documentada": TOLERANCIA,
    "iteraciones_sin_mejora_documentadas": (
        ITERACIONES_SIN_MEJORA
    ),
    "early_stopping": False,
    "normalizacion": "division de los pixeles por 16",
    "version_python": platform.python_version(),
    "version_numpy": np.__version__,
    "version_pandas": pd.__version__,
    "version_scikit_learn": sklearn.__version__,
}


# ============================================================
# 7. Verificacion previa sin entrenamiento
# ============================================================

if SOLO_VERIFICAR:
    print("VERIFICACION DEL EXPERIMENTO COMPLEMENTARIO")
    print("------------------------------------------")
    print(f"Entrenamiento:       {X_entrenamiento.shape}")
    print(f"Validacion:          {X_validacion.shape}")
    print(f"Anchos:              {len(ANCHOS)} (1 a 25)")
    print(f"Semillas:            {len(SEMILLAS)} (1 a 15)")
    print(f"Puntos de control:   {PUNTOS_CONTROL}")
    print(f"Trayectorias:        {TRAYECTORIAS_ESPERADAS}")
    print(f"Mediciones:          {FILAS_ESPERADAS}")
    print(f"Ultima iteracion:    {max(PUNTOS_CONTROL)}")
    print(f"Activacion:          {ACTIVACION}")
    print(f"Optimizador:         {OPTIMIZADOR}")
    print(
        "Rango entrenamiento: "
        f"[{X_entrenamiento.min()}, {X_entrenamiento.max()}]"
    )
    print(
        "Rango validacion:    "
        f"[{X_validacion.min()}, {X_validacion.max()}]"
    )
    print("Prueba oficial:      no utilizada")
    print("Tiempo estimado:     aproximadamente 1,5 a 2 horas")
    print("\nVerificacion completada.")
    print("No se entreno ninguna red.")
    sys.exit(0)


# ============================================================
# 8. Proteccion de configuracion y resultados
# ============================================================

if RUTA_RESULTADOS.exists() and not RUTA_CONFIGURACION.exists():
    raise RuntimeError(
        "Existe el CSV del experimento complementario, pero no su "
        "configuracion. No es seguro continuar."
    )

if RUTA_CONFIGURACION.exists():
    with RUTA_CONFIGURACION.open("r", encoding="utf-8") as archivo:
        configuracion_anterior = json.load(archivo)

    if configuracion_anterior != configuracion:
        raise RuntimeError(
            "La configuracion actual no coincide con la que inicio "
            "el experimento complementario."
        )
else:
    with RUTA_CONFIGURACION.open("w", encoding="utf-8") as archivo:
        json.dump(
            configuracion,
            archivo,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# 9. Recuperacion de trayectorias completas
# ============================================================

if RUTA_RESULTADOS.exists():
    tabla_existente = pd.read_csv(RUTA_RESULTADOS)
    columnas_faltantes = (
        set(COLUMNAS_RESULTADOS) - set(tabla_existente.columns)
    )
    if columnas_faltantes:
        raise RuntimeError(
            "Faltan columnas en los resultados: "
            f"{columnas_faltantes}"
        )

    if tabla_existente.duplicated(
        subset=["ancho", "semilla", "iteracion"]
    ).any():
        raise RuntimeError("Existen mediciones duplicadas.")

    puntos_esperados = set(PUNTOS_CONTROL)
    trayectorias_completas = set()
    for (ancho, semilla), grupo in tabla_existente.groupby(
        ["ancho", "semilla"]
    ):
        if set(grupo["iteracion"]) == puntos_esperados:
            trayectorias_completas.add((int(ancho), int(semilla)))

    mascara_completas = tabla_existente.apply(
        lambda fila: (
            int(fila["ancho"]),
            int(fila["semilla"]),
        ) in trayectorias_completas,
        axis=1,
    )
    resultados = tabla_existente.loc[
        mascara_completas,
        COLUMNAS_RESULTADOS,
    ].to_dict("records")

    # Una trayectoria incompleta no puede reanudarse sin el estado
    # del optimizador. Se elimina y se repetira desde su semilla.
    if len(resultados) != len(tabla_existente):
        guardar_resultados(resultados)
else:
    resultados = []
    trayectorias_completas = set()

trayectorias_esperadas = {
    (ancho, semilla)
    for ancho in ANCHOS
    for semilla in SEMILLAS
}

if not trayectorias_completas.issubset(trayectorias_esperadas):
    raise RuntimeError(
        "El archivo contiene una trayectoria fuera del protocolo."
    )


# ============================================================
# 10. Ejecucion de trayectorias pendientes
# ============================================================

print("EXPERIMENTO COMPLEMENTARIO DE TRAYECTORIAS")
print("------------------------------------------")
print(f"Trayectorias completas: {len(trayectorias_completas)}")
print(
    "Trayectorias pendientes: "
    f"{TRAYECTORIAS_ESPERADAS - len(trayectorias_completas)}"
)

for ancho in ANCHOS:
    for semilla in SEMILLAS:
        trayectoria = (ancho, semilla)
        if trayectoria in trayectorias_completas:
            continue

        numero_trayectoria = len(trayectorias_completas) + 1
        print(
            f"\nTrayectoria {numero_trayectoria}/"
            f"{TRAYECTORIAS_ESPERADAS}: "
            f"ancho={ancho:2d}, semilla={semilla:2d}"
        )

        modelo = crear_modelo(ancho, semilla)
        tiempo_acumulado = 0.0
        tiempo_intervalo = 0.0

        for epoca in range(1, max(PUNTOS_CONTROL) + 1):
            inicio_entrenamiento = perf_counter()
            if epoca == 1:
                modelo.partial_fit(
                    X_entrenamiento,
                    y_entrenamiento,
                    classes=CLASES,
                )
            else:
                modelo.partial_fit(X_entrenamiento, y_entrenamiento)

            duracion_epoca = perf_counter() - inicio_entrenamiento
            tiempo_intervalo += duracion_epoca
            tiempo_acumulado += duracion_epoca

            assert modelo.n_iter_ == 1
            assert modelo.t_ == epoca * len(X_entrenamiento)
            assert len(modelo.loss_curve_) == epoca

            if epoca not in PUNTOS_CONTROL:
                continue

            inicio_evaluacion = perf_counter()
            prediccion_entrenamiento = modelo.predict(X_entrenamiento)
            prediccion_validacion = modelo.predict(X_validacion)
            precision_entrenamiento = accuracy_score(
                y_entrenamiento,
                prediccion_entrenamiento,
            )
            precision_validacion = accuracy_score(
                y_validacion,
                prediccion_validacion,
            )
            tiempo_evaluacion = perf_counter() - inicio_evaluacion

            cantidad_parametros = sum(
                matriz.size for matriz in modelo.coefs_
            ) + sum(vector.size for vector in modelo.intercepts_)
            assert cantidad_parametros == 75 * ancho + 10

            assert 0.0 <= precision_entrenamiento <= 1.0
            assert 0.0 <= precision_validacion <= 1.0
            assert np.isfinite(modelo.loss_)
            assert modelo.loss_ >= 0.0
            assert tiempo_intervalo >= 0.0
            assert tiempo_acumulado >= 0.0
            assert tiempo_evaluacion >= 0.0

            resultados.append(
                {
                    "ancho": ancho,
                    "semilla": semilla,
                    "iteracion": epoca,
                    "parametros": cantidad_parametros,
                    "precision_entrenamiento": (
                        precision_entrenamiento
                    ),
                    "precision_validacion": precision_validacion,
                    "perdida_entrenamiento": modelo.loss_,
                    "tiempo_intervalo_segundos": tiempo_intervalo,
                    "tiempo_acumulado_segundos": tiempo_acumulado,
                    "tiempo_evaluacion_segundos": tiempo_evaluacion,
                    "fecha_hora_medicion": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                }
            )
            guardar_resultados(resultados)

            print(
                f"  iteracion={epoca:3d}, "
                f"entrenamiento={precision_entrenamiento:.4f}, "
                f"validacion={precision_validacion:.4f}, "
                f"perdida={modelo.loss_:.6f}, "
                f"tiempo={tiempo_acumulado:.2f} s"
            )
            tiempo_intervalo = 0.0

        trayectorias_completas.add(trayectoria)


# ============================================================
# 11. Verificacion final
# ============================================================

tabla_final = pd.read_csv(RUTA_RESULTADOS)
assert len(tabla_final) == FILAS_ESPERADAS
assert not tabla_final.duplicated(
    subset=["ancho", "semilla", "iteracion"]
).any()

assert set(
    map(
        tuple,
        tabla_final[["ancho", "semilla"]]
        .drop_duplicates()
        .to_numpy(),
    )
) == trayectorias_esperadas

for _, grupo in tabla_final.groupby(["ancho", "semilla"]):
    assert set(grupo["iteracion"]) == set(PUNTOS_CONTROL)
    assert grupo.sort_values("iteracion")[
        "tiempo_acumulado_segundos"
    ].is_monotonic_increasing

assert (
    tabla_final["parametros"]
    == 75 * tabla_final["ancho"] + 10
).all()
assert tabla_final["precision_entrenamiento"].between(0.0, 1.0).all()
assert tabla_final["precision_validacion"].between(0.0, 1.0).all()

print("\nEXPERIMENTO COMPLEMENTARIO COMPLETADO")
print("-------------------------------------")
print(f"Trayectorias: {TRAYECTORIAS_ESPERADAS}")
print(f"Mediciones:   {len(tabla_final)}")
print(f"Archivo:      {RUTA_RESULTADOS}")
