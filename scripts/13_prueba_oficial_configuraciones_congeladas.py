from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


# ============================================================
# 1. Modo de ejecucion
# ============================================================

# Debe permanecer en True durante la verificacion previa. En este
# modo se comprueban datos y protocolo, pero no se entrena ninguna
# red ni se calcula precision sobre la prueba oficial.
SOLO_VERIFICAR = False


# ============================================================
# 2. Rutas
# ============================================================

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_PROCESADOS = RAIZ / "datos" / "procesados"
CARPETA_RESULTADOS = RAIZ / "resultados"

RUTA_ENTRENAMIENTO = CARPETA_PROCESADOS / "entrenamiento_interno.csv"
RUTA_VALIDACION = CARPETA_PROCESADOS / "validacion.csv"
RUTA_PRUEBA = CARPETA_PROCESADOS / "prueba_oficial.csv"

RUTA_CONFIGURACION_CONGELADA = (
    CARPETA_RESULTADOS / "configuracion_prueba_oficial_congelada.json"
)
RUTA_MANIFIESTO = (
    CARPETA_RESULTADOS / "registro_congelamiento_prueba_oficial.sha256"
)

RUTA_RESULTADOS = (
    CARPETA_RESULTADOS / "prueba_oficial_75_ejecuciones.csv"
)
RUTA_RESULTADOS_TEMPORAL = (
    CARPETA_RESULTADOS / "prueba_oficial_75_ejecuciones_temporal.csv"
)
RUTA_CONFIGURACION_EJECUCION = (
    CARPETA_RESULTADOS / "configuracion_ejecucion_prueba_oficial.json"
)
RUTA_RESUMEN = (
    CARPETA_RESULTADOS / "prueba_oficial_resumen_configuraciones.csv"
)
RUTA_DIFERENCIAS = (
    CARPETA_RESULTADOS / "prueba_oficial_diferencias_pareadas.csv"
)
RUTA_RESUMEN_DIFERENCIAS = (
    CARPETA_RESULTADOS / "prueba_oficial_resumen_diferencias.csv"
)


# ============================================================
# 3. Huellas congeladas
# ============================================================

HASH_ENTRENAMIENTO_ESPERADO = (
    "9746E6212B7D65FE41196B29508BFB52"
    "F9EA89A47AF2825EAFDE2C3BF56AB540"
)
HASH_VALIDACION_ESPERADO = (
    "65A5489166C35ED0343FE153AAC609998"
    "97E36DF3A28FB9763777D04C46A7F78"
)
HASH_PRUEBA_ESPERADO = (
    "2AC6123EC4B84031C33E9D009BC7E3F69"
    "BE77E4A5D1E5032ABB9E9CDE694F6C3"
)
HASH_CONFIGURACION_CONGELADA_ESPERADO = (
    "D15E4CC7A1EEB9D714005C40645FC651"
    "E7089ECDB3C3E44C59D0B57FE5A4C999"
)


# ============================================================
# 4. Columnas y constantes
# ============================================================

COLUMNAS_PIXELES = [
    f"pixel_{numero:02d}" for numero in range(1, 65)
]
CLASES = np.arange(10)
T_CRITICO_95_14_GL = 2.1447866879

COLUMNAS_RESULTADOS = [
    "id_configuracion",
    "analisis",
    "rol",
    "ancho",
    "semilla",
    "tipo_detencion",
    "iteraciones_objetivo",
    "iteraciones_ejecutadas",
    "parametros",
    "precision_entrenamiento",
    "precision_prueba_oficial",
    "perdida_final",
    "tiempo_segundos",
    "advertencia_convergencia",
    "fecha_hora_finalizacion",
]

IDS_ESPERADOS = [
    "principal_compacto_H14",
    "principal_referencia_H25",
    "complementario_parametros_H10_T200",
    "complementario_intermedio_H11_T100",
    "complementario_iteraciones_H14_T50",
]


# ============================================================
# 5. Funciones auxiliares
# ============================================================

def calcular_sha256(ruta: Path) -> str:
    calculador = sha256()
    with ruta.open("rb") as archivo:
        while bloque := archivo.read(1024 * 1024):
            calculador.update(bloque)
    return calculador.hexdigest().upper()


def cargar_json(ruta: Path) -> dict:
    with ruta.open("r", encoding="utf-8") as archivo:
        return json.load(archivo)


def cargar_manifiesto(ruta: Path) -> dict[str, str]:
    huellas = {}
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea:
            continue
        huella, nombre = linea.split(maxsplit=1)
        huellas[nombre.replace("\\", "/")] = huella.upper()
    return huellas


def cantidad_parametros(modelo: MLPClassifier) -> int:
    return sum(matriz.size for matriz in modelo.coefs_) + sum(
        vector.size for vector in modelo.intercepts_
    )


def crear_modelo_principal(
    ancho: int,
    semilla: int,
    comunes: dict,
    protocolo: dict,
) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(ancho,),
        activation=comunes["activacion"],
        solver=comunes["solver"],
        alpha=comunes["alpha"],
        batch_size=comunes["batch_size"],
        learning_rate_init=comunes["learning_rate_init"],
        max_iter=protocolo["max_iter"],
        shuffle=comunes["shuffle"],
        random_state=semilla,
        early_stopping=protocolo["early_stopping"],
        tol=protocolo["tol"],
        n_iter_no_change=protocolo["n_iter_no_change"],
    )


def crear_modelo_complementario(
    ancho: int,
    semilla: int,
    comunes: dict,
    protocolo_principal: dict,
) -> MLPClassifier:
    generador = np.random.RandomState(semilla)
    return MLPClassifier(
        hidden_layer_sizes=(ancho,),
        activation=comunes["activacion"],
        solver=comunes["solver"],
        alpha=comunes["alpha"],
        batch_size=comunes["batch_size"],
        learning_rate_init=comunes["learning_rate_init"],
        max_iter=1,
        shuffle=comunes["shuffle"],
        random_state=generador,
        early_stopping=False,
        tol=protocolo_principal["tol"],
        n_iter_no_change=protocolo_principal["n_iter_no_change"],
    )


def guardar_resultados(resultados: list[dict]) -> None:
    tabla = pd.DataFrame(resultados, columns=COLUMNAS_RESULTADOS)
    tabla = tabla.sort_values(
        ["id_configuracion", "semilla"]
    ).reset_index(drop=True)
    tabla.to_csv(RUTA_RESULTADOS_TEMPORAL, index=False)
    RUTA_RESULTADOS_TEMPORAL.replace(RUTA_RESULTADOS)


def resumir_configuraciones(tabla: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for id_configuracion, grupo in tabla.groupby(
        "id_configuracion", sort=False
    ):
        precisiones = grupo["precision_prueba_oficial"].to_numpy()
        media = float(np.mean(precisiones))
        desvio = float(np.std(precisiones, ddof=1))
        error_estandar = desvio / np.sqrt(len(precisiones))
        margen = T_CRITICO_95_14_GL * error_estandar
        analisis = grupo["analisis"].iloc[0]

        filas.append(
            {
                "id_configuracion": id_configuracion,
                "analisis": analisis,
                "rol": grupo["rol"].iloc[0],
                "ancho": int(grupo["ancho"].iloc[0]),
                "iteraciones_objetivo": int(
                    grupo["iteraciones_objetivo"].iloc[0]
                ),
                "parametros": int(grupo["parametros"].iloc[0]),
                "n": len(precisiones),
                "precision_prueba_media": media,
                "precision_prueba_desvio": desvio,
                "precision_prueba_minima": float(np.min(precisiones)),
                "precision_prueba_maxima": float(np.max(precisiones)),
                "precision_prueba_error_estandar": error_estandar,
                "t_critico_95": T_CRITICO_95_14_GL,
                "ic95_prueba_inferior": media - margen,
                "ic95_prueba_superior": media + margen,
                "objetivo_096_aplica": analisis == "complementario",
                "media_supera_096": (
                    media >= 0.96 if analisis == "complementario" else None
                ),
                "ic95_inferior_supera_096": (
                    media - margen >= 0.96
                    if analisis == "complementario"
                    else None
                ),
                "tiempo_medio_segundos": float(
                    grupo["tiempo_segundos"].mean()
                ),
                "iteraciones_ejecutadas_media": float(
                    grupo["iteraciones_ejecutadas"].mean()
                ),
                "advertencias_convergencia": int(
                    grupo["advertencia_convergencia"].sum()
                ),
            }
        )
    return pd.DataFrame(filas)


def construir_diferencias_pareadas(
    tabla: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparaciones = [
        (
            "principal_H25_menos_H14",
            "principal_referencia_H25",
            "principal_compacto_H14",
        ),
        (
            "complementario_H11T100_menos_H10T200",
            "complementario_intermedio_H11_T100",
            "complementario_parametros_H10_T200",
        ),
        (
            "complementario_H14T50_menos_H10T200",
            "complementario_iteraciones_H14_T50",
            "complementario_parametros_H10_T200",
        ),
        (
            "complementario_H14T50_menos_H11T100",
            "complementario_iteraciones_H14_T50",
            "complementario_intermedio_H11_T100",
        ),
    ]

    filas = []
    for nombre, id_a, id_b in comparaciones:
        a = tabla.loc[
            tabla["id_configuracion"] == id_a,
            ["semilla", "precision_prueba_oficial"],
        ].rename(columns={"precision_prueba_oficial": "precision_a"})
        b = tabla.loc[
            tabla["id_configuracion"] == id_b,
            ["semilla", "precision_prueba_oficial"],
        ].rename(columns={"precision_prueba_oficial": "precision_b"})
        pares = a.merge(b, on="semilla", validate="one_to_one")
        assert len(pares) == 15
        for fila in pares.itertuples(index=False):
            filas.append(
                {
                    "comparacion": nombre,
                    "configuracion_a": id_a,
                    "configuracion_b": id_b,
                    "semilla": int(fila.semilla),
                    "precision_a": float(fila.precision_a),
                    "precision_b": float(fila.precision_b),
                    "diferencia_a_menos_b": float(
                        fila.precision_a - fila.precision_b
                    ),
                }
            )

    detalle = pd.DataFrame(filas)
    resumen_filas = []
    for nombre, grupo in detalle.groupby("comparacion", sort=False):
        diferencias = grupo["diferencia_a_menos_b"].to_numpy()
        media = float(np.mean(diferencias))
        desvio = float(np.std(diferencias, ddof=1))
        error_estandar = desvio / np.sqrt(len(diferencias))
        margen = T_CRITICO_95_14_GL * error_estandar
        resumen_filas.append(
            {
                "comparacion": nombre,
                "configuracion_a": grupo["configuracion_a"].iloc[0],
                "configuracion_b": grupo["configuracion_b"].iloc[0],
                "n_pares": len(diferencias),
                "diferencia_media_a_menos_b": media,
                "diferencia_desvio": desvio,
                "diferencia_error_estandar": error_estandar,
                "t_critico_95": T_CRITICO_95_14_GL,
                "ic95_diferencia_inferior": media - margen,
                "ic95_diferencia_superior": media + margen,
            }
        )
    return detalle, pd.DataFrame(resumen_filas)


# ============================================================
# 6. Verificar configuracion congelada y manifiesto
# ============================================================

assert calcular_sha256(
    RUTA_CONFIGURACION_CONGELADA
) == HASH_CONFIGURACION_CONGELADA_ESPERADO

manifiesto = cargar_manifiesto(RUTA_MANIFIESTO)
assert manifiesto[
    "resultados/configuracion_prueba_oficial_congelada.json"
] == HASH_CONFIGURACION_CONGELADA_ESPERADO

configuracion_congelada = cargar_json(RUTA_CONFIGURACION_CONGELADA)
assert configuracion_congelada[
    "estado_prueba_oficial_al_congelar"
] == "no_utilizada_para_seleccion"
assert configuracion_congelada["semillas"] == list(range(1, 16))

configuraciones = configuracion_congelada["configuraciones"]
assert [configuracion["id"] for configuracion in configuraciones] == (
    IDS_ESPERADOS
)
assert len(configuraciones) == 5
for configuracion in configuraciones:
    assert configuracion["parametros"] == 75 * configuracion["ancho"] + 10


# ============================================================
# 7. Cargar y verificar los datos finales
# ============================================================

hash_entrenamiento = calcular_sha256(RUTA_ENTRENAMIENTO)
hash_validacion = calcular_sha256(RUTA_VALIDACION)
hash_prueba = calcular_sha256(RUTA_PRUEBA)

assert hash_entrenamiento == HASH_ENTRENAMIENTO_ESPERADO
assert hash_validacion == HASH_VALIDACION_ESPERADO
assert hash_prueba == HASH_PRUEBA_ESPERADO

entrenamiento = pd.read_csv(RUTA_ENTRENAMIENTO)
validacion = pd.read_csv(RUTA_VALIDACION)
prueba = pd.read_csv(RUTA_PRUEBA)

assert len(entrenamiento) == 3058
assert len(validacion) == 765
assert len(prueba) == 1797

assert entrenamiento["id_original"].is_unique
assert validacion["id_original"].is_unique
assert prueba["id_original"].is_unique

ids_entrenamiento = set(entrenamiento["id_original"])
ids_validacion = set(validacion["id_original"])
ids_prueba = set(prueba["id_original"])
assert ids_entrenamiento.isdisjoint(ids_validacion)
assert ids_entrenamiento.isdisjoint(ids_prueba)
assert ids_validacion.isdisjoint(ids_prueba)

entrenamiento_final = pd.concat(
    [entrenamiento, validacion], ignore_index=True
).sort_values("id_original").reset_index(drop=True)

ids_tra_esperados = [f"TRA_{numero:04d}" for numero in range(1, 3824)]
ids_tes_esperados = [f"TES_{numero:04d}" for numero in range(1, 1798)]
assert entrenamiento_final["id_original"].tolist() == ids_tra_esperados
assert prueba["id_original"].tolist() == ids_tes_esperados
assert set(entrenamiento_final["archivo_origen"]) == {"optdigits.tra"}
assert set(prueba["archivo_origen"]) == {"optdigits.tes"}

assert set(entrenamiento_final["clase"].unique()) == set(CLASES)
assert set(prueba["clase"].unique()) == set(CLASES)
assert not entrenamiento_final[COLUMNAS_PIXELES].isna().any().any()
assert not prueba[COLUMNAS_PIXELES].isna().any().any()

X_entrenamiento = (
    entrenamiento_final[COLUMNAS_PIXELES].to_numpy(dtype=float) / 16.0
)
y_entrenamiento = entrenamiento_final["clase"].to_numpy()
X_prueba = prueba[COLUMNAS_PIXELES].to_numpy(dtype=float) / 16.0
y_prueba = prueba["clase"].to_numpy()

assert X_entrenamiento.shape == (3823, 64)
assert X_prueba.shape == (1797, 64)
assert np.isfinite(X_entrenamiento).all()
assert np.isfinite(X_prueba).all()
assert 0.0 <= X_entrenamiento.min() <= X_entrenamiento.max() <= 1.0
assert 0.0 <= X_prueba.min() <= X_prueba.max() <= 1.0


# ============================================================
# 8. Configuracion de ejecucion
# ============================================================

semillas = configuracion_congelada["semillas"]
comunes = configuracion_congelada["hiperparametros_comunes"]
protocolo_principal = configuracion_congelada["protocolo_principal"]
protocolo_complementario = configuracion_congelada[
    "protocolo_complementario"
]

configuracion_ejecucion = {
    "tipo": "comprobacion final con prueba oficial",
    "configuracion_congelada": RUTA_CONFIGURACION_CONGELADA.name,
    "sha256_configuracion_congelada": (
        HASH_CONFIGURACION_CONGELADA_ESPERADO
    ),
    "archivo_entrenamiento_interno": RUTA_ENTRENAMIENTO.name,
    "archivo_validacion": RUTA_VALIDACION.name,
    "archivo_prueba_oficial": RUTA_PRUEBA.name,
    "sha256_entrenamiento_interno": hash_entrenamiento,
    "sha256_validacion": hash_validacion,
    "sha256_prueba_oficial": hash_prueba,
    "entrenamiento_final_observaciones": len(entrenamiento_final),
    "prueba_oficial_observaciones": len(prueba),
    "orden_entrenamiento_final": "id_original TRA_0001 a TRA_3823",
    "normalizacion": "division de los pixeles por 16",
    "configuraciones": configuraciones,
    "semillas": semillas,
    "hiperparametros_comunes": comunes,
    "protocolo_principal": protocolo_principal,
    "protocolo_complementario": protocolo_complementario,
    "ejecuciones_esperadas": len(configuraciones) * len(semillas),
    "version_python": platform.python_version(),
    "version_numpy": np.__version__,
    "version_pandas": pd.__version__,
    "version_scikit_learn": sklearn.__version__,
}


# ============================================================
# 9. Verificacion previa sin entrenamiento ni prediccion
# ============================================================

if SOLO_VERIFICAR:
    print("VERIFICACION DE LA PRUEBA OFICIAL CONGELADA")
    print("-------------------------------------------")
    print(f"Entrenamiento final: {X_entrenamiento.shape}")
    print(f"Prueba oficial:      {X_prueba.shape}")
    print(f"Configuraciones:     {len(configuraciones)}")
    print(f"Semillas:            {len(semillas)}")
    print(
        "Ejecuciones:         "
        f"{len(configuraciones) * len(semillas)}"
    )
    print("\nConfiguraciones congeladas:")
    for configuracion in configuraciones:
        print(
            f"- {configuracion['id']}: "
            f"H={configuracion['ancho']}, "
            f"iteraciones={configuracion['iteraciones']}, "
            f"parametros={configuracion['parametros']}"
        )
    print("\nHashes de datos y configuracion: OK")
    print("Orden TRA_0001 a TRA_3823:       OK")
    print("Orden TES_0001 a TES_1797:       OK")
    print("No se entreno ninguna red.")
    print("No se calculo precision sobre la prueba oficial.")
    sys.exit(0)


# ============================================================
# 10. Proteger configuracion y recuperar ejecuciones
# ============================================================

if RUTA_RESULTADOS.exists() and not RUTA_CONFIGURACION_EJECUCION.exists():
    raise RuntimeError(
        "Existen resultados sin su configuracion de ejecucion. "
        "No es seguro continuar."
    )

if RUTA_CONFIGURACION_EJECUCION.exists():
    configuracion_anterior = cargar_json(RUTA_CONFIGURACION_EJECUCION)
    if configuracion_anterior != configuracion_ejecucion:
        raise RuntimeError(
            "La configuracion actual no coincide con la que inicio "
            "la comprobacion oficial."
        )
else:
    with RUTA_CONFIGURACION_EJECUCION.open(
        "w", encoding="utf-8"
    ) as archivo:
        json.dump(
            configuracion_ejecucion,
            archivo,
            indent=2,
            ensure_ascii=False,
        )

if RUTA_RESULTADOS.exists():
    tabla_existente = pd.read_csv(RUTA_RESULTADOS)
    columnas_faltantes = set(COLUMNAS_RESULTADOS) - set(
        tabla_existente.columns
    )
    if columnas_faltantes:
        raise RuntimeError(
            f"Faltan columnas en los resultados: {columnas_faltantes}"
        )
    if tabla_existente.duplicated(
        subset=["id_configuracion", "semilla"]
    ).any():
        raise RuntimeError("Existen ejecuciones duplicadas.")
    resultados = tabla_existente[COLUMNAS_RESULTADOS].to_dict("records")
else:
    resultados = []

completadas = {
    (fila["id_configuracion"], int(fila["semilla"]))
    for fila in resultados
}
esperadas = {
    (configuracion["id"], semilla)
    for configuracion in configuraciones
    for semilla in semillas
}
if not completadas.issubset(esperadas):
    raise RuntimeError(
        "Los resultados contienen una combinacion no congelada."
    )


# ============================================================
# 11. Ejecutar las combinaciones pendientes
# ============================================================

print("COMPROBACION CON PRUEBA OFICIAL")
print("--------------------------------")
print(f"Completadas anteriormente: {len(completadas)}")
print(f"Pendientes:                {len(esperadas) - len(completadas)}")

for configuracion in configuraciones:
    for semilla in semillas:
        clave = (configuracion["id"], semilla)
        if clave in completadas:
            continue

        numero = len(resultados) + 1
        ancho = int(configuracion["ancho"])
        analisis = configuracion["analisis"]

        print(
            f"[{numero:02d}/{len(esperadas)}] "
            f"{configuracion['id']}, semilla={semilla:2d}...",
            end=" ",
            flush=True,
        )

        inicio = perf_counter()
        hubo_advertencia = False

        if analisis == "principal":
            modelo = crear_modelo_principal(
                ancho,
                semilla,
                comunes,
                protocolo_principal,
            )
            with warnings.catch_warnings(record=True) as advertencias:
                warnings.simplefilter("always", ConvergenceWarning)
                modelo.fit(X_entrenamiento, y_entrenamiento)
            hubo_advertencia = any(
                issubclass(
                    advertencia.category,
                    ConvergenceWarning,
                )
                for advertencia in advertencias
            )
            tipo_detencion = "convergencia_hasta_maximo"
            iteraciones_objetivo = int(protocolo_principal["max_iter"])
            iteraciones_ejecutadas = int(modelo.n_iter_)
        elif analisis == "complementario":
            modelo = crear_modelo_complementario(
                ancho,
                semilla,
                comunes,
                protocolo_principal,
            )
            iteraciones_objetivo = int(configuracion["iteraciones"])
            for epoca in range(1, iteraciones_objetivo + 1):
                if epoca == 1:
                    modelo.partial_fit(
                        X_entrenamiento,
                        y_entrenamiento,
                        classes=CLASES,
                    )
                else:
                    modelo.partial_fit(X_entrenamiento, y_entrenamiento)
                assert modelo.n_iter_ == 1
                assert modelo.t_ == epoca * len(X_entrenamiento)
                assert len(modelo.loss_curve_) == epoca
            tipo_detencion = "presupuesto_fijo"
            iteraciones_ejecutadas = iteraciones_objetivo
        else:
            raise RuntimeError(f"Analisis desconocido: {analisis}")

        tiempo_segundos = perf_counter() - inicio
        parametros = cantidad_parametros(modelo)
        assert parametros == configuracion["parametros"]

        precision_entrenamiento = accuracy_score(
            y_entrenamiento,
            modelo.predict(X_entrenamiento),
        )
        precision_prueba = accuracy_score(
            y_prueba,
            modelo.predict(X_prueba),
        )

        assert 0.0 <= precision_entrenamiento <= 1.0
        assert 0.0 <= precision_prueba <= 1.0
        assert np.isfinite(modelo.loss_)
        assert modelo.loss_ >= 0.0

        resultados.append(
            {
                "id_configuracion": configuracion["id"],
                "analisis": analisis,
                "rol": configuracion["rol"],
                "ancho": ancho,
                "semilla": semilla,
                "tipo_detencion": tipo_detencion,
                "iteraciones_objetivo": iteraciones_objetivo,
                "iteraciones_ejecutadas": iteraciones_ejecutadas,
                "parametros": parametros,
                "precision_entrenamiento": precision_entrenamiento,
                "precision_prueba_oficial": precision_prueba,
                "perdida_final": float(modelo.loss_),
                "tiempo_segundos": tiempo_segundos,
                "advertencia_convergencia": hubo_advertencia,
                "fecha_hora_finalizacion": datetime.now().isoformat(
                    timespec="seconds"
                ),
            }
        )
        completadas.add(clave)
        guardar_resultados(resultados)

        print(
            f"prueba={precision_prueba:.4f}, "
            f"iteraciones={iteraciones_ejecutadas}, "
            f"tiempo={tiempo_segundos:.2f} s"
        )


# ============================================================
# 12. Verificacion final y resumenes
# ============================================================

tabla_final = pd.read_csv(RUTA_RESULTADOS)
assert len(tabla_final) == 75
assert not tabla_final.duplicated(
    subset=["id_configuracion", "semilla"]
).any()
assert set(
    map(
        tuple,
        tabla_final[["id_configuracion", "semilla"]].to_numpy(),
    )
) == esperadas
assert (
    tabla_final["parametros"]
    == 75 * tabla_final["ancho"] + 10
).all()
assert tabla_final["precision_entrenamiento"].between(0.0, 1.0).all()
assert tabla_final["precision_prueba_oficial"].between(0.0, 1.0).all()

resumen = resumir_configuraciones(tabla_final)
detalle_diferencias, resumen_diferencias = construir_diferencias_pareadas(
    tabla_final
)

assert len(resumen) == 5
assert (resumen["n"] == 15).all()
assert len(detalle_diferencias) == 60
assert len(resumen_diferencias) == 4

resumen.to_csv(RUTA_RESUMEN, index=False)
detalle_diferencias.to_csv(RUTA_DIFERENCIAS, index=False)
resumen_diferencias.to_csv(RUTA_RESUMEN_DIFERENCIAS, index=False)

print("\nCOMPROBACION OFICIAL COMPLETADA")
print("--------------------------------")
print(f"Ejecuciones: {len(tabla_final)}")
print(f"Resumenes:   {len(resumen)}")
print(f"Archivo:     {RUTA_RESULTADOS}")
print("\nRESULTADOS POR CONFIGURACION")
print(
    resumen[
        [
            "id_configuracion",
            "precision_prueba_media",
            "ic95_prueba_inferior",
            "ic95_prueba_superior",
        ]
    ].to_string(index=False)
)
