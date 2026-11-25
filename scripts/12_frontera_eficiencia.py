import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as distribucion_t


OBJETIVOS = (0.90, 0.95, 0.96)
CRITERIOS = ("media", "ic95_inferior")
ITERACIONES_OBSERVADAS = (25, 50, 100, 200, 400, 800)


def validar_mediciones(datos):
    requeridas = {
        "ancho",
        "semilla",
        "iteracion",
        "parametros",
        "precision_validacion",
        "tiempo_acumulado_segundos",
    }
    faltantes = requeridas.difference(datos.columns)
    if faltantes:
        raise ValueError(
            "Faltan columnas: " + ", ".join(sorted(faltantes))
        )
    if len(datos) != 2250:
        raise ValueError(f"Se esperaban 2250 mediciones y hay {len(datos)}.")
    if datos[list(requeridas)].isna().any().any():
        raise ValueError("Hay valores faltantes en las mediciones.")
    if datos.duplicated(["ancho", "semilla", "iteracion"]).any():
        raise ValueError("Hay claves ancho-semilla-iteración duplicadas.")
    if tuple(sorted(datos["ancho"].unique())) != tuple(range(1, 26)):
        raise ValueError("Los anchos no coinciden con 1 a 25.")
    if tuple(sorted(datos["semilla"].unique())) != tuple(range(1, 16)):
        raise ValueError("Las semillas no coinciden con 1 a 15.")
    if tuple(sorted(datos["iteracion"].unique())) != ITERACIONES_OBSERVADAS:
        raise ValueError("Los puntos de control no coinciden con el protocolo.")


def construir_resumen(datos):
    resumen = (
        datos.groupby(["ancho", "iteracion"], as_index=False)
        .agg(
            parametros=("parametros", "first"),
            n=("precision_validacion", "size"),
            precision_media=("precision_validacion", "mean"),
            precision_desvio=("precision_validacion", "std"),
            precision_minima=("precision_validacion", "min"),
            precision_maxima=("precision_validacion", "max"),
            tiempo_acumulado_medio=("tiempo_acumulado_segundos", "mean"),
        )
        .sort_values(["ancho", "iteracion"])
        .reset_index(drop=True)
    )
    resumen["error_estandar"] = (
        resumen["precision_desvio"] / np.sqrt(resumen["n"])
    )
    resumen["t_critico"] = distribucion_t.ppf(
        0.975,
        resumen["n"] - 1,
    )
    margen = resumen["t_critico"] * resumen["error_estandar"]
    resumen["ic95_inferior"] = resumen["precision_media"] - margen
    resumen["ic95_superior"] = resumen["precision_media"] + margen
    return resumen


def es_no_dominado(fila, candidatos):
    no_mayor_costo = (
        (candidatos["parametros"] <= fila["parametros"])
        & (candidatos["iteracion"] <= fila["iteracion"])
    )
    estrictamente_mejor = (
        (candidatos["parametros"] < fila["parametros"])
        | (candidatos["iteracion"] < fila["iteracion"])
    )
    return not (no_mayor_costo & estrictamente_mejor).any()


def extraer_frontera(candidatos):
    if candidatos.empty:
        resultado = candidatos.copy()
        resultado["es_frontera"] = pd.Series(dtype=bool)
        return resultado

    resultado = candidatos.copy()
    resultado["es_frontera"] = [
        es_no_dominado(fila, candidatos)
        for _, fila in candidatos.iterrows()
    ]
    return resultado


def construir_fronteras_observadas(datos, resumen):
    filas_candidatos = []
    filas_frontera = []

    for objetivo in OBJETIVOS:
        cumplimiento = (
            datos.assign(
                cumple=(datos["precision_validacion"] >= objetivo).astype(int)
            )
            .groupby(["ancho", "iteracion"], as_index=False)
            .agg(
                semillas_cumplen=("cumple", "sum"),
                proporcion_semillas_cumplen=("cumple", "mean"),
            )
        )
        base = resumen.merge(
            cumplimiento,
            on=["ancho", "iteracion"],
            how="left",
            validate="one_to_one",
        )

        for criterio in CRITERIOS:
            columna = (
                "precision_media"
                if criterio == "media"
                else "ic95_inferior"
            )
            candidatos = base.loc[base[columna] >= objetivo].copy()
            candidatos.insert(0, "criterio", criterio)
            candidatos.insert(0, "objetivo", objetivo)
            evaluados = extraer_frontera(candidatos)
            filas_candidatos.append(evaluados)
            filas_frontera.append(
                evaluados.loc[evaluados["es_frontera"]].copy()
            )

    candidatos = pd.concat(filas_candidatos, ignore_index=True)
    frontera = pd.concat(filas_frontera, ignore_index=True)
    return candidatos, frontera


def cargar_parametros_interaccion(ruta):
    parametros = pd.read_csv(ruta)
    seleccion = parametros.loc[
        parametros["modelo"] == "interaccion_saturada",
        ["parametro", "valor"],
    ]
    valores = dict(zip(seleccion["parametro"], seleccion["valor"]))
    esperados = {"P0", "A", "a", "b", "c"}
    if set(valores) != esperados:
        raise ValueError(
            "Los parámetros de interacción no coinciden con los esperados."
        )
    return valores


def estimar_iteraciones(parametros):
    p0 = parametros["P0"]
    amplitud = parametros["A"]
    tasa_h = parametros["a"]
    tasa_t = parametros["b"]
    exponente = parametros["c"]
    filas = []

    for objetivo in OBJETIVOS:
        for ancho in range(1, 26):
            cantidad_parametros = 75 * ancho + 10
            saturacion_ancho = p0 + amplitud * (
                1.0 - np.exp(-tasa_h * ancho)
            )
            denominador = amplitud * (
                1.0 - np.exp(-tasa_h * ancho)
            )
            q = (objetivo - p0) / denominador

            if not 0.0 < q < 1.0:
                iteraciones = np.nan
                estado = "objetivo_supera_saturacion"
            else:
                iteraciones = (
                    -800.0
                    * np.log(1.0 - q)
                    / (tasa_t * np.power(ancho, exponente))
                )
                if iteraciones < 25.0:
                    estado = "antes_del_primer_checkpoint"
                elif iteraciones > 800.0:
                    estado = "fuera_del_rango_superior"
                else:
                    estado = "dentro_del_dominio"

            filas.append(
                {
                    "objetivo": objetivo,
                    "ancho": ancho,
                    "parametros": cantidad_parametros,
                    "saturacion_estimada": saturacion_ancho,
                    "q": q,
                    "iteraciones_estimadas": iteraciones,
                    "iteraciones_enteras_superiores": (
                        np.ceil(iteraciones)
                        if np.isfinite(iteraciones)
                        else np.nan
                    ),
                    "estado": estado,
                }
            )

    estimaciones = pd.DataFrame(filas)
    validas = estimaciones.loc[
        estimaciones["estado"] == "dentro_del_dominio"
    ].copy()
    fronteras = []
    for objetivo, grupo in validas.groupby("objetivo"):
        grupo = grupo.copy()
        grupo["iteracion"] = grupo["iteraciones_estimadas"]
        grupo = extraer_frontera(grupo)
        fronteras.append(grupo.loc[grupo["es_frontera"]].copy())
    if fronteras:
        frontera_modelo = pd.concat(fronteras, ignore_index=True)
    else:
        frontera_modelo = validas.assign(
            iteracion=pd.Series(dtype=float),
            es_frontera=pd.Series(dtype=bool),
        )
    return estimaciones, frontera_modelo


def construir_resumen_objetivos(candidatos, frontera, frontera_modelo):
    filas = []
    for objetivo in OBJETIVOS:
        for criterio in CRITERIOS:
            candidatos_actuales = candidatos.loc[
                (candidatos["objetivo"] == objetivo)
                & (candidatos["criterio"] == criterio)
            ]
            frontera_actual = frontera.loc[
                (frontera["objetivo"] == objetivo)
                & (frontera["criterio"] == criterio)
            ]
            filas.append(
                {
                    "objetivo": objetivo,
                    "criterio": criterio,
                    "candidatos_elegibles": len(candidatos_actuales),
                    "puntos_frontera": len(frontera_actual),
                    "menor_ancho_frontera": (
                        frontera_actual["ancho"].min()
                        if not frontera_actual.empty
                        else np.nan
                    ),
                    "menor_iteracion_frontera": (
                        frontera_actual["iteracion"].min()
                        if not frontera_actual.empty
                        else np.nan
                    ),
                }
            )

    resumen = pd.DataFrame(filas)
    cantidad_modelo = (
        frontera_modelo.groupby("objetivo").size().rename("puntos_modelo")
    )
    resumen = resumen.merge(
        cantidad_modelo,
        on="objetivo",
        how="left",
    )
    resumen["puntos_modelo"] = (
        resumen["puntos_modelo"].fillna(0).astype(int)
    )
    return resumen


def validar_fronteras(candidatos, frontera):
    for objetivo in OBJETIVOS:
        for criterio in CRITERIOS:
            grupo = candidatos.loc[
                (candidatos["objetivo"] == objetivo)
                & (candidatos["criterio"] == criterio)
            ]
            frente = frontera.loc[
                (frontera["objetivo"] == objetivo)
                & (frontera["criterio"] == criterio)
            ]
            if grupo.empty:
                if not frente.empty:
                    raise AssertionError("Existe frontera sin candidatos.")
                continue
            if frente.empty:
                raise AssertionError("Hay candidatos pero no existe frontera.")
            for _, fila in frente.iterrows():
                if not es_no_dominado(fila, grupo):
                    raise AssertionError("La frontera contiene un punto dominado.")
            for _, fila in grupo.loc[~grupo["es_frontera"]].iterrows():
                if es_no_dominado(fila, grupo):
                    raise AssertionError(
                        "Un punto no dominado quedó fuera de la frontera."
                    )


def crear_graficos(salida, candidatos, frontera, estimaciones):
    figura, ejes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for columna, objetivo in enumerate(OBJETIVOS):
        for fila_indice, criterio in enumerate(CRITERIOS):
            eje = ejes[fila_indice, columna]
            grupo = candidatos.loc[
                (candidatos["objetivo"] == objetivo)
                & (candidatos["criterio"] == criterio)
            ]
            frente = frontera.loc[
                (frontera["objetivo"] == objetivo)
                & (frontera["criterio"] == criterio)
            ].sort_values("parametros")
            eje.scatter(
                grupo["parametros"],
                grupo["iteracion"],
                color="0.75",
                s=24,
                label="Elegibles",
            )
            eje.plot(
                frente["parametros"],
                frente["iteracion"],
                marker="o",
                color="tab:red",
                label="Frontera",
            )
            eje.set_title(f"Objetivo {objetivo:.0%} — {criterio}")
            eje.set_xlabel("Parámetros entrenables")
            eje.set_ylabel("Iteraciones")
            eje.set_yscale("log", base=2)
            eje.set_yticks(ITERACIONES_OBSERVADAS)
            eje.set_yticklabels(ITERACIONES_OBSERVADAS)
            eje.grid(alpha=0.25)
    ejes[0, 0].legend()
    figura.suptitle("Fronteras observadas de eficiencia")
    figura.tight_layout()
    figura.savefig(salida / "fronteras_observadas.png", dpi=180)
    plt.close(figura)

    figura, eje = plt.subplots(figsize=(10, 6))
    for objetivo in OBJETIVOS:
        grupo = estimaciones.loc[estimaciones["objetivo"] == objetivo]
        eje.plot(
            grupo["ancho"],
            grupo["iteraciones_estimadas"],
            marker="o",
            label=f"Objetivo {objetivo:.0%}",
        )
    eje.axhline(25, color="black", linestyle="--", linewidth=1)
    eje.axhline(800, color="black", linestyle="--", linewidth=1)
    eje.set_yscale("log")
    eje.set_xlabel("Ancho")
    eje.set_ylabel("Iteraciones estimadas")
    eje.set_title("Interpolación del modelo de interacción")
    eje.grid(alpha=0.25)
    eje.legend()
    figura.tight_layout()
    figura.savefig(salida / "iteraciones_estimadas_modelo.png", dpi=180)
    plt.close(figura)


def main():
    analizador = argparse.ArgumentParser()
    analizador.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    analizador.add_argument("--output", type=Path, default=None)
    argumentos = analizador.parse_args()

    raiz = argumentos.root.resolve()
    salida = (
        argumentos.output.resolve()
        if argumentos.output is not None
        else raiz / "resultados" / "complementario_frontera_eficiencia"
    )
    salida.mkdir(parents=True, exist_ok=True)

    ruta_mediciones = (
        raiz / "resultados" / "complementario_experimento_2250_mediciones.csv"
    )
    ruta_parametros = (
        raiz
        / "resultados"
        / "complementario_modelos_definitivos"
        / "parametros_modelos.csv"
    )
    datos = pd.read_csv(ruta_mediciones)
    validar_mediciones(datos)
    resumen = construir_resumen(datos)
    candidatos, frontera = construir_fronteras_observadas(datos, resumen)
    parametros = cargar_parametros_interaccion(ruta_parametros)
    estimaciones, frontera_modelo = estimar_iteraciones(parametros)
    resumen_objetivos = construir_resumen_objetivos(
        candidatos,
        frontera,
        frontera_modelo,
    )
    validar_fronteras(candidatos, frontera)

    resumen.to_csv(salida / "resumen_150_puntos_frontera.csv", index=False)
    candidatos.to_csv(salida / "candidatos_observados.csv", index=False)
    frontera.to_csv(salida / "frontera_observada.csv", index=False)
    estimaciones.to_csv(salida / "estimaciones_modelo_interaccion.csv", index=False)
    frontera_modelo.to_csv(salida / "frontera_modelo_interaccion.csv", index=False)
    resumen_objetivos.to_csv(salida / "resumen_objetivos.csv", index=False)
    crear_graficos(salida, candidatos, frontera, estimaciones)

    if len(resumen) != 150 or not (resumen["n"] == 15).all():
        raise AssertionError("El resumen no contiene 150 puntos con n=15.")
    if len(estimaciones) != 75:
        raise AssertionError("Deben existir 75 estimaciones del modelo.")
    if set(frontera["objetivo"].unique()) != set(OBJETIVOS):
        raise AssertionError("Falta algún objetivo en la frontera observada.")
    if not resumen_objetivos["puntos_frontera"].gt(0).all():
        raise AssertionError("Algún criterio quedó sin frontera.")
    if not frontera_modelo.empty:
        if not frontera_modelo["estado"].eq("dentro_del_dominio").all():
            raise AssertionError("La frontera del modelo salió del dominio.")

    print("FRONTERA DE EFICIENCIA COMPLETADA")
    print("--------------------------------")
    print("\nRESUMEN POR OBJETIVO")
    print(resumen_objetivos.to_string(index=False))
    print("\nFRONTERA OBSERVADA")
    columnas = [
        "objetivo",
        "criterio",
        "ancho",
        "iteracion",
        "parametros",
        "precision_media",
        "ic95_inferior",
        "semillas_cumplen",
        "proporcion_semillas_cumplen",
        "tiempo_acumulado_medio",
    ]
    print(frontera[columnas].to_string(index=False))
    print(f"\nArchivos creados en:\n{salida}")


if __name__ == "__main__":
    main()
