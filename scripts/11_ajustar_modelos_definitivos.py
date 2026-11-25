import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANCHOS_ESPERADOS = tuple(range(1, 26))
SEMILLAS_ESPERADAS = tuple(range(1, 16))
ITERACIONES_ESPERADAS = (25, 50, 100, 200, 400, 800)
INICIOS_AJUSTE_MEDIO = 80
INICIOS_POR_SEMILLA = 40


def cargar_modelos(ruta_script):
    especificacion = importlib.util.spec_from_file_location(
        "modelos_candidatos_piloto",
        ruta_script,
    )
    if especificacion is None or especificacion.loader is None:
        raise RuntimeError("No se pudo cargar el módulo de modelos.")

    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def validar_datos(datos):
    columnas_requeridas = {
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
    }
    faltantes = columnas_requeridas.difference(datos.columns)
    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas: "
            + ", ".join(sorted(faltantes))
        )

    if len(datos) != 2250:
        raise ValueError(f"Se esperaban 2250 filas y hay {len(datos)}.")
    if datos[list(columnas_requeridas)].isna().any().any():
        raise ValueError("Existen valores faltantes en columnas requeridas.")
    if datos.duplicated(["ancho", "semilla", "iteracion"]).any():
        raise ValueError("Existen claves ancho-semilla-iteración duplicadas.")
    if tuple(sorted(datos["ancho"].unique())) != ANCHOS_ESPERADOS:
        raise ValueError("Los anchos no coinciden con 1 a 25.")
    if tuple(sorted(datos["semilla"].unique())) != SEMILLAS_ESPERADAS:
        raise ValueError("Las semillas no coinciden con 1 a 15.")
    if tuple(sorted(datos["iteracion"].unique())) != ITERACIONES_ESPERADAS:
        raise ValueError("Los puntos de control no coinciden con el protocolo.")
    if not datos["precision_validacion"].between(0.0, 1.0).all():
        raise ValueError("Hay precisiones de validación fuera de [0, 1].")

    cantidades = datos.groupby(["ancho", "iteracion"]).size()
    if len(cantidades) != 150 or not (cantidades == 15).all():
        raise ValueError("Cada combinación ancho-iteración debe tener 15 semillas.")


def nombres_parametros(modulo, nombre):
    if nombre in modulo.MODELOS_NO_LINEALES:
        return modulo.MODELOS_NO_LINEALES[nombre].parametros
    return modulo.PARAMETROS_CUADRATICO


def ajustar_todos_con_inicios(
    modulo,
    ancho,
    iteracion,
    observado,
    semilla,
    cantidad_inicios,
):
    resultados = {}
    for nombre in modulo.MODELOS_NO_LINEALES:
        resultados[nombre] = modulo.ajustar_no_lineal(
            nombre,
            ancho,
            iteracion,
            observado,
            cantidad_inicios=cantidad_inicios,
            semilla=semilla,
        )
    resultados[modulo.NOMBRE_CUADRATICO] = modulo.ajustar_cuadratico(
        ancho,
        iteracion,
        observado,
    )
    return resultados


def distancia_relativa_a_limites(modulo, nombre, parametros):
    if nombre not in modulo.MODELOS_NO_LINEALES:
        return np.nan

    especificacion = modulo.MODELOS_NO_LINEALES[nombre]
    inferior = np.asarray(especificacion.limites_inferiores, dtype=float)
    superior = np.asarray(especificacion.limites_superiores, dtype=float)
    parametros = np.asarray(parametros, dtype=float)
    amplitud = superior - inferior
    distancia = np.minimum(parametros - inferior, superior - parametros)
    return float(np.min(distancia / amplitud))


def construir_resumen(datos):
    resumen = (
        datos.groupby(["ancho", "iteracion"], as_index=False)
        .agg(
            parametros=("parametros", "first"),
            n=("precision_validacion", "size"),
            precision_entrenamiento_media=(
                "precision_entrenamiento",
                "mean",
            ),
            precision_entrenamiento_desvio=(
                "precision_entrenamiento",
                "std",
            ),
            precision_validacion_media=("precision_validacion", "mean"),
            precision_validacion_desvio=("precision_validacion", "std"),
            precision_validacion_minima=("precision_validacion", "min"),
            precision_validacion_maxima=("precision_validacion", "max"),
            perdida_media=("perdida_entrenamiento", "mean"),
            perdida_desvio=("perdida_entrenamiento", "std"),
            tiempo_acumulado_medio=("tiempo_acumulado_segundos", "mean"),
        )
        .sort_values(["ancho", "iteracion"])
        .reset_index(drop=True)
    )
    resumen["brecha_entrenamiento_validacion"] = (
        resumen["precision_entrenamiento_media"]
        - resumen["precision_validacion_media"]
    )
    resumen["error_estandar_validacion"] = (
        resumen["precision_validacion_desvio"] / np.sqrt(resumen["n"])
    )
    return resumen


def evaluar_ajustes(modulo, resumen, ajustes):
    observado = resumen["precision_validacion_media"].to_numpy()
    filas_metricas = []
    filas_parametros = []
    filas_residuos = []

    for nombre, ajuste in ajustes.items():
        parametros_nombres = nombres_parametros(modulo, nombre)
        metricas = modulo.calcular_metricas(
            observado,
            ajuste["pronostico"],
            len(parametros_nombres),
        )
        residuos = observado - ajuste["pronostico"]

        filas_metricas.append(
            {
                "modelo": nombre,
                "cantidad_parametros": len(parametros_nombres),
                **metricas,
                "inicios_convergentes": ajuste["inicios_convergentes"],
                "soluciones_cercanas": ajuste["soluciones_cercanas"],
                "rango_relativo_maximo_multinicio": ajuste[
                    "rango_relativo_maximo"
                ],
                "distancia_relativa_minima_a_limites": (
                    distancia_relativa_a_limites(
                        modulo,
                        nombre,
                        ajuste["parametros"],
                    )
                ),
                "pronostico_minimo_observado": float(
                    np.min(ajuste["pronostico"])
                ),
                "pronostico_maximo_observado": float(
                    np.max(ajuste["pronostico"])
                ),
                "residuos_fuera_mas_2rmse": int(
                    np.sum(np.abs(residuos) > 2.0 * metricas["rmse"])
                ),
            }
        )

        for parametro, valor in zip(
            parametros_nombres,
            ajuste["parametros"],
        ):
            filas_parametros.append(
                {
                    "modelo": nombre,
                    "parametro": parametro,
                    "valor": valor,
                }
            )

        for posicion, fila in resumen.iterrows():
            pronostico = float(ajuste["pronostico"][posicion])
            observado_actual = float(fila["precision_validacion_media"])
            filas_residuos.append(
                {
                    "modelo": nombre,
                    "ancho": int(fila["ancho"]),
                    "iteracion": int(fila["iteracion"]),
                    "observado": observado_actual,
                    "pronostico": pronostico,
                    "residuo": observado_actual - pronostico,
                    "residuo_cuadrado": (observado_actual - pronostico) ** 2,
                }
            )

    metricas = pd.DataFrame(filas_metricas).sort_values("rmse")
    metricas.insert(0, "posicion_rmse", range(1, len(metricas) + 1))
    return (
        metricas,
        pd.DataFrame(filas_parametros),
        pd.DataFrame(filas_residuos),
    )


def evaluar_estabilidad(modulo, datos):
    filas = []

    for semilla in SEMILLAS_ESPERADAS:
        subconjunto = datos.loc[datos["semilla"] == semilla].sort_values(
            ["ancho", "iteracion"]
        )
        ancho = subconjunto["ancho"].to_numpy(dtype=float)
        iteracion = subconjunto["iteracion"].to_numpy(dtype=float)
        observado = subconjunto["precision_validacion"].to_numpy(dtype=float)
        ajustes = ajustar_todos_con_inicios(
            modulo,
            ancho,
            iteracion,
            observado,
            semilla=2026 + semilla,
            cantidad_inicios=INICIOS_POR_SEMILLA,
        )

        for nombre, ajuste in ajustes.items():
            parametros_nombres = nombres_parametros(modulo, nombre)
            metricas = modulo.calcular_metricas(
                observado,
                ajuste["pronostico"],
                len(parametros_nombres),
            )
            fila = {
                "semilla": semilla,
                "modelo": nombre,
                **metricas,
                "pronostico_minimo": float(np.min(ajuste["pronostico"])),
                "pronostico_maximo": float(np.max(ajuste["pronostico"])),
                "inicios_convergentes": ajuste["inicios_convergentes"],
                "soluciones_cercanas": ajuste["soluciones_cercanas"],
                "rango_relativo_maximo_multinicio": ajuste[
                    "rango_relativo_maximo"
                ],
            }
            fila.update(
                {
                    parametro: valor
                    for parametro, valor in zip(
                        parametros_nombres,
                        ajuste["parametros"],
                    )
                }
            )
            filas.append(fila)

    return pd.DataFrame(filas)


def resumir_estabilidad(estabilidad):
    columnas_metricas = [
        "rmse",
        "mae",
        "r2",
        "aicc",
        "residuo_maximo_absoluto",
        "pronostico_minimo",
        "pronostico_maximo",
    ]
    filas = []
    for modelo, grupo in estabilidad.groupby("modelo"):
        for variable in columnas_metricas:
            serie = grupo[variable].dropna()
            filas.append(
                {
                    "modelo": modelo,
                    "variable": variable,
                    "media": serie.mean(),
                    "desvio": serie.std(ddof=1),
                    "minimo": serie.min(),
                    "maximo": serie.max(),
                }
            )
    return pd.DataFrame(filas)


def diagnosticos_agrupados(residuos):
    por_iteracion = (
        residuos.groupby(["modelo", "iteracion"], as_index=False)
        .agg(
            residuo_medio=("residuo", "mean"),
            rmse=("residuo_cuadrado", lambda x: np.sqrt(x.mean())),
            residuo_maximo_absoluto=("residuo", lambda x: np.abs(x).max()),
        )
    )
    por_ancho = (
        residuos.groupby(["modelo", "ancho"], as_index=False)
        .agg(
            residuo_medio=("residuo", "mean"),
            rmse=("residuo_cuadrado", lambda x: np.sqrt(x.mean())),
            residuo_maximo_absoluto=("residuo", lambda x: np.abs(x).max()),
        )
    )
    return por_iteracion, por_ancho


def crear_graficos(salida, resumen, ajustes, residuos):
    observado = resumen["precision_validacion_media"].to_numpy()
    figura, ejes = plt.subplots(2, 2, figsize=(13, 10))
    for eje, (nombre, ajuste) in zip(ejes.ravel(), ajustes.items()):
        pronostico = ajuste["pronostico"]
        minimo = min(observado.min(), pronostico.min())
        maximo = max(observado.max(), pronostico.max())
        eje.scatter(observado, pronostico, alpha=0.65, s=25)
        eje.plot([minimo, maximo], [minimo, maximo], color="black")
        eje.set_title(nombre.replace("_", " "))
        eje.set_xlabel("Precisión observada")
        eje.set_ylabel("Precisión pronosticada")
        eje.grid(alpha=0.25)
    figura.suptitle("Modelos candidatos — experimento definitivo")
    figura.tight_layout()
    figura.savefig(salida / "observado_vs_pronosticado.png", dpi=180)
    plt.close(figura)

    figura, ejes = plt.subplots(2, 2, figsize=(14, 10))
    for eje, nombre in zip(ejes.ravel(), ajustes):
        tabla = (
            residuos.loc[residuos["modelo"] == nombre]
            .pivot(index="ancho", columns="iteracion", values="residuo")
            .reindex(index=ANCHOS_ESPERADOS, columns=ITERACIONES_ESPERADAS)
        )
        limite = np.abs(tabla.to_numpy()).max()
        imagen = eje.imshow(
            tabla.to_numpy(),
            aspect="auto",
            cmap="coolwarm",
            vmin=-limite,
            vmax=limite,
            origin="lower",
        )
        eje.set_title(nombre.replace("_", " "))
        eje.set_xlabel("Iteraciones")
        eje.set_ylabel("Ancho")
        eje.set_xticks(range(len(ITERACIONES_ESPERADAS)))
        eje.set_xticklabels(ITERACIONES_ESPERADAS)
        eje.set_yticks(range(0, 25, 4))
        eje.set_yticklabels(range(1, 26, 4))
        figura.colorbar(imagen, ax=eje, shrink=0.8)
    figura.suptitle("Mapa de residuos — observado menos pronosticado")
    figura.tight_layout()
    figura.savefig(salida / "mapas_residuos.png", dpi=180)
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
        else raiz / "resultados" / "complementario_modelos_definitivos"
    )
    salida.mkdir(parents=True, exist_ok=True)

    modulo = cargar_modelos(
        raiz / "scripts" / "09_evaluar_modelos_candidatos_piloto.py"
    )
    ruta_datos = (
        raiz / "resultados" / "complementario_experimento_2250_mediciones.csv"
    )
    datos = pd.read_csv(ruta_datos)
    validar_datos(datos)
    resumen = construir_resumen(datos)

    ancho = resumen["ancho"].to_numpy(dtype=float)
    iteracion = resumen["iteracion"].to_numpy(dtype=float)
    observado = resumen["precision_validacion_media"].to_numpy(dtype=float)
    ajustes = ajustar_todos_con_inicios(
        modulo,
        ancho,
        iteracion,
        observado,
        semilla=2026,
        cantidad_inicios=INICIOS_AJUSTE_MEDIO,
    )

    metricas, parametros, residuos = evaluar_ajustes(
        modulo,
        resumen,
        ajustes,
    )
    estabilidad = evaluar_estabilidad(modulo, datos)
    estabilidad_resumen = resumir_estabilidad(estabilidad)
    residuos_iteracion, residuos_ancho = diagnosticos_agrupados(residuos)

    resumen.to_csv(salida / "resumen_150_puntos.csv", index=False)
    metricas.to_csv(salida / "comparacion_modelos.csv", index=False)
    parametros.to_csv(salida / "parametros_modelos.csv", index=False)
    residuos.to_csv(salida / "residuos_modelos.csv", index=False)
    estabilidad.to_csv(salida / "estabilidad_por_semilla.csv", index=False)
    estabilidad_resumen.to_csv(
        salida / "resumen_estabilidad_metricas.csv",
        index=False,
    )
    residuos_iteracion.to_csv(
        salida / "diagnostico_residuos_por_iteracion.csv",
        index=False,
    )
    residuos_ancho.to_csv(
        salida / "diagnostico_residuos_por_ancho.csv",
        index=False,
    )
    crear_graficos(salida, resumen, ajustes, residuos)

    if len(resumen) != 150 or not (resumen["n"] == 15).all():
        raise AssertionError("El resumen definitivo no tiene 150 puntos con n=15.")
    if len(metricas) != 4:
        raise AssertionError("La comparación no contiene los cuatro modelos.")
    if len(parametros) != 20:
        raise AssertionError("La tabla de parámetros no contiene 20 filas.")
    if len(residuos) != 600:
        raise AssertionError("La tabla de residuos no contiene 600 filas.")
    if len(estabilidad) != 60:
        raise AssertionError("La estabilidad no contiene 60 ajustes.")
    for tabla in (metricas, parametros, residuos, estabilidad):
        if tabla.empty:
            raise AssertionError("Se generó una tabla vacía.")
    columnas_metricas_obligatorias = [
        columna
        for columna in metricas.columns
        if columna != "distancia_relativa_minima_a_limites"
    ]
    if metricas[columnas_metricas_obligatorias].isna().any().any():
        raise AssertionError("Hay métricas definitivas faltantes.")
    distancia_cuadratico = metricas.loc[
        metricas["modelo"] == modulo.NOMBRE_CUADRATICO,
        "distancia_relativa_minima_a_limites",
    ]
    if len(distancia_cuadratico) != 1 or not distancia_cuadratico.isna().all():
        raise AssertionError(
            "La distancia a límites debe ser no aplicable en el cuadrático."
        )
    if parametros.isna().any().any():
        raise AssertionError("Hay parámetros definitivos faltantes.")

    print("AJUSTE DEFINITIVO DE MODELOS COMPLETADO")
    print("--------------------------------------")
    print(metricas.to_string(index=False))
    print(f"\nArchivos creados en:\n{salida}")


if __name__ == "__main__":
    main()
