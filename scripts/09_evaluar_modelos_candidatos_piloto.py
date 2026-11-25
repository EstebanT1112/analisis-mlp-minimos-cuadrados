import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass(frozen=True)
class ModeloNoLineal:
    nombre: str
    parametros: tuple[str, ...]
    limites_inferiores: tuple[float, ...]
    limites_superiores: tuple[float, ...]


MODELOS_NO_LINEALES = {
    "aditivo_saturado": ModeloNoLineal(
        nombre="aditivo_saturado",
        parametros=("P0", "AH", "a", "AT", "b"),
        limites_inferiores=(0.0, 0.0, 1e-6, 0.0, 1e-6),
        limites_superiores=(1.0, 1.0, 10.0, 1.0, 50.0),
    ),
    "multiplicativo_saturado": ModeloNoLineal(
        nombre="multiplicativo_saturado",
        parametros=("P0", "A", "a", "b"),
        limites_inferiores=(0.0, 0.0, 1e-6, 1e-6),
        limites_superiores=(1.0, 1.0, 10.0, 50.0),
    ),
    "interaccion_saturada": ModeloNoLineal(
        nombre="interaccion_saturada",
        parametros=("P0", "A", "a", "b", "c"),
        limites_inferiores=(0.0, 0.0, 1e-6, 1e-6, 0.0),
        limites_superiores=(1.0, 1.0, 10.0, 50.0, 8.0),
    ),
}

NOMBRE_CUADRATICO = "cuadratico_logit"
PARAMETROS_CUADRATICO = (
    "beta0",
    "beta_u",
    "beta_v",
    "beta_u2",
    "beta_v2",
    "beta_uv",
)


def modelo_aditivo(parametros, ancho, tiempo_normalizado):
    p0, amplitud_h, tasa_h, amplitud_t, tasa_t = parametros
    return (
        p0
        + amplitud_h * (1.0 - np.exp(-tasa_h * ancho))
        + amplitud_t * (
            1.0 - np.exp(-tasa_t * tiempo_normalizado)
        )
    )


def modelo_multiplicativo(parametros, ancho, tiempo_normalizado):
    p0, amplitud, tasa_h, tasa_t = parametros
    return p0 + amplitud * (
        1.0 - np.exp(-tasa_h * ancho)
    ) * (
        1.0 - np.exp(-tasa_t * tiempo_normalizado)
    )


def modelo_interaccion(parametros, ancho, tiempo_normalizado):
    p0, amplitud, tasa_h, tasa_t, exponente = parametros
    return p0 + amplitud * (
        1.0 - np.exp(-tasa_h * ancho)
    ) * (
        1.0
        - np.exp(
            -tasa_t
            * tiempo_normalizado
            * np.power(ancho, exponente)
        )
    )


FUNCIONES_NO_LINEALES = {
    "aditivo_saturado": modelo_aditivo,
    "multiplicativo_saturado": modelo_multiplicativo,
    "interaccion_saturada": modelo_interaccion,
}


def matriz_cuadratica(ancho, iteracion):
    u = np.log(ancho)
    v = np.log(iteracion / 25.0)
    return np.column_stack(
        [
            np.ones_like(u),
            u,
            v,
            u**2,
            v**2,
            u * v,
        ]
    )


def pronosticar_cuadratico(parametros, ancho, iteracion):
    valor_logit = matriz_cuadratica(ancho, iteracion) @ parametros
    return 1.0 / (1.0 + np.exp(-valor_logit))


def generar_inicio_aleatorio(especificacion, generador):
    inferior = np.asarray(especificacion.limites_inferiores)
    superior = np.asarray(especificacion.limites_superiores)
    inicio = generador.uniform(inferior, superior)

    for posicion, nombre in enumerate(especificacion.parametros):
        if nombre in {"a", "b"}:
            inicio[posicion] = np.exp(
                generador.uniform(
                    np.log(inferior[posicion]),
                    np.log(superior[posicion]),
                )
            )

    return inicio


def ajustar_no_lineal(
    nombre,
    ancho,
    iteracion,
    observado,
    cantidad_inicios=80,
    semilla=2026,
):
    especificacion = MODELOS_NO_LINEALES[nombre]
    funcion = FUNCIONES_NO_LINEALES[nombre]
    tiempo_normalizado = iteracion / 800.0
    inferior = np.asarray(especificacion.limites_inferiores)
    superior = np.asarray(especificacion.limites_superiores)
    generador = np.random.default_rng(semilla)
    soluciones = []

    for _ in range(cantidad_inicios):
        inicio = generar_inicio_aleatorio(
            especificacion,
            generador,
        )
        resultado = least_squares(
            lambda parametros: (
                funcion(
                    parametros,
                    ancho,
                    tiempo_normalizado,
                )
                - observado
            ),
            x0=inicio,
            bounds=(inferior, superior),
            max_nfev=100000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )

        if resultado.success and np.isfinite(resultado.cost):
            pronostico = funcion(
                resultado.x,
                ancho,
                tiempo_normalizado,
            )
            sse = np.sum((observado - pronostico) ** 2)
            soluciones.append((sse, resultado.x, pronostico))

    if not soluciones:
        raise RuntimeError(f"No convergió el modelo {nombre}")

    soluciones.sort(key=lambda elemento: elemento[0])
    mejor_sse, mejores_parametros, mejor_pronostico = soluciones[0]

    tolerancia = mejor_sse * 1.001 + 1e-12
    cercanas = [
        parametros
        for sse, parametros, _ in soluciones
        if sse <= tolerancia
    ]
    matriz_cercanas = np.asarray(cercanas)
    rango = np.ptp(matriz_cercanas, axis=0)
    escala = np.maximum(np.abs(mejores_parametros), 1e-8)
    rango_relativo_maximo = np.max(rango / escala)

    return {
        "parametros": mejores_parametros,
        "pronostico": mejor_pronostico,
        "sse": mejor_sse,
        "inicios_convergentes": len(soluciones),
        "soluciones_cercanas": len(cercanas),
        "rango_relativo_maximo": rango_relativo_maximo,
    }


def ajustar_cuadratico(ancho, iteracion, observado):
    matriz = matriz_cuadratica(ancho, iteracion)
    if not np.all((observado > 0.0) & (observado < 1.0)):
        raise ValueError(
            "El ajuste logit requiere precisiones entre 0 y 1."
        )

    observado_logit = np.log(observado / (1.0 - observado))
    parametros, _, rango, valores_singulares = np.linalg.lstsq(
        matriz,
        observado_logit,
        rcond=None,
    )
    valor_logit = matriz @ parametros
    pronostico = 1.0 / (1.0 + np.exp(-valor_logit))
    condicion = valores_singulares.max() / valores_singulares.min()

    return {
        "parametros": parametros,
        "pronostico": pronostico,
        "sse": np.sum((observado - pronostico) ** 2),
        "inicios_convergentes": 1,
        "soluciones_cercanas": 1,
        "rango_relativo_maximo": 0.0,
        "rango_matriz": int(rango),
        "numero_condicion": condicion,
    }


def calcular_metricas(observado, pronostico, cantidad_parametros):
    residuos = observado - pronostico
    sse = np.sum(residuos**2)
    st = np.sum((observado - observado.mean()) ** 2)
    cantidad = len(observado)
    rmse = np.sqrt(sse / cantidad)
    mae = np.mean(np.abs(residuos))
    r2 = 1.0 - sse / st
    aicc = (
        cantidad * np.log(sse / cantidad)
        + 2.0 * cantidad_parametros
        + (
            2.0
            * cantidad_parametros
            * (cantidad_parametros + 1)
            / (cantidad - cantidad_parametros - 1)
        )
    )

    return {
        "sse": sse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "aicc": aicc,
        "residuo_medio": residuos.mean(),
        "residuo_maximo_absoluto": np.max(np.abs(residuos)),
    }


def ajustar_todos(ancho, iteracion, observado, semilla):
    resultados = {}

    for nombre in MODELOS_NO_LINEALES:
        resultados[nombre] = ajustar_no_lineal(
            nombre,
            ancho,
            iteracion,
            observado,
            semilla=semilla,
        )

    resultados[NOMBRE_CUADRATICO] = ajustar_cuadratico(
        ancho,
        iteracion,
        observado,
    )

    return resultados


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
        else raiz
        / "resultados"
        / "complementario_piloto_modelos_candidatos"
    )
    salida.mkdir(parents=True, exist_ok=True)

    ruta_datos = (
        raiz
        / "resultados"
        / "complementario_piloto_trayectorias.csv"
    )
    datos = pd.read_csv(ruta_datos)

    assert len(datos) == 126
    assert not datos.duplicated(
        ["ancho", "semilla", "iteracion"]
    ).any()

    medias = (
        datos.groupby(["ancho", "iteracion"], as_index=False)
        ["precision_validacion"]
        .mean()
        .sort_values(["ancho", "iteracion"])
        .reset_index(drop=True)
    )

    ancho = medias["ancho"].to_numpy(dtype=float)
    iteracion = medias["iteracion"].to_numpy(dtype=float)
    observado = medias["precision_validacion"].to_numpy()

    ajustes_medios = ajustar_todos(
        ancho,
        iteracion,
        observado,
        semilla=2026,
    )

    filas_metricas = []
    filas_parametros = []
    filas_residuos = []

    for nombre, ajuste in ajustes_medios.items():
        nombres_parametros = (
            MODELOS_NO_LINEALES[nombre].parametros
            if nombre in MODELOS_NO_LINEALES
            else PARAMETROS_CUADRATICO
        )
        metricas = calcular_metricas(
            observado,
            ajuste["pronostico"],
            len(nombres_parametros),
        )

        filas_metricas.append(
            {
                "modelo": nombre,
                "cantidad_parametros": len(nombres_parametros),
                **metricas,
                "inicios_convergentes": ajuste[
                    "inicios_convergentes"
                ],
                "soluciones_cercanas": ajuste[
                    "soluciones_cercanas"
                ],
                "rango_relativo_maximo_multinicio": ajuste[
                    "rango_relativo_maximo"
                ],
                "pronostico_minimo_observado": np.min(
                    ajuste["pronostico"]
                ),
                "pronostico_maximo_observado": np.max(
                    ajuste["pronostico"]
                ),
            }
        )

        for parametro, valor in zip(
            nombres_parametros,
            ajuste["parametros"],
        ):
            filas_parametros.append(
                {
                    "modelo": nombre,
                    "parametro": parametro,
                    "valor": valor,
                }
            )

        for posicion, fila in medias.iterrows():
            pronostico = ajuste["pronostico"][posicion]
            filas_residuos.append(
                {
                    "modelo": nombre,
                    "ancho": int(fila["ancho"]),
                    "iteracion": int(fila["iteracion"]),
                    "observado": fila["precision_validacion"],
                    "pronostico": pronostico,
                    "residuo": (
                        fila["precision_validacion"] - pronostico
                    ),
                }
            )

    tabla_metricas = pd.DataFrame(filas_metricas).sort_values("rmse")
    tabla_parametros = pd.DataFrame(filas_parametros)
    tabla_residuos = pd.DataFrame(filas_residuos)

    tabla_metricas.to_csv(salida / "comparacion_modelos.csv", index=False)
    tabla_parametros.to_csv(salida / "parametros_modelos.csv", index=False)
    tabla_residuos.to_csv(salida / "residuos_modelos.csv", index=False)

    filas_estabilidad = []

    for semilla in sorted(datos["semilla"].unique()):
        subconjunto = datos.loc[datos["semilla"] == semilla].sort_values(
            ["ancho", "iteracion"]
        )
        ancho_semilla = subconjunto["ancho"].to_numpy(dtype=float)
        iteracion_semilla = subconjunto["iteracion"].to_numpy(dtype=float)
        observado_semilla = subconjunto[
            "precision_validacion"
        ].to_numpy()

        ajustes_semilla = ajustar_todos(
            ancho_semilla,
            iteracion_semilla,
            observado_semilla,
            semilla=2026 + int(semilla),
        )

        for nombre, ajuste in ajustes_semilla.items():
            nombres_parametros = (
                MODELOS_NO_LINEALES[nombre].parametros
                if nombre in MODELOS_NO_LINEALES
                else PARAMETROS_CUADRATICO
            )
            metricas = calcular_metricas(
                observado_semilla,
                ajuste["pronostico"],
                len(nombres_parametros),
            )

            fila = {
                "semilla": int(semilla),
                "modelo": nombre,
                "rmse": metricas["rmse"],
                "r2": metricas["r2"],
            }
            fila.update(
                {
                    parametro: valor
                    for parametro, valor in zip(
                        nombres_parametros,
                        ajuste["parametros"],
                    )
                }
            )
            filas_estabilidad.append(fila)

    tabla_estabilidad = pd.DataFrame(filas_estabilidad)
    tabla_estabilidad.to_csv(
        salida / "estabilidad_por_semilla.csv",
        index=False,
    )

    figura, ejes = plt.subplots(2, 2, figsize=(13, 10))

    for eje, nombre in zip(ejes.ravel(), ajustes_medios):
        ajuste = ajustes_medios[nombre]
        pronostico = ajuste["pronostico"]
        minimo = min(observado.min(), pronostico.min())
        maximo = max(observado.max(), pronostico.max())
        eje.scatter(observado, pronostico, alpha=0.8)
        eje.plot([minimo, maximo], [minimo, maximo], color="black")
        eje.set_title(nombre.replace("_", " "))
        eje.set_xlabel("Precisión observada")
        eje.set_ylabel("Precisión pronosticada")
        eje.grid(alpha=0.25)

    figura.suptitle("Pronósticos de los modelos candidatos sobre el piloto")
    figura.tight_layout()
    figura.savefig(
        salida / "observado_vs_pronosticado.png",
        dpi=180,
    )
    plt.close(figura)

    figura, ejes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)

    for eje, nombre in zip(ejes.ravel(), ajustes_medios):
        residuos = tabla_residuos.loc[
            tabla_residuos["modelo"] == nombre
        ]
        for ancho_actual in sorted(residuos["ancho"].unique()):
            curva = residuos.loc[residuos["ancho"] == ancho_actual]
            eje.plot(
                curva["iteracion"],
                curva["residuo"],
                marker="o",
                label=f"H={ancho_actual}",
            )
        eje.axhline(0.0, color="black", linewidth=1)
        eje.set_xscale("log", base=2)
        eje.set_xticks([25, 50, 100, 200, 400, 800])
        eje.set_xticklabels([25, 50, 100, 200, 400, 800])
        eje.set_title(nombre.replace("_", " "))
        eje.set_xlabel("Iteraciones")
        eje.set_ylabel("Residuo")
        eje.grid(alpha=0.25)

    ejes.ravel()[0].legend(ncol=2, fontsize=8)
    figura.suptitle("Residuos de los modelos candidatos")
    figura.tight_layout()
    figura.savefig(salida / "residuos_modelos.png", dpi=180)
    plt.close(figura)

    assert len(tabla_metricas) == 4
    assert len(tabla_parametros) == 20
    assert len(tabla_residuos) == 168
    assert len(tabla_estabilidad) == 12
    assert not tabla_metricas.isna().any().any()
    assert not tabla_parametros.isna().any().any()

    print("EVALUACIÓN DE MODELOS CANDIDATOS COMPLETADA")
    print("-----------------------------------------")
    print(tabla_metricas.to_string(index=False))
    print(f"\nArchivos creados en:\n{salida}")


if __name__ == "__main__":
    main()
