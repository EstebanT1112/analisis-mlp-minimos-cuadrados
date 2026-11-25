# Análisis del ancho de una MLP mediante mínimos cuadrados

Este repositorio contiene los datos, scripts, resultados y documentos finales de un estudio sobre la relación entre el ancho de la capa oculta de una red neuronal multicapa (MLP) y su precisión de clasificación.

El análisis utiliza el conjunto público **Optical Recognition of Handwritten Digits (Optdigits)** de UCI. Se estudian redes con una capa oculta de 1 a 25 neuronas, manteniendo constantes los demás hiperparámetros. Cada configuración se repite con 15 semillas para medir la variabilidad.

El trabajo se divide en dos partes:

1. **Análisis principal:** relación entre el ancho de la capa oculta y la precisión media de validación. Se comparan un ajuste lineal, un ajuste de crecimiento saturado y un ajuste exponencial saturado.
2. **Análisis complementario:** relación conjunta entre ancho, cantidad de iteraciones y precisión. Se comparan modelos lineales y saturados, y se estudian configuraciones eficientes según precisión, cantidad de parámetros e iteraciones.

La selección de modelos se realizó con validación. El archivo oficial de prueba se mantuvo reservado hasta congelar las configuraciones finales.

## Estructura

```text
.
├── datos/
│   ├── originales/       # Archivos originales de UCI
│   ├── procesados/       # División reproducible usada en los experimentos
│   └── README.md         # Procedencia, licencia e integridad de los datos
├── documentos/
│   ├── Grupo 03 Analisis MLP mediante Minimos Cuadrados.pdf
│   ├── Análisis principal MLP - Grupo 03.xlsx
│   └── Análisis complementario MLP - Grupo 03.xlsx
├── resultados/
│   ├── complementario_modelos_definitivos/
│   ├── complementario_frontera_eficiencia/
│   └── ...
├── scripts/
└── requirements.txt
```

Los dos libros de cálculo contienen la trazabilidad del análisis por mínimos cuadrados, las métricas, los residuos, las comparaciones y los gráficos utilizados en el paper.

## Requisitos

- Python 3.12
- Dependencias indicadas en `requirements.txt`

Preparación recomendada en Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Secuencia de reproducción

Desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe .\scripts\01_preparar_datos.py
.\.venv\Scripts\python.exe .\scripts\06_experimento_definitivo.py
.\.venv\Scripts\python.exe .\scripts\10_experimento_complementario.py
.\.venv\Scripts\python.exe .\scripts\11_ajustar_modelos_definitivos.py
.\.venv\Scripts\python.exe .\scripts\12_frontera_eficiencia.py
.\.venv\Scripts\python.exe .\scripts\13_prueba_oficial_configuraciones_congeladas.py
```

Los experimentos de entrenamiento pueden tardar varios minutos. Los archivos de resultados ya generados se incluyen para permitir la inspección y verificación sin repetir todo el costo computacional.

El archivo `09_evaluar_modelos_candidatos_piloto.py` se conserva porque contiene funciones de ajuste reutilizadas por `11_ajustar_modelos_definitivos.py`; no es necesario ejecutarlo por separado para reproducir el análisis final.

## Datos y trazabilidad

- `optdigits.tra`: 3.823 observaciones, utilizadas para entrenamiento interno y validación.
- `optdigits.tes`: 1.797 observaciones, reservado para la comprobación final.
- Cada observación contiene 64 características de entrada y la clase del dígito.
- Los valores de entrada se normalizan al intervalo de 0 a 1.
- La partición de entrenamiento y validación se realiza de manera reproducible y conserva la proporción de las clases.

Los archivos procesados y los resultados se mantienen en formato CSV para que cada valor presentado en las planillas pueda rastrearse hasta las ejecuciones originales.

## Fuente del dataset

D. Dua y C. Graff, *Optical Recognition of Handwritten Digits*, UCI Machine Learning Repository. DOI: [10.24432/C50P49](https://doi.org/10.24432/C50P49).

El dataset se distribuye bajo licencia [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

## Nota sobre la licencia del código

Este repositorio no declara por el momento una licencia para el código. La licencia CC BY 4.0 indicada arriba corresponde al dataset de UCI, no necesariamente a los scripts desarrollados para el trabajo.
