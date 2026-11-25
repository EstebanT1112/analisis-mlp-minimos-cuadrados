# Datos utilizados

## Procedencia

Los datos pertenecen al conjunto **Optical Recognition of Handwritten Digits (Optdigits)** del UCI Machine Learning Repository:

- Página oficial: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- DOI permanente: https://doi.org/10.24432/C50P49
- Licencia: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Archivos originales

- `originales/optdigits.tra`: conjunto oficial de entrenamiento, con 3.823 observaciones.
- `originales/optdigits.tes`: conjunto oficial de prueba, con 1.797 observaciones.

Cada fila contiene 64 valores que representan una imagen de 8 por 8 píxeles y una última columna con la clase del dígito, de 0 a 9. Las intensidades originales se encuentran entre 0 y 16.

## Archivos procesados

El conjunto `optdigits.tra` se dividió en:

- `procesados/entrenamiento_interno.csv`: 3.058 observaciones.
- `procesados/validacion.csv`: 765 observaciones.

Además se incluyen:

- `procesados/prueba_oficial.csv`: conversión trazable de `optdigits.tes`.
- `procesados/particion_datos.csv`: registro de la partición asignada a cada observación.

La división mantiene la proporción de las diez clases y utiliza una semilla fija para poder reproducirse. Las características se normalizan dividiendo sus valores por 16, de modo que queden en el intervalo de 0 a 1.

## Uso metodológico

La validación se utilizó para construir los ajustes, comparar modelos y seleccionar configuraciones. La prueba oficial permaneció reservada hasta que las configuraciones quedaron definidas, evitando emplearla para tomar decisiones durante el análisis.
