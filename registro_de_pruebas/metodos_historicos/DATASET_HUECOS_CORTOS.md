# Versión 3: huecos cortos y anomalías climáticas

Propuesta propia: reservar la imputación para huecos cortos y no reconstruir años enteros sin medición. Para el primer o segundo mes consecutivo ausente se calcula la mediana de valores **originalmente observados** del mismo mes, 12/24/36/48 meses antes. Si no hay antecedentes, se conserva el nulo.

Los antecedentes pueden incluir datos originales de estaciones anteriores a 2001, siempre dentro de la ventana de cuatro años y anteriores al objetivo.

Se evalúa la racha desde el pasado: no utiliza observaciones futuras para decidir si un hueco termina. Por ello puede rellenar los primeros dos meses tras terminar una estación, pero nunca prolonga toda la serie. No usa valores previamente imputados como fuentes.

## Archivos y trazabilidad

- `data/processed/experimento_clima/historico/huecos_cortos.csv`.
- `data/processed/experimento_clima/oficial/huecos_cortos.csv`.
- `huecos_cortos_auditoria.csv`, en cada carpeta.

La auditoría indica cuántos antecedentes observados contribuyeron a cada mediana; `fuente_fecha` conserva el antecedente más reciente de esos valores. Las columnas de trazabilidad nunca entran como predictores.

Se rechazan antecedentes incompatibles con la otra temperatura disponible y valores negativos de lluvia. Se conservan los nulos cuando no existe un candidato físicamente consistente.

## Transformación de entrenamiento

Antes de ajustar la rama climática se resta a cada rezago la mediana del mes calendario calculada únicamente con su entrenamiento. Esta anomalía busca separar desviaciones climáticas del ciclo anual ya representado por seno/coseno. No está precomputada en el CSV porque hacerlo con todo el historial filtraría información futura.

Las filas con clima todavía incompleto usan el modelo de respaldo de casos. El límite de dos meses y la ventana de cuatro años son decisiones exploratorias, no garantías científicas. Esta versión se compara con las otras y con `solo_casos` sobre los mismos meses; ver [guía completa](EXPERIMENTO_CLIMA.md).
