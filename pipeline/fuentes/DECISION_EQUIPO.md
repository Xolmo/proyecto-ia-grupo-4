# Fuente primaria: decisiones comunicadas por el integrante solicitante

Fecha de registro: 17 de noviembre de 2026, zona America/Lima. Fuente: instrucciones del integrante en este chat. No se asigna un nombre real que no haya sido confirmado.

## conservar-v1

El integrante indicó: «nos quedamos con el V1» y «esta es mi decisión como un miembro del equipo». Se registra su decisión de conservar esta propuesta con autoría del integrante solicitante. Esta fuente acredita una decisión del integrante; no documenta una votación del equipo.

## modelos-explicables

El integrante solicitó comparar Random Forest, árboles, regresión, redes neuronales y convolucionales. Posteriormente explicó la razón del método: «son los que el equipo puede explicar». Esa es la justificación de la selección de familias; no se atribuye a los papers que todos esos algoritmos sean superiores o que todos aparezcan en ellos.

## imputacion-52

El integrante pidió aplicar la técnica estacional a «52% de los datos faltantes», conservar una versión sin imputación y ensayar una tercera propuesta. La interpretación ejecutada y comunicada fue rellenar el 52 % de celdas originalmente nulas, no el 52 % del dataset completo. Se conserva esa definición, con su redondeo a celdas.

## modificaciones-minimas

El integrante indicó: «modifica lo mínimo posible el entrenamiento que ya hay». Por eso los experimentos se añadieron como scripts independientes y se mantuvieron referencias comparables. Al organizar la propuesta se adaptó la entrada y la ubicación, sin cambiar la arquitectura, hiperparámetros ni resultados del V1.

## excluir-dos-etapas

El integrante rechazó la propuesta de clasificar primero cero/positivo y después estimar casos positivos. No se implementó ese modelo. El respaldo por falta de clima de V1 decide según disponibilidad de entradas; no clasifica cero/positivo ni modifica las dos tareas originales.

## registro-y-retiro

El integrante pidió borrar los demás pipelines y conservar sus casos probados como matrices MD/CSV. Antes de retirar implementaciones se copiaron métricas, búsquedas, métodos y metadatos con comprobación de hashes. La referencia histórica está en `../../registro_de_pruebas/README.md`, fuera del paquete activo. V2 y la ingeniería posterior no son entradas ejecutables del V1.

## revision-final-mlp

22 de noviembre de 2026, preparación del entregable final. Se aprobó corregir la falta de convergencia de la MLP (41 de 41 ajustes alcanzaban el límite de 150 iteraciones) activando early stopping, sin entrenar modelos nuevos. También se añadieron la tabla comparativa y los gráficos de `results/finales/` (`codigo/resultados_finales.py`) y el análisis de `ANALISIS_RESULTADOS.md`.
