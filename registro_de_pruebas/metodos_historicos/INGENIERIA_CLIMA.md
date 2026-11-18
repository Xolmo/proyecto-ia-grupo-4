# Ingeniería climática: estudio de ocho pipelines

Implementación independiente: `src/ingenieria_clima.py`. Preserva V1, V2, sus datasets y los notebooks. Se ensayan hipótesis compatibles con los datos disponibles; no se afirma haber agotado todas las representaciones posibles ni encontrado un óptimo global.

## Relectura de los originales y decisiones

| Fuente del repositorio | Qué se verificó | Aplicación aquí |
|---|---|---|
| [Adams et al.](../papers/paper_integrante%201.pdf), PDF pp. 4–6, §2.1–2.2 | Meteorología rezagada, interacciones climáticas/contextuales; ejercicio retrospectivo con clima del mes objetivo | Rezagos solo de meses anteriores; retardos ponderados e interacciones como hipótesis propias. Se excluye lag0 del objetivo |
| [Donizette et al.](../papers/paper_integrante%202.pdf), pp. 8–9 | Unión mensual, imputación del mismo mes de años previos, selección de subconjuntos de entradas y redes | Conservación de tres regímenes de datos V1 y comparación de representaciones compactas |
| [Araque y Paipa](../papers/paper_integrante%203.pdf), pp. 43–44, 65–69 | Agregación temporal, retardos distribuidos propuestos, selección de rezagos, evaluación restringida por cobertura | Ventanas de acumulación, bloques de retardos, selección supervisada y evaluación histórica con clima observado |
| [El Alaoui et al.](../papers/paper_integrante%204.pdf), §4.6–4.7 y discusión p. 15 | Asociaciones ambientales separadas de SARIMA, que no incorpora clima; sugiere futuras extensiones | Control sin clima y análisis de mejora real frente a contexto temporal/distrital |

Las distribuciones ponderadas, extremos diarios, anomalías, cuadrados y PCA son adaptaciones exploratorias propias, no reproducciones exactas de técnicas ejecutadas en los cuatro artículos. F3 no implementa un DLNM formal. No se copian directamente desfases de dengue como si fueran parámetros biológicos de leishmaniosis.

## Matriz de versiones

| ID | Representación climática |
|---|---|
| F0 | Control: casos previos, media3, mes cíclico, distrito y conteos relativos; sin clima |
| F1 | Anomalías de lluvia acumulada y temperaturas/amplitud medias de tres meses |
| F2 | Acumulados/medias de 1, 3 y 6 meses, contraste entre meses 1–3 y 4–6, medias ponderadas y amplitud |
| F3 | Medias de seis rezagos con pesos recientes y demorados, más términos cuadráticos |
| F4 | Anomalías trimestrales × casos recientes relativos, × seno/coseno del mes, y lluvia × temperatura |
| F5 | Fracciones de días lluviosos, lluvia > percentil95, calor > percentil90, frío < percentil10; máximo diario de lluvia y variabilidad térmica del mes anterior |
| F6 | Seis entradas climáticas/calidad seleccionadas mediante información mutua, ajustada solo con entrenamiento |
| F7 | Tres componentes principales del bloque multiescala, con escalamiento y PCA ajustados solo con entrenamiento |

Las anomalías mensuales se calculan contra observaciones del distrito y mes en entrenamiento, con referencia general de entrenamiento cuando falta la distrital. Se utiliza desviación con mínimo de una unidad y recorte de anomalías a ±20 para estabilidad numérica. No se recortan los casos objetivo.

Las representaciones climáticas añaden calidad de asignación (distancias y concentración de casos por estación) y, salvo F5, fracción de entradas rezagadas imputadas. F6 puede seleccionar estas columnas: la lista exacta se conserva, sin atribuirla automáticamente a efectos meteorológicos.

F5 aprende los percentiles de cada estación con todos sus días disponibles anteriores al corte; no utiliza el futuro ni presupone temperaturas biológicas universales. Exige 80 % de días válidos por variable para aceptar un mes. Los extremos diarios nunca se inventan a partir de la imputación mensual. El último mes disponible se desplaza un mes para predecir el siguiente.

No se inventan humedad, ONI, NDVI ni otras variables que no están presentes.

## Diseño de comparación

- Fuentes mensuales: `sin_imputar`, `estacional_52` y `huecos_cortos` de V1. El presupuesto de imputación no cambia.
- Modelos: regresión Ridge/logística, árbol de decisión, Random Forest, MLP y CNN temporal.
- Tareas: conteo de casos y clasificación de alerta, sin modelo en dos etapas.
- Histórico: validación 2010–2011 y prueba 2012–2014; 46 distritos.
- Oficial: validación 2016–2018 y prueba 2019–2024; 57 distritos.
- Todas las versiones predicen las mismas filas, incluso cuando no tienen clima; en tal caso usan el control F0 de su familia.
- Las ventanas, configuraciones y épocas de cada familia se fijan desde su control `solo_casos` de V2. Así se prueba ingeniería de variables sin volver a buscar hiperparámetros diferentes en cada representación. Esto no equivale a optimizar exhaustivamente cada red nueva.
- F0 de prueba reutiliza las predicciones verificadas de V2; las predicciones de validación se reconstruyen con esos mismos ajustes.
- La selección de seis columnas y PCA se ajustan de nuevo solo con entrenamiento en cada fase. Nunca se seleccionan columnas con la prueba.
- El umbral de probabilidad/puntaje maximiza F1 de validación para cada combinación. La definición distrital de alerta permanece idéntica a V1/V2.

Son 22 condiciones por protocolo (F0 una vez y F1–F7 por tres fuentes), cinco familias y dos tareas: **440 combinaciones**. La cobertura efectiva de clima se informa por combinación.

## Reproducción

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -u src/ingenieria_clima.py
.venv/Scripts/python.exe src/reporte_ingenieria_clima.py
```

Requiere los CSV V1, `data/processed/clima_diario.csv` de la integración auxiliar y los resultados V2. No altera esos archivos ni requiere datos externos nuevos.

## Salidas

En `results/ingenieria_clima/`: métricas completas, predicciones por fila, listado de features seleccionados, matrices de RMSE/MAE/PR-AUC/F1/recall, selección por validación y reporte Markdown. Las matrices conservan protocolo y fuente, evitando mezclar prueba histórica con oficial o seleccionar retrospectivamente la mejor imputación por modelo.

La prueba oficial no contiene nuevas mediciones meteorológicas; cualquier mejora allí puede involucrar datos propagados o el respaldo de casos. La prueba ya fue examinada anteriormente: los resultados son exploratorios y no constituyen una nueva validación externa ni evidencia causal de clima.
