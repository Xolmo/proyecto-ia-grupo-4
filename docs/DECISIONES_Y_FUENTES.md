# Decisiones de arquitectura y método: V1

Cada entrada explica **por qué**, identifica si su origen es bibliográfico, del equipo o de implementación y enlaza la evidencia. Un antecedente bibliográfico motiva una hipótesis; no convierte los valores elegidos localmente en parámetros demostrados por ese paper.

## Fuentes

- **P1:** [Adams et al., original](../papers/paper_integrante1.pdf#page=5), §2.1–2.2, PDF pp. 4–6: casos mensuales, meteorología rezagada, ventanas temporales y modelos separados de regresión/clasificación.
- **P2:** [Donizette, Rocco y de Queiroz, original](../papers/paper_integrante2.pdf#page=8), pp. 8–9: datos mensuales, relleno con el mismo mes 12/24/36/48 meses antes, combinaciones de entradas y comparación de redes/SVR.
- **P3:** [Araque y Paipa, tesis original](../papers/paper_integrante3.pdf#page=65), pp. 43–44, 65–69 y 82–84: efectos climáticos demorados, cobertura restringida, comparaciones temporales y límites de la generalización. Los resultados son de dengue; no se trasladan sus rezagos como leyes de leishmaniosis.
- **P4:** [El Alaoui et al., original](../papers/paper_integrante4.pdf#page=15), §4.6–4.7 y discusión: SARIMA usa casos históricos; la integración exógena queda como extensión. Correlación ambiental no equivale a mejora predictiva demostrada.
- **E:** [Decisiones del integrante](DECISION_EQUIPO.md), fuente primaria local de la selección de familias y de conservar V1.
- **DATOS:** [Descripción CDC](../data/raw/descripcion_dataset_leishmaniosis.md), [fuentes geográficas](../data/raw/external/FUENTES.md), [EDA epidemiológico](../notebooks/01_eda.ipynb), [EDA climático](../notebooks/01b_eda_clima.ipynb).
- **IMPLEMENTACIÓN:** [preprocesamiento](../src/preprocessing.py), [integración](../src/integracion_clima.py), [V1](../src/experimento_clima.py), [modelos](../src/models.py), [evaluación](../src/evaluation.py), [comparación de 70 distritos](../src/comparacion_70_distritos.py).

<a id="d01"></a>
## D01. Conservar V1 y atribuir la propuesta

**Decisión:** V1 queda como propuesta del integrante solicitante, con fecha 17-11. **Por qué:** es su decisión explícita después de revisar las comparaciones. Es una elección del equipo comunicada por ese integrante, no una superioridad estadística establecida. Fuente: [E, conservar V1](DECISION_EQUIPO.md#conservar-v1). Resultado relevante: [comparación equivalente](../results/comparacion_70_distritos/metricas.csv).

<a id="d02"></a>
## D02. Cajamarca y unidad distrito × mes

**Decisión:** usar Cajamarca, casos 2000–2024 y panel mensual desde 2001 tras exigir historia de rezagos. **Por qué:** allí hay estaciones asociables a los distritos y la agregación mensual reduce dispersión semanal, conservando resolución local. El distrito identifica series; V1 no incorpora su código como variable numérica. Fuente: [EDA del equipo](../notebooks/01_eda.ipynb), [cobertura SENAMHI](../notebooks/01b_eda_clima.ipynb), [P1, unidad mensual](../papers/paper_integrante1.pdf#page=5) y [P2, unión mensual](../papers/paper_integrante2.pdf#page=8). Implementación: `build_monthly_panel`.

<a id="d03"></a>
## D03. Limpieza, calendario y picos

**Decisión:** normalizar ubigeo a seis dígitos, edades y faltantes descriptivos; aproximar fecha desde semana, completar meses sin registros con cero, mantener duplicados potencialmente válidos y brotes. **Por qué:** ubigeos consistentes permiten unir fuentes; duplicados de atributos no identifican necesariamente la misma persona; recortar picos eliminaría el fenómeno de interés. Las fechas son aproximadas y cero significa ausencia de registros, no ausencia garantizada de enfermedad. Fuente y explicación: docstrings de [clean_cases y build_monthly_panel](../src/preprocessing.py), [descripción original](../data/raw/descripcion_dataset_leishmaniosis.md). La edad imputada no entra en los modelos.

<a id="d04"></a>
## D04. Mínimo de 50 casos y alcance de cada ejecución

**Decisión:** conservar el umbral local de 50 casos, separando los 70 distritos originales de los 46/57 del experimento. **Por qué:** evitar series extremadamente escasas y poder comparar las mismas filas. El valor 50 es una decisión heredada, no el umbral de P1, que usa otro criterio. En el experimento se selecciona antes del ajuste inicial; en la reproducción de 70 se conserva el alcance original, seleccionado con todo el historial. Esta última condición introduce información del periodo futuro en la elegibilidad y debe declararse. Fuente: [constante y filtro del equipo](../src/preprocessing.py), [build_datasets](../src/experimento_clima.py), [comparación 70](../src/comparacion_70_distritos.py), [P1, criterio distinto](../papers/paper_integrante1.pdf#page=5).

<a id="d05"></a>
## D05. Coordenadas de localidades y coincidencia de nombres

**Decisión:** normalizar nombres, buscar localidad exacta o parecida dentro del distrito; si no se encuentra, usar capital distrital. **Por qué:** la ubicación corresponde al probable lugar de infección y los nombres tienen errores; usar siempre la capital ignoraría focos alejados. El corte de similitud 0,85 es una heurística local para limitar coincidencias incorrectas, no un parámetro epidemiológico validado. Fuente: [procedencia INEI y estaciones](../data/raw/external/FUENTES.md), explicación de `coordenadas_localidades` y `CORTE_NOMBRE` en [integración](../src/integracion_clima.py).

<a id="d06"></a>
## D06. Estación próxima, Haversine y asignación distrital

**Decisión:** calcular distancia Haversine por localidad y elegir para cada distrito la estación que concentra más casos históricos de sus localidades. **Por qué:** priorizar la cercanía a las áreas con casos, en vez de una estación cercana solo a la capital; Haversine utiliza las coordenadas disponibles y no requiere un modelo de interpolación sin calibración. Se buscan estaciones de lluvia y temperatura separadamente porque no todas miden ambas. Fuente: [datos y límites espaciales del EDA](../notebooks/01b_eda_clima.ipynb) y explicación de `estacion_mas_cercana`/`asignar_distritos` en [código](../src/integracion_clima.py). Es una aproximación geográfica local, no una recomendación atribuida a los papers; no controla altura ni microclimas.

<a id="d07"></a>
## D07. Cobertura y calidad mensual del clima

**Decisión:** invalidar -99,9, lluvia negativa y pares térmicos inconsistentes; aceptar un mes por variable con al menos 80 % de días válidos; temperatura media mensual y lluvia estimada como media diaria × días del mes. **Por qué:** no tratar faltantes como observaciones ni comparar meses con cobertura muy desigual. El 80 % y la aptitud térmica de al menos 50 % de meses son controles locales, no normas SENAMHI ni garantías de exactitud. Con días faltantes, la lluvia mensual es una estimación, no una suma observada completa. Fuente: [leer_estacion, clima_mensual y resumen_estaciones](../src/integracion_clima.py); precedente de agregación coherente por variable en [P3, pp. 43–44](../papers/paper_integrante3.pdf#page=43).

<a id="d08"></a>
## D08. Qué se une y por qué se conservan nulos

**Decisión:** unir el panel y la estación por `ubigeo` y `fecha_mes`, conservar meses sin clima y no utilizar Cusco como predictor de Cajamarca. **Por qué:** la unión debe representar el mismo lugar y periodo; eliminar meses sin clima cambiaría selectivamente el conjunto de evaluación. Se preservan lluvia total, tmax y tmin como variables primarias; lluvia media y temperatura media se derivan para evitar imputar cantidades dependientes de forma contradictoria. Fuente: [dataset_final](../src/integracion_clima.py), [seasonal_fill](../src/experimento_clima.py), [fuentes](../data/raw/external/FUENTES.md), [P2, alineación de fuentes](../papers/paper_integrante2.pdf#page=8), [P3, restricción por cobertura](../papers/paper_integrante3.pdf#page=65).

<a id="d09"></a>
## D09. Presupuesto del 52 %

**Decisión:** rellenar `round(0.52 × nulos)` entre lluvia total/tmax/tmin, con presupuesto global por protocolo y recorrido cronológico determinista. **Por qué:** reproducir exactamente la hipótesis solicitada y auditarla sin contar dos veces variables derivadas. Origen de la elección: [E, petición de imputación](DECISION_EQUIPO.md#imputacion-52). Antecedente: [P2, tabla 5 y preprocesamiento](../papers/paper_integrante2.pdf#page=8), donde 52,08 % es humedad faltante de una ciudad; **no** es un límite seguro demostrado. Alcanzar 52 % no certifica validez de la reconstrucción.

<a id="d10"></a>
## D10. Imputación estacional, recursividad y trazabilidad

**Decisión:** buscar el mismo mes 12/24/36/48 meses antes, usar el primer antecedente válido y permitir fuentes imputadas marcándolas. Nunca usar un valor posterior. **Por qué:** preservar el ciclo anual y materializar el presupuesto solicitado cuando hay huecos largos; se audita la recursividad para mostrar cuánto se copia una estimación. Fuente de la búsqueda estacional: [P2, p. 8](../papers/paper_integrante2.pdf#page=8). El presupuesto exacto, la propagación explícita y las banderas son adaptación local de [seasonal_fill](../src/experimento_clima.py), no una seguridad probada por P2. Se rechazan candidatos físicamente incompatibles y no se modifican observaciones para acomodarlos. Evidencia: [auditoría de 70 distritos](../results/comparacion_70_distritos/auditoria_imputacion.csv).

<a id="d11"></a>
## D11. Controles sin imputación y huecos cortos

**Decisión de método:** comparar el 52 % con nulos intactos y con relleno de como máximo dos meses consecutivos usando medianas de antecedentes originales del mismo mes. **Por qué:** distinguir el efecto de rellenar datos de una mejora aparente por cambiar cobertura; una alternativa conservadora evita prolongar años completos con estimaciones. Origen: [E, tres versiones solicitadas](DECISION_EQUIPO.md#imputacion-52). La mediana, ventana de cuatro años y límite de dos meses son hipótesis locales documentadas en [seasonal_fill](../src/experimento_clima.py); P2 solo motiva la semejanza estacional. Estos controles pertenecen al método de V1, no son V2 ni los pipelines retirados.

<a id="d12"></a>
## D12. Siete variables de casos

**Decisión:** `lag1`, `lag2`, `lag3`, `lag12`, `media3`, `mes_sin`, `mes_cos`, calculadas por distrito; media3 desplaza antes de promediar. **Por qué:** captar persistencia, respuesta corta y ciclo anual con entradas disponibles antes del mes objetivo. Seno/coseno mantienen cercanía entre diciembre y enero. Fuente: [P1, casos rezagados](../papers/paper_integrante1.pdf#page=5), [P2, lags 1–3](../papers/paper_integrante2.pdf#page=8), [P3, codificación cíclica](../papers/paper_integrante3.pdf#page=67), [EDA anual](../notebooks/01_eda.ipynb). Los cinco conteos relativos y one-hot distrital de V2 no forman parte del V1 conservado.

<a id="d13"></a>
## D13. Nueve entradas climáticas rezagadas

**Decisión:** lags 1/2/3 de lluvia total, tmax y tmin; no usar clima del mes a predecir. **Por qué:** permitir efectos demorados con un bloque pequeño y conocido al emitir la predicción. P1 motiva retardos, pero su estudio retrospectivo también utiliza valores observados del mes objetivo; aquí se excluyen para evitar esa disponibilidad irreal. Fuente: [P1, §2.2 y advertencia de indexación](../papers/paper_integrante1.pdf#page=6), [P3, demoras y colinealidad](../papers/paper_integrante3.pdf#page=43). Elegir tres meses es una adaptación local, no el óptimo biológico demostrado para Cajamarca. V1 recalcula lags dentro del panel publicado; puede perder las primeras tres filas climáticas de cada distrito.

<a id="d14"></a>
## D14. Objetivos y definición de alerta

**Decisión:** regresión de casos y clasificación de `casos > media + 2 × desviación` distritales del entrenamiento. **Por qué:** estudiar exactitud de cantidades y detección de meses inusuales como tareas separadas. El umbral distrital se recalcula en cada corte del experimento, antes de evaluar el siguiente periodo. Fuente: doble tarea en [P1](../papers/paper_integrante1.pdf#page=5); fórmula y su carácter provisional en [add_alert](../src/preprocessing.py) y [labels](../src/evaluation.py). La fórmula es experimental y no se presenta como norma del MINSA ni como umbral del paper.

<a id="d15"></a>
## D15. Partición temporal y dos protocolos

**Decisión de método:** histórico: ajuste hasta 2009, validación 2010–2011, prueba 2012–2014; oficial: ajuste hasta 2015, validación 2016–2018, prueba 2019–2024, con refit hasta 2011/2018. **Por qué:** separar selección y evaluación respetando el tiempo y estudiar clima observado donde sí existe. Es una división local, motivada por cobertura, no las fechas exactas de los papers. Fuente: [P1, ventanas](../papers/paper_integrante1.pdf#page=5), [P3, coincidencia de cobertura](../papers/paper_integrante3.pdf#page=65), [EDA climático](../notebooks/01b_eda_clima.ipynb), [PROTOCOLS y run_experiments](../src/experimento_clima.py).

<a id="d16"></a>
## D16. Por qué comparar esos modelos

**Decisión de método:** probar Ridge/logística, árbol, Random Forest, MLP y CNN temporal. **Por qué:** son familias que el integrante identifica como explicables por el equipo; permiten contrastar linealidad, reglas, ensambles y redes bajo el mismo diseño. La razón curricular proviene de [E, modelos explicables](DECISION_EQUIPO.md#modelos-explicables). [P2](../papers/paper_integrante2.pdf#page=9) motiva comparar redes; [P4](../papers/paper_integrante4.pdf#page=15) motiva referencias simples; **no se afirma** que los cuatro papers entrenaron estas cinco familias ni que probarlas garantiza superioridad.

<a id="d17"></a>
## D17. Arquitecturas tabulares y búsqueda acotada

**Decisión:** dos configuraciones por familia: Ridge alpha 1/10; logística C 0,1/1; árbol profundidad 3/5 con 20 por hoja; RF 100 árboles, profundidad 8 y 5/20 por hoja; MLP 32 o 32/16 neuronas, alpha 0,01 y early stopping (10 % interno del ajuste, 20 épocas de paciencia, máximo 1 000 iteraciones). **Por qué:** comparación exploratoria acotada y reproducible, con capacidad limitada para reducir sobreajuste y mantener explicación manejable. Son elecciones locales de [make_model](../src/models.py), evaluadas en [validacion.csv](../results/experimento_clima/validacion.csv), bajo el criterio de [E](DECISION_EQUIPO.md#modelos-explicables). P2 motiva variar capacidad; no respalda específicamente estos números. Revisión del entregable final (22-11-2026): con el límite original de 150 iteraciones los 41 ajustes MLP terminaban sin converger; el equipo aprobó activar early stopping, con lo que los 42 ajustes convergen entre 24 y 187 iteraciones ([diagnóstico](../results/experimento_clima/diagnostico_ajuste.csv), [E, revisión final](DECISION_EQUIPO.md#revision-final-mlp)). Los demás modelos no cambian.

<a id="d18"></a>
## D18. Arquitectura de CNN

**Decisión:** Conv1D de kernel 2 sobre tres meses, 8/16 filtros, ReLU y flatten; concatenar contexto lag12/media3/seno/coseno; capa densa de 16 y salida de uno. Canal de casos y, cuando hay clima, tres canales meteorológicos; orden temporal pasado→presente. Adam lr 0,005, weight decay 0,001, 50 épocas, lotes 512. **Por qué:** aplicar convolución a una secuencia temporal real, con contexto anual explícito y una arquitectura pequeña que el equipo pueda explicar. Fuente del requisito de comparar convolucionales: [E](DECISION_EQUIPO.md#modelos-explicables). Arquitectura exacta y números: decisión local implementada en [TemporalCNN](../src/models.py), no arquitectura extraída de P2, que compara otras redes. Salida sigmoide para clasificación; lineal con recorte inferior a cero para conteos.

<a id="d19"></a>
## D19. Escalamiento y desbalance

**Decisión:** ajustar StandardScaler solo con entrenamiento en Ridge/logística, MLP y CNN; no escalar árboles. Clasificación con pesos balanceados o razón negativo/positivo en la pérdida CNN; no aplicar SMOTE. **Por qué:** estabilizar entradas numéricas y atender alertas raras sin fabricar mezclas de distritos/meses. Fuente metodológica: advertencias de [apply_smote y escaladores del equipo](../src/preprocessing.py); escala uniforme como motivación en [P2, p. 8](../papers/paper_integrante2.pdf#page=8). StandardScaler, pesos y no-SMOTE son decisiones de implementación de [make_model y TemporalCNN](../src/models.py), [fit_one](../src/experimento_clima.py), no el MinMax exacto de P2. El puntaje balanceado no es una probabilidad calibrada.

<a id="d20"></a>
## D20. Respaldo por ausencia de clima

**Decisión:** si faltan entradas climáticas, usar el modelo de casos de la misma familia; ajustar rama climática solo con filas completas, al menos 100 y ambas clases cuando corresponde. **Por qué:** emitir resultados sobre las mismas filas sin imputación oculta ni eliminación selectiva de meses difíciles. El mínimo 100 es un control de estabilidad local, no un umbral demostrado por un paper. Fuente: [train_predict](../src/experimento_clima.py) y ausencia de cobertura en [EDA](../notebooks/01b_eda_clima.ipynb). No es el modelo cero/positivo de dos etapas rechazado por [E](DECISION_EQUIPO.md#excluir-dos-etapas): la condición es disponibilidad de datos, no una predicción intermedia de casos.

<a id="d21"></a>
## D21. Selección de configuración y corte del clasificador

**Decisión:** seleccionar menor RMSE de validación para cantidades y mayor average precision para alertas; elegir corte de puntaje que maximiza F1 de validación. Reentrenar antes de la prueba y mantener el corte. **Por qué:** evitar seleccionar por resultados finales y distinguir ranking de decisiones binarias. El corte del clasificador (0–1) no es el umbral distrital de casos de D14. Fuente: [run_experiments](../src/experimento_clima.py) y [choose_threshold](../src/evaluation.py), comparaciones de [P2](../papers/paper_integrante2.pdf#page=8) y [P3](../papers/paper_integrante3.pdf#page=65). Estas métricas/criterios concretos son una adaptación del experimento. Los ganadores observados retrospectivamente se registran como descriptivos, no se presentan como selecciones independientes.

<a id="d22"></a>
## D22. Métricas, referencias y evaluación de un paso

**Decisión:** RMSE, MAE y MASE contra error estacional de entrenamiento; precisión, recall, F1, average precision y ROC-AUC; referencias último mes/año anterior/cero. **Por qué:** los ceros abundantes pueden favorecer métricas parciales; hace falta medir tanto cantidades como falsas alarmas y comparación trivial. Fuente: [P2, RMSE](../papers/paper_integrante2.pdf#page=8), [P3, comparaciones y métricas](../papers/paper_integrante3.pdf#page=65), [P4, referencias temporales](../papers/paper_integrante4.pdf#page=15), [metrics](../src/evaluation.py). Los lags en prueba usan meses anteriores observados: es evaluación de un paso, no pronóstico recursivo de seis años desde 2018.

<a id="d23"></a>
## D23. Comparación equivalente de 70 distritos

**Decisión final de presentación:** informar juntos árbol original, RF solo casos y RF con 52 %, sobre las mismas 5 040 filas de 2019–2024. **Por qué:** la comparación anterior entre 57 y 70 distritos no permitía aislar diferencias; se reproduce exactamente el RMSE del árbol original antes de aceptar el contraste. Fuente: comprobación en [comparacion_70_distritos.py](../src/comparacion_70_distritos.py), [referencia original](../results/arbol_metricas.csv) y [resultados equivalentes](../results/comparacion_70_distritos/metricas.csv). El dataset de 70 conserva asignaciones y elegibilidad del flujo original; la asignación espacial se basó hasta 2018, de modo que su uso durante validación 2016–2018 es una limitación heredada. No se presenta como una nueva validación estricta de asignación geográfica. La comparación de 70 es de regresión; las alertas de 57 distritos pertenecen al otro protocolo.

<a id="d24"></a>
## D24. Por qué se probaron y retiraron los otros módulos

**Decisión de método:** V2 probó relativos/distrito, ventanas 5/10 años, parada temporal MLP y cortes alternativos; la ingeniería posterior comparó anomalías, acumulados, no linealidad, interacciones, extremos diarios, selección y PCA. **Por qué:** comprobar si representaciones explicables y clima coherente mejoraban frente a casos solos, sin asumir que mayor complejidad ayuda. Motivaciones: retardos e interacciones de [P1](../papers/paper_integrante1.pdf#page=5), combinaciones de entradas de [P2](../papers/paper_integrante2.pdf#page=9), cobertura y rezagos de [P3](../papers/paper_integrante3.pdf#page=43), referencias simples de [P4](../papers/paper_integrante4.pdf#page=15), y [criterio explicable del integrante](DECISION_EQUIPO.md#modelos-explicables). Las recetas exactas de PCA, información mutua, cuadrados y extremos fueron hipótesis locales, no métodos ejecutados por todos los papers. Se retiraron sus implementaciones por [decisión del integrante](DECISION_EQUIPO.md#registro-y-retiro), preservando matrices y métodos en el [registro histórico](../registro_de_pruebas/README.md).

<a id="d25"></a>
## D25. Qué justifica conservar V1 y qué no

**Decisión:** conservar la propuesta sin afirmar que imputar logró una ventaja sólida. **Por qué:** sobre 70 distritos RF cambia RMSE 0,8915→0,8908, pero MAE 0,5647→0,5658; la elección pertenece al integrante y no demuestra superioridad general del clima. Fuente: [métricas](../results/comparacion_70_distritos/metricas.csv), [decisión de conservación](DECISION_EQUIPO.md#conservar-v1). El 52 % utiliza valores propagados, las estaciones no cubren 2019–2024 y las pruebas ya fueron exploradas: no hay nueva validación externa ni garantía de operación sanitaria.

<a id="d26"></a>
## D26. Reproducibilidad, empaquetado y retiro verificable

**Decisión:** código, entradas, fuentes, resultados y versiones de dependencias dentro de esta carpeta; semilla 42, CPU con dos hilos; registro de hashes y pruebas de integridad antes de eliminar alternativas. **Por qué:** evitar depender de scripts borrados y permitir comprobar qué se conservó y qué se ensayó. Fuente del objetivo: [E, modificaciones mínimas y retiro](DECISION_EQUIPO.md#modificaciones-minimas), [registro de retiro](DECISION_EQUIPO.md#registro-y-retiro). Implementación: [entrada única](../src/ejecutar.py), [dependencias exactas](../requirements-lock.txt), [metadatos V1](../results/experimento_clima/ejecucion.json), [metadatos 70](../results/comparacion_70_distritos/ejecucion.json). Los hashes detectan modificaciones posteriores, no demuestran por sí mismos calidad predictiva.
