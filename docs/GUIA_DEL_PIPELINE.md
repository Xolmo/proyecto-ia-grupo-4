# Pipeline V1 (propuesta del 17-11)

**Autoría de la propuesta conservada:** integrante solicitante. **Decisión registrada:** 17 de noviembre de 2026. Se conserva V1 por su decisión como miembro del equipo, con la comparación equivalente de 70 distritos como resultado principal. La decisión y su motivo están documentados en [D01](DECISIONES_Y_FUENTES.md#d01) y [D25](DECISIONES_Y_FUENTES.md#d25).

## Dónde está cada parte

```text
src/
├── ejecutar.py                 # entrada única (verificar, datos, experimento, 70 distritos)
├── preprocessing.py            # limpieza de casos y panel distrito × mes
├── integracion_clima.py        # clima SENAMHI, coordenadas y asignación de estaciones
├── models.py                   # los cinco modelos
├── evaluation.py               # etiqueta de alerta, umbral y métricas
├── experimento_clima.py        # imputación, entrenamiento y selección por validación
├── comparacion_70_distritos.py # árbol original vs Random Forest en 70 distritos
├── reporte_experimento_clima.py, verificar_experimento_clima.py
├── resultados_finales.py       # metrics.csv, plots/ y model_comparison.txt
└── tests/                      # pruebas de integridad
data/raw/, data/processed/      # entradas originales y procesadas
results/                        # métricas, predicciones, gráficos y resumen
docs/                           # esta guía, decisiones, análisis
MANIFIESTO.json                 # hashes para verificar la copia
```

Todo el código resuelve las rutas respecto de la raíz del repositorio. Justificación del empaquetado: [D26](DECISIONES_Y_FUENTES.md#d26).

## Flujo y explicación de cada parte

### 1. Cargar y limpiar los casos

`clean_cases` lee CDC-MINSA, normaliza identificadores y descriptores y aproxima fechas a partir de semanas. No se recortan brotes ni se eliminan automáticamente registros con atributos repetidos porque pueden representar personas distintas. La imputación descriptiva de edad no interviene en las entradas predictoras. Motivo y fuentes: [D02–D03](DECISIONES_Y_FUENTES.md#d03).

### 2. Construir distrito × mes

Se filtra Cajamarca, se seleccionan distritos con al menos 50 casos según el alcance correspondiente y se completa el calendario mensual con ceros de notificación. Las versiones comparadas utilizan las mismas filas dentro de su protocolo. El alcance original de 70 y el experimental de 46/57 no se mezclan. Motivo y limitaciones: [D02](DECISIONES_Y_FUENTES.md#d02), [D04](DECISIONES_Y_FUENTES.md#d04).

### 3. Localizar los focos de casos

Se asocian localidades con centros poblados del INEI mediante nombres normalizados y coincidencia exacta o aproximada; se usa la capital como respaldo geográfico. Esto busca representar la probable zona de infección y no solo el centro administrativo. Motivo del cruce y de sus parámetros: [D05](DECISIONES_Y_FUENTES.md#d05).

### 4. Asignar estaciones de clima

Se calcula cercanía Haversine por localidad y se elige la estación que concentra más casos históricos del distrito, separando lluvia y temperatura. Es una aproximación basada en los datos disponibles; no una interpolación validada ni un control de altitud. La explicación de por qué se hizo ese cruce está en [D06](DECISIONES_Y_FUENTES.md#d06).

### 5. Depurar y agregar el clima

Se invalidan códigos de faltantes e inconsistencias; un mes exige al menos 80 % de días válidos por variable. Se promedian temperaturas y se estima lluvia mensual a partir de media diaria por días. No hay estaciones que cubran el periodo oficial de prueba. Criterios y razón de cada uno: [D07](DECISIONES_Y_FUENTES.md#d07).

### 6. Unir casos y clima

Se une por distrito (`ubigeo`) y mes (`fecha_mes`). Los meses sin clima permanecen presentes para no cambiar el conjunto evaluado. Se distinguen variables primarias y derivadas, evitando rellenarlas de forma contradictoria. Motivo de las claves y de conservar nulos: [D08](DECISIONES_Y_FUENTES.md#d08).

### 7. Aplicar la hipótesis de imputación seleccionada

La versión del 52 % rellena ese porcentaje de celdas originalmente nulas entre lluvia total/tmax/tmin. Busca el mismo mes 12/24/36/48 meses antes, permite propagación auditada y rechaza valores físicamente inconsistentes. No utiliza valores futuros. El 52 % no es un límite seguro demostrado: origen y alcance en [D09](DECISIONES_Y_FUENTES.md#d09), procedimiento y fuentes en [D10](DECISIONES_Y_FUENTES.md#d10).

Los controles sin imputación, huecos cortos y solo casos siguen disponibles dentro de V1 para reproducir el método que sustentó la decisión. Su propósito está en [D11](DECISIONES_Y_FUENTES.md#d11); no son los pipelines V2 o F0–F7 retirados.

### 8. Generar variables predictoras

Casos: `lag1`, `lag2`, `lag3`, `lag12`, `media3`, `mes_sin`, `mes_cos`. Clima: lags 1/2/3 de lluvia total, tmax y tmin. No se usa clima contemporáneo del objetivo. Los cálculos se hacen por distrito. Fuentes y razones: [D12](DECISIONES_Y_FUENTES.md#d12), [D13](DECISIONES_Y_FUENTES.md#d13).

### 9. Definir objetivos y separar fechas

La regresión estima casos mensuales y la clasificación detecta meses por encima de media distrital más dos desviaciones de entrenamiento. Se respetan fechas para ajustar, validar y evaluar. El protocolo histórico prueba clima observado; el oficial conserva 2019–2024. La fórmula es provisional. Motivos: [D14](DECISIONES_Y_FUENTES.md#d14), [D15](DECISIONES_Y_FUENTES.md#d15).

### 10. Comparar modelos que el equipo puede explicar

Ridge/logística, árbol, Random Forest, MLP y CNN temporal permiten comparar referencias simples, reglas, ensambles y redes. La justificación de qué familias se eligieron proviene del integrante: son las que el equipo puede explicar; no de que todos los papers utilizaran todas esas técnicas. Fuentes: [D16](DECISIONES_Y_FUENTES.md#d16), [arquitecturas e hiperparámetros D17](DECISIONES_Y_FUENTES.md#d17), [CNN D18](DECISIONES_Y_FUENTES.md#d18).

### 11. Entrenar y seleccionar sin elegir por prueba

Los escaladores se ajustan con entrenamiento, se trata el desbalance con pesos y se comparan dos configuraciones por familia. RMSE selecciona regresión y average precision selecciona clasificación. El corte de puntaje se elige con F1 de validación. Motivos y límites: [D19](DECISIONES_Y_FUENTES.md#d19), [D21](DECISIONES_Y_FUENTES.md#d21).

### 12. Predecir cuando falta clima

Se utiliza la rama climática solo cuando sus entradas están completas y existe suficiente entrenamiento; de lo contrario se usa el modelo de casos de la misma familia. La disponibilidad decide el respaldo, no una clasificación intermedia cero/positivo. Por qué se hace y qué modelo no se implementó: [D20](DECISIONES_Y_FUENTES.md#d20).

### 13. Evaluar y conservar trazabilidad

Se guardan RMSE/MAE/MASE, métricas de alerta, predicciones, cobertura y auditoría de imputaciones. Los rezagos observados durante prueba permiten evaluar un paso temporal. No se afirma pronosticar seis años desde 2018 sin actualizar entradas. Motivos: [D22](DECISIONES_Y_FUENTES.md#d22).

### 14. Conservar V1 y registrar alternativas

V2 y los pipelines posteriores se probaron para contrastar hipótesis explicables, no porque su superioridad estuviera garantizada. Sus implementaciones se retiran y sus métricas, métodos, diagnósticos y matrices se conservan en [registro de pruebas](../registro_de_pruebas/README.md). Por qué se ensayaron, por qué se retiran y quién decidió: [D24](DECISIONES_Y_FUENTES.md#d24), [D01](DECISIONES_Y_FUENTES.md#d01).

## Resultado principal: comparación equivalente de 70 distritos

| Modelo | RMSE ↓ | MAE ↓ | Filas de prueba |
|---|---:|---:|---:|
| Árbol original, solo casos | 0,907046 | 0,576065 | 5 040 |
| Random Forest, solo casos | 0,891514 | 0,564662 | 5 040 |
| Random Forest, imputación estacional del 52 % | 0,890787 | 0,565816 | 5 040 |

La reducción de RMSE atribuible al contraste de imputación es aproximadamente 0,08 %; MAE empeora ligeramente. Se conserva V1 por decisión del integrante, sin presentar ese cambio como ventaja sólida general. Evidencia y restricciones de esta comparación: [D23](DECISIONES_Y_FUENTES.md#d23), [D25](DECISIONES_Y_FUENTES.md#d25), [CSV completo](../results/comparacion_70_distritos/metricas.csv).

La comparación de los cinco modelos en las dos tareas, con sus gráficos, está en [results/tabla_comparativa.md](../results/tabla_comparativa.md) y se analiza en [ANALISIS_RESULTADOS.md](ANALISIS_RESULTADOS.md).

El modo de 70 distritos compara **regresión**. Las métricas de alerta y las otras familias del experimento se preservan bajo `results/experimento_clima/`, con sus cohortes históricas de 46 y oficiales de 57 distritos. No deben atribuirse a los 70.

## Ejecutar desde la raíz del repositorio

Con Python 3.12 y dependencias instaladas:

```powershell
python -m pip install -r requirements-lock.txt
python src/ejecutar.py --verificar
python -m unittest discover -s src/tests -v

# Reproduce árbol original + RF casos + RF con 52 %, sobre 70 distritos.
python src/ejecutar.py --modo equipo70

# Regenera la integración auxiliar desde las entradas originales incluidas.
python src/ejecutar.py --modo datos

# Reproduce el experimento completo V1 de cinco familias y dos tareas.
python src/ejecutar.py --modo experimento-v1 --reutilizar-datos

# Genera results/metrics.csv, results/plots/ y results/model_comparison.txt (requiere el paso anterior).
python src/resultados_finales.py
```

`ejecutar.py` resuelve las entradas respecto de la raíz, incluso si se invoca desde otro directorio. Cada ejecución escribe en `results/`.

`--verificar` compara la copia conservada contra `MANIFIESTO.json`; una modificación legítima posterior puede invalidar el hash y requiere registrar una nueva revisión. Dependencias, prueba y semilla: [D26](DECISIONES_Y_FUENTES.md#d26).

## Límites que permanecen en V1

- El clima termina como máximo en octubre de 2015; los valores posteriores imputados no son observaciones nuevas ([D07](DECISIONES_Y_FUENTES.md#d07), [D10](DECISIONES_Y_FUENTES.md#d10)).
- El alcance original de 70 y su asignación geográfica conserva limitaciones temporales heredadas; reproducir un resultado no elimina esas limitaciones ([D04](DECISIONES_Y_FUENTES.md#d04), [D23](DECISIONES_Y_FUENTES.md#d23)).
- La MLP usa early stopping desde la revisión final; antes alcanzaba el límite de 150 iteraciones sin converger. La arquitectura retirada de V2 no se incorpora ([D17](DECISIONES_Y_FUENTES.md#d17)).
- La alerta es experimental, el puntaje no está calibrado y la prueba ya fue revisada: no es validación clínica ni externa nueva ([D14](DECISIONES_Y_FUENTES.md#d14), [D19](DECISIONES_Y_FUENTES.md#d19), [D25](DECISIONES_Y_FUENTES.md#d25)).
