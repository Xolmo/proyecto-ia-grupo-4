# V2: contexto distrital, clima coherente y redes con parada temporal

Extensión aditiva en `src/experimento_clima_v2.py`. No modifica el experimento V1, sus resultados, los tres datasets climáticos fuente ni los notebooks originales. No hay modelo en dos etapas.

## Qué variables se utilizan

- Se conservan `lag1`, `lag2`, `lag3`, `lag12`, `media3`, `mes_sin`, `mes_cos`.
- Se incorporan cinco conteos relativos: rezagos 1/2/3/12 y media3 divididos entre el umbral distrital del entrenamiento, con denominador mínimo de un caso para evitar división por cero o escalas extremas.
- El distrito se representa con columnas indicadoras (one-hot). El ubigeo no se interpreta como una magnitud numérica. El mes conserva su codificación cíclica.
- El clima se resume en lluvia total de los tres meses anteriores, temperatura máxima y mínima medias de esos meses y amplitud térmica media.
- Cada resumen climático se transforma en anomalía: diferencia respecto al promedio histórico de ese distrito y mes, dividida por su desviación histórica. El denominador tiene un mínimo de una unidad. Si faltan estadísticas de distrito/mes, se utilizan las distritales y luego las generales del entrenamiento.
- Las referencias climáticas se calculan usando solo observaciones de `sin_imputar` hasta el corte de entrenamiento. No se aprende una normal climática a partir de estimaciones repetidas.
- Se añade la proporción de los nueve insumos climáticos rezagados que fue imputada. Los meses incompletos no se rellenan ocultamente: activan el respaldo de casos.

Las nueve columnas de clima bruto no se vuelven a introducir como entradas del modelo. En CNN la convolución recorre tres meses de casos; distrito, variables relativas, estacionalidad y clima resumido entran en una rama de contexto denso.

## Qué significa entrenar con datos recientes

Se comparan todo el historial disponible, los últimos diez años y los últimos cinco años. El modelo sigue utilizando casos del mes anterior y el rezago anual en todas las alternativas. Son decisiones distintas: una elige filas para entrenar; otra construye predictores de cada fila.

Por ejemplo, el ajuste final oficial hasta diciembre de 2018 usa 2001–2018, 2009–2018 o 2014–2018. La ventana y los hiperparámetros se seleccionan exclusivamente en validación. El histórico y la prueba oficial mantienen los cortes de V1.

La definición de alerta real y las referencias distritales se mantienen fijas dentro de cada corte para todas las ventanas. Así no se compara un modelo de cinco años contra uno de diez usando diferentes etiquetas. Las estadísticas de referencia pueden resumir la historia anterior al comienzo de la ventana de ejemplos de entrenamiento.

## Dos umbrales diferentes

1. **Umbral de casos:** media más dos desviaciones del distrito en el conjunto de referencia anterior al corte. Define si un mes fue realmente una alerta experimental; no es una norma sanitaria.
2. **Corte del clasificador:** número entre cero y uno aplicado a su puntaje. Con corte 0,60, un puntaje 0,65 activa alerta y uno 0,55 no. Es un hiperparámetro de decisión, no un porcentaje de datos ni una tasa de aprendizaje. El puntaje no se interpreta como probabilidad calibrada, especialmente tras balancear clases.

Se informan dos políticas para el segundo umbral: maximizar F1 en validación o buscar precisión de al menos 20 % en validación, maximizando recall entre cortes que cumplen. El 20 % es un objetivo exploratorio, no un requisito clínico ni una garantía en prueba. Si no existe corte factible se informa `validation_feasible=False` y se utiliza un corte que no emite alertas. No cambia la definición de alerta real.

La desactivación por política imposible se representa con `threshold=inf`, un centinela que no se interpreta como probabilidad.

## Ajuste de MLP

Se prueban una capa de 32 y dos capas de 32/16. Se escalan entradas y, para regresión, el objetivo usando solo entrenamiento. Se permiten hasta 600 épocas, se revisa la pérdida de validación cada cinco y se detiene tras 50 épocas sin mejora suficiente. La regresión monitoriza MSE; clasificación, log-loss.

Se conserva el estado de mejor validación. Al reentrenar con ajuste más validación se utiliza exactamente ese número de épocas; no se consulta la prueba para detenerse. Se registra si se alcanza el máximo sin parada. Es parada temporal por generalización, no una garantía de óptimo numérico o global. CNN conserva 50 épocas; árboles, bosque y regresión conservan sus dos configuraciones de V1.

## Reproducción y salidas

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -u src/experimento_clima_v2.py
.venv/Scripts/python.exe src/reporte_clima_v2.py
```

Requiere los CSV de V1 en `data/processed/experimento_clima/{historico,oficial}/`. Se utiliza el entorno y las dependencias del experimento anterior. `--protocol historico` o `--protocol oficial` permite ejecutar solo una evaluación; sus resultados reemplazan la carpeta V2, no V1.

Se generan bajo `results/experimento_clima_v2/`:

- `metricas.csv`: resultados de las configuraciones seleccionadas y ambas políticas de umbral.
- `validacion.csv`: seis candidatos por combinación (tres ventanas por dos configuraciones).
- `convergencia_mlp.csv`: épocas ejecutadas, elegidas y ajustes que alcanzaron el límite.
- `predicciones.csv`: resultados fila a fila, sobre los mismos objetivos y filas que V1.
- `ejecucion.json`: configuración y advertencias de alcance.
- `RESULTADOS.md`: comparación V1/V2, resultados de todos los modelos y explicación de los cortes.
- `verificacion.json`: comprobación de que se conservaron exactamente las filas y etiquetas de V1.

En `data/processed/experimento_clima/v2/{historico,oficial}/` se exportan las variables de cada variante. **Estos CSV son una vista del diseño final aprendido hasta 2011/2018**, para auditoría y predicción de prueba; no deben usarse para validar 2010–2011/2016–2018 directamente. El script reconstruye el diseño con el corte inicial al validar, evitando usar estadísticas posteriores.

## Interpretación

La prueba ya fue examinada en V1. Por tanto, estas mejoras constituyen desarrollo exploratorio sobre datos conocidos; no son una validación externa nueva. El histórico permite estudiar clima observado; la prueba oficial no contiene nuevas mediciones climáticas. No se atribuye una mejora al clima si la rama climática nunca se utilizó.
