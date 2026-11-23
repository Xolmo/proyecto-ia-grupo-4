# Réplica del paper 2 en Cajamarca: análisis

**Paper:** Donizette, Rocco y de Queiroz (2025), *Predicting leishmaniasis outbreaks in Brazil using machine learning models based on disease surveillance and meteorological data*, Operations Research for Health Care 44, 100453 ([PDF](../papers/paper_integrante2.pdf), pp. 8–10).
**Código:** [`src/modelo_paper2.py`](../src/modelo_paper2.py) · **Comparación:** [`src/comparacion_final.py`](../src/comparacion_final.py) · **Notebook:** [`notebooks/06_paper_integrante2_redes.ipynb`](../notebooks/06_paper_integrante2_redes.ipynb) · **Tablas:** [`results/comparacion_final/`](../results/comparacion_final/tabla_comparativa_final.md)

## 1. Qué se replicó

El paper predice los casos mensuales de leishmaniasis de seis ciudades de Brasil con cuatro redes y un SVR. Aquí se implementaron las tres redes feedforward:

| Modelo | Arquitectura |
|---|---|
| **SFNN1** | 1 capa oculta densa (4–8 neuronas), Dropout 0,2, salida lineal de 1 neurona |
| **SFNN2** | igual, con 12–28 neuronas |
| **DFNN** | 2 capas ocultas densas, cada una con Dropout 0,2, salida lineal |

Se conservan del paper: MinMaxScaler (0–1), pérdida MSE, cuadrícula de neuronas, activación (ReLU/Tanh), optimizador (Adam/RMSprop), 100 o 300 épocas y RMSE como criterio de elección.

## 2. En qué se diferencia (y por qué)

| Paper | Aquí | Motivo |
|---|---|---|
| `GridSearchCV` de 5 folds | Elección con la validación temporal del proyecto | Los folds aleatorios mezclan futuro y pasado |
| Cuadrícula completa de neuronas, 16–24 h por ciudad | SFNN2 = 12/20/28, DFNN = 4/12/28; lotes de 256 | Que corra en minutos |
| 127 combinaciones de variables | 4 conjuntos: `casos3`, `casos7`, y ambos con clima | Costo; ver §3 |
| Clima del mismo mes (insolación, humedad) | Rezagos 1–3 de lluvia, tmax y tmin | SENAMHI no trae insolación ni humedad; el clima del mes objetivo no se conoce al predecir |
| Keras / TensorFlow | PyTorch | Ya es dependencia del proyecto (CNN); evita sumar TensorFlow |
| LSTM y SVR | No implementados | Fuera del temario del curso (decisión del equipo); la CNN temporal del equipo cubre una red secuencial |
| 6 ciudades de Brasil, regresión | 57 distritos de Cajamarca, regresión + alerta | La alerta se evalúa usando los casos predichos como puntaje |

Conjuntos de variables: `casos3` = `lag1`–`lag3` (como el paper); `casos7` = las 7 variables de casos del equipo (`lag1`–`lag3`, `lag12`, `media3`, `mes_sin`, `mes_cos`), para comparar con los demás modelos con las mismas entradas; `_clima` agrega 9 rezagos de lluvia, tmax y tmin.

## 3. Protocolos

| | Oficial | Histórico |
|---|---|---|
| Ajuste / validación / prueba | ≤2015 / 2016–2018 / 2019–2024 | ≤2009 / 2010–2011 / 2012–2014 |
| Filas de prueba, alertas | 4 104, 120 | 1 656, 23 |
| Clima en prueba | No observado (SENAMHI termina en oct. 2015) | Observado (72 % de las filas) |

Con la configuración elegida con la semilla 42 se reentrena con 5 semillas para medir la variación (`RMSE_media_semillas`, `RMSE_sd_semillas`). Los modelos del equipo se entrenaron una sola vez.

## 4. Resultados (protocolo oficial, prueba 2019–2024)

| Modelo | RMSE valid. | RMSE prueba | media ± sd (5 sem.) | PR-AUC | ROC-AUC | EER | Accuracy | Recall |
|---|---|---|---|---|---|---|---|---|
| **DFNN `casos7` (paper 2)** | **1,115** | **0,916** | 0,922 ± 0,016 | **0,158** | **0,796** | **0,259** | 0,826 | 0,583 |
| SFNN2 `casos7` (paper 2) | 1,116 | 0,960 | 0,932 ± 0,020 | 0,138 | 0,794 | 0,267 | 0,814 | 0,558 |
| SFNN1 `casos7` (paper 2) | 1,121 | 0,971 | 0,938 ± 0,022 | 0,137 | 0,771 | 0,326 | 0,819 | 0,550 |
| MLP (equipo) | 1,131 | 0,922 | — | 0,121 | 0,752 | 0,301 | 0,816 | 0,542 |
| Random Forest (equipo) | 1,127 | 0,926 | — | 0,120 | 0,747 | 0,307 | 0,802 | 0,550 |
| CNN temporal (equipo) | 1,128 | 0,938 | — | 0,112 | 0,745 | 0,309 | 0,853 | 0,417 |
| Ridge / logística (equipo) | 1,129 | 0,940 | — | 0,107 | 0,690 | 0,408 | 0,874 | 0,392 |
| Árbol de decisión (equipo) | 1,153 | 0,941 | — | 0,093 | 0,702 | 0,375 | 0,889 | 0,358 |
| SFNN1 `casos3` (como el paper) | 1,147 | 0,976 | 0,955 ± 0,014 | 0,125 | 0,727 | 0,316 | 0,779 | 0,600 |
| SFNN2 `casos3` | 1,148 | 0,983 | 0,957 ± 0,018 | 0,124 | 0,726 | 0,321 | 0,803 | 0,542 |
| DFNN `casos3` | 1,145 | 1,002 | 0,956 ± 0,026 | 0,124 | 0,727 | 0,327 | 0,659 | 0,700 |
| Regla «siempre cero» | — | 1,100 | — | — | — | — | — | — |
| Regla «último mes» | — | 1,187 | — | — | — | — | — | — |

(Las filas «+ clima» del oficial son idénticas a «solo casos»: no hay clima observado en 2019–2024 y el modelo usa el respaldo sin clima.)

## 5. Hallazgos

1. **Mejor por validación: la DFNN `casos7`**, de las redes del paper 2. Pero **no se puede afirmar que sea mejor** que la MLP o el Random Forest: su RMSE medio entre semillas (0,922 ± 0,016) es igual al de la MLP (0,922) y a 0,004 del Random Forest (0,926), diferencias menores que la variación entre semillas. La semilla 42 (0,916) fue favorable.
2. **En la alerta la DFNN queda por delante** (PR-AUC 0,158 vs 0,121; EER 0,259 vs 0,301). Es una sola semilla, así que es una señal más débil de lo que parece.
3. **Las entradas pesan más que la arquitectura.** Con solo `lag1`–`lag3` (como el paper) las redes dan ≈0,955 de media; con las 7 variables de casos bajan a ≈0,92–0,94. Con 3 variables la red más simple (SFNN1) iguala a las demás; con 7, la más profunda (DFNN) es la mejor.
4. **El clima no ayuda, ni siquiera observado.** En el histórico (clima observado), la media de 5 semillas empeora en las seis combinaciones, entre +0,013 y +0,025 de RMSE. Coincide con los cinco modelos del equipo (empeoran entre +0,004 y +0,023). En el oficial no se puede medir.
5. **Todos superan a las reglas simples en RMSE** (0,92–1,00 vs 1,10 de «siempre cero»), pero «siempre cero» sigue ganando en MAE (0,432), porque el 75 % de las filas tiene 0 casos.
6. **Histórico:** las diferencias entre modelos son menores que la variación entre semillas (RMSE 0,998–1,03). El mejor por validación fue el Random Forest del equipo (1,329; prueba 1,016). Con solo 23 alertas, las métricas de alerta (PR-AUC 0,02–0,04) no permiten comparar.

### Accuracy y EER

El *accuracy* se calculó para completar la comparación, pero **no sirve para elegir**: solo el 2,9 % de las filas de prueba son alertas, así que «nunca alertar» obtiene 97,1 %, más que cualquier modelo. Por eso se usan PR-AUC y EER. **EER** es la tasa de error en el punto donde las falsas alarmas igualan a los brotes no detectados; menor es mejor.

## 6. Limitaciones

- **El clima no se puede evaluar en 2019–2024:** SENAMHI termina en octubre de 2015. La conclusión sobre el clima viene del protocolo histórico (2012–2014, 46 distritos).
- **Una sola partición y una semilla en los modelos del equipo.** Las diferencias menores a ~0,02 de RMSE no son interpretables.
- **Réplica parcial:** sin LSTM ni SVR, con un subconjunto de variables y una cuadrícula reducida; no reproduce los valores del paper (otra enfermedad, país, escala y variables).
- **La alerta del paper 2 es derivada** de la regresión (casos predichos como puntaje), no un clasificador entrenado para ello, y su umbral se elige en validación.
- **Los casos están subestimados en las semanas recientes** y la ubicación es el probable sitio de infección (ver [README](../README.md)).

## 7. Reproducir

```bash
pip install -r requirements.txt
python src/modelo_paper2.py                        # oficial -> results/paper2/ (~50 min)
python src/modelo_paper2.py --protocolo historico  # histórico -> results/paper2_historico/ (~35 min)
python src/comparacion_final.py                    # tablas y gráficos -> results/comparacion_final/
python -m unittest discover -s src/tests -v
```

Semilla 42 para la selección. Los dos comandos de entrenamiento son independientes y pueden correr en paralelo.
