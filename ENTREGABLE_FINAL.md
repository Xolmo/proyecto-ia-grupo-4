# Entregable final (semana 15) — checklist

**Proyecto:** Predicción de leishmaniosis cutánea en Cajamarca · Grupo 4 · Inteligencia Artificial Aplicada (1INF62).

**Integrantes:** Dhanny Sebastian Ramirez Rivera, Sergio Estéfano Huamán Noriega, Leonardo Jaime Flores Vera y Franz Javier Vilcapoma Madueño.

| Requisito | Estado | Dónde |
|---|---|---|
| Código limpio, comentado y reproducible | ✅ | `src/`; se ejecuta sin errores con los comandos del README; pruebas en `src/tests/` |
| `requirements.txt` con todas las dependencias | ✅ | `requirements.txt` |
| Todos los modelos implementados (mínimo 2) | ✅ 5 modelos | `src/models.py`: lineal (Ridge / regresión logística), árbol de decisión, Random Forest, MLP y CNN temporal |
| Métricas y evaluación | ✅ | `src/evaluation.py` (RMSE, MAE, MASE, PR-AUC, F1, recall, precisión) |
| Tabla comparativa de métricas | ✅ | `results/metrics.csv` (también `results/tabla_comparativa.md`): 5 modelos del equipo, redes del paper 2 y SARIMA del paper 4 en las mismas filas de prueba |
| Gráficos | ✅ | `results/plots/` (`fig1`–`fig5` de resultados finales, más EDA, árbol y réplicas de los papers) |
| Resumen de resultados y análisis | ✅ | `results/model_comparison.txt`, con una sección por paper (el XGBoost del paper 1 se resume aparte, evaluado en 70 distritos) |
| Por qué un modelo funcionó mejor, limitaciones e insights | ✅ | `results/model_comparison.txt`, `results/analisis/ANALISIS_RESULTADOS.md` y `report.pdf` |
| Implementación y comparación en notebook | ✅ | `notebooks/02_modelos.ipynb` (ejecutado, con salidas) |
| Paper / reporte académico (3–5 páginas) | ✅ | `report.pdf`: 5 páginas en formato IEEE de la plantilla del curso, con resumen, declaración de uso de IA, contribuciones y referencias |
| Réplica del paper 1 (XGBoost, Adams et al.) | ✅ | `src/p1_adams.py`, `notebooks/05_paper_integrante1_xgboost.ipynb`, `results/p1_adams/`, `results/plots/p1_*.png` |
| Réplica del paper 2 (redes SFNN/DFNN) y comparación con el equipo | ✅ | `src/modelo_paper2.py`, `src/comparacion_final.py`, `notebooks/06_paper_integrante2_redes.ipynb`, `results/comparacion_final/`, `results/analisis/ANALISIS_PAPER2.md` |
| Réplica del paper 4 (SARIMA, El Alaoui et al.) | ✅ | `src/modelo_paper4.py`, `notebooks/07_paper_integrante4_sarima.ipynb`, `results/paper4/`, `results/paper4_historico/`, `results/analisis/ANALISIS_PAPER4.md`; incluido en `results/comparacion_final/` |
| Presentación (máximo 5 minutos) | ⏳ | `presentation.pdf` |
| Papers revisados (1 por integrante) | ✅ | `papers/paper_integrante1.pdf` a `paper_integrante4.pdf` |

## Resultado principal

Prueba 2019–2024, 57 distritos, 4 104 filas distrito × mes:

- **Modelo final:** la DFNN del paper 2 con las 7 variables de casos, que tiene el menor RMSE de validación de todos los modelos (1,115). Es el modelo que analiza `report.pdf`.
  - Casos: RMSE 0,916 (0,922 ± 0,016 con 5 semillas), un 17 % menor que la regla "siempre cero".
  - Alertas: PR-AUC 0,158, más de 5 veces la del azar (0,029). Detecta 70 de 120 brotes, con 664 falsas alarmas.
- **Mejor del equipo:** el Random Forest con solo casos (RMSE 0,926, PR-AUC 0,120), empatado con la MLP (0,922). Ambos quedan a menos de 0,01 de RMSE de la DFNN, así que en casos los tres son equivalentes. La DFNN es mejor en alertas.
- **Clima:** no mejoró de forma consistente, porque las observaciones terminan en 2015.
- **Otros papers:** el XGBoost del paper 1 empata con el Random Forest en 70 distritos, y el SARIMA del paper 4 queda por detrás.

## Cómo verificar

```bash
pip install -r requirements.txt
python -m unittest discover -s src/tests -v
python src/ejecutar.py --modo experimento-v1 --reutilizar-datos
python src/resultados_finales.py
```

Estos comandos regeneran `results/metrics.csv`, `results/plots/` y `results/model_comparison.txt` con los mismos valores (semilla 42).

## Cambios respecto al entregable parcial

- Se implementaron 4 modelos más, además del árbol de decisión de la semana 8, y se integró el clima.
- Los distritos y los umbrales de alerta se calculan solo con el tramo de ajuste, para no usar información del periodo de prueba.
- La MLP usa early stopping. Con el límite original de 150 iteraciones, ningún ajuste convergía.
- Se reorganizó el repositorio según la estructura indicada por el curso: se retiraron el registro histórico de pruebas, el manifiesto de hashes y la documentación que no usan los notebooks; el análisis extendido pasó a `results/analisis/` y todos los gráficos a `results/plots/`.
