# Registro histórico de pruebas — decisión 17-11

Se conserva V1 como propuesta del integrante solicitante. Las implementaciones de V2 y de ingeniería posterior se retiran por su instrucción; este registro mantiene la evidencia numérica y metodológica. La propuesta ejecutable está en [pipeline propuesta 17-11](../pipeline%20propuesta%2017-11/GUIA_DEL_PIPELINE.md).

## Qué se probó realmente

| Campaña | Filas de métricas conservadas | Qué representan | Estado |
|---|---:|---|---|
| V1 | 86 | 80 combinaciones de familia/tarea/fuente/protocolo y 6 referencias | Conservado |
| V2 | 120 | 80 combinaciones, con dos políticas de corte para las 40 de clasificación; búsqueda de 480 candidatos ventana/configuración | Implementación retirada |
| Ingeniería climática | 440 | 8 representaciones, tres fuentes para las climáticas, cinco familias, dos tareas y dos protocolos | Implementación retirada |
| Comparación equivalente de 70 distritos | 3 | Árbol original y RF con/sin imputación sobre las mismas 5 040 filas | Conservado como contraste V1 |

Son **649 filas de resultados**, no 649 modelos independientes ni 649 pruebas externas. Algunas filas son referencias o políticas distintas del mismo clasificador. KNN, LSTM, Transformer y otras opciones discutidas que no se ejecutaron no se registran como entrenadas.

## Matrices y trazabilidad

- [Registro completo](registro_completo.csv): conserva columnas originales y añade campaña, estado, modelo y claves de comparación.
- [Matriz RMSE](matriz_RMSE.csv), [MAE](matriz_MAE.csv), [MASE](matriz_MASE.csv).
- [Matriz PR-AUC](matriz_PR_AUC.csv), [F1](matriz_F1.csv), [recall](matriz_recall.csv), [precisión](matriz_precision.csv).
- [Cobertura climática efectiva](matriz_climate_fraction.csv).
- `campanas/`: copia exacta de métricas, búsquedas, features seleccionados, diagnósticos, metadatos, informes y verificaciones de cada campaña.
- `metodos_historicos/`: explicaciones previas de V1/V2/ingeniería y de las tres versiones de datos. Se conservan como documentos históricos; sus comandos y rutas antiguas no son instrucciones actuales.
- `material_original/`: líneas base y árbol del repositorio previo, como evidencia de referencia.
- [Integridad de copias](integridad_copias.json): ruta de origen, archivo archivado, tamaño y SHA-256, verificados antes de retirar archivos.
- [Implementaciones retiradas](implementaciones_retiradas.json): nombres y hashes de código retirado; no contiene sus implementaciones ejecutables.

Las matrices mantienen campaña, protocolo, tarea, versión, fuente y política de corte. No se promedian protocolos diferentes ni se oculta una política de umbral dentro de otra. La PR-AUC del registro corresponde a `average_precision_score`, como se implementó en las campañas.

## Motivo de probar y motivo de conservar

La selección de familias se basó en que el equipo puede explicarlas, según el integrante. Las variantes posteriores exploraron si clima resumido, interacciones, extremos diarios o menor dimensión añadían información sobre los casos. Su base bibliográfica, hipótesis locales y límites están explicados en [D16 y D24 del registro de decisiones](../pipeline%20propuesta%2017-11/DECISIONES_Y_FUENTES.md#d24).

La decisión final de conservar V1 se atribuye al integrante solicitante, no se presenta como unanimidad ni como demostración de superioridad en todas las métricas. Fuente primaria: [decisión comunicada](../pipeline%20propuesta%2017-11/fuentes/DECISION_EQUIPO.md#conservar-v1).

La comparación de 70 distritos reprodujo RMSE 0,907046 del árbol original, 0,891514 de RF solo casos y 0,890787 de RF con imputación. Las otras campañas de 46/57 distritos no son comparables directamente con esos números. Fuente: [CSV equivalente](campanas/comparacion_70_distritos/metricas.csv).

Los conjuntos de prueba ya fueron explorados; el registro no convierte esos resultados en nuevas validaciones independientes. Las implementaciones alternativas fueron eliminadas como archivos concretos verificados. Sus resultados completos, incluidas predicciones y datos derivados, se conservaron adicionalmente bajo `artefactos_historicos/`, sin código ejecutable de V2/ingeniería; las matrices de esta carpeta siguen siendo el índice principal.

La revisión automática rechazó una eliminación amplia de carpetas de documentación, pruebas y temporales por posible contenido no relacionado. Se resolvió reduciendo el retiro a implementaciones identificadas por nombre y hash, y archivando los directorios de resultados sin destruir contenido. [Archivos verificados retirados](retiro_archivos_verificados.json) y [movimientos al registro](movimientos_registro.json).
