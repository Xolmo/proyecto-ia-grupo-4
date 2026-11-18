# Leishmaniosis cutánea en Cajamarca — Grupo 4

La propuesta conservada es **V1**, por decisión comunicada por el integrante solicitante el 17 de noviembre de 2026. Código, entradas, resultados y fuentes están reunidos en [pipeline propuesta 17-11](pipeline%20propuesta%2017-11/GUIA_DEL_PIPELINE.md).

## Entradas principales

- [Guía del V1](pipeline%20propuesta%2017-11/GUIA_DEL_PIPELINE.md): pasos y ejecución.
- [Decisiones y fuentes](pipeline%20propuesta%2017-11/DECISIONES_Y_FUENTES.md): explicación y citas de arquitectura y método.
- [Entrada única](pipeline%20propuesta%2017-11/codigo/ejecutar.py): paquete autocontenido.
- [Registro de pruebas](registro_de_pruebas/README.md): 649 filas históricas, matrices, búsquedas y motivos de lo ensayado. Las implementaciones V2 y de ingeniería posterior se retiraron.

## Comparación equivalente: 70 distritos

| Modelo | RMSE | MAE |
|---|---:|---:|
| Árbol original, casos | 0,907046 | 0,576065 |
| Random Forest, casos | 0,891514 | 0,564662 |
| Random Forest, imputación estacional 52 % | 0,890787 | 0,565816 |

Prueba 2019–2024, 5 040 filas. La imputación mejora RMSE aproximadamente 0,08 % y empeora ligeramente MAE; conservar V1 no demuestra superioridad general. [Fuente y límites](pipeline%20propuesta%2017-11/DECISIONES_Y_FUENTES.md#d23).

## Material previo del equipo

Se conservan `data/raw/`, `papers/`, `notebooks/` y las utilidades originales `src/preprocessing.py`/`src/integracion_clima.py` como materiales y herramientas de datos del equipo. La propuesta activa se ejecuta desde la carpeta indicada arriba. Las métricas previas de líneas base y árbol permanecen como referencias.

Integrantes del proyecto: Dhanny Sebastian Ramirez Rivera, Sergio Estéfano Huamán Noriega, Leonardo Jaime Flores Vera y Vilcapoma Madueño Franz Javier. La autoría de la propuesta V1 se registra como integrante solicitante, sin inferir cuál de esos nombres corresponde al usuario.
