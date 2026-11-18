# Versión 1: imputación estacional del 52 % de faltantes

## Definición

Se imputan exactamente `round(0.52 * N)` celdas, donde `N` es la cantidad inicial de nulos entre `pp_total_mm`, `tmax_media` y `tmin_media`. El presupuesto es global por protocolo, con redondeo a la celda más próxima; no significa 52 % de todas las filas ni 52 % por columna.

Para cada celda se busca el mismo mes 12, 24, 36 o 48 meses antes, en ese orden. Se usa el primer antecedente disponible, sin mirar fechas futuras. Se permite copiar un antecedente ya imputado para materializar la variante solicitada, y se marca explícitamente; esta propagación puede repetir artificialmente ciclos anuales.

Se aprovechan también registros originales de estaciones anteriores al inicio del panel epidemiológico, si están dentro de esos cuatro años de búsqueda. Esto no añade casos ni usa datos futuros.

Las celdas se recorren por fecha, ubigeo y variable en orden determinista. Al alcanzar el presupuesto se conservan los nulos restantes. Si no hay suficientes antecedentes para cumplirlo, el programa falla explícitamente: no inventa otra técnica para alcanzar el porcentaje.

Se rechazan candidatos con lluvia negativa o temperaturas incompatibles con la contraparte disponible (`tmin > tmax`). En ese caso se intenta otro año anterior; la celda solo se cuenta si el valor pasa el control. Las observaciones originales nunca se cambian para acomodar una imputación.

## Archivos

- `data/processed/experimento_clima/historico/estacional_52.csv`.
- `data/processed/experimento_clima/oficial/estacional_52.csv`.
- Auditoría por celda en `estacional_52_auditoria.csv`, en cada carpeta.

Las columnas `*_imputado`, `*_fuente_fecha` y `*_fuente_imputada` permiten distinguir observaciones y estimaciones. La auditoría registra también valor y número de fuentes. Lluvia media diaria y temperatura media se derivan de las variables primarias; sus nulos no entran nuevamente en el denominador. Los rezagos se recalculan después de imputar.

## Uso y límites

Los nulos restantes no se rellenan durante el entrenamiento: activan la estrategia de respaldo descrita en [la guía del experimento](EXPERIMENTO_CLIMA.md). Los objetivos de alerta se recalculan por partición; no se reutilizan umbrales ajustados con el futuro.

Inspiración: Donizette et al., `papers/paper_integrante 2.pdf`, página 8, tabla 5 y preprocesamiento. La variante toma su búsqueda estacional; el presupuesto exacto del 52 % y la propagación explícitamente auditada son decisiones de este experimento, no un umbral de seguridad validado por el artículo. Se omite el relleno con valores posteriores que ese estudio usó para faltantes iniciales.
