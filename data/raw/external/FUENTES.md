# Fuentes de coordenadas

Las usa `src/integracion_clima.py` para encontrar la estación más cercana a cada localidad. Se consultaron el 11 de octubre de 2026.

| Archivo | Contenido | Fuente |
|---|---|---|
| `estaciones_senamhi.csv` | Coordenadas de las 34 estaciones de `data/clima/` | Mapa de estaciones de SENAMHI (https://www.senamhi.gob.pe/mapas/mapa-estaciones-2/). Querocotillo y Acomayo no figuran en el mapa; sus coordenadas vienen del Atlas Climático de SENAMHI (https://www.senamhi.gob.pe/pdf/anexos-atlas-climatico.pdf), redondeadas a 0,1° (unos 11 km). |
| `centros_poblados_cajamarca.csv` | 6 513 centros poblados de Cajamarca (INEI) con latitud y longitud | `ubigeo_ccpp.csv` de https://github.com/jmcastagnetto/ubigeo-peru-aumentado (licencia MIT), filtrado a Cajamarca |
| `distritos_cajamarca.csv` | 127 distritos de Cajamarca con las coordenadas de su capital | `ubigeo_distrito.csv` del mismo repositorio, filtrado a Cajamarca |

**Nota sobre Cilete.** La estación `qc00153203 - cilete` corresponde a la estación pluviométrica **Chilete** de SENAMHI (código 107058, provincia de Contumazá). La lluvia anual del archivo (198 mm) coincide con la normal que el Atlas Climático publica para Chilete (198 mm).
