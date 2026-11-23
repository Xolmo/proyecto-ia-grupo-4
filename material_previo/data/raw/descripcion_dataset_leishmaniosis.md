# Descripción del dataset: Vigilancia epidemiológica de leishmaniosis (Perú, 2000–2024)

## Origen

- **Nombre del archivo:** `datos_abiertos_vigilancia_leishmaniosis_2000_2024.csv`
- **Publicador:** Centro Nacional de Epidemiología, Prevención y Control de Enfermedades (CDC Perú), Ministerio de Salud (MINSA).
- **Portal:** Plataforma Nacional de Datos Abiertos, https://datosabiertos.gob.pe/dataset/vigilancia-epidemiológica-de-leishmaniosis
- **Fuente primaria:** Red Nacional de Epidemiología (RENACE), conformada por establecimientos de salud del MINSA, EsSalud y otros del sector que notifican casos semanalmente.
- **Permisos:** datos públicos y anonimizados (sin nombres ni documentos de identidad). No se requiere autorización adicional; basta citar la fuente.
- **Advertencia del publicador:** la información es provisional y sujeta a modificación. Las semanas más recientes pueden estar subestimadas por retraso en la notificación.

## Estructura

El archivo es un **registro individual por caso notificado** (no una serie agregada).

| Característica | Valor |
|---|---|
| Filas (casos) | 156 819 |
| Columnas | 14 |
| Periodo | 2000–2024 (semanas epidemiológicas 1–53) |
| Cobertura | 23 departamentos, 171 provincias, 1 033 distritos (1 100 ubigeos) |
| Tamaño | 18,3 MB |

| Columna | Tipo | Descripción |
|---|---|---|
| `departamento`, `provincia`, `distrito` | texto | División político-administrativa |
| `localidad`, `localcod` | texto | Centro poblado y su código |
| `enfermedad` | texto | Siempre `LEISHMANIASIS CUTANEA` |
| `ano`, `semana` | entero | Año y semana epidemiológica |
| `diagnostic` | texto | Código CIE-10; siempre `B55.1` |
| `diresa` | numérico | Dirección Regional de Salud (33 valores) |
| `ubigeo` | entero | Código INEI del distrito |
| `edad`, `tipo_edad` | entero, texto | Edad y su unidad (A = años, M = meses, D = días) |
| `sexo` | texto | M (64 %) o F (36 %) |

## Características relevantes para el modelado

- **Es leishmaniosis cutánea.** Todos los registros corresponden a `B55.1`. No hay forma visceral ni mucocutánea separada, lo que la distingue de la mayoría de la literatura internacional (leishmaniosis visceral).
- **Distribución geográfica concentrada.** Cusco, Madre de Dios, Cajamarca, San Martín y Piura reúnen el 48 % de los casos.
- **Serie relativamente estable.** Entre 4 452 y 9 762 casos por año, con picos en 2007 y 2011 y descenso desde 2019.
- **Muy dispersa a escala fina.** Solo el 6,8 % de las combinaciones ubigeo × año × semana tiene al menos un caso. De los 1 100 ubigeos, 659 acumulan 10 o más casos y 314 acumulan 100 o más. Conviene filtrar distritos con pocos casos, como hace el paper de referencia (Adams et al., 2025).
- **Etiqueta de ubicación.** Según la documentación del CDC, el lugar corresponde al probable sitio de infección, no necesariamente a la residencia del paciente.

## Problemas de calidad detectados

- **`localcod`:** 32 % de valores nulos (50 513). Además tiene tipos mezclados.
- **`localidad`:** 8 837 nulos (5,6 %).
- **`diresa`:** 6 nulos.
- **`edad`:** contiene valores atípicos (máximo 4 321 569; 4 registros con más de 110 años en `tipo_edad` = A). Requiere normalizar a años y descartar errores.
- **Duplicados exactos:** 1 119 filas. Al ser registros individuales sin identificador, pueden ser casos distintos con idénticos atributos; se recomienda revisarlos antes de eliminarlos.
- **Semana 53:** 481 registros; debe tratarse según el calendario epidemiológico de cada año.
- **Metadatos del portal:** la ficha del dataset contiene errores de copiado (describe el dengue). Validar siempre contra el contenido real del CSV.

## Uso previsto

Agregar los casos por distrito y semana (o mes) para obtener una serie temporal, incorporar variables climáticas rezagadas (temperatura, precipitación, ICEN/ONI) y población distrital del INEI, y entrenar modelos de regresión y clasificación para anticipar brotes con varias semanas de antelación.