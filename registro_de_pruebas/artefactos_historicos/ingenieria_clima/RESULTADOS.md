# Matrices de ingeniería climática

440 combinaciones, ocho representaciones y tres fuentes climáticas. F0 es un control sin clima. Hiperparámetros, ventana y épocas se fijan desde el control V2 para aislar las representaciones. Todos los resultados corresponden a las mismas filas y etiquetas de las pruebas anteriores.

## Versiones

| ID | Features |
| --- | --- |
| F0 | Solo casos y contexto distrital |
| F1 | Anomalías de tres meses |
| F2 | Acumulados y contrastes 1/3/6 meses |
| F3 | Retardos ponderados y efectos no lineales |
| F4 | Interacciones clima, casos y estación anual |
| F5 | Extremos diarios observados |
| F6 | Selección supervisada de seis features climáticos |
| F7 | Tres componentes principales climáticos |

## historico: RMSE

| Versión / fuente | lineal | arbol | random_forest | mlp | cnn |
| --- | --- | --- | --- | --- | --- |
| F0 / solo_casos | 1.0617 | 1.0354 | 0.9973 | 1.0451 | 1.0385 |
| F1 / estacional_52 | 1.0697 | 1.0451 | 1.0084 | 1.0660 | 1.0952 |
| F1 / huecos_cortos | 1.0733 | 1.0397 | 1.0169 | 1.0680 | 1.1271 |
| F1 / sin_imputar | 1.0659 | 1.0354 | 1.0217 | 1.0548 | 1.0830 |
| F2 / estacional_52 | 1.0772 | 1.0200 | 1.0262 | 1.0963 | 1.1333 |
| F2 / huecos_cortos | 1.0812 | 1.0498 | 1.0353 | 1.0818 | 1.1582 |
| F2 / sin_imputar | 1.0746 | 1.0448 | 1.0328 | 1.0743 | 1.1353 |
| F3 / estacional_52 | 1.0902 | 1.0414 | 1.0179 | 1.1188 | 1.1438 |
| F3 / huecos_cortos | 1.0982 | 1.0165 | 1.0243 | 1.1163 | 1.1558 |
| F3 / sin_imputar | 1.0943 | 1.0209 | 1.0288 | 1.0899 | 1.1478 |
| F4 / estacional_52 | 1.0685 | 1.0495 | 1.0279 | 1.1210 | 1.1358 |
| F4 / huecos_cortos | 1.0704 | 1.0529 | 1.0351 | 1.1215 | 1.1564 |
| F4 / sin_imputar | 1.0657 | 1.0527 | 1.0374 | 1.1096 | 1.1150 |
| F5 / estacional_52 | 1.0772 | 1.0657 | 1.0101 | 1.0588 | 1.0579 |
| F5 / huecos_cortos | 1.0772 | 1.0657 | 1.0101 | 1.0588 | 1.0579 |
| F5 / sin_imputar | 1.0772 | 1.0657 | 1.0101 | 1.0588 | 1.0579 |
| F6 / estacional_52 | 1.0726 | 1.0496 | 1.0151 | 1.0679 | 1.0584 |
| F6 / huecos_cortos | 1.0778 | 1.0513 | 1.0173 | 1.0761 | 1.0849 |
| F6 / sin_imputar | 1.0719 | 1.0352 | 1.0211 | 1.0571 | 1.0537 |
| F7 / estacional_52 | 1.0735 | 1.0381 | 1.0014 | 1.0545 | 1.0605 |
| F7 / huecos_cortos | 1.0762 | 1.0476 | 1.0104 | 1.0577 | 1.0752 |
| F7 / sin_imputar | 1.0739 | 1.0349 | 1.0119 | 1.0613 | 1.0699 |

## historico: PR_AUC

| Versión / fuente | lineal | arbol | random_forest | mlp | cnn |
| --- | --- | --- | --- | --- | --- |
| F0 / solo_casos | 0.1076 | 0.0354 | 0.0896 | 0.0746 | 0.1165 |
| F1 / estacional_52 | 0.1044 | 0.0280 | 0.0784 | 0.0463 | 0.0665 |
| F1 / huecos_cortos | 0.1064 | 0.0780 | 0.0866 | 0.0575 | 0.1103 |
| F1 / sin_imputar | 0.1100 | 0.0506 | 0.1052 | 0.0581 | 0.1269 |
| F2 / estacional_52 | 0.0903 | 0.0293 | 0.0624 | 0.0320 | 0.0488 |
| F2 / huecos_cortos | 0.0940 | 0.0684 | 0.0872 | 0.0389 | 0.0803 |
| F2 / sin_imputar | 0.1028 | 0.0532 | 0.0947 | 0.0383 | 0.0999 |
| F3 / estacional_52 | 0.0778 | 0.0296 | 0.0518 | 0.0592 | 0.0567 |
| F3 / huecos_cortos | 0.0819 | 0.0682 | 0.0716 | 0.0651 | 0.0903 |
| F3 / sin_imputar | 0.0902 | 0.0588 | 0.0865 | 0.0642 | 0.1208 |
| F4 / estacional_52 | 0.0873 | 0.0265 | 0.0753 | 0.0329 | 0.0828 |
| F4 / huecos_cortos | 0.0951 | 0.0769 | 0.0968 | 0.0478 | 0.1158 |
| F4 / sin_imputar | 0.1065 | 0.0516 | 0.0948 | 0.0499 | 0.1308 |
| F5 / estacional_52 | 0.1042 | 0.0900 | 0.0989 | 0.0784 | 0.1635 |
| F5 / huecos_cortos | 0.1042 | 0.0900 | 0.0989 | 0.0784 | 0.1635 |
| F5 / sin_imputar | 0.1042 | 0.0900 | 0.0989 | 0.0784 | 0.1635 |
| F6 / estacional_52 | 0.1137 | 0.0358 | 0.0873 | 0.0658 | 0.1283 |
| F6 / huecos_cortos | 0.0998 | 0.0731 | 0.0826 | 0.0426 | 0.1227 |
| F6 / sin_imputar | 0.1082 | 0.0498 | 0.0918 | 0.0400 | 0.1311 |
| F7 / estacional_52 | 0.1111 | 0.0255 | 0.0781 | 0.0626 | 0.0866 |
| F7 / huecos_cortos | 0.1033 | 0.0836 | 0.0864 | 0.0606 | 0.1105 |
| F7 / sin_imputar | 0.1058 | 0.0460 | 0.0962 | 0.0644 | 0.1282 |

## oficial: RMSE

| Versión / fuente | lineal | arbol | random_forest | mlp | cnn |
| --- | --- | --- | --- | --- | --- |
| F0 / solo_casos | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F1 / estacional_52 | 0.9446 | 0.9562 | 0.9293 | 0.9568 | 0.9648 |
| F1 / huecos_cortos | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F1 / sin_imputar | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F2 / estacional_52 | 0.9448 | 0.9581 | 0.9300 | 0.9609 | 0.9627 |
| F2 / huecos_cortos | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F2 / sin_imputar | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F3 / estacional_52 | 0.9442 | 0.9555 | 0.9288 | 0.9566 | 0.9650 |
| F3 / huecos_cortos | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F3 / sin_imputar | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F4 / estacional_52 | 0.9442 | 0.9596 | 0.9319 | 0.9573 | 0.9634 |
| F4 / huecos_cortos | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F4 / sin_imputar | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F5 / estacional_52 | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F5 / huecos_cortos | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F5 / sin_imputar | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F6 / estacional_52 | 0.9451 | 0.9535 | 0.9285 | 0.9568 | 0.9633 |
| F6 / huecos_cortos | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F6 / sin_imputar | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F7 / estacional_52 | 0.9449 | 0.9571 | 0.9307 | 0.9606 | 0.9622 |
| F7 / huecos_cortos | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |
| F7 / sin_imputar | 0.9451 | 0.9539 | 0.9305 | 0.9565 | 0.9641 |

## oficial: PR_AUC

| Versión / fuente | lineal | arbol | random_forest | mlp | cnn |
| --- | --- | --- | --- | --- | --- |
| F0 / solo_casos | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F1 / estacional_52 | 0.1850 | 0.1305 | 0.1949 | 0.0595 | 0.0565 |
| F1 / huecos_cortos | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F1 / sin_imputar | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F2 / estacional_52 | 0.1853 | 0.1264 | 0.1938 | 0.0511 | 0.0592 |
| F2 / huecos_cortos | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F2 / sin_imputar | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F3 / estacional_52 | 0.1854 | 0.1307 | 0.1962 | 0.0540 | 0.0567 |
| F3 / huecos_cortos | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F3 / sin_imputar | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F4 / estacional_52 | 0.1834 | 0.1261 | 0.1924 | 0.0533 | 0.0568 |
| F4 / huecos_cortos | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F4 / sin_imputar | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F5 / estacional_52 | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F5 / huecos_cortos | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F5 / sin_imputar | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F6 / estacional_52 | 0.1851 | 0.1326 | 0.1907 | 0.0540 | 0.0579 |
| F6 / huecos_cortos | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F6 / sin_imputar | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F7 / estacional_52 | 0.1847 | 0.1318 | 0.1898 | 0.0575 | 0.0557 |
| F7 / huecos_cortos | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |
| F7 / sin_imputar | 0.1855 | 0.1322 | 0.1940 | 0.0562 | 0.0569 |

## Selección y límites

`seleccion_por_validacion.csv` elige sin mirar prueba; `mejor_observado_prueba.csv` es descriptivo y no una nueva selección independiente. Las matrices completas de MAE, F1, precisión, recall y cobertura están en los CSV.

Los extremos diarios solo existen donde hubo mediciones. En 2019–2024 no hay estaciones nuevas: cobertura positiva en otros pipelines puede corresponder a imputaciones propagadas. Un cambio de RMSE o PR-AUC no demuestra que se recuperó el clima real ni que el efecto sea causal.

No se usó el clima contemporáneo del objetivo, ni se inventaron humedad, ONI o NDVI. Selección y PCA se ajustaron únicamente en entrenamiento; no se implementó un modelo en dos etapas.

Prueba previamente examinada, una semilla y presupuesto de modelos fijado: estudio exploratorio, sin garantía de óptimo global ni significancia estadística. Guía, trazabilidad bibliográfica y reproducción: `docs/INGENIERIA_CLIMA.md`.

## Comparación con el trabajo anterior

Menor RMSE observado en V1: 0.9229; en esta ingeniería: 0.9285. No se reemplaza el regresor V1: las nuevas representaciones no lo superaron en esta prueba.

Las matrices describen todos los modelos, no solo los ganadores. El mayor valor observado en prueba no equivale a una selección independiente por validación. El avance histórico con extremos diarios tampoco demuestra capacidad operativa posterior a 2015.