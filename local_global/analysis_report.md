# ¿La mejor reconstrucción global preserva también regiones críticas y endmembers?

Análisis descriptivo de seis sistemas en tres datasets, a partir de los resultados guardados.
Todos los criterios siguientes son errores: valores menores indican mejor resultado.

## Respuesta a la pregunta

**Hay concordancia para regiones agregadas, pero no una equivalencia general entre reconstrucción global, conservación local de cada material y endmembers.**

- El ganador global también gana en reconstrucción de bordes en 3/3 escenas y de píxeles mezclados en 3/3 con las máscaras principales.
- Al separar regiones por material dominante, ese ganador solo minimiza el RMSE local en 4/11 combinaciones material-escena; las otras 7 tienen otro ganador.
- El menor RMSE global coincide con el menor mSAD en 2/3 escenas y con el menor SAD de cada material en 3/11 combinaciones.
- El ganador global minimiza el RMSE de abundancia de cada material en 6/11 combinaciones.
- En Urban, aunque DINO gana ambos criterios agregados, los rankings de reconstrucción global y abundancia en bordes presentan rho = 0.0857; coincidir en el primer puesto no implica concordancia del orden completo.
- La sensibilidad también importa: revisar `sensitivity.csv`; en Jasper Ridge, con mezcla más estricta (máxima abundancia < 0.7), gana PCA y el ganador global queda en posición 2.

Los conteos por material son descriptivos y no representan muestras independientes.


## Resultado por dataset

### Samson

Mejor reconstrucción global: **DINO**, RMSE = **0.099493**.

| metric | best_method | best_value | global_winner_rank | global_winner_gap_percent |
| --- | --- | --- | --- | --- |
| Global reconstruction RMSE | DINO | 0.0995 | 1.0000 | 0.0000 |
| Boundary reconstruction RMSE | DINO | 0.1249 | 1.0000 | 0.0000 |
| Mixed-pixel reconstruction RMSE | DINO | 0.1253 | 1.0000 | 0.0000 |
| Boundary abundance error (mean pixel RMSE) | DINO | 0.1061 | 1.0000 | 0.0000 |
| Mean endmember SAD | AE-Patch | 2.7889 | 4.0000 | 53.0851 |
| Worst endmember SAD | AE-Patch | 4.4537 | 2.0000 | 39.5572 |

`global_winner_rank` es la posición del ganador global en cada criterio; `global_winner_gap_percent` es su exceso de error respecto al mejor método de ese criterio.

### Jasper Ridge

Mejor reconstrucción global: **MAE**, RMSE = **0.049648**.

| metric | best_method | best_value | global_winner_rank | global_winner_gap_percent |
| --- | --- | --- | --- | --- |
| Global reconstruction RMSE | MAE | 0.0496 | 1.0000 | 0.0000 |
| Boundary reconstruction RMSE | MAE | 0.0596 | 1.0000 | 0.0000 |
| Mixed-pixel reconstruction RMSE | MAE | 0.0592 | 1.0000 | 0.0000 |
| Boundary abundance error (mean pixel RMSE) | MAE | 0.1115 | 1.0000 | 0.0000 |
| Mean endmember SAD | MAE | 7.0758 | 1.0000 | 0.0000 |
| Worst endmember SAD | ViT / MAE | 9.6608 | 1.5000 | 0.0000 |

`global_winner_rank` es la posición del ganador global en cada criterio; `global_winner_gap_percent` es su exceso de error respecto al mejor método de ese criterio.

### Urban

Mejor reconstrucción global: **DINO**, RMSE = **0.100733**.

| metric | best_method | best_value | global_winner_rank | global_winner_gap_percent |
| --- | --- | --- | --- | --- |
| Global reconstruction RMSE | DINO | 0.1007 | 1.0000 | 0.0000 |
| Boundary reconstruction RMSE | DINO | 0.0978 | 1.0000 | 0.0000 |
| Mixed-pixel reconstruction RMSE | DINO | 0.0816 | 1.0000 | 0.0000 |
| Boundary abundance error (mean pixel RMSE) | DINO | 0.1768 | 1.0000 | 0.0000 |
| Mean endmember SAD | DINO | 6.9435 | 1.0000 | 0.0000 |
| Worst endmember SAD | DINO | 11.4745 | 1.0000 | 0.0000 |

`global_winner_rank` es la posición del ganador global en cada criterio; `global_winner_gap_percent` es su exceso de error respecto al mejor método de ese criterio.

## Coincidencia de ganadores

| metric | sum | count |
| --- | --- | --- |
| Boundary abundance error (mean pixel RMSE) | 3 | 3 |
| Boundary reconstruction RMSE | 3 | 3 |
| Mean endmember SAD | 2 | 3 |
| Mixed-pixel reconstruction RMSE | 3 | 3 |
| Worst endmember SAD | 2 | 3 |

`sum` cuenta los datasets en que el ganador global también minimiza el criterio local o espectral.

## Asociación entre rankings (seis métodos por dataset)

Spearman positivo indica que menor error global tiende a acompañar menor error en el otro criterio. La coincidencia exacta del ganador y la asociación del ranking son preguntas distintas.

| dataset | comparison | spearman_rho |
| --- | --- | --- |
| Samson | Boundary reconstruction RMSE | 1.0000 |
| Samson | Mixed-pixel reconstruction RMSE | 1.0000 |
| Samson | Critical-region reconstruction RMSE | 1.0000 |
| Samson | Macro material reconstruction RMSE | 1.0000 |
| Samson | Global abundance error (mean pixel RMSE) | 0.4857 |
| Samson | Boundary abundance error (mean pixel RMSE) | 0.5429 |
| Samson | Critical-region abundance error (mean pixel RMSE) | 0.5429 |
| Samson | Mean endmember SAD | -0.3714 |
| Samson | Worst endmember SAD | 0.1429 |
| Samson | 95th-percentile reconstruction error | 1.0000 |
| Jasper Ridge | Boundary reconstruction RMSE | 1.0000 |
| Jasper Ridge | Mixed-pixel reconstruction RMSE | 0.9429 |
| Jasper Ridge | Critical-region reconstruction RMSE | 1.0000 |
| Jasper Ridge | Macro material reconstruction RMSE | 1.0000 |
| Jasper Ridge | Global abundance error (mean pixel RMSE) | 1.0000 |
| Jasper Ridge | Boundary abundance error (mean pixel RMSE) | 0.8286 |
| Jasper Ridge | Critical-region abundance error (mean pixel RMSE) | 0.8286 |
| Jasper Ridge | Mean endmember SAD | 0.9429 |
| Jasper Ridge | Worst endmember SAD | 0.9276 |
| Jasper Ridge | 95th-percentile reconstruction error | 0.9429 |
| Urban | Boundary reconstruction RMSE | 1.0000 |
| Urban | Mixed-pixel reconstruction RMSE | 1.0000 |
| Urban | Critical-region reconstruction RMSE | 1.0000 |
| Urban | Macro material reconstruction RMSE | 0.9429 |
| Urban | Global abundance error (mean pixel RMSE) | 0.0857 |
| Urban | Boundary abundance error (mean pixel RMSE) | 0.0857 |
| Urban | Critical-region abundance error (mean pixel RMSE) | 0.0857 |
| Urban | Mean endmember SAD | 0.9429 |
| Urban | Worst endmember SAD | 0.8857 |
| Urban | 95th-percentile reconstruction error | 0.9429 |

## Materiales: conectar cada firma con su región y su abundancia

SAD corresponde a una firma de endmember de toda la escena. El error de abundancia del material se mide en todos los píxeles; el RMSE local usa la región donde ese material domina según el GT.

| dataset | material | global_winner | global_winner_SAD | best_SAD_method | best_SAD | global_winner_abundance_RMSE | best_abundance_method | best_abundance_RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Samson | Soil | DINO | 2.9093 | ViT / MAE | 1.8929 | 0.1035 | DINO | 0.1035 |
| Samson | Tree | DINO | 3.6835 | AE / AE-Patch | 1.2550 | 0.0782 | DINO | 0.0782 |
| Samson | Water | DINO | 6.2154 | AE-Patch | 4.4537 | 0.1148 | AE | 0.1115 |
| Jasper Ridge | Tree | MAE | 9.6608 | PCA / AE | 8.9315 | 0.1562 | AE-Patch | 0.1465 |
| Jasper Ridge | Water | MAE | 8.8647 | MAE | 8.8647 | 0.1209 | MAE | 0.1209 |
| Jasper Ridge | Soil | MAE | 7.0643 | AE-Patch | 6.7838 | 0.1206 | AE-Patch | 0.1200 |
| Jasper Ridge | Road | MAE | 2.7132 | MAE | 2.7132 | 0.0640 | ViT | 0.0609 |
| Urban | Asphalt | DINO | 8.2106 | AE | 7.5404 | 0.2053 | PCA | 0.1491 |
| Urban | Grass | DINO | 1.9304 | DINO | 1.9304 | 0.2488 | DINO | 0.2488 |
| Urban | Tree | DINO | 6.1583 | MAE | 3.5227 | 0.1841 | DINO | 0.1841 |
| Urban | Roof | DINO | 11.4745 | AE-Patch | 5.3892 | 0.0800 | DINO | 0.0800 |

### Excepciones en regiones dominadas por un material

Un buen resultado en la máscara conjunta de bordes o mezcla no implica ganar en cada material. Los siguientes casos tienen menor RMSE local con otro método que con el ganador global:

| dataset | material | pixels | global_winner | global_winner_local_RMSE | best_local_method | best_local_RMSE | global_winner_gap_percent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Samson | Soil | 3015 | DINO | 0.0975 | AE | 0.0889 | 9.6899 |
| Samson | Water | 2344 | DINO | 0.0171 | MAE | 0.0152 | 13.0377 |
| Jasper Ridge | Tree | 3493 | MAE | 0.0588 | AE-Patch | 0.0471 | 24.8081 |
| Jasper Ridge | Water | 3326 | MAE | 0.0205 | AE | 0.0116 | 76.3488 |
| Jasper Ridge | Road | 753 | MAE | 0.0789 | PCA | 0.0761 | 3.7129 |
| Urban | Tree | 24805 | DINO | 0.0874 | AE | 0.0826 | 5.8766 |
| Urban | Roof | 7162 | DINO | 0.1315 | AE-Patch | 0.1037 | 26.7959 |

## Regiones y sensibilidad

Las máscaras se fijaron con referencia, sin seleccionar zonas a partir del error de un método. Se evalúan también materiales minoritarios sin exigir un tamaño mínimo de ROI.

| dataset | region | pixels | best_reconstruction_method | best_abundance_method | global_winner_reconstruction_rank | global_winner_abundance_rank |
| --- | --- | --- | --- | --- | --- | --- |
| Samson | boundary_r0 | 1403 | DINO | DINO | 1.0000 | 1.0000 |
| Samson | boundary_r1 | 2910 | DINO | DINO | 1.0000 | 1.0000 |
| Samson | boundary_r2 | 3919 | DINO | DINO | 1.0000 | 1.0000 |
| Samson | mixed_t070 | 2443 | DINO | DINO | 1.0000 | 1.0000 |
| Samson | mixed_t080 | 3455 | DINO | DINO | 1.0000 | 1.0000 |
| Samson | mixed_t090 | 4897 | DINO | DINO | 1.0000 | 1.0000 |
| Jasper Ridge | boundary_r0 | 3267 | MAE | MAE | 1.0000 | 1.0000 |
| Jasper Ridge | boundary_r1 | 5709 | MAE | MAE | 1.0000 | 1.0000 |
| Jasper Ridge | boundary_r2 | 6862 | MAE | MAE | 1.0000 | 1.0000 |
| Jasper Ridge | mixed_t070 | 3041 | PCA | MAE | 2.0000 | 1.0000 |
| Jasper Ridge | mixed_t080 | 4147 | MAE | MAE | 1.0000 | 1.0000 |
| Jasper Ridge | mixed_t090 | 5868 | MAE | MAE | 1.0000 | 1.0000 |
| Urban | boundary_r0 | 38337 | DINO | DINO | 1.0000 | 1.0000 |
| Urban | boundary_r1 | 65553 | DINO | DINO | 1.0000 | 1.0000 |
| Urban | boundary_r2 | 76892 | DINO | DINO | 1.0000 | 1.0000 |
| Urban | mixed_t070 | 42816 | DINO | DINO | 1.0000 | 1.0000 |
| Urban | mixed_t080 | 58972 | DINO | DINO | 1.0000 | 1.0000 |
| Urban | mixed_t090 | 72559 | DINO | DINO | 1.0000 | 1.0000 |

### Cobertura de las máscaras y contribución al error del ganador global

| dataset | method | region | scene_percent | squared_error_percent | error_enrichment |
| --- | --- | --- | --- | --- | --- |
| Samson | DINO | boundary_r1 | 32.2438 | 50.8225 | 1.5762 |
| Samson | DINO | critical | 42.9584 | 65.6362 | 1.5279 |
| Jasper Ridge | MAE | boundary_r1 | 57.0900 | 82.2734 | 1.4411 |
| Jasper Ridge | MAE | critical | 60.7000 | 85.0989 | 1.4020 |
| Urban | DINO | boundary_r1 | 69.5530 | 65.5524 | 0.9425 |
| Urban | DINO | critical | 81.4544 | 69.9390 | 0.8586 |

Las máscaras agregadas pueden abarcar una fracción amplia de la escena; su concordancia con el ranking global no sustituye el análisis por material. Un `error_enrichment` mayor que uno indica mayor error cuadrático medio dentro de la máscara que en el conjunto de la escena. Una gran contribución total al error no implica enriquecimiento si la máscara también ocupa una gran superficie. No se debe afirmar que los errores siempre se concentran en bordes: esa afirmación depende del método y de la escena.

## Alcance y precauciones de interpretación

- Una reconstrucción espectral pequeña puede coexistir con firmas o abundancias incorrectas; las métricas se comparan directamente y no se asume equivalencia entre ellas.
- Las correlaciones describen seis sistemas por escena. No hay repeticiones independientes ni pruebas de significación; los píxeles no se tratan como réplicas independientes.
- Las máscaras se pueden solapar. Sus porcentajes de error cuadrático no deben sumarse. Solo `critical` y `noncritical` constituyen una partición disjunta.
- El decil de señal baja es un proxy de brillo, no una anotación de sombra. Los bordes de etiquetas dominantes no abarcan todas las transiciones continuas.
- Estas asociaciones no identifican atención, contexto espacial ni autodestilación como causas. Harían falta ablaciones controladas para esa atribución.
- El GT de abundancias y firmas se usa como referencia del benchmark; no se demuestra una verdad física independiente mediante estas comparaciones.
- Los resultados son comparaciones de sistemas completos dentro de cada dataset. Las imágenes se normalizan por escena; no se interpretan RMSE entre escenas como unidades físicas comunes.

## Auditoría de métricas y archivos

La columna histórica `rRMSE` del proyecto coincide con el RMSE convencional del cubo normalizado. Se exportan `reconstruction_rmse` y `relative_rmse` (cociente de normas) por separado. En cada dataset su denominador es constante entre métodos, por lo que sus rankings coinciden.

La columna histórica `aRMSE` es la **media de los RMSE por píxel**, no la raíz del error cuadrático medio agrupado. Ambos se exportan y no se intercambian.

### Consistencia de las referencias con el cubo observado

Se calculó también la reconstrucción usando directamente las abundancias y firmas GT, sin ajuste ni recalibración. Esta es una comprobación del modelo de referencia bajo las mismas escalas que carga el proyecto, no un método competidor:

| dataset | reference_mixture_rmse | reference_mixture_relative_rmse | reference_mixture_sam_deg |
| --- | --- | --- | --- |
| Samson | 0.3678 | 1.5054 | 2.3182 |
| Jasper Ridge | 0.0469 | 0.1614 | 6.5959 |
| Urban | 0.0848 | 0.4171 | 4.3123 |

**El producto de firmas y abundancias GT no reproduce exactamente el cubo observado.** En particular, el residual de Samson es grande frente a los residuos de los sistemas evaluados. Este diagnóstico no identifica su causa; no se han cambiado las escalas ni se ha supuesto una calibración radiométrica común entre firmas y cubo. Por tanto, los desacuerdos entre RMSE de reconstrucción, SAD y abundancias se interpretan como desacuerdos entre los objetivos de este benchmark, sin atribuirlos exclusivamente a pérdida de información del modelo. SAD compara forma y no valida por sí solo amplitud.

Consultar `metric_audit.csv`, `region_definitions.csv`, `rank_correlations.csv`, `endmember_metrics.csv` y `provenance.json` para reproducir cada afirmación.

Los empates en los rankings reciben el rango promedio; los ganadores empatados se indican juntos en `winner_agreement.csv`.
