# Relación entre reconstrucción global, regiones y endmembers

Este análisis responde una pregunta: ¿el ganador global también gana en cada
material y en la fidelidad media de las firmas? Se reutilizan los resultados
existentes, sin entrenar modelos ni generar figuras adicionales.

## Tabla propuesta para el paper

| Dataset | Lowest global RMSE | Material regions also won | Lowest mean SAD |
| --- | --- | --- | --- |
| Samson | DINO | 1 / 3 | AE-Patch |
| Jasper Ridge | MAE | 1 / 4 | MAE |
| Urban | DINO | 2 / 4 | DINO |

Cada región reúne todos los píxeles donde un material tiene la mayor abundancia
GT. La tercera columna cuenta materiales, no porcentaje de píxeles; los empates
locales también cuentan como coincidencias. La última columna compara el SAD
promedio de los endmembers, no exige ganar en cada firma individual.

## Párrafo ilustrativo para el paper

In Urban, DINO achieves the lowest global reconstruction RMSE. However, AE-Patch has lower RMSE in roof-dominant pixels (0.1037 versus 0.1315). The corresponding Roof endmember SAD values are 5.39 and 11.47 degrees, respectively. Thus, scene-level reconstruction quality does not guarantee the best preservation of every material region or endmember.

## Reproducción y datos

Desde la raíz del repositorio: `python local_global/analyze_local_global.py`.
También funciona desde otra carpeta usando la ruta completa del script.
Dependencias: NumPy, pandas, SciPy y PyYAML del entorno del proyecto.

`material_results.csv` contiene los seis métodos para los 11 materiales de las
tres escenas (66 filas): dataset, método, material, número de píxeles, RMSE
global, RMSE local y SAD en grados. El RMSE global se repite en las filas de los
materiales del mismo método; no son observaciones globales independientes.

El script verifica RMSE global y mSAD frente al resumen original. Los mapas y
curvas de `paper_figures` aportan la evidencia visual; esta carpeta no los duplica.

## Notas de interpretación

- RMSE es el error convencional del cubo normalizado; el CSV original lo llama
  `rRMSE`, aunque no calcula el cociente de normas del RMSE relativo.
- SAD compara forma espectral. Una buena media no garantiza la mejor firma de
  cada material ni la preservación de su amplitud.
- Son comparaciones descriptivas de los sistemas guardados. La revisión previa
  de coherencia de escalas del GT sigue pendiente; simplificar no la corrige.
