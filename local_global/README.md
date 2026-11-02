# Resumen local–global

Ejecutar: `python local_global/analyze_local_global.py`.

Entregables: código, `local_global_summary.png` y `local_global_summary.tex` (figura y un párrafo corto en inglés). El CSV conserva los 66 resultados de respaldo.

Para incluirlo desde la raíz del proyecto: `\input{local_global/local_global_summary.tex}`; requiere graphicx.

Las fracciones cuentan materiales, no píxeles; los empates locales cuentan como coincidencias. RMSE es convencional, aunque el resumen original lo denomina rRMSE. SAD compara forma espectral. La revisión previa de escalas del GT sigue pendiente.

Dependencias: NumPy, pandas, SciPy, PyYAML y Matplotlib. El script verifica RMSE global y mSAD contra el resumen original.
