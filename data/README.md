# Datos
Estructura esperada:

```
data/raw/
├── M2-result.csv   M3-result.csv   M4-result.csv   M5-result.csv
├── M12-result.csv  M13-result.csv  M22-result.csv NGC104-result.csv
├── (opcionales) M6, M7, M44, NGC3532, ngc2808, ngc4755, ngc6694
└── (preliminar) ngc4755.csv  ngc6694.csv  ngc2808.csv
iso_files/MIST_v1.2_vvcrit0.0_UBVRIplus/*.iso.cmd
```

Los nombres de archivo se pueden cambiar en `configs/config.yaml` (`clusters.<nombre>.file`).

## Gaia DR3

Una consulta por cúmulo en el archivo de Gaia (búsqueda básica circular): centro por nombre o coordenadas, radio en minutos de arco tomado de SIMBAD, hasta 10 000 fuentes, parámetros por defecto, descarga en **CSV**. Columnas usadas:

`phot_g_mean_mag`, `phot_bp_mean_mag`, `phot_rp_mean_mag`, `pmra` y `distance_gspphot`.

Se eliminan las filas con valores nulos en esas columnas (lista configurable en `binaries.required_columns`).

## Isócronas MIST

Isócronas de la grilla MIST v1.2, sin rotación (`vvcrit0.0`), sistema fotométrico **UBV(RI)+** (incluye Gaia EDR3), como archivos `.iso.cmd`: una por cada [Fe/H] de la grilla, con edad `log(edad) = 10.3`. Se descargan desde el sitio del proyecto MIST: <https://waps.cfa.harvard.edu/MIST/>.
