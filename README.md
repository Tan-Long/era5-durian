# ERA5 Durian Orchard Climate

Scripts and Excel outputs for extracting ERA5/ERA5-Land climate variables for the durian orchard samples in `SAMPLE_Fruit_morphology.xlsx`.

The separate cinnamon–durian dataset (`DATA-2526CM-Pseudo-14Sep-La-Ce.xlsx`) is not part of this repository.

The orchard-age workbook contains:

- `Orchard_summary`: one row per orchard.
- `Annual_orchard_age`: one row per orchard-age year.
- `Monthly_orchard_age`: exactly `orchard age × 12` monthly rows per orchard.

The sampling month is excluded; each climate window ends at the last complete month before sampling.

## Run

Use the local virtual environment and configure Copernicus CDS credentials in `.env`:

```bash
.venv/bin/python extract_era5land_monthly_orchard_age.py
```

Raw NetCDF downloads and `.env` are intentionally not committed.
