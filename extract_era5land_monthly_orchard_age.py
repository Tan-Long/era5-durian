#!/usr/bin/env python3
from pathlib import Path
import math
import shutil

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr


INPUT = Path("SAMPLE_Fruit_morphology_ERA5Land_CDS_140d.xlsx")
AGES = Path("Tuổi vườn.xlsx")
RAW = Path("/Volumes/SoilTECH/era5land_monthly_orchard_age")
OUTPUT = Path("SAMPLE_Fruit_morphology_ERA5Land_CDS_140d_orchard_age.xlsx")
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "potential_evaporation",
    "surface_solar_radiation_downwards",
]


def client():
    config = {
        key.strip(): value.strip()
        for key, value in (
            line.split(": ", 1) for line in Path(".env").read_text().splitlines() if ": " in line
        )
    }
    return cdsapi.Client(url=config["url"], key=config["key"])


def download():
    RAW.mkdir(exist_ok=True)
    requests = {
        "era5land_monthly_2004_2025.nc": ([str(year) for year in range(2004, 2026)], [f"{m:02d}" for m in range(1, 13)]),
        "era5land_monthly_2026_01_07.nc": (["2026"], [f"{m:02d}" for m in range(1, 8)]),
    }
    for filename, (years, months) in requests.items():
        target = RAW / filename
        if target.exists() and target.stat().st_size > 10_000:
            continue
        client().retrieve(
            "reanalysis-era5-land-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": VARIABLES,
                "year": years,
                "month": months,
                "time": ["00:00"],
                "area": [14.1, 107.7, 12.0, 108.4],
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            str(target),
        )


def load_orchards():
    samples = pd.read_excel(INPUT, sheet_name="Summary_140d")
    ids = samples["Fruit-ID"].str.extract(r"26TAN-(?:(N\d+)-)?(V\d+)-C\d+")
    assert ids.notna().iloc[:, 1].all()
    samples[["Group", "Farm-ID"]] = ids
    samples["Group"] = samples.Group.fillna("")
    samples["farm_key"] = samples.Group + samples["Farm-ID"]
    samples["sampling_date_used"] = pd.to_datetime(samples.sampling_date_used)

    ages = pd.read_excel(AGES, header=3).iloc[:, :4]
    ages.columns = ["farm_key", "owner", "planting_year_raw", "orchard_age_years"]
    ages = ages.dropna(subset=["farm_key"])
    ages["farm_key"] = ages.farm_key.astype(str).str.strip()
    ages["owner"] = ages.owner.fillna("")
    ages["planting_year"] = pd.to_numeric(ages.planting_year_raw, errors="coerce")
    ages["planting_year"] = ages.planting_year.fillna(2026 - ages.orchard_age_years).astype(int)

    grouped = samples.groupby("farm_key")
    assert grouped["sampling_date_used"].apply(lambda values: values.dt.to_period("M").nunique()).eq(1).all()
    orchards = grouped.agg(
        Group=("Group", "first"),
        **{
            "Farm-ID": ("Farm-ID", "first"),
            "sample_count": ("Fruit-ID", "count"),
            "sampling_date_used": ("sampling_date_used", "min"),
            "Lat": ("Lat", "mean"),
            "Long": ("Long", "mean"),
        },
    ).reset_index()
    orchards = orchards.merge(
        ages[["farm_key", "owner", "planting_year", "orchard_age_years"]],
        on="farm_key",
        how="left",
        validate="one_to_one",
    )
    assert len(orchards) == len(ages) == 43 and orchards.orchard_age_years.notna().all()
    return orchards


def annual_means(monthly):
    weighted = monthly.assign(
        temperature_x_days=monthly.temperature_mean_C * monthly.days,
        humidity_x_days=monthly.RH_proxy_pct * monthly.days,
    )
    annual = weighted.groupby(["farm_key", "age_year_period"], as_index=False).agg(
        owner=("owner", "first"),
        Group=("Group", "first"),
        **{
            "Farm-ID": ("Farm-ID", "first"),
            "orchard_age_years": ("orchard_age_years", "first"),
            "period_start": ("month", "min"),
            "period_end": ("month", "max"),
            "valid_months": ("month", "count"),
            "valid_days": ("days", "sum"),
            "temperature_x_days": ("temperature_x_days", "sum"),
            "humidity_x_days": ("humidity_x_days", "sum"),
            "rainfall_total_mm": ("rainfall_mm_month", "sum"),
            "PEV_total_mm": ("PEV_mm_month", "sum"),
            "sunshine_equivalent_total_hours": ("sunshine_equivalent_hours_month", "sum"),
        },
    )
    annual["orchard_age_end_years"] = np.minimum(
        annual.age_year_period * 12, annual.orchard_age_years * 12
    ) / 12
    annual["temperature_mean_C"] = annual.temperature_x_days / annual.valid_days
    annual["RH_proxy_mean_pct"] = annual.humidity_x_days / annual.valid_days
    return annual.drop(columns=["temperature_x_days", "humidity_x_days"])


def main():
    download()
    orchards = load_orchards()
    paths = [RAW / "era5land_monthly_2004_2025.nc", RAW / "era5land_monthly_2026_01_07.nc"]
    ds = xr.concat([xr.open_dataset(path) for path in paths], dim="valid_time").sortby("valid_time")
    times = pd.DatetimeIndex(ds.valid_time.values).to_period("M").to_timestamp()
    rows = []

    for _, orchard in orchards.iterrows():
        month_count = round(float(orchard.orchard_age_years) * 12)
        window_end = orchard.sampling_date_used.to_period("M").start_time - pd.offsets.MonthBegin()
        window_start = window_end - pd.offsets.MonthBegin(month_count - 1)
        keep = (times >= window_start) & (times <= window_end)
        point = ds.sel(latitude=orchard.Lat, longitude=orchard.Long, method="nearest").isel(valid_time=keep)
        months = times[keep]
        assert len(months) == month_count
        days = months.days_in_month.to_numpy()
        temperature = point.t2m.values - 273.15
        dewpoint = point.d2m.values - 273.15
        humidity = np.clip(
            100 * np.exp(17.62 * dewpoint / (243.12 + dewpoint))
            / np.exp(17.62 * temperature / (243.12 + temperature)),
            0,
            100,
        )
        rows.append(
            pd.DataFrame(
                {
                    "farm_key": orchard.farm_key,
                    "owner": orchard.owner,
                    "Group": orchard.Group,
                    "Farm-ID": orchard["Farm-ID"],
                    "orchard_age_years": orchard.orchard_age_years,
                    "age_month": np.arange(1, month_count + 1),
                    "age_year_period": np.ceil(np.arange(1, month_count + 1) / 12).astype(int),
                    "month": months,
                    "days": days,
                    "grid_lat": float(point.latitude),
                    "grid_long": float(point.longitude),
                    "temperature_mean_C": temperature,
                    "RH_proxy_pct": humidity,
                    "rainfall_mm_month": point.tp.values * 1000 * days,
                    "PEV_mm_month": -point.pev.values * 1000 * days,
                    "sunshine_equivalent_hours_month": point.ssrd.values * days / 3_600_000,
                }
            )
        )

    monthly = pd.concat(rows, ignore_index=True)
    annual = annual_means(monthly)
    summary = orchards.copy()
    summary["window_end"] = summary.sampling_date_used.dt.to_period("M").dt.start_time - pd.offsets.MonthBegin()
    summary["valid_months"] = (summary.orchard_age_years * 12).round().astype(int)
    summary["window_start"] = [
        end - pd.offsets.MonthBegin(count - 1)
        for end, count in zip(summary.window_end, summary.valid_months)
    ]
    summary["annual_periods"] = summary.orchard_age_years.apply(math.ceil)

    for frame in (summary, annual, monthly):
        numeric = frame.select_dtypes("number").columns
        frame[numeric] = frame[numeric].round(3)

    readme = pd.DataFrame(
        [
            ["Preserved sheets", "Summary_140d, Daily_140d and README are copied unchanged from the input workbook."],
            ["Source", "Copernicus CDS ERA5-Land monthly averaged reanalysis, nearest 0.1° grid cell."],
            ["Unit", "One climate series per orchard (farm_key), not repeated for each sampled fruit."],
            ["Period", "Exactly orchard age × 12 complete months ending one month before sampling."],
            ["Annual", "Consecutive 12-month orchard-age periods; a fractional final age is kept as a shorter period."],
            ["Temperature", "Day-weighted mean monthly 2 m temperature, °C."],
            ["Humidity", "RH proxy (%) derived from monthly mean T2m and D2m; not mean hourly RH."],
            ["Rainfall", "Monthly total = monthly mean daily accumulation × days; annual-period value is the sum, mm."],
            ["PEV", "Native potential evaporation, sign reversed positive; annual-period value is the sum, mm."],
            ["Sunshine", "Equivalent full-sun hours from SSRD; not >=120 W/m² hourly sunshine duration."],
        ],
        columns=["Item", "Definition"],
    )

    shutil.copy2(INPUT, OUTPUT)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        summary.to_excel(writer, sheet_name="Orchard_summary", index=False)
        annual.to_excel(writer, sheet_name="Annual_orchard_age", index=False)
        monthly.to_excel(writer, sheet_name="Monthly_orchard_age", index=False)
        readme.to_excel(writer, sheet_name="README_orchard_age", index=False)

    expected_months = round(float((orchards.orchard_age_years * 12).sum()))
    expected_years = int(orchards.orchard_age_years.apply(math.ceil).sum())
    assert len(monthly) == expected_months == 5478
    assert len(annual) == expected_years == 457
    assert monthly.groupby("farm_key").size().eq(orchards.set_index("farm_key").orchard_age_years * 12).all()
    assert annual.valid_months.between(1, 12).all()
    assert not annual.isna().any().any() and not monthly.isna().any().any()
    print(f"Wrote {OUTPUT}: {len(orchards)} orchards, {len(annual)} orchard-year rows, {len(monthly)} orchard-month rows")


if __name__ == "__main__":
    main()
