#!/usr/bin/env python3
from datetime import datetime
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import time

import pandas as pd
from openpyxl import load_workbook

INPUT = Path("SAMPLE_Fruit_morphology.xlsx")
OUTPUT = Path("SAMPLE_Fruit_morphology_ERA5Land_140d.xlsx")


def samples():
    ws = load_workbook(INPUT, read_only=True, data_only=True)["Samples_log"]
    header = [c.value for c in next(ws.iter_rows())]
    rows = []
    for values in ws.iter_rows(min_row=2, values_only=True):
        row = dict(zip(header, values))
        try:
            date = datetime.strptime(str(row["Ngày lấy"]), "%d %b %Y at %H:%M")
        except (TypeError, ValueError):
            continue
        if row["Lat"] is None or row["Long"] is None:
            continue
        rows.append({**row, "sampling_date": pd.Timestamp(date.date())})
    return pd.DataFrame(rows)


def download(points):
    result = []
    for start in range(0, len(points), 10):
        batch = points.iloc[start : start + 10]
        query = urllib.parse.urlencode(
            {
                "latitude": ",".join(map(str, batch.Lat)),
                "longitude": ",".join(map(str, batch.Long)),
                "start_date": "2026-02-21",
                "end_date": "2026-08-18",
                "hourly": "relative_humidity_2m",
                "daily": "temperature_2m_mean,temperature_2m_min,temperature_2m_max,precipitation_sum,et0_fao_evapotranspiration_sum,sunshine_duration",
                "timezone": "Asia/Ho_Chi_Minh",
                "models": "era5_seamless",
            }
        )
        url = "https://archive-api.open-meteo.com/v1/archive?" + query
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=180) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))
        result.extend(payload if isinstance(payload, list) else [payload])
    return result


def main():
    df = samples()
    payloads = download(df)
    records = []

    for (_, sample), payload in zip(df.iterrows(), payloads):
        hourly = pd.DataFrame(payload["hourly"]).assign(time=lambda x: pd.to_datetime(x.time)).set_index("time")
        daily = pd.DataFrame(payload["daily"]).assign(time=lambda x: pd.to_datetime(x.time)).set_index("time")
        daily["RH_mean_pct"] = hourly.relative_humidity_2m.resample("D").mean()
        start, end = sample.sampling_date - pd.Timedelta(days=140), sample.sampling_date - pd.Timedelta(days=1)
        daily = daily.loc[str(start.date()):str(end.date())].rename(columns={
            "temperature_2m_mean": "temperature_mean_C",
            "temperature_2m_min": "temperature_min_C",
            "temperature_2m_max": "temperature_max_C",
            "precipitation_sum": "rain_mm",
            "et0_fao_evapotranspiration_sum": "PEV_mm",
            "sunshine_duration": "sunshine_hours",
        })
        daily.sunshine_hours /= 3600
        for date, row in daily.iterrows():
            records.append({"Fruit-ID": sample["Fruit-ID"], "date": date, **row.to_dict()})

    daily = pd.DataFrame(records)
    summary = daily.groupby("Fruit-ID", as_index=False).agg(
        valid_days=("date", "count"),
        temperature_mean_C=("temperature_mean_C", "mean"),
        temperature_min_C=("temperature_min_C", "min"),
        temperature_max_C=("temperature_max_C", "max"),
        RH_mean_pct=("RH_mean_pct", "mean"),
        rain_total_mm=("rain_mm", "sum"),
        PEV_total_mm=("PEV_mm", "sum"),
        sunshine_total_hours=("sunshine_hours", "sum"),
    )
    meta = df[["Fruit-ID", "Ngày lấy", "Lat", "Long"]].copy()
    meta["window_start"] = df.sampling_date - pd.Timedelta(days=140)
    meta["window_end"] = df.sampling_date - pd.Timedelta(days=1)
    summary = meta.merge(summary, on="Fruit-ID", how="left")

    for col in summary.select_dtypes("number"):
        summary[col] = summary[col].round(3)
    for col in daily.select_dtypes("number"):
        daily[col] = daily[col].round(3)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary_140d", index=False)
        daily.to_excel(writer, sheet_name="Daily_140d", index=False)
        pd.DataFrame(
            [
                ["Window", "D-140 through D-1; sampling date excluded; calendar days use UTC+7."],
                ["Source", "Open-Meteo ERA5-Seamless: ERA5-Land temperature/humidity; ERA5 precipitation/radiation, retrieved 2026-09-27."],
                ["Sunshine", "WMO definition: direct normal irradiance > 120 W/m²; summed hours."],
                ["PEV", "FAO-56 reference evapotranspiration ET0 supplied for ERA5-Land (mm), used as PEV."],
                ["Skipped", "7 rows without sampling date; all 121 dated rows had coordinates."],
            ],
            columns=["Item", "Definition"],
        ).to_excel(writer, sheet_name="README", index=False)

    assert len(summary) == 121 and summary.valid_days.eq(140).all()
    print(f"Wrote {OUTPUT}: {len(summary)} samples, {len(daily)} daily rows")


if __name__ == "__main__":
    main()
