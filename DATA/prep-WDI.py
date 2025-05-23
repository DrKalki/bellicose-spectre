import pandas as pd
import numpy as np

# Load the WDI data
wdi = pd.read_csv("WDI.csv")

needed_series = [
    "Government Effectiveness: Estimate",
    "GDP per capita (constant 2015 US$)",
    "Population, total"
]
wdi = wdi[wdi["Series Name"].isin(needed_series)]

# Melt to long
wdi_long = wdi.melt(
    id_vars=["Country Name", "Country Code", "Series Name"],
    var_name="year",
    value_name="value"
)

# Clean: replace '..' with np.nan and convert to float
wdi_long["value"] = wdi_long["value"].replace("..", np.nan)
wdi_long["value"] = pd.to_numeric(wdi_long["value"], errors="coerce")

# Extract year as before
wdi_long["year"] = wdi_long["year"].str.extract(r'(\d{4})').astype(float).astype('Int64')

# Pivot
wdi_pivot = wdi_long.pivot_table(
    index=["Country Name", "Country Code", "year"],
    columns="Series Name",
    values="value"
).reset_index()

wdi_pivot = wdi_pivot.rename(columns={
    "Country Name": "country",
    "Country Code": "country_code",
    "Government Effectiveness: Estimate": "gov_effect",
    "GDP per capita (constant 2015 US$)": "gdp_pc",
    "Population, total": "pop"
})

# Compute logs
wdi_pivot["log_gdp_pc"] = np.log(wdi_pivot["gdp_pc"])
wdi_pivot["log_pop"] = np.log(wdi_pivot["pop"])

# Drop rows with NA for any key variable (listwise deletion)
wdi_pivot = wdi_pivot.dropna(subset=["gov_effect", "gdp_pc", "pop", "log_gdp_pc", "log_pop"])

# Output
wdi_pivot.to_csv("WDI_prepared.csv", index=False)
print("Done! Wrote 'WDI_prepared.csv'")