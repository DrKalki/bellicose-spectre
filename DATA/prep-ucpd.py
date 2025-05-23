import pandas as pd

# Load the UCDP termination data (full dataset)
df = pd.read_csv("ucpd-termination.csv")

# Group by country name and year, aggregate max recurrence (captures any recurrence per country-year)
agg = (
    df.groupby(['location', 'year'], as_index=False)
    .agg({'recur': 'max'})
    .rename(columns={'location': 'country', 'recur': 'recurrence'})
)

# (Optional: drop rows with missing country or year, but should not be necessary)
agg = agg.dropna(subset=['country', 'year'])

# Output to CSV
agg.to_csv("UCDP_prepared.csv", index=False)
print("Done! Wrote 'UCDP_prepared.csv'")