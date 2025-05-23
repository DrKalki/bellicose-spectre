import pandas as pd

wdi = pd.read_csv("WDI_prepared.csv")
ucdp = pd.read_csv("UCDP_prepared.csv")
sipri = pd.read_csv("SIPRI_prepared.csv")

# Merge stepwise, starting with WDI and SIPRI
panel = pd.merge(wdi, sipri, how="inner", on=["country", "year"])
panel = pd.merge(panel, ucdp, how="inner", on=["country", "year"])

panel.to_csv("analysis_panel.csv", index=False)
print("Panel shape:", panel.shape)
print(panel.head())