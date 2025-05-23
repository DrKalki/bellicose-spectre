import pandas as pd

# Load SIPRI data with correct header
sipri = pd.read_csv('SIPRI.csv', header=10, encoding='latin1')

# ---- Step 1: Filter for imports only ----
# There is NO explicit "trade_flow" column in your dataset,
# so by SIPRI convention, "Recipient" countries are always importers.
# If you later add export data, adjust here.

# ---- Step 2: Extract/import the columns you care about ----
# We'll treat 'Recipient' as 'importer', and use delivery year and TIV.
sipri = sipri.rename(columns={
    'Recipient': 'importer',
    'Year(s) of delivery': 'year',
    'SIPRI TIV of delivered weapons': 'TIV'
})

# ---- Step 3: Clean up year field ----
# Some year fields have "2017; 2018; 2019", so extract the first year
sipri['year'] = sipri['year'].astype(str).str.extract(r'(\d{4})')
sipri['year'] = pd.to_numeric(sipri['year'], errors='coerce').astype('Int64')

# ---- Step 4: Identify new vs second-hand ----
# We'll use 'status' to determine second hand (1) or new (0)
def is_second_hand(status):
    if isinstance(status, str) and 'second hand' in status.lower():
        return 1
    elif isinstance(status, str) and 'new' in status.lower():
        return 0
    return pd.NA

sipri['second_hand'] = sipri['status'].apply(is_second_hand)

# ---- Step 5: Clean up and convert TIV ----
sipri['TIV'] = pd.to_numeric(sipri['TIV'], errors='coerce')
sipri = sipri.dropna(subset=['importer', 'year', 'second_hand', 'TIV'])

# ---- Step 6: Group by importer, year, second_hand ----
grouped = (
    sipri.groupby(['importer', 'year', 'second_hand'], as_index=False)
    .agg({'TIV': 'sum'})
)

# ---- Step 7: Pivot to wide format (country-year columns for new/second-hand) ----
pivot = (
    grouped
    .pivot_table(index=['importer', 'year'], columns='second_hand', values='TIV', fill_value=0)
    .reset_index()
)

pivot = pivot.rename(
    columns={
        1: 'second_hand_arms_tiv',
        0: 'new_arms_tiv',
        'importer': 'country'
    }
)

# ---- Step 8: Output ----
pivot.to_csv('SIPRI_prepared.csv', index=False)
print("Done! Wrote 'SIPRI_prepared.csv'")