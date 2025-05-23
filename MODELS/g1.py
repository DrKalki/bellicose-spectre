import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("synthbs.csv")
df = df.sort_values(['country', 'year'])
df['new_arms_tiv_lag'] = df.groupby('country')['new_arms_tiv'].shift(1)
df['second_hand_arms_tiv_lag'] = df.groupby('country')['second_hand_arms_tiv'].shift(1)
df = df.dropna(subset=['new_arms_tiv_lag', 'second_hand_arms_tiv_lag', 'gov_effect', 'log_gdp_pc', 'log_pop', 'recurrence'])

features = ['second_hand_arms_tiv_lag', 'new_arms_tiv_lag', 'gov_effect', 'log_gdp_pc', 'log_pop']
X = df[features].astype(float)
y = df['recurrence'].astype(int)
X = sm.add_constant(X)
logit = sm.Logit(y, X)
result = logit.fit()

marginal_effects = result.get_margeff()
margeff_summary = marginal_effects.summary_frame()

# Bar plot of average marginal effects
margeff_summary = margeff_summary.loc[features]
margeff_summary['Variable'] = margeff_summary.index

plt.figure(figsize=(8,5))
plt.barh(margeff_summary['Variable'], margeff_summary['dy/dx'], xerr=1.96*margeff_summary['Std. Err.'])
plt.axvline(0, color='red', linestyle='dashed')
plt.xlabel("Average Marginal Effect on Recurrence Probability")
plt.title("Marginal Effects from Logit Model")
plt.tight_layout()
plt.savefig("marginal_effects_barplot.png")
plt.show()