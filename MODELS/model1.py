import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# 1. Load data and sort
df = pd.read_csv("synthbs.csv")
df = df.sort_values(['country', 'year'])

# 2. Create lags for arms transfers
df['second_hand_arms_tiv_lag'] = df.groupby('country')['second_hand_arms_tiv'].shift(1)
df['new_arms_tiv_lag'] = df.groupby('country')['new_arms_tiv'].shift(1)

# 3. Drop rows with missing values (necessary for lags and controls)
df = df.dropna(subset=[
    'second_hand_arms_tiv_lag', 'new_arms_tiv_lag',
    'gov_effect', 'log_gdp_pc', 'log_pop', 'recurrence'
])

# 4. Define features
features = [
    'second_hand_arms_tiv_lag',
    'new_arms_tiv_lag',
    'gov_effect',
    'log_gdp_pc',
    'log_pop'
]
X = df[features].astype(float)
y = df['recurrence']

# Ensure outcome is numeric
if y.dtype == bool or y.dtype == object:
    y = y.astype(int)

# Double check for NA
assert not X.isna().any().any(), "NA values remain in X"
assert not y.isna().any(), "NA values remain in y"

# 5. Add intercept
X_const = sm.add_constant(X)

# 6. Fit logit model
logit = sm.Logit(y, X_const)
result = logit.fit()

# 7. Save regression table
results_df = pd.DataFrame({
    'coef': result.params,
    'std_err': result.bse,
    'z': result.tvalues,
    'pval': result.pvalues,
    'ci_lower': result.conf_int()[0],
    'ci_upper': result.conf_int()[1]
})
results_df.to_csv('logit_results.csv')

# 8. Print and save descriptive stats
with open('descriptive_stats.txt', 'w') as f:
    f.write("--- DESCRIPTIVE STATS ---\n")
    f.write("Second-hand arms TIV (lagged):\n")
    f.write(str(df['second_hand_arms_tiv_lag'].describe()) + "\n\n")
    f.write("New arms TIV (lagged):\n")
    f.write(str(df['new_arms_tiv_lag'].describe()) + "\n\n")
    f.write("Recurrence value counts:\n")
    f.write(str(df['recurrence'].value_counts()) + "\n\n")
print(open('descriptive_stats.txt').read())

# 9. Marginal effects plots (predicted probability vs. arms variables)
for var in ['second_hand_arms_tiv_lag', 'new_arms_tiv_lag']:
    x_vals = np.linspace(X[var].min(), X[var].max(), 100)
    X_pred = X.mean().to_dict()
    prob = []
    for x in x_vals:
        X_pred[var] = x
        temp = pd.DataFrame([X_pred])
        temp = sm.add_constant(temp)
        prob.append(result.predict(temp)[0])
    plt.figure()
    plt.plot(x_vals, prob, label=f'Effect of {var}')
    plt.xlabel(var)
    plt.ylabel('Predicted Probability of Recurrence')
    plt.title(f'Marginal Effect: {var}')
    plt.savefig(f'marginal_effect_{var}.png')
    plt.close()

# 10. Coefficient plot
plt.figure()
coefs = result.params[1:]  # exclude intercept
errs = result.bse[1:]
variables = coefs.index
plt.errorbar(coefs, variables, xerr=1.96*errs, fmt='o', capsize=5)
plt.axvline(0, color='grey', linestyle='--')
plt.xlabel('Coefficient')
plt.title('Logit Coefficients (95% CI)')
plt.tight_layout()
plt.savefig('coef_plot.png')
plt.close()

# 11. Distribution plots
plt.figure()
df.boxplot(column=['second_hand_arms_tiv_lag', 'new_arms_tiv_lag'], by='recurrence')
plt.title('Distribution of Arms TIV (Lagged) by Recurrence')
plt.suptitle('')
plt.ylabel('TIV')
plt.savefig('tiv_distribution_by_recurrence.png')
plt.close()

print("Saved regression table, descriptive stats, and graphs!")