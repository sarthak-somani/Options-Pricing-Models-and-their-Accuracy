import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# --- 1. Load the Prepared Data ---
# This assumes you have the 'prepared_nifty_options_data.csv' file in the same directory.
file_path = 'Options-Pricing-Models-and-their-Accuracy/prepared_nifty_options_data.csv'
df = pd.read_csv(file_path, parse_dates=['date', 'expiry'])

# --- 2. Define Constants and Calculate Volatility ---
# Risk-free rate based on the 91-day T-bill yield for the period.
risk_free_rate = 0.054597

# Calculate Historical Volatility (sigma)
# First, sort data by date to ensure correct rolling calculation.
df = df.sort_values(by='date')
# Calculate daily log returns of the underlying asset.
df['log_return'] = np.log(df['underlying_value'] / df['underlying_value'].shift(1))
# Calculate rolling standard deviation of log returns (21-day window for ~1 month).
# Then, annualize it by multiplying by the square root of 252 (trading days in a year).
df['volatility'] = df['log_return'].rolling(window=21).std() * np.sqrt(252)

# Drop rows with NaN values created by the rolling calculation.
df.dropna(inplace=True)


# --- 3. Black-Scholes Model Implementation ---
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculates the price of a European call option using the Black-Scholes formula.
    S: Underlying asset price
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free interest rate
    sigma: Volatility of the underlying asset
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

# --- 4. Apply the Model to the Dataset ---
# Calculate the theoretical Black-Scholes price for each option.
df['bs_price'] = black_scholes_call(
    df['underlying_value'],
    df['strike_price'],
    df['time_to_expiration'],
    risk_free_rate,
    df['volatility']
)

# --- 5. Analyze Accuracy ---
# Calculate the pricing error.
df['error'] = df['bs_price'] - df['close']
df['abs_percentage_error'] = np.abs(df['error'] / df['close']) * 100

# Calculate Mean Absolute Percentage Error (MAPE).
mape = df['abs_percentage_error'].mean()

print("--- Black-Scholes Model Accuracy Assessment ---")
print(f"Risk-Free Rate (r) Used: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print("\nThis means, on average, the Black-Scholes model's price was off by about "
      f"{mape:.2f}% from the actual market price for the options in your dataset.")

# --- 6. Visualize the Results ---
# To make the plot readable, we'll take a random sample of 500 data points.
sample_df = df.sample(n=500, random_state=42)

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot of Black-Scholes Price vs. Actual Market Price.
ax.scatter(sample_df['close'], sample_df['bs_price'], alpha=0.6, edgecolors='w', label='Model vs. Market')

# Add a line for perfect correlation (y=x).
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()]),
]
ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Correlation')

ax.set_xlabel('Actual Market Price (Close)', fontsize=12)
ax.set_ylabel('Calculated Black-Scholes Price', fontsize=12)
ax.set_title('Black-Scholes Model Price vs. Actual Market Price (Sample of 500 Options)', fontsize=14)
ax.legend()
ax.grid(True)

plt.tight_layout()
# Save the plot to a file.
plt.savefig('black_scholes_accuracy_plot.png')

print("\n- A plot visualizing the model's accuracy has been generated and saved as 'black_scholes_accuracy_plot.png'.")
print("- The closer the points are to the red dashed line, the more accurate the model's pricing.")

# Display a sample of the final data with results.
print("\n--- Sample of Final Results ---")
print(df[['date', 'strike_price', 'close', 'volatility', 'bs_price', 'abs_percentage_error']].head())