import pandas as pd
import numpy as np
import re

# Loading the  CSV file sourced from the NSE website.
raw_file_path = 'Options-Pricing-Models-and-their-Accuracy/OPTIDX_NIFTY_CE_10-May-2025_TO_10-Aug-2025.csv'

# low_memory=False to prevent potential data type warnings with large files.
df = pd.read_csv(raw_file_path, low_memory=False)

# --- 1. Clean Column Names ---
# This function standardizes the column names to make them easier to work with.
def clean_col_names(df):
    columns = df.columns
    new_columns = []
    for col in columns:
        # Convert to lowercase.
        new_col = col.lower()
        # Remove special characters and leading/trailing spaces.
        new_col = new_col.strip()
        new_col = re.sub(r'[^a-zA-Z0-9\s]', '', new_col)
        # Replace spaces with underscores.
        new_col = re.sub(r'\s+', '_', new_col)
        new_columns.append(new_col)
    df.columns = new_columns
    return df

df = clean_col_names(df)

# --- 2. Convert Data Types ---
# Convert date columns from text to a proper datetime format.
df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
df['expiry'] = pd.to_datetime(df['expiry'], format='%d-%b-%Y')

# Identify all columns that should be numeric.
numeric_cols = ['open', 'high', 'low', 'ltp', 'no_of_contracts',
                'turnover_in_lakhs', 'premium_turnover_in_lakhs',
                'open_int', 'change_in_oi', 'underlying_value']

# Convert these columns to numbers. 'errors='coerce'' will turn any
# non-numeric values (like '-') into a Not a Number (NaN) placeholder.
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --- 3. Calculate Time to Expiration ---
# This calculates the time to expiration (T) in years, a required
# input for the Black-Scholes model.
# We add 1 day to the difference as per market convention.
df['time_to_expiration'] = (df['expiry'] - df['date']).dt.days / 365.25

# --- 4. Filter Data for Backtesting ---
# Remove rows where critical data for the model is missing.
critical_cols = ['close', 'strike_price', 'underlying_value', 'time_to_expiration']
df.dropna(subset=critical_cols, inplace=True)

# Filter out options that were not traded (no contracts) or have no open interest.
df = df[(df['no_of_contracts'] > 0) & (df['open_int'] > 0)]

# Remove options that have already expired (time to expiration is zero or negative).
df = df[df['time_to_expiration'] > 0]

# --- 5. Save the Prepared Data ---
prepared_file_path = 'prepared_nifty_options_data.csv'
df.to_csv(prepared_file_path, index=False)

print(f"Data cleaning and preparation complete.")
print(f"The prepared data, with {df.shape[0]} rows, has been saved to '{prepared_file_path}'")