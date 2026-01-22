import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def replace_outliers_with_rolling_mean(df, column, window=7, threshold=2):
    # Calculate rolling mean and standard deviation with center alignment
    rolling_mean = df[column].rolling(window, center=True, min_periods=1).mean()
    rolling_std = df[column].rolling(window, center=True, min_periods=1).std()

    def is_outlier(idx):
        value = df.at[idx, column]
        mean = rolling_mean.iat[idx]
        std = rolling_std.iat[idx]
        if pd.isna(std) or std == 0:
            return False
        return abs(value - mean) > threshold * std

    # Identify outlier indices
    outliers = [i for i in range(len(df)) if is_outlier(i)]
    print(len(outliers))
    # Replace outliers with mean of previous and next valid prices
    for i in outliers:
        prev_idx = i - 1
        next_idx = i + 1
        # Find previous valid index not flagged as outlier
        while prev_idx in outliers and prev_idx >= 0:
            prev_idx -= 1
        # Find next valid index not flagged as outlier
        while next_idx in outliers and next_idx < len(df):
            next_idx += 1
        if prev_idx >= 0 and next_idx < len(df):
            df.at[i, column] = (df.at[prev_idx, column] + df.at[next_idx, column]) / 2
        elif prev_idx >= 0:
            df.at[i, column] = df.at[prev_idx, column]
        elif next_idx < len(df):
            df.at[i, column] = df.at[next_idx, column]
        else:
            # No valid neighbors, keep original value
            pass

    return df

# Example usage
# example_data = {'close': [100, 102, 101, 500, 103, 104, 102, 101, 99, 105]}
df = pd.read_csv('final_new_train.csv')[['gbp']]

print("Before:")
print(df)
# plt.plot(df['gbp'])

df_cleaned = replace_outliers_with_rolling_mean(df, 'gbp', window=5, threshold=2)

print("After:")
print(df_cleaned)

plt.plot(df_cleaned['gbp'])
plt.show()

