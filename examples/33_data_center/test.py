import pandas as pd
import yaml

# Load the CSV file
df = pd.read_csv("10mw_80per_compute_load.csv")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Resample to hourly and take the mean
hourly_power = df['power_W'].resample('h').mean()

# Scale by 10 and convert from W to MW
hourly_power = (hourly_power * 10) / 1e6

# Drop the last value
hourly_power = hourly_power[:-1]

# Convert to a list for YAML
hourly_power_list = hourly_power.tolist()

# Save to YAML file
with open('hourly_power_load.yaml', 'w') as f:
    yaml.dump({'hourly_power_MW': hourly_power_list}, f, default_flow_style=False)

print(f"Successfully processed {len(hourly_power_list)} hours of data")
print(f"Saved to hourly_power_load.yaml")
