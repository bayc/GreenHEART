"""
Example script demonstrating the PUE/WUE-based data center model.

This script shows how to use the new DataCenterPUEWUEPerformanceModel and
DataCenterPUEWUECostModel which use Power Usage Effectiveness (PUE) and
Water Usage Effectiveness (WUE) metrics to calculate facility power and
water consumption from an IT workload.

The key metrics are:
- PUE: Total facility power / IT equipment power (typically 1.1-2.5)
- WUE: Water used per kWh of IT equipment power (typically 0.3-2.0 L/kWh)
"""

import numpy as np
import pandas as pd
import yaml
from h2integrate.core.h2integrate_model import H2IntegrateModel

# Load the CSV file
df = pd.read_csv("10mw_80per_compute_load_training.csv")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Resample to hourly and take the mean
it_workload_mw = df['power_W'].resample('h').mean()

# Scale by 10 and convert from W to MW
it_workload_mw = (it_workload_mw * 10) / 1e6

# Drop the last value
it_workload_mw = it_workload_mw[:-1]

# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(hourly_power.index, hourly_power.values, marker="o", label="Hourly Power Load (MW)", alpha=0.8)
# plt.show()
# lll

# Load the CSV file
df = pd.read_csv("1mw_80per_compute_load_inference.csv")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Resample to hourly and take the mean
hourly_power_inference = df['power_W'].resample('h').mean()

# Scale by 10 and convert from W to MW
hourly_power_inference = (hourly_power_inference * 100) / 1e6


# Create hourly IT workload profile (80 MW constant load)
# This could also be a time-varying profile
n_hours = 8760  # Full year
# it_workload_mw = np.full(n_hours, 80.0)  # 80 MW constant compute load

# Alternatively, create a varying profile:
# it_workload_mw = 80.0 * (0.7 + 0.3 * np.sin(np.linspace(0, 2*np.pi, n_hours)))

# Load the configuration file
config_file = "data_center_pue_wue.yaml"

# Create the H2Integrate model
h2i = H2IntegrateModel(config_file)

# Setup the model
h2i.setup()

# Set the compute IT workload profile
# print(f"Setting IT workload to {it_workload_mw[0]:.1f} MW average load...")
h2i.prob.set_val("data_center_pue_wue.compute_it_workload", it_workload_mw, units="MW")

# Run the model
print("Running model...")
h2i.run()

# Post-process the results
h2i.post_process()

# Extract results
print("\n" + "="*60)
print("DATA CENTER PERFORMANCE AND COST ANALYSIS")
print("="*60)

# Get performance metrics
total_facility_power = h2i.prob.get_val("data_center_pue_wue.total_facility_power", units="MW")
water_used = h2i.prob.get_val("data_center_pue_wue.water_consumed", units="galUS/h")
pue = h2i.prob.get_val("data_center_pue_wue.pue")[0]
wue = h2i.prob.get_val("data_center_pue_wue.wue")[0]

# Get cost metrics
capex = h2i.prob.get_val("data_center_pue_wue.CapEx", units="USD")[0]
opex = h2i.prob.get_val("data_center_pue_wue.OpEx", units="USD/year")[0]

print(f"\nINPUT PARAMETERS:")
print(f"  IT Equipment Capacity: 100 MW")
print(f"  Average IT Workload: {it_workload_mw.mean():.1f} MW")
print(f"  Power Usage Effectiveness (PUE): {pue:.2f}")
print(f"  Water Usage Effectiveness (WUE): {wue:.2f} L/kWh")

print(f"\nPERFORMANCE RESULTS (Annual):")
print(f"  Average Facility Power: {total_facility_power.mean():.1f} MW")
print(f"  Total Facility Power: {(total_facility_power.sum() * 8760/8760):.1f} MWh")
print(f"  Average Water Usage: {water_used.mean():.1f} galUS/h")
print(f"  Total Water Usage: {(water_used.sum() * 8760/8760):.1f} galUS")

print(f"\nCOST ANALYSIS:")
print(f"  Capital Expenditure (CapEx): ${capex:,.0f}")
print(f"  Operating Expenditure (OpEx): ${opex:,.0f}")
print(f"  Total Cost (CapEx + OpEx): ${capex + opex:,.0f}")
print(f"  Cost per MW-year: ${(capex + opex) / 100:,.0f}")

# Calculate cost breakdown
electricity_rate = h2i.prob.get_val("data_center_pue_wue.electricity_rate")[0]
water_rate = h2i.prob.get_val("data_center_pue_wue.water_rate", units="USD/galUS")[0]
total_power_kwh = (total_facility_power.sum() * 8760/8760) * 1000  # Convert MWh to kWh
total_water_gal = water_used.sum() * 8760/8760

electricity_cost = total_power_kwh * electricity_rate
water_cost = total_water_gal * water_rate

print(f"\nOPERATING COST BREAKDOWN:")
print(f"  Electricity Cost: ${electricity_cost:,.0f} ({electricity_cost/(electricity_cost+water_cost)*100:.1f}%)")
print(f"  Water Cost: ${water_cost:,.0f} ({water_cost/(electricity_cost+water_cost)*100:.1f}%)")
print(f"  Fixed O&M: ${opex - electricity_cost - water_cost:,.0f}")

print(f"\nUNIT COSTS:")
print(f"  Electricity Rate: ${electricity_rate:.4f}/kWh")
print(f"  Water Rate: ${water_rate:.4f}/gallon")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)

# Optional: Create a DataFrame with hourly results
df_results = pd.DataFrame({
    'hour': np.arange(n_hours),
    'it_workload_mw': it_workload_mw,
    'total_facility_power_mw': total_facility_power,
    'water_consumed_galus_per_h': water_used,
})

print(f"\nHourly statistics (first 24 hours):")
print(df_results.head(24).to_string(index=False))

# Summary statistics
print(f"\n\nSummary Statistics:")
print(f"  IT Workload (MW):        mean={it_workload_mw.mean():.2f}, min={it_workload_mw.min():.2f}, max={it_workload_mw.max():.2f}")
print(f"  Facility Power (MW):     mean={total_facility_power.mean():.2f}, min={total_facility_power.min():.2f}, max={total_facility_power.max():.2f}")
print(f"  Water Usage (galUS/h):   mean={water_used.mean():.0f}, min={water_used.min():.0f}, max={water_used.max():.0f}")
