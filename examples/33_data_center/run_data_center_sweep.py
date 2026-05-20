import matplotlib.pyplot as plt
import numpy as np
from h2integrate.core.h2integrate_model import H2IntegrateModel

import pandas as pd
import yaml
import itertools
from pathlib import Path

# Sweep of data center sizes, relative to Hermantown, statewide average size, and up-sized trend
# Evaoprative cooling vs advanced cooling
# PUE across different data center sizes and cooling approaches



# Load the CSV file
df = pd.read_csv("10mw_80per_compute_load_training.csv")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Resample to hourly and take the mean
hourly_power_training = df['power_W'].resample('h').mean()

# Scale by 10 and convert from W to MW
hourly_power_training = (hourly_power_training * 10) / 1e6

# Drop the last value
hourly_power_training = hourly_power_training[:-1]

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

# Define parameter sweep configurations
parameter_sweep = {
    "compute_electrical_efficiency": [0.95, 0.97, 0.99],
    "cooling_load_ratio": [0.15, 0.2, 0.25],
    "water_use_gal_per_mwh": [240, 265, 290],
    "electricity_buy_price": [0.04, 0.05, 0.06],
}

# Generate all combinations of parameters
param_keys = list(parameter_sweep.keys())
param_values = list(parameter_sweep.values())
param_combinations = list(itertools.product(*param_values))

print(f"Running parameter sweep with {len(param_combinations)} combinations...")
print(f"Parameters: {param_keys}")
print()

water_costs_training = []
electricity_costs_training = []
water_usages_training = []
electricity_usages_training = []

water_costs_inference = []
electricity_costs_inference = []
water_usages_inference = []
electricity_usages_inference = []

sweep_results = {
    "params": [],
    "water_cost_training": [],
    "electricity_cost_training": [],
    "water_usage_training": [],
    "electricity_usage_training": [],
    "water_cost_inference": [],
    "electricity_cost_inference": [],
    "water_usage_inference": [],
    "electricity_usage_inference": [],
}

# Run sweep for each parameter combination
for idx, params in enumerate(param_combinations):
    param_dict = dict(zip(param_keys, params))
    print(f"[{idx + 1}/{len(param_combinations)}] Running with parameters: {param_dict}")
    
    # Create temporary config file
    config_path = "data_center_sweep_temp.yaml"
    with open("data_center_advanced.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Update parameters in the config
    config["technology_config"] = "tech_config_sweep_temp.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    # Create temporary tech config
    tech_config_path = "tech_config_sweep_temp.yaml"
    with open("tech_config_advanced.yaml", "r") as f:
        tech_config = yaml.safe_load(f)
    
    # Update technology parameters
    tech_config["technologies"]["data_center"]["model_inputs"]["performance_parameters"][
        "compute_electrical_efficiency"
    ] = param_dict["compute_electrical_efficiency"]
    tech_config["technologies"]["data_center"]["model_inputs"]["performance_parameters"][
        "cooling_load_ratio"
    ] = param_dict["cooling_load_ratio"]
    tech_config["technologies"]["data_center"]["model_inputs"]["performance_parameters"][
        "water_use_gal_per_mwh"
    ] = param_dict["water_use_gal_per_mwh"]
    tech_config["technologies"]["grid_buy"]["model_inputs"]["cost_parameters"][
        "electricity_buy_price"
    ] = param_dict["electricity_buy_price"]
    
    with open(tech_config_path, "w") as f:
        yaml.dump(tech_config, f)
    
    try:
        # Training run
        h2i = H2IntegrateModel(config_path)
        h2i.setup()
        h2i.prob.set_val("data_center.compute_load_demand", hourly_power_training.values, units="MW")
        h2i.run()
        h2i.post_process()
        
        water_cost_train = h2i.prob.get_val("water_feedstock.VarOpEx", units="USD/yr")[0]
        electricity_cost_train = h2i.prob.get_val("grid_buy.VarOpEx", units="USD/yr")[0]
        water_usage_train = h2i.prob.get_val("data_center.water_consumed", units="galUS/h")
        electricity_usage_train = h2i.prob.get_val("grid_buy.electricity_out", units="MW")

        # Inference run
        h2i = H2IntegrateModel(config_path)
        h2i.setup()
        h2i.prob.set_val("data_center.compute_load_demand", hourly_power_inference.values, units="MW")
        h2i.run()
        h2i.post_process()
        
        water_cost_inf = h2i.prob.get_val("water_feedstock.VarOpEx", units="USD/yr")[0]
        electricity_cost_inf = h2i.prob.get_val("grid_buy.VarOpEx", units="USD/yr")[0]
        water_usage_inf = h2i.prob.get_val("data_center.water_consumed", units="galUS/h")
        electricity_usage_inf = h2i.prob.get_val("grid_buy.electricity_out", units="MW")

        # Store results
        sweep_results["params"].append(param_dict)
        sweep_results["water_cost_training"].append(water_cost_train)
        sweep_results["electricity_cost_training"].append(electricity_cost_train)
        sweep_results["water_usage_training"].append(water_usage_train)
        sweep_results["electricity_usage_training"].append(electricity_usage_train)
        sweep_results["water_cost_inference"].append(water_cost_inf)
        sweep_results["electricity_cost_inference"].append(electricity_cost_inf)
        sweep_results["water_usage_inference"].append(water_usage_inf)
        sweep_results["electricity_usage_inference"].append(electricity_usage_inf)

        print(f"  ✓ Training - Water: ${water_cost_train:.2f}/yr, Electricity: ${electricity_cost_train:.2f}/yr")
        print(f"  ✓ Inference - Water: ${water_cost_inf:.2f}/yr, Electricity: ${electricity_cost_inf:.2f}/yr")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()

# Clean up temporary files
Path(config_path).unlink(missing_ok=True)
Path(tech_config_path).unlink(missing_ok=True)

# Save results to CSV
results_df = pd.DataFrame({
    "compute_electrical_efficiency": [p.get("compute_electrical_efficiency") for p in sweep_results["params"]],
    "cooling_load_ratio": [p.get("cooling_load_ratio") for p in sweep_results["params"]],
    "water_use_gal_per_mwh": [p.get("water_use_gal_per_mwh") for p in sweep_results["params"]],
    "electricity_buy_price": [p.get("electricity_buy_price") for p in sweep_results["params"]],
    "water_cost_training": sweep_results["water_cost_training"],
    "electricity_cost_training": sweep_results["electricity_cost_training"],
    "water_usage_training": sweep_results["water_usage_training"],
    "electricity_usage_training": sweep_results["electricity_usage_training"],
    "water_cost_inference": sweep_results["water_cost_inference"],
    "electricity_cost_inference": sweep_results["electricity_cost_inference"],
    "water_usage_inference": sweep_results["water_usage_inference"],
    "electricity_usage_inference": sweep_results["electricity_usage_inference"],
})

results_df.to_csv("data_center_sweep_results.csv", index=False)
results_df.to_excel("data_center_sweep_results.xlsx", index=False)
print("Results saved to data_center_sweep_results.csv")
print("Results saved to data_center_sweep_results.xlsx")
print()
print(results_df)

