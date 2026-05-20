"""
Parameter sweep for different data center sizes (20 MW, 160 MW, 300 MW)
Includes analysis of how size impacts costs and efficiency
"""

import matplotlib.pyplot as plt
import numpy as np
from h2integrate.core.h2integrate_model import H2IntegrateModel

import pandas as pd
import yaml
import itertools
from pathlib import Path

# Load the CSV files for training workload
df_train = pd.read_csv("10mw_80per_compute_load_training.csv")
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
df_train.set_index('timestamp', inplace=True)
hourly_power_training = df_train['power_W'].resample('h').mean()
hourly_power_training = (hourly_power_training * 10) / 1e6
hourly_power_training = hourly_power_training[:-1]

# Load the CSV files for inference workload
df_inf = pd.read_csv("1mw_80per_compute_load_inference.csv")
df_inf['timestamp'] = pd.to_datetime(df_inf['timestamp'])
df_inf.set_index('timestamp', inplace=True)
hourly_power_inference = df_inf['power_W'].resample('h').mean()
hourly_power_inference = (hourly_power_inference * 100) / 1e6

# Define data center sizes to sweep
dc_sizes_mw = [20, 160, 300]

# Define other parameters to sweep
parameter_sweep = {
    "dc_size_mw": dc_sizes_mw,
    "cooling_approach": ["evaporative", "advanced"],  # PUE will vary by approach
    "electricity_buy_price": [0.04, 0.05, 0.06],
}

# Generate all combinations
param_keys = list(parameter_sweep.keys())
param_values = list(parameter_sweep.values())
param_combinations = list(itertools.product(*param_values))

print("="*70)
print(f"Data Center Size Parameter Sweep")
print("="*70)
print(f"Running {len(param_combinations)} combinations...")
print(f"Data Center Sizes: {dc_sizes_mw} MW")
print(f"Cooling Approaches: {parameter_sweep['cooling_approach']}")
print(f"Electricity Prices: {parameter_sweep['electricity_buy_price']} $/kWh")
print("="*70)
print()

sweep_results = {
    "dc_size_mw": [],
    "cooling_approach": [],
    "electricity_buy_price": [],
    "capex_training": [],
    "opex_training": [],
    "water_cost_training": [],
    "electricity_cost_training": [],
    "total_cost_training": [],
    "lcoc_training": [],
    "capex_inference": [],
    "opex_inference": [],
    "water_cost_inference": [],
    "electricity_cost_inference": [],
    "total_cost_inference": [],
    "lcoc_inference": [],
    "error": [],
}

# PUE (Power Usage Effectiveness) values by size and cooling approach
# PUE = total facility power / IT equipment power
# Smaller datacenters tend to have worse PUE, larger ones better
pue_values = {
    20: {"evaporative": 1.35, "advanced": 1.25},
    160: {"evaporative": 1.20, "advanced": 1.10},
    300: {"evaporative": 1.15, "advanced": 1.05},
}

# CAPEX values by size (economies of scale)
capex_per_mw_by_size = {
    20: 4.5e6,      # Smaller: higher cost per MW
    160: 4.0625e6,  # Medium
    300: 3.8e6,     # Larger: economies of scale
}

def calculate_water_usage(compute_load_mw, pue, water_intensity_gal_per_mwh=265):
    """Calculate water usage based on PUE and compute load"""
    # Total facility power needed
    facility_power = compute_load_mw * pue
    # Cooling load is the difference
    cooling_power = facility_power - compute_load_mw
    # Water consumption during cooling
    water_consumed = cooling_power * water_intensity_gal_per_mwh
    return water_consumed

# Run sweep for each parameter combination
for idx, params in enumerate(param_combinations):
    dc_size = params[0]
    cooling = params[1]
    elec_price = params[2]
    
    print(f"[{idx + 1}/{len(param_combinations)}] DC Size: {dc_size} MW, "
          f"Cooling: {cooling}, Elec Price: ${elec_price:.2f}/kWh")
    
    pue = pue_values[dc_size][cooling]
    capex_per_mw = capex_per_mw_by_size[dc_size]
    
    try:
        # Create temporary config file
        config_path = "data_center_size_sweep_temp.yaml"
        with open("data_center_advanced.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Update config to point to temp tech config
        config["technology_config"] = "tech_config_size_sweep_temp.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Create temporary tech config
        tech_config_path = "tech_config_size_sweep_temp.yaml"
        with open("tech_config_advanced.yaml", "r") as f:
            tech_config = yaml.safe_load(f)
        
        # Update technology parameters based on size
        tech_config["technologies"]["data_center"]["model_inputs"]["shared_parameters"]["system_capacity_mw"] = float(dc_size)
        tech_config["technologies"]["data_center"]["model_inputs"]["cost_parameters"]["capex_per_mw"] = capex_per_mw
        tech_config["technologies"]["data_center"]["model_inputs"]["performance_parameters"]["compute_electrical_efficiency"] = 1.0 / pue
        tech_config["technologies"]["grid_buy"]["model_inputs"]["cost_parameters"]["electricity_buy_price"] = elec_price
        
        # Water usage intensity depends on cooling approach
        if cooling == "evaporative":
            water_intensity = 290  # More water-intensive
        else:  # advanced
            water_intensity = 220  # More efficient
        
        tech_config["technologies"]["data_center"]["model_inputs"]["performance_parameters"]["water_use_gal_per_mwh"] = water_intensity
        
        with open(tech_config_path, "w") as f:
            yaml.dump(tech_config, f)
        
        # ===== TRAINING RUN =====
        h2i_train = H2IntegrateModel(config_path)
        h2i_train.setup()
        
        # Scale training load to the data center size
        # Original is 10 MW, scale to desired size
        training_load_scaled = hourly_power_training.values * (dc_size / 10.0)
        h2i_train.prob.set_val("data_center.compute_load_demand", training_load_scaled, units="MW")
        
        h2i_train.run()
        h2i_train.post_process()
        
        capex_train = h2i_train.prob.get_val("data_center.CapEx", units="USD")[0]
        opex_train = h2i_train.prob.get_val("data_center.OpEx", units="USD/yr")[0]
        water_cost_train = h2i_train.prob.get_val("water_feedstock.VarOpEx", units="USD/yr")[0]
        electricity_cost_train = h2i_train.prob.get_val("grid_buy.VarOpEx", units="USD/yr")[0]
        lcoc_train = h2i_train.prob.get_val("finance_subgroup_compute_load.LCOC", units="USD/kW/h")[0]
        
        total_cost_train = opex_train + water_cost_train + electricity_cost_train
        
        # ===== INFERENCE RUN =====
        h2i_inf = H2IntegrateModel(config_path)
        h2i_inf.setup()
        
        # Scale inference load to the data center size
        inference_load_scaled = hourly_power_inference.values * (dc_size / 1.0)
        h2i_inf.prob.set_val("data_center.compute_load_demand", inference_load_scaled, units="MW")
        
        h2i_inf.run()
        h2i_inf.post_process()
        
        capex_inf = h2i_inf.prob.get_val("data_center.CapEx", units="USD")[0]
        opex_inf = h2i_inf.prob.get_val("data_center.OpEx", units="USD/yr")[0]
        water_cost_inf = h2i_inf.prob.get_val("water_feedstock.VarOpEx", units="USD/yr")[0]
        electricity_cost_inf = h2i_inf.prob.get_val("grid_buy.VarOpEx", units="USD/yr")[0]
        lcoc_inf = h2i_inf.prob.get_val("finance_subgroup_compute_load.LCOC", units="USD/kW/h")[0]
        
        total_cost_inf = opex_inf + water_cost_inf + electricity_cost_inf
        
        # Store results
        sweep_results["dc_size_mw"].append(dc_size)
        sweep_results["cooling_approach"].append(cooling)
        sweep_results["electricity_buy_price"].append(elec_price)
        sweep_results["capex_training"].append(capex_train)
        sweep_results["opex_training"].append(opex_train)
        sweep_results["water_cost_training"].append(water_cost_train)
        sweep_results["electricity_cost_training"].append(electricity_cost_train)
        sweep_results["total_cost_training"].append(total_cost_train)
        sweep_results["lcoc_training"].append(lcoc_train)
        sweep_results["capex_inference"].append(capex_inf)
        sweep_results["opex_inference"].append(opex_inf)
        sweep_results["water_cost_inference"].append(water_cost_inf)
        sweep_results["electricity_cost_inference"].append(electricity_cost_inf)
        sweep_results["total_cost_inference"].append(total_cost_inf)
        sweep_results["lcoc_inference"].append(lcoc_inf)
        sweep_results["error"].append("")
        
        print(f"  ✓ Training - Total Cost: ${total_cost_train/1e6:.2f}M/yr, LCOC: ${lcoc_train:.4f}/kWh")
        print(f"  ✓ Inference - Total Cost: ${total_cost_inf/1e6:.2f}M/yr, LCOC: ${lcoc_inf:.4f}/kWh")
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        sweep_results["dc_size_mw"].append(dc_size)
        sweep_results["cooling_approach"].append(cooling)
        sweep_results["electricity_buy_price"].append(elec_price)
        for key in sweep_results:
            if key not in ["dc_size_mw", "cooling_approach", "electricity_buy_price", "error"]:
                sweep_results[key].append(None)
        sweep_results["error"].append(str(e))
    
    print()

# Clean up temporary files
Path("data_center_size_sweep_temp.yaml").unlink(missing_ok=True)
Path("tech_config_size_sweep_temp.yaml").unlink(missing_ok=True)

# Save results to CSV
results_df = pd.DataFrame(sweep_results)
results_df.to_csv("data_center_size_sweep_results.csv", index=False)
print("Results saved to data_center_size_sweep_results.csv")
print()

# Print summary statistics
print("="*70)
print("SUMMARY BY DATA CENTER SIZE")
print("="*70)
for size in dc_sizes_mw:
    subset = results_df[results_df['dc_size_mw'] == size]
    if len(subset) > 0:
        print(f"\n{size} MW Data Center:")
        print(f"  Training - Avg LCOC: ${subset['lcoc_training'].mean():.4f}/kWh, "
              f"Total Cost: ${subset['total_cost_training'].mean()/1e6:.2f}M/yr")
        print(f"  Inference - Avg LCOC: ${subset['lcoc_inference'].mean():.4f}/kWh, "
              f"Total Cost: ${subset['total_cost_inference'].mean()/1e6:.2f}M/yr")

print("\n" + "="*70)
print("SUMMARY BY COOLING APPROACH")
print("="*70)
for approach in ["evaporative", "advanced"]:
    subset = results_df[results_df['cooling_approach'] == approach]
    if len(subset) > 0:
        print(f"\n{approach.capitalize()} Cooling:")
        print(f"  Training - Avg LCOC: ${subset['lcoc_training'].mean():.4f}/kWh, "
              f"Avg Water Cost: ${subset['water_cost_training'].mean()/1e6:.2f}M/yr")
        print(f"  Inference - Avg LCOC: ${subset['lcoc_inference'].mean():.4f}/kWh, "
              f"Avg Water Cost: ${subset['water_cost_inference'].mean()/1e6:.2f}M/yr")

print("\n" + "="*70)
print(f"Complete! Processed {len(results_df)} parameter combinations.")
print("="*70)
print("\nDataFrame preview:")
print(results_df.head(10))
