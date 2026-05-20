# Data Center PUE/WUE Model Documentation

## Overview

The new **DataCenterPUEWUEPerformanceModel** and **DataCenterPUEWUECostModel** provide a streamlined approach to modeling data center power and water consumption using industry-standard metrics:

- **PUE (Power Usage Effectiveness)**: Ratio of total facility power to IT equipment power
- **WUE (Water Usage Effectiveness)**: Water consumption per unit of IT equipment power (L/kWh)

These metrics allow data center operators to quickly model facility performance without needing detailed knowledge of cooling systems, electrical efficiency, or other subsystem parameters.

## Key Features

### Performance Model Features
- Simple PUE-based power calculation
- WUE-based water consumption calculation
- Automatic cap of IT workload at system capacity
- Tracks unmet electricity and water demands
- Computes annual performance metrics and capacity factors

### Cost Model Features
- Separate electricity and water cost tracking
- Capital costs based on IT equipment capacity
- Fixed O&M costs
- Variable costs for electricity and water usage
- Clear cost breakdown and cost accounting

## Model Parameters

### Performance Model Configuration

```yaml
performance_parameters:
  pue: 1.5                          # Power Usage Effectiveness (typical: 1.1-2.5)
  wue: 0.7                          # Water Usage Effectiveness in L/kWh (typical: 0.3-2.0)
  compute_it_workload_profile: 80.0 # IT workload in MW (constant or array)
  system_capacity_mw: 100.          # Maximum IT equipment capacity in MW
```

#### Parameter Explanations

**PUE (Power Usage Effectiveness)**
- Dimensionless ratio: Total Facility Power ÷ IT Equipment Power
- **1.1**: Highly efficient data center (rare)
- **1.5**: Good efficiency (modern facilities)
- **2.0**: Average efficiency
- **3.0**: Poor efficiency (older facilities)
- Formula: `Total Facility Power = IT Workload × PUE`

**WUE (Water Usage Effectiveness)**
- Units: Liters per kWh of IT equipment power
- **0.3 L/kWh**: Efficient water recycling systems
- **0.7 L/kWh**: Typical cooling systems
- **2.0 L/kWh**: High water consumption

**compute_it_workload_profile**
- Can be a scalar (constant load) or array (time-varying)
- In MW, matching simulation timesteps

**system_capacity_mw**
- Maximum IT equipment capacity
- Workload is capped at this value if exceeded

### Cost Model Configuration

```yaml
cost_parameters:
  electricity_rate: 0.05            # $/kWh
  water_rate: 0.001                 # $/galUS
  capex_per_mw: 1.0e6               # $/MW of IT capacity
  fixed_opex_per_mw_per_year: 5.0e4 # $/MW/year
```

#### Parameter Explanations

**electricity_rate**
- Price of electricity in USD/kWh
- Applies to total facility power consumption
- Typical range: $0.04 - $0.15/kWh depending on location

**water_rate**
- Price of water in USD per gallon (US gallons)
- Typical range: $0.0005 - $0.005/gal

**capex_per_mw**
- Capital cost per MW of IT equipment capacity
- Includes servers, networking, power distribution
- Typical range: $0.5M - $2.0M per MW

**fixed_opex_per_mw_per_year**
- Fixed operating costs (maintenance, labor, etc.) per MW annually
- Typical range: $30k - $100k per MW/year

## Configuration File Example

### Technology Configuration (tech_config_pue_wue.yaml)

```yaml
name: technology_config
technologies:
  data_center_pue_wue:
    performance_model:
      model: DataCenterPUEWUEPerformanceModel
    cost_model:
      model: DataCenterPUEWUECostModel
    model_inputs:
      shared_parameters:
        system_capacity_mw: 100.
      performance_parameters:
        pue: 1.5
        wue: 0.7
        compute_it_workload_profile: 80.0
      cost_parameters:
        electricity_rate: 0.05
        water_rate: 0.001
        capex_per_mw: 1.0e6
        fixed_opex_per_mw_per_year: 5.0e4
```

### Plant Configuration (plant_config_pue_wue.yaml)

```yaml
technology_interconnections:
  - [data_center_pue_wue, grid_buy, [unmet_electricity_demand, electricity_set_point]]
  - [water_feedstock, data_center_pue_wue, water, pipe]
plant:
  plant_life: 15
  simulation:
    n_timesteps: 8760
    dt: 3600
```

## Usage Example

### Basic Python Script

```python
import numpy as np
from h2integrate.core.h2integrate_model import H2IntegrateModel

# Create hourly IT workload profile (80 MW constant load)
it_workload_mw = np.full(8760, 80.0)

# Create and setup model
h2i = H2IntegrateModel("plant_config_pue_wue.yaml")
h2i.setup()

# Set IT workload
h2i.prob.set_val("data_center_pue_wue.compute_it_workload", it_workload_mw, units="MW")

# Run model
h2i.run()
h2i.post_process()

# Extract results
power = h2i.prob.get_val("data_center_pue_wue.total_facility_power", units="MW")
water = h2i.prob.get_val("data_center_pue_wue.water_used", units="galUS/h")
capex = h2i.prob.get_val("data_center_pue_wue.CapEx", units="USD")[0]
opex = h2i.prob.get_val("data_center_pue_wue.OpEx", units="USD")[0]

print(f"Average Facility Power: {power.mean():.1f} MW")
print(f"Total Annual Water: {(water.sum() * 8760/8760):.0f} gallons")
print(f"Total Cost: ${capex + opex:,.0f}")
```

## Model Outputs

### Performance Outputs (Hourly)

| Output | Units | Description |
|--------|-------|-------------|
| `total_facility_power` | MW | Total facility power consumption (electricity_used) |
| `water_used` | galUS/h | Water consumption rate |
| `unmet_electricity_demand` | MW | Unmet electricity demand |
| `unmet_water_demand` | galUS/h | Unmet water demand |

### Performance Outputs (Summary)

| Output | Units | Description |
|--------|-------|-------------|
| `total_power_produced` | MWh | Total facility power over simulation period |
| `total_water_consumed` | galUS | Total water consumed over simulation period |
| `annual_power_produced` | MWh/year | Annualized facility power |
| `annual_water_consumed` | galUS/year | Annualized water consumption |
| `capacity_factor` | dimensionless | Facility power as fraction of max capacity |

### Cost Outputs

| Output | Units | Description |
|--------|-------|-------------|
| `CapEx` | USD | Capital expenditure (IT equipment) |
| `OpEx` | USD | Operating expenditure (fixed O&M + electricity + water) |

## Calculation Details

### Power Calculation

```
Total Facility Power = IT Workload × PUE

Example:
- IT Workload: 80 MW
- PUE: 1.5
- Total Facility Power: 80 × 1.5 = 120 MW
```

### Water Calculation

```
Water Demand (L/timestep) = IT Workload (kW) × WUE (L/kWh) × (timestep_duration_seconds / 3600)

Example:
- IT Workload: 80 MW = 80,000 kW
- WUE: 0.7 L/kWh
- Timestep: 1 hour
- Water Demand: 80,000 × 0.7 × (3600/3600) = 56,000 liters
- In gallons: 56,000 / 3.785 = 14,792 gallons
```

### Cost Calculation

```
CapEx = capex_per_mw × system_capacity_mw

Fixed OpEx = fixed_opex_per_mw_per_year × system_capacity_mw

Electricity Cost = total_facility_power_kwh × electricity_rate

Water Cost = total_water_consumed_gal × water_rate

Total OpEx = Fixed OpEx + Electricity Cost + Water Cost
```

## Comparison with Original DataCenterPerformanceModel

| Feature | Original Model | PUE/WUE Model |
|---------|---|---|
| Configuration Complexity | High (efficiency, cooling ratio, etc.) | Simple (just PUE and WUE) |
| Input Parameters | 4 parameters | 2 parameters |
| Use Case | Detailed system modeling | Quick feasibility studies |
| Industry Standard | Custom | PUE/WUE industry metrics |
| Accuracy | High with proper calibration | Good for typical scenarios |

## Typical PUE/WUE Ranges

### PUE Ranges by Data Center Type

- **Hyperscale Cloud**: 1.1 - 1.2
- **Modern Efficient**: 1.2 - 1.5
- **Standard**: 1.5 - 2.0
- **Legacy**: 2.0 - 3.0

### WUE Ranges by Cooling Type

- **Dry Cooling**: 0.0 L/kWh
- **Hybrid Cooling**: 0.1 - 0.5 L/kWh
- **Wet Cooling**: 0.5 - 1.5 L/kWh
- **Low Efficiency**: 1.5 - 2.0+ L/kWh

## Running the Example

```bash
cd /home/cbay/code/H2Integrate/examples/33_data_center

# Option 1: Use the provided script
python run_data_center_pue_wue.py

# Option 2: Modify and run your own Python script
python your_script.py
```

## Troubleshooting

### Issue: "Model not found" error
**Solution**: Ensure `DataCenterPUEWUEPerformanceModel` and `DataCenterPUEWUECostModel` are properly imported in your configuration files.

### Issue: Unmet demands appearing
**Solution**: Check that `grid_buy` or `water_feedstock` capacities are sufficient. Increase `electricity_in` or `water_in` availability.

### Issue: Costs seem incorrect
**Solution**: Verify electricity_rate and water_rate units ($/kWh and $/gallon). Check that fixed_opex is for the full project life or adjust accordingly.

## Further Reading

- PUE Definition: https://www.datacenterfriendly.com/
- WUE Definition: https://www.datacenterfriendly.com/
- H2Integrate Documentation: Check the docs/ folder
