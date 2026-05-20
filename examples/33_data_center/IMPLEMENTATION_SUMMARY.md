# Data Center PUE/WUE Model Implementation Summary

## What Was Created

I've successfully created a new data center model system for H2Integrate that uses **Power Usage Effectiveness (PUE)** and **Water Usage Effectiveness (WUE)** metrics to determine facility power and water consumption from a given compute IT workload.

## Components Created

### 1. Core Model Classes (`h2integrate/converters/data_center/data_center.py`)

Added four new classes to the existing data_center.py file:

#### DataCenterPUEWUEPerformanceConfig
- Configuration class using `@define` decorator from `attrs`
- Parameters:
  - `pue`: Power Usage Effectiveness (float, typically 1.1-3.0)
  - `wue`: Water Usage Effectiveness in L/kWh (float, typically 0.3-2.0)
  - `compute_it_workload_profile`: IT workload profile in MW
  - `system_capacity_mw`: Maximum IT equipment capacity in MW

#### DataCenterPUEWUEPerformanceModel
- Extends `PerformanceModelBaseClass` (OpenMDAO component)
- **Inputs**:
  - `compute_it_workload`: IT equipment power demand (MW)
  - `pue`, `wue`, `system_capacity`: Configuration parameters
  - `electricity_in`, `water_in`: Available resources (MW, galUS/h)
- **Outputs**:
  - `total_facility_power`: Total facility power consumption (MW)
  - `water_used`: Water consumption (galUS/h)
  - `unmet_electricity_demand`, `unmet_water_demand`: Tracking shortfalls
  - Summary metrics: annual production, capacity factor, etc.

**Key Calculations**:
```
Total Facility Power = IT Workload × PUE
Water Demand (L) = IT Workload (kW) × WUE (L/kWh) × Duration (h)
```

#### DataCenterPUEWUECostConfig
- Configuration for cost model with parameters:
  - `electricity_rate`: $/kWh
  - `water_rate`: $/gallon
  - `capex_per_mw`: Capital cost per MW of IT capacity
  - `fixed_opex_per_mw_per_year`: Fixed O&M per MW annually

#### DataCenterPUEWUECostModel
- Extends `CostModelBaseClass` (OpenMDAO component)
- **Inputs**: Performance model outputs + configuration parameters
- **Outputs**:
  - `CapEx`: Total capital expenditure
  - `OpEx`: Total operating expenditure (fixed O&M + electricity + water costs)

### 2. Configuration Files

#### `tech_config_pue_wue.yaml`
- Example technology configuration showing:
  - Model registration (DataCenterPUEWUEPerformanceModel, DataCenterPUEWUECostModel)
  - Parameter examples with realistic values
  - Integration with water_feedstock and grid_buy components

#### `plant_config_pue_wue.yaml`
- Plant-level configuration showing:
  - Technology interconnections for the PUE/WUE model
  - Proper connection to grid and water systems
  - Simulation parameters (8760 hourly timesteps)

### 3. Example Scripts

#### `run_data_center_pue_wue.py`
- Comprehensive example script demonstrating:
  - How to instantiate the H2Integrate model
  - Setting custom IT workload profiles
  - Running the simulation
  - Extracting and analyzing results
  - Cost breakdown analysis
  - Summary statistics with pandas DataFrames

### 4. Documentation

#### `README_PUE_WUE_MODEL.md`
- **140+ lines** of comprehensive documentation including:
  - Overview of PUE and WUE metrics
  - Detailed parameter explanations with typical ranges
  - Complete configuration examples
  - Usage guide with Python code examples
  - Model outputs table with descriptions
  - Detailed calculation explanations
  - Comparison with original DataCenterPerformanceModel
  - Typical PUE/WUE ranges by data center and cooling type
  - Troubleshooting guide

## Key Features

✅ **Simple Interface**: Users only need to provide PUE, WUE, and IT workload

✅ **Industry Standard**: Uses widely-recognized PUE/WUE metrics

✅ **Flexible Workload**: Supports both constant loads and time-varying profiles

✅ **Cost Accounting**: Tracks electricity, water, and O&M costs separately

✅ **Resource Constraints**: Handles unmet demands when resources are insufficient

✅ **Annual Scaling**: Automatically scales simulation results to annual metrics

✅ **OpenMDAO Integration**: Fully integrated with H2Integrate's component framework

## How to Use

### Basic Usage

```python
import numpy as np
from h2integrate.core.h2integrate_model import H2IntegrateModel

# Create workload profile
workload = np.full(8760, 80.0)  # 80 MW constant

# Create and run model
h2i = H2IntegrateModel("plant_config_pue_wue.yaml")
h2i.setup()
h2i.prob.set_val("data_center_pue_wue.compute_it_workload", workload, units="MW")
h2i.run()
h2i.post_process()

# Extract results
power = h2i.prob.get_val("data_center_pue_wue.total_facility_power", units="MW")
water = h2i.prob.get_val("data_center_pue_wue.water_used", units="galUS/h")
costs = h2i.prob.get_val("data_center_pue_wue.OpEx", units="USD")[0]
```

### Configuration Customization

Users can adjust in YAML files:
- **pue**: 1.2 (efficient) to 3.0 (legacy)
- **wue**: 0.3 (dry cooling) to 2.0 (wet cooling)
- **electricity_rate**: $0.04 - $0.15 per kWh
- **water_rate**: $0.0005 - $0.005 per gallon
- **Workload profiles**: Constant or hourly varying arrays

## File Locations

```
/home/cbay/code/H2Integrate/
├── h2integrate/
│   └── converters/
│       └── data_center/
│           └── data_center.py [MODIFIED - added 4 new classes]
└── examples/
    └── 33_data_center/
        ├── tech_config_pue_wue.yaml [NEW]
        ├── plant_config_pue_wue.yaml [NEW]
        ├── run_data_center_pue_wue.py [NEW]
        └── README_PUE_WUE_MODEL.md [NEW]
```

## Model Calculations

### Power Consumption
```
Total Facility Power = IT Equipment Workload × PUE

Example: 80 MW × 1.5 = 120 MW facility power
```

### Water Consumption
```
Water (L) = IT Workload (kW) × WUE (L/kWh) × Time (hours)
Water (gallons) = Water (L) / 3.785

Example: 80,000 kW × 0.7 L/kWh × 1 h / 3.785 = 14,792 gal
```

### Cost Calculation
```
CapEx = capex_per_mw × system_capacity_mw

OpEx = fixed_opex_per_mw_per_year × system_capacity_mw 
     + (total_facility_power_kwh × electricity_rate)
     + (total_water_consumed_gal × water_rate)
```

## Integration Points

The new models integrate with:
- **FeedstockPerformanceModel / FeedstockCostModel**: For water supply
- **GridPerformanceModel / GridCostModel**: For electricity supply
- **H2IntegrateModel**: Main simulation engine
- **OpenMDAO**: Component framework

## Testing & Validation

The models:
- ✅ Follow H2Integrate's BaseClass structure
- ✅ Use proper OpenMDAO component patterns
- ✅ Include parameter validation (gt_zero, gte_zero)
- ✅ Support hourly timestep convention
- ✅ Track unmet demands and capacity factors
- ✅ Properly convert units (MW ↔ kWh, L ↔ gallons)

## Next Steps for Users

1. **Review the documentation**: Read `README_PUE_WUE_MODEL.md`
2. **Run the example**: Execute `run_data_center_pue_wue.py`
3. **Customize configuration**: Adjust `tech_config_pue_wue.yaml` for your scenario
4. **Analyze results**: Use the output metrics for feasibility studies or optimization
5. **Integrate into larger models**: Connect to wind, solar, or other resources

## Technical Notes

- **Time step**: Designed for 3600-second (hourly) timesteps
- **Simulation length**: Typical 8760 hours (1 year)
- **Unit conversions**: 
  - 1 MWh = 1000 kWh
  - 1 galUS = 3.785 liters
  - Liters per kWh converted to gallons per hour
- **Capacity factors**: Calculated as actual power / maximum capacity
- **Annual scaling**: Uses `fraction_of_year_simulated` for extrapolation
