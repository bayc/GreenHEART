"""
Script to create visualizations from data center parameter sweep results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load results
results_df = pd.read_csv("data_center_sweep_results.csv")

print("Dataset shape:", results_df.shape)
print("\nColumn names:", results_df.columns.tolist())
print("\nFirst few rows:")
print(results_df.head())
print("\nData summary:")
print(results_df.describe())

# Create output directory for plots
output_dir = Path("sweep_plots")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# Plot 1: Water Cost vs Water Usage (Training)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Group by water_use_gal_per_mwh and calculate mean water costs
water_cost_by_usage = results_df.groupby('water_use_gal_per_mwh')['water_cost_training'].mean()
electricity_cost_by_usage = results_df.groupby('water_use_gal_per_mwh')['electricity_cost_training'].mean()

x = np.arange(len(water_cost_by_usage))
width = 0.35

bars1 = ax.bar(x - width/2, water_cost_by_usage.values / 1e6, width, label='Water Cost', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, electricity_cost_by_usage.values / 1e6, width, label='Electricity Cost', alpha=0.8, color='darkorange')

ax.set_xlabel('Water Usage (gal/MWh)', fontsize=12, fontweight='bold')
ax.set_ylabel('Annual Cost (Million USD/yr)', fontsize=12, fontweight='bold')
ax.set_title('Training: Water Usage Impact on Costs', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(water_cost_by_usage.index.astype(int))
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.2f}M',
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "01_water_usage_impact_training.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 01_water_usage_impact_training.png")
plt.close()

# ============================================================================
# Plot 2: Water Cost vs Water Usage (Inference)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

water_cost_by_usage_inf = results_df.groupby('water_use_gal_per_mwh')['water_cost_inference'].mean()
electricity_cost_by_usage_inf = results_df.groupby('water_use_gal_per_mwh')['electricity_cost_inference'].mean()

x = np.arange(len(water_cost_by_usage_inf))
bars1 = ax.bar(x - width/2, water_cost_by_usage_inf.values / 1e6, width, label='Water Cost', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, electricity_cost_by_usage_inf.values / 1e6, width, label='Electricity Cost', alpha=0.8, color='darkorange')

ax.set_xlabel('Water Usage (gal/MWh)', fontsize=12, fontweight='bold')
ax.set_ylabel('Annual Cost (Million USD/yr)', fontsize=12, fontweight='bold')
ax.set_title('Inference: Water Usage Impact on Costs', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(water_cost_by_usage_inf.index.astype(int))
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.2f}M',
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "02_water_usage_impact_inference.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 02_water_usage_impact_inference.png")
plt.close()

# ============================================================================
# Plot 3: Efficiency Impact on Water Costs (Training)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

efficiency_costs_train = results_df.groupby('compute_electrical_efficiency')['water_cost_training'].mean()
x = np.arange(len(efficiency_costs_train))

bars = ax.bar(x, efficiency_costs_train.values / 1e6, alpha=0.8, color='forestgreen', width=0.5)
ax.set_xlabel('Compute Electrical Efficiency', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Water Cost (Million USD/yr)', fontsize=12, fontweight='bold')
ax.set_title('Training: Efficiency Impact on Water Costs', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{e:.0%}' for e in efficiency_costs_train.index])
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'${height:.2f}M',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "03_efficiency_impact_training.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 03_efficiency_impact_training.png")
plt.close()

# ============================================================================
# Plot 4: Efficiency Impact on Water Costs (Inference)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

efficiency_costs_inf = results_df.groupby('compute_electrical_efficiency')['water_cost_inference'].mean()
x = np.arange(len(efficiency_costs_inf))

bars = ax.bar(x, efficiency_costs_inf.values / 1e6, alpha=0.8, color='forestgreen', width=0.5)
ax.set_xlabel('Compute Electrical Efficiency', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Water Cost (Million USD/yr)', fontsize=12, fontweight='bold')
ax.set_title('Inference: Efficiency Impact on Water Costs', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{e:.0%}' for e in efficiency_costs_inf.index])
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'${height:.2f}M',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "04_efficiency_impact_inference.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 04_efficiency_impact_inference.png")
plt.close()

# ============================================================================
# Plot 5: Heatmap - Water Cost vs Efficiency and Water Usage (Training)
# ============================================================================
# Create pivot table
pivot_train = results_df.pivot_table(
    values='water_cost_training',
    index='compute_electrical_efficiency',
    columns='water_use_gal_per_mwh',
    aggfunc='mean'
) / 1e6

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot_train, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax, 
            cbar_kws={'label': 'Water Cost (Million USD/yr)'})
ax.set_xlabel('Water Usage (gal/MWh)', fontsize=12, fontweight='bold')
ax.set_ylabel('Compute Electrical Efficiency', fontsize=12, fontweight='bold')
ax.set_title('Training: Water Cost Heatmap\n(Efficiency vs Water Usage)', fontsize=14, fontweight='bold')
ax.set_yticklabels([f'{y:.0%}' for y in pivot_train.index], rotation=0)

plt.tight_layout()
plt.savefig(output_dir / "05_heatmap_efficiency_water_usage_training.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 05_heatmap_efficiency_water_usage_training.png")
plt.close()

# ============================================================================
# Plot 6: Heatmap - Water Cost vs Efficiency and Water Usage (Inference)
# ============================================================================
pivot_inf = results_df.pivot_table(
    values='water_cost_inference',
    index='compute_electrical_efficiency',
    columns='water_use_gal_per_mwh',
    aggfunc='mean'
) / 1e6

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot_inf, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
            cbar_kws={'label': 'Water Cost (Million USD/yr)'})
ax.set_xlabel('Water Usage (gal/MWh)', fontsize=12, fontweight='bold')
ax.set_ylabel('Compute Electrical Efficiency', fontsize=12, fontweight='bold')
ax.set_title('Inference: Water Cost Heatmap\n(Efficiency vs Water Usage)', fontsize=14, fontweight='bold')
ax.set_yticklabels([f'{y:.0%}' for y in pivot_inf.index], rotation=0)

plt.tight_layout()
plt.savefig(output_dir / "06_heatmap_efficiency_water_usage_inference.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 06_heatmap_efficiency_water_usage_inference.png")
plt.close()

# ============================================================================
# Plot 7: Training vs Inference Cost Comparison
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Group by efficiency
efficiency_groups = results_df.groupby('compute_electrical_efficiency')[['water_cost_training', 'water_cost_inference']].mean() / 1e6

x = np.arange(len(efficiency_groups))
width = 0.35

bars1 = ax1.bar(x - width/2, efficiency_groups['water_cost_training'].values, width, 
               label='Training', alpha=0.8, color='steelblue')
bars2 = ax1.bar(x + width/2, efficiency_groups['water_cost_inference'].values, width,
               label='Inference', alpha=0.8, color='coral')

ax1.set_xlabel('Compute Electrical Efficiency', fontsize=12, fontweight='bold')
ax1.set_ylabel('Water Cost (Million USD/yr)', fontsize=12, fontweight='bold')
ax1.set_title('Training vs Inference Water Costs by Efficiency', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{e:.0%}' for e in efficiency_groups.index])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Difference plot
difference = (efficiency_groups['water_cost_inference'] - efficiency_groups['water_cost_training']).values
colors = ['green' if d >= 0 else 'red' for d in difference]
bars = ax2.bar(x, difference, alpha=0.8, color=colors, width=0.5)

ax2.set_xlabel('Compute Electrical Efficiency', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cost Difference (Million USD/yr)', fontsize=12, fontweight='bold')
ax2.set_title('Inference - Training Cost Difference', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{e:.0%}' for e in efficiency_groups.index])
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:.2f}M',
            ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "07_training_vs_inference_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 07_training_vs_inference_comparison.png")
plt.close()

# ============================================================================
# Plot 8: Cooling Load Impact (Training)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

cooling_costs = results_df.groupby('cooling_load_ratio')['water_cost_training'].mean() / 1e6
x = np.arange(len(cooling_costs))

bars = ax.bar(x, cooling_costs.values, alpha=0.8, color='mediumpurple', width=0.5)
ax.set_xlabel('Cooling Load Ratio', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Water Cost (Million USD/yr)', fontsize=12, fontweight='bold')
ax.set_title('Training: Cooling Load Ratio Impact on Water Costs', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cooling_costs.index)
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'${height:.2f}M',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "08_cooling_ratio_impact_training.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 08_cooling_ratio_impact_training.png")
plt.close()

# ============================================================================
# Plot 9: 3D Surface Plot - Water Cost as function of Efficiency and Water Usage
# ============================================================================
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 6))

# Training
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(
    pivot_train.columns.values[np.newaxis, :].repeat(len(pivot_train), axis=0),
    pivot_train.index.values[:, np.newaxis].repeat(len(pivot_train.columns), axis=1),
    pivot_train.values,
    cmap='viridis', alpha=0.8
)
ax1.set_xlabel('Water Usage (gal/MWh)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Efficiency', fontsize=10, fontweight='bold')
ax1.set_zlabel('Water Cost ($M/yr)', fontsize=10, fontweight='bold')
ax1.set_title('Training: Water Cost Surface', fontsize=12, fontweight='bold')

# Inference
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(
    pivot_inf.columns.values[np.newaxis, :].repeat(len(pivot_inf), axis=0),
    pivot_inf.index.values[:, np.newaxis].repeat(len(pivot_inf.columns), axis=1),
    pivot_inf.values,
    cmap='viridis', alpha=0.8
)
ax2.set_xlabel('Water Usage (gal/MWh)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Efficiency', fontsize=10, fontweight='bold')
ax2.set_zlabel('Water Cost ($M/yr)', fontsize=10, fontweight='bold')
ax2.set_title('Inference: Water Cost Surface', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "09_3d_surface_water_costs.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 09_3d_surface_water_costs.png")
plt.close()

# ============================================================================
# Plot 10: Summary Statistics Table as Image
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Create summary statistics
summary_data = []
for eff in sorted(results_df['compute_electrical_efficiency'].unique()):
    subset = results_df[results_df['compute_electrical_efficiency'] == eff]
    summary_data.append([
        f'{eff:.0%}',
        f"${subset['water_cost_training'].min()/1e6:.2f}M",
        f"${subset['water_cost_training'].max()/1e6:.2f}M",
        f"${subset['water_cost_training'].mean()/1e6:.2f}M",
        f"${subset['water_cost_inference'].min()/1e6:.2f}M",
        f"${subset['water_cost_inference'].max()/1e6:.2f}M",
        f"${subset['water_cost_inference'].mean()/1e6:.2f}M",
    ])

columns = ['Efficiency', 'Train Min', 'Train Max', 'Train Mean', 'Infer Min', 'Infer Max', 'Infer Mean']
table = ax.table(cellText=summary_data, colLabels=columns, cellLoc='center', loc='center',
                colWidths=[0.12, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data) + 1):
    for j in range(len(columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('#ffffff')

plt.title('Data Center Sweep Results - Summary Statistics by Efficiency', 
         fontsize=14, fontweight='bold', pad=20)
plt.savefig(output_dir / "10_summary_statistics_table.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 10_summary_statistics_table.png")
plt.close()

print("\n" + "="*60)
print("All plots saved to:", output_dir)
print("="*60)
print(f"\nGenerated {len(list(output_dir.glob('*.png')))} visualization plots!")
