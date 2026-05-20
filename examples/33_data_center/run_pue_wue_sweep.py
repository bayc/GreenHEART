"""
PUE / WUE parameter sweep for the data center model.

Runs the H2Integrate data-center model over a list of specific (PUE, WUE)
pairs and produces comparison plots of:
  - Annual facility power consumed (MWh)
  - Annual water consumed (million gallons)
  - Annual electricity cost ($M)
  - Annual water cost ($M)
  - Total annual OpEx ($M)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from h2integrate.core.h2integrate_model import H2IntegrateModel

# ============================================================
# SWEEP PARAMETERS  — add / remove (pue, wue) pairs here
# ============================================================
# Each tuple is (PUE, WUE [L/kWh])
# PUE_WUE_PAIRS = [
#     (1.1, 0.0),
#     (1.2, 0.3),
#     (1.3, 0.5),
#     (1.4, 0.7),
#     (1.5, 0.7),
#     (1.6, 1.0),
#     (1.8, 1.0),
#     (2.0, 1.5),
#     (2.5, 2.0),
# ]

PUE_WUE_PAIRS_EFFICIENT = [
    (1.10, 0.07),
    (1.08, 1.73),
    (1.38, 1.05),
    (1.27, 1.99),
    (1.46, 2.37),
    (1.38, 0.014),
    (1.46, 0.02),
    (1.66, 2.66),
    (1.62, 0.025),
    (1.86, 0.025),
]

PUE_WUE_PAIRS_INEFFICIENT = [
    (1.20, 0.52),
    (1.19, 2.60),
    (1.65, 2.12),
    (1.50, 2.98),
    (1.74, 3.54),
    (1.59, 0.055),
    (1.79, 0.10),
    (1.97, 4.01),
    (1.92, 0.103),
    (2.28, 0.103),
]

CONFIG_FILE = "data_center_pue_wue.yaml"

# ============================================================
# LOAD IT WORKLOADS
# ============================================================
dt = 3600  # seconds per timestep

df_train = pd.read_csv("10mw_80per_compute_load_training.csv")
df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
df_train.set_index("timestamp", inplace=True)
workload_training = (df_train["power_W"].resample("h").mean() * 10 / 1e6)[:-1].to_numpy()

df_infer = pd.read_csv("1mw_80per_compute_load_inference.csv")
df_infer["timestamp"] = pd.to_datetime(df_infer["timestamp"])
df_infer.set_index("timestamp", inplace=True)
workload_inference = (df_infer["power_W"].resample("h").mean() * 100 / 1e6).to_numpy()

WORKLOADS = {
    "training":  workload_training,
    "inference": workload_inference,
}

for name, wl in WORKLOADS.items():
    print(f"Workload '{name}': {wl.mean():.2f} MW mean, {len(wl)} hours")

# ============================================================
# RUN SWEEP  (over both workloads)
# ============================================================
records = []

total_runs = len(PUE_WUE_PAIRS) * len(WORKLOADS)
run = 0
for workload_name, it_workload_mw in WORKLOADS.items():
    print(f"\n--- Workload: {workload_name} ---")
    for pue, wue in PUE_WUE_PAIRS:
        run += 1
        print(f"  [{run}/{total_runs}] PUE={pue:.2f}  WUE={wue:.2f} L/kWh", end="  ... ", flush=True)

        h2i = H2IntegrateModel(CONFIG_FILE)
        h2i.setup()

        # Override PUE and WUE
        h2i.prob.set_val("data_center_pue_wue.pue", pue)
        h2i.prob.set_val("data_center_pue_wue.wue", wue)
        h2i.prob.set_val("data_center_pue_wue.compute_it_workload", it_workload_mw, units="MW")

        h2i.run()

        # --- extract results ---
        facility_power_mw = h2i.prob.get_val(
            "data_center_pue_wue.total_facility_power", units="MW"
        )
        water_gal_per_h = h2i.prob.get_val(
            "data_center_pue_wue.water_consumed", units="galUS/h"
        )
        capex = float(h2i.prob.get_val("data_center_pue_wue.CapEx", units="USD")[0])
        opex = float(h2i.prob.get_val("data_center_pue_wue.OpEx", units="USD/year")[0])

        electricity_rate = float(h2i.prob.get_val("data_center_pue_wue.electricity_rate")[0])
        water_rate = float(h2i.prob.get_val("data_center_pue_wue.water_rate", units="USD/galUS")[0])

        annual_power_mwh = float(facility_power_mw.sum() * (dt / 3600))
        annual_water_gal = float(water_gal_per_h.sum() * (dt / 3600))

        elec_cost = annual_power_mwh * 1000 * electricity_rate   # MWh → kWh → USD
        water_cost = annual_water_gal * water_rate

        records.append(
            {
                "workload": workload_name,
                "pue": pue,
                "wue": wue,
                "annual_power_mwh": annual_power_mwh,
                "annual_water_mgal": annual_water_gal / 1e6,
                "electricity_cost_musd": elec_cost / 1e6,
                "water_cost_musd": water_cost / 1e6,
                "opex_musd": opex / 1e6,
                "capex_usd": capex,
            }
        )
        print("done")

results = pd.DataFrame(records)
results.to_csv("pue_wue_sweep_results.csv", index=False)
print("\nResults saved to pue_wue_sweep_results.csv")

for wl_name, sub in results.groupby("workload"):
    sub.to_csv(f"pue_wue_sweep_results_{wl_name}.csv", index=False)
    print(f"Saved pue_wue_sweep_results_{wl_name}.csv")

# ============================================================
# PLOTS
# ============================================================
METRICS = [
    ("annual_power_mwh",      "Annual Facility Power (MWh)",      "MWh"),
    ("annual_water_mgal",     "Annual Water Consumed (M gallons)", "M gal"),
    ("electricity_cost_musd", "Annual Electricity Cost ($M)",      "$M"),
    ("water_cost_musd",       "Annual Water Cost ($M)",            "$M"),
    ("opex_musd",             "Annual OpEx ($M)",                  "$M"),
]

WORKLOAD_NAMES = list(WORKLOADS.keys())
CMAP = plt.colormaps["tab10"]
WL_COLORS = {name: CMAP(i / len(WORKLOAD_NAMES)) for i, name in enumerate(WORKLOAD_NAMES)}
WL_HATCHES = {"training": "", "inference": "//"}

# Build a short label per pair (same for both workloads)
pair_labels = [f"PUE={p:.2g}\nWUE={w:.2g}" for p, w in PUE_WUE_PAIRS]
n_pairs = len(PUE_WUE_PAIRS)
x = np.arange(n_pairs)
bar_width = 0.35

# ------------------------------------------------------------------
# Plot A: grouped bar chart — one subplot per metric, grouped by pair
#         with one bar per workload within each pair
# ------------------------------------------------------------------
fig_a, axes_a = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 5), sharey=False)
fig_a.suptitle("Results by (PUE, WUE) Pair — Training vs Inference", fontsize=13)

for ax, (col, ylabel, _) in zip(axes_a, METRICS):
    for i, wl_name in enumerate(WORKLOAD_NAMES):
        sub = results[results["workload"] == wl_name].reset_index(drop=True)
        offset = (i - (len(WORKLOAD_NAMES) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            sub[col],
            width=bar_width,
            label=wl_name,
            color=WL_COLORS[wl_name],
            hatch=WL_HATCHES.get(wl_name, ""),
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontsize=9)
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(fontsize=8)

fig_a.tight_layout()
fig_a.savefig("sweep_bar_chart.png", dpi=150, bbox_inches="tight")
print("Saved sweep_bar_chart.png")

# ------------------------------------------------------------------
# Plot B: scatter — power vs water, one series per workload
# ------------------------------------------------------------------
fig_b, ax_b = plt.subplots(figsize=(8, 5))
for wl_name in WORKLOAD_NAMES:
    sub = results[results["workload"] == wl_name]
    sc = ax_b.scatter(
        sub["annual_power_mwh"],
        sub["annual_water_mgal"],
        c=sub["pue"],
        s=50 + sub["wue"] * 80,
        cmap="plasma",
        vmin=results["pue"].min(),
        vmax=results["pue"].max(),
        edgecolors=WL_COLORS[wl_name],
        linewidths=2,
        zorder=3,
        label=wl_name,
        marker="o" if wl_name == "training" else "s",
    )
    for _, row in sub.iterrows():
        ax_b.annotate(
            f"({row['pue']:.2g},{row['wue']:.2g})",
            xy=(row["annual_power_mwh"], row["annual_water_mgal"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            color=WL_COLORS[wl_name],
        )

plt.colorbar(sc, ax=ax_b, label="PUE")
ax_b.set_xlabel("Annual Facility Power (MWh)")
ax_b.set_ylabel("Annual Water Consumed (M gallons)")
ax_b.set_title("Power vs Water — colour=PUE, size∝WUE, shape=workload")
ax_b.legend()
ax_b.grid(True, alpha=0.4)
fig_b.tight_layout()
fig_b.savefig("sweep_power_vs_water.png", dpi=150, bbox_inches="tight")
print("Saved sweep_power_vs_water.png")

# ------------------------------------------------------------------
# Plot C: stacked cost bars — one figure per workload, side by side
# ------------------------------------------------------------------
fig_c, axes_c = plt.subplots(1, len(WORKLOAD_NAMES),
                              figsize=(max(7, n_pairs * 0.9) * len(WORKLOAD_NAMES), 5),
                              sharey=True)
fig_c.suptitle("Stacked Annual OpEx by (PUE, WUE) Pair", fontsize=13)

for ax, wl_name in zip(axes_c, WORKLOAD_NAMES):
    sub = results[results["workload"] == wl_name].reset_index(drop=True)
    fixed_om = sub["opex_musd"] - sub["electricity_cost_musd"] - sub["water_cost_musd"]
    ax.bar(x, sub["electricity_cost_musd"], label="Electricity", color="#4C72B0")
    ax.bar(x, sub["water_cost_musd"],       label="Water",       color="#55A868",
           bottom=sub["electricity_cost_musd"])
    ax.bar(x, fixed_om,                     label="Fixed O&M",   color="#C44E52",
           bottom=sub["electricity_cost_musd"] + sub["water_cost_musd"])
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel("Annual Cost ($M)")
    ax.set_title(wl_name.capitalize())
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.4)

fig_c.tight_layout()
fig_c.savefig("sweep_stacked_costs.png", dpi=150, bbox_inches="tight")
print("Saved sweep_stacked_costs.png")

plt.show()
print("\nSweep complete.")
print("\nSweep complete.")
