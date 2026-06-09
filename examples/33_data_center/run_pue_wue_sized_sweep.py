"""
PUE / WUE range sweep with size-dependent workload scaling.

Each (PUE, WUE, size) tuple specifies the data center size class, which
determines how the base IT workload CSV is scaled before being passed to
the model:
    large   -> scale factor 1.0   (full dataset)
    midsize -> scale factor 0.2
    small   -> scale factor 0.01

Runs the sweep for two efficiency bounds (efficient / inefficient) and both
the training and inference compute workload profiles, then produces range-band
plots comparing the results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from h2integrate.core.h2integrate_model import H2IntegrateModel

# ============================================================
# SIZE SCALE FACTORS
# ============================================================
SIZE_SCALE = {
    "large":   1.0,
    "midsize": 0.2,
    # "small":   0.01,
}

SIZE_COLORS = {
    "large":   "#1f77b4",
    "midsize": "#ff7f0e",
    # "small":   "#2ca02c",
}

SIZE_MARKERS = {
    "large":   "o",
    "midsize": "s",
    # "small":   "^",
}

# ============================================================
# (PUE, WUE, size) PAIRS
# ============================================================
PUE_WUE_PAIRS_EFFICIENT = [
    (1.10, 0.07,  "large"),
    (1.08, 1.73,  "large"),
    (1.38, 1.05,  "midsize"),
    (1.27, 1.99,  "midsize"),
    # (1.46, 2.37,  "midsize"),
    # (1.38, 0.014, "midsize"),
    # (1.46, 0.02,  "midsize"),
    # (1.66, 2.66,  "small"),
    # (1.62, 0.025, "small"),
    # (1.86, 0.025, "small"),
]

PUE_WUE_PAIRS_INEFFICIENT = [
    (1.20, 0.52,  "large"),
    (1.19, 2.60,  "large"),
    (1.65, 2.12,  "midsize"),
    (1.50, 2.98,  "midsize"),
    # (1.74, 3.54,  "midsize"),
    # (1.59, 0.055, "midsize"),
    # (1.79, 0.10,  "midsize"),
    # (1.97, 4.01,  "small"),
    # (1.92, 0.103, "small"),
    # (2.28, 0.103, "small"),
]

assert len(PUE_WUE_PAIRS_EFFICIENT) == len(PUE_WUE_PAIRS_INEFFICIENT), (
    "EFFICIENT and INEFFICIENT lists must have the same length."
)

BOUNDS = {
    "efficient":   PUE_WUE_PAIRS_EFFICIENT,
    "inefficient": PUE_WUE_PAIRS_INEFFICIENT,
}

CONFIG_FILE = "data_center_pue_wue.yaml"

# ============================================================
# LOAD BASE IT WORKLOADS  (unscaled)
# ============================================================
dt = 3600  # seconds per timestep

df_train = pd.read_csv("10mw_80per_compute_load_training.csv")
df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
df_train.set_index("timestamp", inplace=True)
base_training = (df_train["power_W"].resample("h").mean() * 10 / 1e6)[:-1].to_numpy()

df_infer = pd.read_csv("1mw_80per_compute_load_inference.csv")
df_infer["timestamp"] = pd.to_datetime(df_infer["timestamp"])
df_infer.set_index("timestamp", inplace=True)
base_inference = (df_infer["power_W"].resample("h").mean() * 100 / 1e6).to_numpy()

BASE_WORKLOADS = {
    "training":  base_training,
    "inference": base_inference,
}

for name, wl in BASE_WORKLOADS.items():
    print(f"Base workload '{name}': {wl.mean():.3f} MW mean (unscaled), {len(wl)} hours")

# ============================================================
# RUN SWEEP
# ============================================================
records = []

n_pairs = len(PUE_WUE_PAIRS_EFFICIENT)
total_runs = len(BASE_WORKLOADS) * len(BOUNDS) * n_pairs
run = 0

for workload_name, base_wl in BASE_WORKLOADS.items():
    print(f"\n=== Workload: {workload_name} ===")
    for bound_name, pairs in BOUNDS.items():
        print(f"  -- Bound: {bound_name} --")
        for pair_idx, (pue, wue, size) in enumerate(pairs):
            run += 1
            scale = SIZE_SCALE[size]
            it_workload_mw = base_wl * scale

            print(
                f"  [{run}/{total_runs}] pair={pair_idx}  size={size}  "
                f"scale={scale}  PUE={pue:.3f}  WUE={wue:.3f} L/kWh",
                end="  ... ",
                flush=True,
            )

            h2i = H2IntegrateModel(CONFIG_FILE)
            h2i.setup()

            h2i.prob.set_val("data_center_pue_wue.pue", pue)
            h2i.prob.set_val("data_center_pue_wue.wue", wue)
            h2i.prob.set_val(
                "data_center_pue_wue.compute_it_workload", it_workload_mw, units="MW"
            )

            h2i.run()

            facility_power_mw = h2i.prob.get_val(
                "data_center_pue_wue.total_facility_power", units="MW"
            )
            water_gal_per_h = h2i.prob.get_val(
                "data_center_pue_wue.water_consumed", units="galUS/h"
            )
            capex = float(h2i.prob.get_val("data_center_pue_wue.CapEx", units="USD")[0])
            opex  = float(h2i.prob.get_val("data_center_pue_wue.OpEx",  units="USD/year")[0])

            electricity_rate = float(
                h2i.prob.get_val("data_center_pue_wue.electricity_rate")[0]
            )
            water_rate = float(
                h2i.prob.get_val("data_center_pue_wue.water_rate", units="USD/galUS")[0]
            )

            annual_power_mwh = float(facility_power_mw.sum() * (dt / 3600))
            annual_water_gal = float(water_gal_per_h.sum() * (dt / 3600))
            elec_cost  = annual_power_mwh * 1000 * electricity_rate
            water_cost = annual_water_gal * water_rate

            records.append(
                {
                    "workload":              workload_name,
                    "bound":                 bound_name,
                    "pair_idx":              pair_idx,
                    "size":                  size,
                    "scale":                 scale,
                    "pue":                   pue,
                    "wue":                   wue,
                    "annual_power_mwh":      annual_power_mwh,
                    "annual_water_mgal":     annual_water_gal / 1e6,
                    "electricity_cost_musd": elec_cost  / 1e6,
                    "water_cost_musd":       water_cost / 1e6,
                    "opex_musd":             opex / 1e6,
                    "capex_usd":             capex,
                }
            )
            print("done")

results = pd.DataFrame(records)
results.to_csv("pue_wue_sized_sweep_results.csv", index=False)
print("\nResults saved to pue_wue_sized_sweep_results.csv")

# ============================================================
# PLOTS
# ============================================================
METRICS = [
    ("annual_power_mwh",      "Annual Facility Power (MWh)",      "MWh"),
    ("annual_water_mgal",     "Annual Water Consumed (M gallons)", "M gal"),
    # ("electricity_cost_musd", "Annual Electricity Cost ($M)",      "$M"),
    # ("water_cost_musd",       "Annual Water Cost ($M)",            "$M"),
    # ("opex_musd",             "Annual OpEx ($M)",                  "$M"),
]

WORKLOAD_NAMES = list(BASE_WORKLOADS.keys())
SIZE_CLASSES   = list(SIZE_SCALE.keys())  # large, midsize, small

WL_LINESTYLES = {"training": "-", "inference": "--"}
x = np.arange(n_pairs) + 1

# ------------------------------------------------------------------
# Plot A: range bands per metric
#   rows = workloads, cols = metrics
#   shaded band between efficient and inefficient bounds
#   line colour = size class
# ------------------------------------------------------------------
fig_a, axes_a = plt.subplots(
    len(WORKLOAD_NAMES), len(METRICS),
    figsize=(4.5 * len(METRICS), 4.5 * len(WORKLOAD_NAMES)),
    sharey="col",
)
fig_a.suptitle("Efficient–Inefficient Range by Pair (colour = size class)", fontsize=13)

for row_i, wl_name in enumerate(WORKLOAD_NAMES):
    for col_i, (col, ylabel, _) in enumerate(METRICS):
        ax = axes_a[row_i, col_i]

        eff_all   = results[(results["workload"] == wl_name) & (results["bound"] == "efficient")].sort_values("pair_idx")
        ineff_all = results[(results["workload"] == wl_name) & (results["bound"] == "inefficient")].sort_values("pair_idx")

        # Plot each size class with its own colour
        for size in SIZE_CLASSES:
            idx_mask = eff_all["size"] == size
            xi       = eff_all.loc[idx_mask, "pair_idx"].to_numpy()
            eff_vals = eff_all.loc[idx_mask, col].to_numpy()
            ineff_vals = ineff_all.loc[ineff_all["size"] == size, col].to_numpy()

            lo = np.minimum(eff_vals, ineff_vals)
            hi = np.maximum(eff_vals, ineff_vals)

            ax.fill_between(xi, lo, hi,
                            color=SIZE_COLORS[size], alpha=0.2)
            ax.plot(xi, ineff_vals, color=SIZE_COLORS[size],
                    ls=":",  marker=SIZE_MARKERS[size], markersize=5,
                    linewidth=1.5, alpha=0.7, label=f"{size} ineff.")
            ax.plot(xi, eff_vals,   color=SIZE_COLORS[size],
                    ls="-",  marker=SIZE_MARKERS[size], markersize=5,
                    linewidth=1.5, label=f"{size} eff.")

        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x], fontsize=8)
        ax.set_xlabel("Case index")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{wl_name.capitalize()} — {ylabel}", fontsize=8)
        ax.grid(True, alpha=0.4)
        if col_i == 0:
            ax.legend(fontsize=6, ncol=2)

fig_a.tight_layout()
fig_a.savefig("sized_sweep_range_bands.png", dpi=150, bbox_inches="tight")
print("Saved sized_sweep_range_bands.png")

# ------------------------------------------------------------------
# Plot B: power vs water scatter with arrows (efficient → inefficient)
#   one subplot per workload, colour = size class
# ------------------------------------------------------------------
fig_b, axes_b = plt.subplots(1, len(WORKLOAD_NAMES),
                              figsize=(7 * len(WORKLOAD_NAMES), 6))
fig_b.suptitle("Power vs Water — efficient→inefficient arrows (colour = size class)", fontsize=13)

for ax, wl_name in zip(axes_b, WORKLOAD_NAMES):
    eff_sub   = results[(results["workload"] == wl_name) & (results["bound"] == "efficient")].sort_values("pair_idx")
    ineff_sub = results[(results["workload"] == wl_name) & (results["bound"] == "inefficient")].sort_values("pair_idx")

    for size in SIZE_CLASSES:
        e_rows = eff_sub[eff_sub["size"] == size]
        i_rows = ineff_sub[ineff_sub["size"] == size]

        for (_, e), (_, i) in zip(e_rows.iterrows(), i_rows.iterrows()):
            ax.annotate(
                "",
                xy=(i["annual_power_mwh"], i["annual_water_mgal"]),
                xytext=(e["annual_power_mwh"], e["annual_water_mgal"]),
                arrowprops=dict(arrowstyle="->", color=SIZE_COLORS[size],
                                alpha=0.55, lw=1.2),
            )

        ax.scatter(e_rows["annual_power_mwh"], e_rows["annual_water_mgal"],
                   color=SIZE_COLORS[size], marker=SIZE_MARKERS[size],
                   s=70, zorder=4, label=f"{size} efficient")
        ax.scatter(i_rows["annual_power_mwh"], i_rows["annual_water_mgal"],
                   color=SIZE_COLORS[size], marker=SIZE_MARKERS[size],
                   s=70, alpha=0.5, zorder=4, label=f"{size} inefficient",
                   edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Annual Facility Power (MWh)")
    ax.set_ylabel("Annual Water Consumed (M gallons)")
    ax.set_title(wl_name.capitalize())
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.4)

fig_b.tight_layout()
fig_b.savefig("sized_sweep_power_vs_water.png", dpi=150, bbox_inches="tight")
print("Saved sized_sweep_power_vs_water.png")

# ------------------------------------------------------------------
# Plot C: grouped bar chart — efficient vs inefficient per metric
#   rows = workloads, cols = metrics, bars grouped by size class
# ------------------------------------------------------------------
n_sizes  = len(SIZE_CLASSES)
bar_w    = 0.35
offsets  = np.linspace(-(n_sizes - 1) / 2, (n_sizes - 1) / 2, n_sizes) * bar_w

fig_c, axes_c = plt.subplots(
    len(WORKLOAD_NAMES), len(METRICS),
    figsize=(4.5 * len(METRICS), 4 * len(WORKLOAD_NAMES)),
    sharey="col",
)
fig_c.suptitle("Efficient vs Inefficient Results (grouped by size class)", fontsize=13)

bound_hatches = {"efficient": "", "inefficient": "//"}

bar_x_labels = [
    "Case 1:\nLarge -\nAir +\nWater Cooled",
    "Case 2:\nLarge -\nWater Cooled",
    "Case 3:\nMedium -\nAir +\nWater Cooled",
    "Case 4:\nMedium -\nWater Cooled",
]

for row_i, wl_name in enumerate(WORKLOAD_NAMES):
    for col_i, (col, ylabel, _) in enumerate(METRICS):
        ax = axes_c[row_i, col_i]
        # Collect pair indices for each size class
        size_indices = {s: [i for i, (_, _, sz) in enumerate(PUE_WUE_PAIRS_EFFICIENT) if sz == s]
                        for s in SIZE_CLASSES}

        bar_x = np.arange(n_pairs)
        for s_i, size in enumerate(SIZE_CLASSES):
            for b_name, hatch in bound_hatches.items():
                sub = (
                    results[
                        (results["workload"] == wl_name) &
                        (results["bound"]    == b_name)
                    ]
                    .sort_values("pair_idx")
                    .reset_index(drop=True)
                )
                vals = sub[col].to_numpy()
                xi   = bar_x[size_indices[size]] + offsets[s_i]
                ax.bar(
                    xi,
                    vals[size_indices[size]],
                    width=bar_w,
                    color=SIZE_COLORS[size],
                    # hatch="//" if b_name == "inefficient" else "",
                    alpha=0.45 if b_name == "inefficient" else 1.0,
                    edgecolor="grey",
                    linewidth=0.4,
                    label=f"{size} {b_name}" if col_i == 0 else "_nolegend_",
                )

        ax.set_xticks(bar_x)
        ax.set_xticklabels(bar_x_labels, fontsize=8)
        # ax.set_xticklabels([str(i + 1) for i in bar_x], fontsize=8)
        ax.set_xlabel("Case")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{wl_name.capitalize()} — {ylabel}", fontsize=8)
        ax.grid(True, axis="y", alpha=0.4)
        if col_i == 0:
            ax.legend(fontsize=6, ncol=2)

fig_c.tight_layout()
fig_c.savefig("sized_sweep_bar_chart.png", dpi=150, bbox_inches="tight")
print("Saved sized_sweep_bar_chart.png")

plt.show()
print("\nSized sweep complete.")
