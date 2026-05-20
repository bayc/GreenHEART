"""
PUE / WUE range sweep for the data center model.

Runs the H2Integrate data-center model over two sets of (PUE, WUE) pairs —
an efficient bound and an inefficient bound — for both the training and
inference compute workloads. Plots show shaded bands between the two bounds
for each metric.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from h2integrate.core.h2integrate_model import H2IntegrateModel

# ============================================================
# BOUND PAIRS  — each list defines one efficiency bound
# Pairs are matched by index: pair i in EFFICIENT corresponds to pair i in INEFFICIENT
# ============================================================
PUE_WUE_PAIRS_EFFICIENT = [
    (1.10, 0.07, 'large'),
    (1.08, 1.73, 'large'),
    (1.38, 1.05, 'midsize'),
    (1.27, 1.99, 'midsize'),
    # (1.46, 2.37, 'midsize'),
    # (1.38, 0.014, 'midsize'),
    # (1.46, 0.02, 'midsize'),
    # (1.66, 2.66, 'small'),
    # (1.62, 0.025, 'small'),
    # (1.86, 0.025, 'small'),
]

PUE_WUE_PAIRS_INEFFICIENT = [
    (1.20, 0.52, 'large'),
    (1.19, 2.60, 'large'),
    (1.65, 2.12, 'midsize'),
    (1.50, 2.98, 'midsize'),
    # (1.74, 3.54, 'midsize'),
    # (1.59, 0.055, 'midsize'),
    # (1.79, 0.10, 'midsize'),
    # (1.97, 4.01, 'small'),
    # (1.92, 0.103, 'small'),
    # (2.28, 0.103, 'small'),
]

assert len(PUE_WUE_PAIRS_EFFICIENT) == len(PUE_WUE_PAIRS_INEFFICIENT), (
    "EFFICIENT and INEFFICIENT pair lists must have the same length."
)

BOUNDS = {
    "efficient":   PUE_WUE_PAIRS_EFFICIENT,
    "inefficient": PUE_WUE_PAIRS_INEFFICIENT,
}

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
# RUN SWEEP
# ============================================================
records = []

n_pairs = len(PUE_WUE_PAIRS_EFFICIENT)
total_runs = len(WORKLOADS) * len(BOUNDS) * n_pairs
run = 0

for workload_name, it_workload_mw in WORKLOADS.items():
    print(f"\n=== Workload: {workload_name} ===")
    for bound_name, pairs in BOUNDS.items():
        print(f"  -- Bound: {bound_name} --")
        for pair_idx, (pue, wue) in enumerate(pairs):
            run += 1
            print(
                f"  [{run}/{total_runs}] pair={pair_idx}  PUE={pue:.3f}  WUE={wue:.3f} L/kWh",
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
            opex = float(h2i.prob.get_val("data_center_pue_wue.OpEx", units="USD/year")[0])

            electricity_rate = float(
                h2i.prob.get_val("data_center_pue_wue.electricity_rate")[0]
            )
            water_rate = float(
                h2i.prob.get_val("data_center_pue_wue.water_rate", units="USD/galUS")[0]
            )

            annual_power_mwh = float(facility_power_mw.sum() * (dt / 3600))
            annual_water_gal = float(water_gal_per_h.sum() * (dt / 3600))
            elec_cost = annual_power_mwh * 1000 * electricity_rate
            water_cost = annual_water_gal * water_rate

            records.append(
                {
                    "workload":  workload_name,
                    "bound":     bound_name,
                    "pair_idx":  pair_idx,
                    "pue":       pue,
                    "wue":       wue,
                    "annual_power_mwh":      annual_power_mwh,
                    "annual_water_mgal":     annual_water_gal / 1e6,
                    "electricity_cost_musd": elec_cost / 1e6,
                    "water_cost_musd":       water_cost / 1e6,
                    "opex_musd":             opex / 1e6,
                    "capex_usd":             capex,
                }
            )
            print("done")

results = pd.DataFrame(records)
results.to_csv("pue_wue_range_sweep_results.csv", index=False)
print("\nResults saved to pue_wue_range_sweep_results.csv")

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

x = np.arange(n_pairs)
pair_x_labels = [f"{i+1}" for i in range(n_pairs)]  # simple index labels on x-axis

WL_STYLES = {
    "training":  {"color": "#1f77b4", "fill_alpha": 0.25, "line_style": "-"},
    "inference": {"color": "#ff7f0e", "fill_alpha": 0.25, "line_style": "--"},
}

# ------------------------------------------------------------------
# Plot A: one figure per metric — shaded band between efficient and
#         inefficient bounds, one series per workload
# ------------------------------------------------------------------
fig_a, axes_a = plt.subplots(1, len(METRICS), figsize=(4.5 * len(METRICS), 5))
fig_a.suptitle("Efficient–Inefficient Range by Pair Index", fontsize=13)

for ax, (col, ylabel, _) in zip(axes_a, METRICS):
    for wl_name, style in WL_STYLES.items():
        eff = (
            results[(results["workload"] == wl_name) & (results["bound"] == "efficient")]
            .sort_values("pair_idx")[col]
            .to_numpy()
        )
        ineff = (
            results[(results["workload"] == wl_name) & (results["bound"] == "inefficient")]
            .sort_values("pair_idx")[col]
            .to_numpy()
        )
        lo = np.minimum(eff, ineff)
        hi = np.maximum(eff, ineff)

        ax.fill_between(x, lo, hi, color=style["color"], alpha=style["fill_alpha"])
        ax.plot(x, ineff, color=style["color"], ls=style["line_style"],
                marker="s", markersize=4, linewidth=1.5, alpha=0.6,
                label=f"{wl_name} inefficient")
        ax.plot(x, eff,   color=style["color"], ls=style["line_style"],
                marker="o", markersize=4, linewidth=1.5, label=f"{wl_name} efficient")
        

    ax.set_xticks(x)
    ax.set_xticklabels(pair_x_labels)
    ax.set_xlabel("Case #")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7)

fig_a.tight_layout()
fig_a.savefig("range_sweep_bands.png", dpi=150, bbox_inches="tight")
print("Saved range_sweep_bands.png")

# ------------------------------------------------------------------
# Plot B: power vs water scatter — efficient/inefficient pairs connected
#         by arrows to show the range, one colour per workload
# ------------------------------------------------------------------
fig_b, ax_b = plt.subplots(figsize=(8, 6))

for wl_name, style in WL_STYLES.items():
    eff_sub   = results[(results["workload"] == wl_name) & (results["bound"] == "efficient")].sort_values("pair_idx")
    ineff_sub = results[(results["workload"] == wl_name) & (results["bound"] == "inefficient")].sort_values("pair_idx")

    # Draw arrows from efficient → inefficient for each pair
    for (_, e_row), (_, i_row) in zip(eff_sub.iterrows(), ineff_sub.iterrows()):
        ax_b.annotate(
            "",
            xy=(i_row["annual_power_mwh"], i_row["annual_water_mgal"]),
            xytext=(e_row["annual_power_mwh"], e_row["annual_water_mgal"]),
            arrowprops=dict(
                arrowstyle="->",
                color=style["color"],
                alpha=0.5,
                lw=1.0,
            ),
        )

    ax_b.scatter(
        eff_sub["annual_power_mwh"],
        eff_sub["annual_water_mgal"],
        color=style["color"],
        marker="o",
        s=60,
        zorder=4,
        label=f"{wl_name} efficient",
    )
    ax_b.scatter(
        ineff_sub["annual_power_mwh"],
        ineff_sub["annual_water_mgal"],
        color=style["color"],
        marker="s",
        s=60,
        alpha=0.6,
        zorder=4,
        label=f"{wl_name} inefficient",
    )

ax_b.set_xlabel("Annual Facility Power (MWh)")
ax_b.set_ylabel("Annual Water Consumed (M gallons)")
ax_b.set_title("Power vs Water — arrows show efficient→inefficient range per pair")
ax_b.legend(fontsize=8)
ax_b.grid(True, alpha=0.4)
fig_b.tight_layout()
fig_b.savefig("range_sweep_power_vs_water.png", dpi=150, bbox_inches="tight")
print("Saved range_sweep_power_vs_water.png")

# ------------------------------------------------------------------
# Plot C: range bar chart — min/max error bars for each metric,
#         one subplot per workload
# ------------------------------------------------------------------
fig_c, axes_c = plt.subplots(
    len(WORKLOADS), len(METRICS),
    figsize=(4 * len(METRICS), 4 * len(WORKLOADS)),
    sharey="col",
)
fig_c.suptitle("Metric Range (efficient–inefficient) per Pair", fontsize=13)

for row_idx, wl_name in enumerate(WORKLOADS):
    eff_sub   = results[(results["workload"] == wl_name) & (results["bound"] == "efficient")].sort_values("pair_idx").reset_index(drop=True)
    ineff_sub = results[(results["workload"] == wl_name) & (results["bound"] == "inefficient")].sort_values("pair_idx").reset_index(drop=True)

    for col_idx, (col, ylabel, _) in enumerate(METRICS):
        ax = axes_c[row_idx, col_idx]
        lo = np.minimum(eff_sub[col].to_numpy(), ineff_sub[col].to_numpy())
        hi = np.maximum(eff_sub[col].to_numpy(), ineff_sub[col].to_numpy())
        mid = (lo + hi) / 2
        err = (hi - lo) / 2

        ax.bar(x, lo, width=0.6, color=WL_STYLES[wl_name]["color"], alpha=0.4, label="efficient")
        ax.bar(x, err, width=0.6, bottom=lo,
               color=WL_STYLES[wl_name]["color"], alpha=0.8, label="range to inefficient")

        ax.set_xticks(x)
        ax.set_xticklabels(pair_x_labels, fontsize=8)
        ax.set_xlabel("Case #")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{wl_name.capitalize()} — {ylabel}", fontsize=8)
        ax.grid(True, axis="y", alpha=0.4)
        if col_idx == 0:
            ax.legend(fontsize=7)

fig_c.tight_layout()
fig_c.savefig("range_sweep_range_bars.png", dpi=150, bbox_inches="tight")
print("Saved range_sweep_range_bars.png")

plt.show()
print("\nRange sweep complete.")
