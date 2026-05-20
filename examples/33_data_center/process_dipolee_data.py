"""
Process DIPOLee data center power profiles.

Loads raw Excel power data from three sources, computes daily and weekly
percentile bands (0/10/25/50/75/90/100), and plots the results.
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# SETUP
# ============================================================
t0 = time.time()

DATA_PATH_SOURCE1 = (
    "/home/cbay/code/data/Samples/NLR/colocation/"
)
DATA_PATH_SOURCE2 = (
    "/home/cbay/code/data/Samples/NLR/inference/"
)
DATA_PATH_SOURCE3 = (
    "/home/cbay/code/data/Samples/ORNL/"
)

SOURCE1_FILES = {
    # "Power1_1MW": "colocation_10MW_2840nodes_20u_power.xlsx",
    # "Power1_2MW": "colocation_10MW_2840nodes_40u_power.xlsx",
    # "Power1_3MW": "colocation_10MW_2840nodes_60u_power.xlsx",
    "Power1_4MW": "colocation_10MW_2840nodes_80u_power.xlsx",
}
SOURCE2_FILES = {
    # "Power2_1MW": "inference_1MW_283nodes_20u_power.xlsx",
    # "Power2_2MW": "inference_1MW_283nodes_40u_power.xlsx",
    # "Power2_3MW": "inference_1MW_283nodes_60u_power.xlsx",
    "Power2_4MW": "inference_1MW_283nodes_80u_power.xlsx",
}
SOURCE3_FILE = "87F42G2C+2QG3-124-131-124-131.xlsx"

# (raw_name, proc_name, tile_id)  tile_id is 1-based, matching MATLAB layout
DATASETS = [
    ("Power1_1MW", "Power1_1", 1),
    ("Power1_2MW", "Power1_2", 4),
    ("Power1_3MW", "Power1_3", 7),
    ("Power1_4MW", "Power1_4", 10),
    ("Power2_1MW", "Power2_1", 2),
    ("Power2_2MW", "Power2_2", 5),
    ("Power2_3MW", "Power2_3", 8),
    ("Power2_4MW", "Power2_4", 11),
    ("Power3_1MW", "Power3_1", 3),
    ("Power3_2MW", "Power3_2", 6),
]

print(f"1. Setup complete ... {time.time() - t0:.2f}s")

# ============================================================
# LOAD DC RAW DATA
# ============================================================

weekly_raw: dict[str, np.ndarray] = {}

# Source 1 — column index 8 (0-based), which is column 9 in MATLAB
for raw_name, filename in SOURCE1_FILES.items():
    df = pd.read_excel(DATA_PATH_SOURCE1 + filename)
    weekly_raw[raw_name] = df.iloc[:, 8].to_numpy(dtype=float)

# Source 2 — column index 8
for raw_name, filename in SOURCE2_FILES.items():
    df = pd.read_excel(DATA_PATH_SOURCE2 + filename)
    weekly_raw[raw_name] = df.iloc[:, 8].to_numpy(dtype=float)

# Source 3 — two sheets with different column indices
df3_1 = pd.read_excel(DATA_PATH_SOURCE3 + SOURCE3_FILE, sheet_name="Data Set 1")
weekly_raw["Power3_1MW"] = df3_1.iloc[:, 10].to_numpy(dtype=float)  # column 11 in MATLAB

df3_2 = pd.read_excel(DATA_PATH_SOURCE3 + SOURCE3_FILE, sheet_name="Data Set 2")
weekly_raw["Power3_2MW"] = df3_2.iloc[:, 1].to_numpy(dtype=float)   # column 2 in MATLAB

print(f"2. Data loading (Raw) complete ... {time.time() - t0:.2f}s")

# ============================================================
# FORMAT AND TRIM DATA
# ============================================================

MINUTES_PER_DAY = 1440
MINUTES_PER_WEEK = 10080

PERCENTILES = [0, 10, 25, 50, 75, 90, 100]
PCTILE_KEYS = ["p0", "p10", "p25", "p50", "p75", "p90", "p100"]


def compute_percentiles(power: np.ndarray, period: int) -> dict[str, np.ndarray]:
    """Reshape *power* into columns of *period* rows and compute row-wise percentiles."""
    n = (len(power) // period) * period
    mat = power[:n].reshape(period, -1, order="F")  # (period, n_periods)
    return {key: np.percentile(mat, pct, axis=1) for key, pct in zip(PCTILE_KEYS, PERCENTILES)}


daily: dict[str, dict] = {}
weekly: dict[str, dict] = {}

for raw_name, proc_name, _ in DATASETS:
    power = weekly_raw[raw_name]
    daily[proc_name] = compute_percentiles(power, MINUTES_PER_DAY)
    weekly[proc_name] = compute_percentiles(power, MINUTES_PER_WEEK)

# Time axes
t_daily = np.arange(MINUTES_PER_DAY) / 60          # hours  0–24
t_weekly = np.arange(MINUTES_PER_WEEK) / MINUTES_PER_DAY  # days  0–7

print(f"3. Data formatting and trimming complete ... {time.time() - t0:.2f}s")

# ============================================================
# HELPER: PLOT PERCENTILE PANEL
# ============================================================

FILL_COLORS = {
    "band_0_100": (0.90, 0.90, 0.90),
    "band_10_90": (0.75, 0.85, 0.95),
    "band_25_75": (0.50, 0.70, 0.90),
}


def plot_percentile_panel(ax: plt.Axes, t: np.ndarray, S: dict) -> list:
    """Fill percentile bands and plot the median on *ax*.

    Returns handles in legend order: [median, 25-75, 10-90, 0-100].
    """
    h1 = ax.fill_between(t, S["p0"],  S["p100"], color=FILL_COLORS["band_0_100"],  label="0–100%")
    h2 = ax.fill_between(t, S["p10"], S["p90"],  color=FILL_COLORS["band_10_90"],  label="10–90%")
    h3 = ax.fill_between(t, S["p25"], S["p75"],  color=FILL_COLORS["band_25_75"],  label="25–75%")
    (h4,) = ax.plot(t, S["p50"], color="blue", linewidth=1, label="50%")
    return [h4, h3, h2, h1]  # legend order matches MATLAB


# ============================================================
# FIGURE 1 — DAILY PROFILES
# ============================================================

fig1, axes1 = plt.subplots(4, 3, figsize=(18, 12), constrained_layout=True)
fig1.suptitle("Daily Power Profiles", fontsize=14)

legend_handles = None

for raw_name, proc_name, tile_id in DATASETS:
    # Convert 1-based tile_id to 0-based (row, col) for a 4×3 grid
    row, col = divmod(tile_id - 1, 3)
    ax = axes1[row, col]
    ax.grid(True)

    S = daily[proc_name]
    handles = plot_percentile_panel(ax, t_daily, S)

    if legend_handles is None:
        legend_handles = handles

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Power (MW)")
    ax.set_title(proc_name)
    ax.set_xticks(range(0, 25))

# Hide any unused subplots (tile positions not covered by datasets)
used_tiles = {tile_id for _, _, tile_id in DATASETS}
for tile_id in range(1, 13):
    if tile_id not in used_tiles:
        row, col = divmod(tile_id - 1, 3)
        axes1[row, col].set_visible(False)

fig1.legend(
    legend_handles,
    ["50%", "25–75%", "10–90%", "0–100%"],
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.02),
)

# ============================================================
# FIGURE 2 — WEEKLY PROFILES
# ============================================================

fig2, axes2 = plt.subplots(4, 3, figsize=(18, 12), constrained_layout=True)
fig2.suptitle("Weekly Power Profiles", fontsize=14)

legend_handles = None

for raw_name, proc_name, tile_id in DATASETS:
    row, col = divmod(tile_id - 1, 3)
    ax = axes2[row, col]
    ax.grid(True)

    S = weekly[proc_name]
    handles = plot_percentile_panel(ax, t_weekly, S)

    if legend_handles is None:
        legend_handles = handles

    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Power (MW)")
    ax.set_title(proc_name)
    ax.set_xticks(range(0, 8))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"])

for tile_id in range(1, 13):
    if tile_id not in used_tiles:
        row, col = divmod(tile_id - 1, 3)
        axes2[row, col].set_visible(False)

fig2.legend(
    legend_handles,
    ["50%", "25–75%", "10–90%", "0–100%"],
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.02),
)

print(f"4. Plots complete ... {time.time() - t0:.2f}s")

plt.show()
