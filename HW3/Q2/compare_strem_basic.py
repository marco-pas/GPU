import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

mpl.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

def save_both(basename):
    """Save current figure as both PNG and PDF."""
    plt.savefig(f"{basename}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{basename}.pdf", bbox_inches="tight")



# Vector lengths for each run (1..9)
vector_lengths = [
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
]

# CPU execution times (ms) per run
cpu_times = [
    0.0006012,
    0.0011342,
    0.0022524,
    0.004701,
    0.0120628,
    0.0239032,
    0.0484616,
    0.0984362,
    0.224036,
]

# GPU basic execution times (ms) per run
gpu_basic_times = [
    0.0196712,
    0.0224866,
    0.0293034,
    0.0451234,
    0.173073,
    0.15221,
    0.276978,
    0.511339,
    0.944627,
]

# Segment sizes present in the GPU-stream experiments
segment_sizes = [8192, 32768, 65536, 131072]

# GPU streams execution times (ms), grouped by segment size.
# Each inner list contains all measurements for that segment size across all runs.

gpu_stream_times_by_segment = [
    # segment size = 8192
    [
        0.0145414,  # run 1
        0.0183648,                                # run 2
        0.0348138,                                # run 3
        0.0642476,                                # run 4
        0.099424,                                 # run 5
        0.217437,                                 # run 6
        0.652576,                                 # run 7
        0.81816,                                  # run 8
        1.63249,                                  # run 9
    ],


    # segment size = 32768
    [
        np.nan,
        np.nan,
        0.0283556,          # run 3
        0.0376852,                                # run 4
        0.0761782,                                # run 5
        0.1384,                                   # run 6
        0.261042,                                 # run 7
        0.465679,                                 # run 8
        1.03946,                                  # run 9
    ],

    # segment size = 65536
    [
        np.nan,
        np.nan,
        np.nan,
        0.038102,                       # run 4
        0.0640932,                                # run 5
        0.123199,                                 # run 6
        0.219497,                                 # run 7
        0.423155,                                 # run 8
        0.786169,                                 # run 9
    ],

    # segment size = 131072
    [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0.073024,                                 # run 5
        0.126419,                                 # run 6
        0.221581,                                 # run 7
        0.38704,                                  # run 8
        0.780559,                                 # run 9
    ],
]


# --- Plot grouped histogram (bar chart) ---

x = np.arange(len(vector_lengths))  # positions 0..8
plt.figure(figsize=(10, 6))
ax1 = plt.gca()

# Optional: log-scale if you want (can comment out)
# ax1.set_yscale("log")

bar_width = 0.13

# Build series: first GPU basic, then each stream segment
series_labels = ["GPU basic"] + [f"GPU streams (seg={sz})" for sz in segment_sizes]
series_values = [gpu_basic_times] + gpu_stream_times_by_segment

n_series = len(series_labels)  # 1 + number of segment sizes

for s, (label, vals) in enumerate(zip(series_labels, series_values)):
    vals = np.asarray(vals, dtype=float)
    offset = (s - (n_series - 1) / 2.0) * bar_width

    # Mask out NaNs so we don't plot bars where there is no measurement
    mask = ~np.isnan(vals)
    ax1.bar(x[mask] + offset,
            vals[mask],
            width=bar_width,
            label=label)
ax1.set_yscale("log")
ax1.set_xticks(x)
ax1.set_xticklabels(vector_lengths, rotation=30)
ax1.set_xlabel("Array length")
ax1.set_ylabel("Time [ms]")
ax1.set_title("Vector add: GPU basic vs GPU streams")
ax1.grid(axis="y", linestyle="--", alpha=0.7)
ax1.legend(loc="upper left")

plt.tight_layout()
save_both("vectoradd_gpu_streams_vs_basic")
