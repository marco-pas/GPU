import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

mpl.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,   # x-axis tick labels
    "ytick.labelsize": 16,    # y-axis tick labels
    "legend.fontsize": 18,
})

def save_both(basename):
    """Save current figure as both PNG and PDF."""
    plt.savefig(f"{basename}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{basename}.pdf", bbox_inches="tight")



array_sizes = [
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
]

time_gpu_basic = [
    0.003932,
    0.004250,
    0.005006,
    0.008204,
    0.012616,
    0.022730,
    0.043382,
    0.084134,
    0.164763,
    0.354967,
]

time_gpu_shared = [
    0.003164,
    0.002893,
    0.003241,
    0.003107,
    0.003456,
    0.002894,
    0.003353,
    0.004151,
    0.004356,
    0.023836,
]

time_cpu = [
    0.000261,
    0.000533,
    0.001093,
    0.002211,
    0.004450,
    0.008927,
    0.017963,
    0.036283,
    0.072363,
    0.146700,
]


x = np.arange(len(array_sizes))                     # categorical positions 0..4
plt.figure(figsize=(9,5))
ax1 = plt.gca()
ax1.set_yscale('log')
bar_width = 0.1

ax1.bar(x - bar_width * 1.5, time_cpu,
        width=bar_width, label="CPU")
ax1.bar(x, time_gpu_basic,
        width=bar_width, label="GPU basic")
ax1.bar(x + bar_width* 1.5, time_gpu_shared,
        width=bar_width,label="GPU shared mem")

#num_ticks = 10
#tick_positions = np.linspace(0, len(x) - 1, num_ticks, dtype=int)
#ax1.set_xticks(tick_positions)
#ax1.set_xticklabels(tick_positions)
x_labels_pow2 = [rf"$2^{{{int(np.log2(v))}}}$" for v in array_sizes]
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels_pow2)
#ax1.set_xticklabels(array_sizes)
ax1.set_xlabel("Array size")
ax1.set_ylabel("Time [ms]")
ax1.set_title(f"Reduction time")
ax1.grid(axis="y", linestyle="--", alpha=0.7)
ax1.legend(loc="upper left")

plt.tight_layout()
save_both(f"reduction_time")

