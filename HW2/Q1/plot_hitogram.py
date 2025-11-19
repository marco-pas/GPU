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

ARRAY_LEN = [1024, 10240, 102400, 1024000]

for array in ARRAY_LEN:

    load_file = f"/home/pennati/phd/applied_gpu/GPU/HW2/Q1/histogram_{array}_bins_4096_uniform.txt"
    with open(load_file, "r") as f:
        lines = [int(line.strip()) for line in f if line.strip()]
    histogram_uniform = lines[:]
    load_file = f"/home/pennati/phd/applied_gpu/GPU/HW2/Q1/histogram_{array}_bins_4096_normal.txt"
    with open(load_file, "r") as f:
        lines = [int(line.strip()) for line in f if line.strip()]
    histogram_normal = lines[:]
    threadsPerBlock = 256
    blocksHist = (array + threadsPerBlock - 1) // threadsPerBlock
    print(f"Array size={array}, threadsPerBlock={threadsPerBlock}, blocksHist={blocksHist}")
    
    x = np.arange(len(histogram_uniform))                     # categorical positions 0..4
    plt.figure(figsize=(9,5))
    ax1 = plt.gca()
    #ax1.set_yscale('log')
    bar_width = 10

    # Bars: x86 and ARM side by side
    ax1.bar(x - bar_width/2, histogram_uniform,
            width=bar_width, label="Uniform")

    ax1.bar(x + bar_width/2, histogram_normal,
            width=bar_width,label="Normal")

    num_ticks = 10
    tick_positions = np.linspace(0, len(x) - 1, num_ticks, dtype=int)

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_positions)
    ax1.set_xlabel("Bins")
    ax1.set_ylabel("Counts")
    ax1.set_title(f"Histogram input size={array}")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.legend(loc="lower left")

    plt.tight_layout()
    save_both(f"histogram_{array}_bins_4096")

