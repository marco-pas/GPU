import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

mpl.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
})

def save_both(basename):
    """Save current figure as both PNG and PDF."""
    plt.savefig(f"{basename}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{basename}.pdf", bbox_inches="tight")


matrix_sizes = [512, 1024, 2048, 4096, 8192]

# Execution times in milliseconds
cpu_times = [
    15.357913,
    330.292115,
    3168.983839,
    33123.290196,
    567238.682415,
]

gpu_basic_times = [
    0.111172,
    0.877801,
    7.093604,
    58.072002,
    861.143680,
]

gpu_tiled_times = [
    0.084544,
    0.455323,
    2.834737,
    22.217420,
    193.184317,
]

gpu_wmma_times = [
    0.021012,
    0.123600,
    0.857015,
    6.687281,
    56.672441,
]

# Error data (Linf norm vs CPU)
gpu_basic_error = [
    0.000046,
    0.000092,
    0.000183,
    0.000366,
    0.000732,
]

gpu_tiled_error = [
    0.000046,
    0.000092,
    0.000183,
    0.000366,
    0.000732,
]

gpu_wmma_error = [
    0.099121,
    0.191711,
    0.380005,
    0.767578,
    1.572754,
]


x = np.arange(len(matrix_sizes))
plt.figure(figsize=(10, 6))
ax1 = plt.gca()

ax1.set_yscale("log")

bar_width = 0.2

ax1.bar(x - 1.5*bar_width, cpu_times,
        width=bar_width, label="CPU")
ax1.bar(x - 0.5*bar_width, gpu_basic_times,
        width=bar_width, label="GPU basic")
ax1.bar(x + 0.5*bar_width, gpu_tiled_times,
        width=bar_width, label="GPU tiled (32x128x16)")
ax1.bar(x + 1.5*bar_width, gpu_wmma_times,
        width=bar_width, label="GPU WMMA TF32")

ax1.set_xticks(x)
ax1.set_xticklabels(matrix_sizes)
ax1.set_xlabel("Matrix size A(N x N), B(N x N), C(N x N) ")
ax1.set_ylabel("Time [ms]")
ax1.set_title("GEMM execution time vs matrix size")
ax1.grid(axis="y", linestyle="--", alpha=0.7)
ax1.legend(loc="upper left")

plt.tight_layout()
save_both("gemm_execution_times")



# Error plot
plt.figure(figsize=(10, 6))
ax2 = plt.gca()

ax2.set_yscale("log")

bar_width = 0.25

ax2.bar(x - bar_width, gpu_basic_error,
        width=bar_width, label="GPU basic")
ax2.bar(x, gpu_tiled_error,
        width=bar_width, label="GPU tiled (32x128x16)")
ax2.bar(x + bar_width, gpu_wmma_error,
        width=bar_width, label="GPU WMMA TF32")

ax2.set_xticks(x)
ax2.set_xticklabels(matrix_sizes)
ax2.set_xlabel("Matrix size N (square matrices N×N)")
ax2.set_ylabel("Linf Error vs CPU")
ax2.set_title("GEMM numerical error vs matrix size")
ax2.grid(axis="y", linestyle="--", alpha=0.7)
ax2.legend(loc="upper left")

plt.tight_layout()
save_both("gemm_error")