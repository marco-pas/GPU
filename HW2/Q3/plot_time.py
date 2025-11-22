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


matrix_sizes = [256, 512, 1024, 1242, 1536, 1846, 2048, 3072]

# Execution times in milliseconds
cpu_times = [
    4.775231,
    93.147872,
    1939.068651,
    520.946261,
    3628.556233,
    1777.546063,
    19401.620757,
    66263.648884,
]

gpu_basic_times = [
    0.020865,
    0.102659,
    0.861556,
    1.605583,
    2.905959,
    6.378825,
    7.583729,
    25.239006,
]

gpu_tiled_32_32_32 = [
    0.080408,
    0.159256,
    0.619943,
    0.820097,
    1.513268,
    2.772962,
    4.582591,
    16.488954,
]

gpu_tiled_32_64_32 = [
    0.065524,
    0.134400,
    0.344592,
    0.668821,
    1.023757,
    1.888929,
    2.634119,
    8.721517,
]

gpu_tiled_64_128_32 = [
    0.052720,
    0.101708,
    0.293139,
    0.542246,
    0.842950,
    1.614666,
    2.042470,
    6.216838,
]


x = np.arange(len(matrix_sizes))
plt.figure(figsize=(10, 6))
ax1 = plt.gca()

ax1.set_yscale("log")

bar_width = 0.15

ax1.bar(x - 2*bar_width, cpu_times,
        width=bar_width, label="CPU")
ax1.bar(x - 1*bar_width, gpu_basic_times,
        width=bar_width, label="GPU basic")
ax1.bar(x, gpu_tiled_32_32_32,
        width=bar_width, label="tiled 32x32x32")
ax1.bar(x + 1*bar_width, gpu_tiled_32_64_32,
        width=bar_width, label="tiled 32x64x32")
ax1.bar(x + 2*bar_width, gpu_tiled_64_128_32,
        width=bar_width, label="tiled 64x128x32")

ax1.set_xticks(x)
ax1.set_xticklabels(matrix_sizes)
ax1.set_xlabel("Matrix size A(N x N), B(N x N), C(N x N) ")
ax1.set_ylabel("Time [ms]")
ax1.set_title("GEMM execution time vs matrix size")
ax1.grid(axis="y", linestyle="--", alpha=0.7)
ax1.legend(loc="upper left")

plt.tight_layout()
save_both("gemm_execution_times")
