import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

def save_both(basename):
    plt.savefig(f"{basename}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{basename}.pdf", bbox_inches="tight")


cpu_times = [
    0.324963,   # CPU execution time
]

gpu_basic_times = [
    0.935294,   # GPU basic execution time
]

# All GPU streams execution times (single list, all segment sizes)
gpu_stream_times = [
    5.42703,    # seg 2048
    1.8499,     # seg 8192
    0.952322,   # seg 32768
    0.768999,   # seg 131072
    0.749797,   # seg 160000
    0.782942,   # seg 524288
]

segment_sizes = [2048, 8192, 32768, 131072, 160000, 524288]
array_len=2097152
# --- Bar plot: segment size vs GPU streamed time ---

x = np.arange(len(segment_sizes))  # 0..5

plt.figure(figsize=(10, 6))
ax = plt.gca()

ax.bar(x, gpu_stream_times, width=0.6)

ax.set_xticks(x)
ax.set_xticklabels(segment_sizes, rotation=30)
ax.set_xlabel("Segment size")
ax.set_ylabel("Time [ms]")
ax.set_title(f"GPU streamed execution time vs segment size - array len={array_len}")
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
save_both("gpu_streams_segment_size_vs_time")
