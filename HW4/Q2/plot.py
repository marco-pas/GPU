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


# Data (dimX = 1024)
time_steps = [100, 1000, 10000, 50000, 100000, 500000]
relative_error = [10.185756, 5.829536, 3.067374, 1.712605, 1.178782, 0.153894]

# Plot error vs time steps
plt.figure(figsize=(10, 6))
ax = plt.gca()

ax.set_xscale("log")
ax.set_yscale("log")

ax.plot(time_steps, relative_error, 'o-', linewidth=2, markersize=8, label="dimX = 1024")

ax.set_xlabel("Number of time steps")
ax.set_ylabel("Relative error")
ax.set_title("Heat equation: convergence vs iterations")
ax.grid(True, linestyle="--", alpha=0.7)
ax.legend(loc="upper right")

plt.tight_layout()
save_both("heat_eq_convergence")