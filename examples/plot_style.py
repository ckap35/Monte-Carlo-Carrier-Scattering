"""Shared styling for the optical-excitation figures."""

from matplotlib import pyplot as plt

BLUE, ORANGE, GREEN, GRAY = "#2563eb", "#d97706", "#0f8c81", "#7c8796"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c3c9d1", "axes.labelcolor": "#263445",
        "xtick.color": "#526071", "ytick.color": "#526071",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "legend.fontsize": 8,
        "savefig.facecolor": "white", "svg.fonttype": "none",
    })


def time_axis(axis, maximum):
    axis.set_xscale("symlog", linthresh=10)
    axis.set_xlim(0, maximum)
    axis.set_xticks([0, 10, 100, 1000, maximum], ["0", "10", "100", "1,000", f"{maximum:,.0f}"])
    axis.set_xlabel("Time (ps)")
    axis.grid(alpha=0.15)
