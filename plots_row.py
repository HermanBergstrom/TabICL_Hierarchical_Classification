import matplotlib.pyplot as plt
import numpy as np

# (means, stds) per dataset; order: [Tabular, Tabular+Raw, Tabular+PCA256]
salary      = ([0.4599, 0.4188, 0.4550], [0.0024, 0.0069, 0.0023])
pad_ufes    = ([0.8400, 0.8299, 0.8664], [0.0033, 0.0030, 0.0040])
news_channel = ([0.5585, 0.5397, 0.5615], [0.0012, 0.0023, 0.0012])
petfinder   = ([0.3745, 0.3648, 0.4395], [0.0006, 0.0012, 0.0027])

datasets = [
    ("PAD-UFES",      pad_ufes,     "image + tabular"),
    ("News Channel",  news_channel, "text + tabular"),
    ("Salary",        salary,       "text + tabular"),
    ("PetFinder",     petfinder,    "image + text + tabular"),
]

bar_labels = ["Tabular", "Tabular\n+ Raw", "Tabular\n+ PCA256"]
x = np.arange(len(bar_labels)) * 0.6
colors = ["#94e2d5", "#fab387", "#fab387"]
hatches = ["", "", "/"]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "serif"
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, (title, (means, errs), subtitle) in zip(axes, datasets):
    pct_values = [v * 100 for v in means]
    stds = [v * 100 for v in errs]
    bars = ax.bar(x, pct_values, yerr=stds, capsize=5, error_kw={"linewidth": 1.5, "color": "black"}, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
        if hatch:
            bar.set_edgecolor("black")
    ax.set_title(title, fontsize=22, pad=28 if subtitle else 10)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=16, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=18)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.spines[["top", "right"]].set_visible(False)

    lo, hi = min(pct_values), max(pct_values)
    margin = (hi - lo) * 0.5 if hi != lo else 0.5
    ax.set_ylim(max(0, lo - margin), hi + margin * 1.2)
    ax.grid(False)

    for bar, val, std in zip(bars, pct_values, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + margin * 0.15,
            f"{val:.2f}%",
            ha="center", va="bottom", fontsize=17, fontweight="bold", color="#333333",
        )

plt.tight_layout(pad=1.5)
plt.savefig("signal_balancing_row.pdf", dpi=150, bbox_inches="tight")
plt.show()
