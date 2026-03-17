import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_histories(mesh_name, base_dir="."):
    history_dir = Path(base_dir) / f"{mesh_name}_weights" / "history"
    if not history_dir.exists():
        print(f"History folder not found: {history_dir}")
        return

    files = sorted(history_dir.glob("*.npz"))

    keep_methods = {
        "large_unpruned",
        "densified",
        "AIRe_0.6",
        "AIRe_0.6_densified",
        "DepGraph_0.6",
        "DepGraph_0.6_densified",
    }

    colors = {
        "large_unpruned": "black",
        "densified": "purple",
        "AIRe_0.6": "tab:orange",
        "AIRe_0.6_densified": "tab:blue",
        "DepGraph_0.6": "tab:red",
        "DepGraph_0.6_densified": "tab:green",
    }

    filtered_files = [
        f for f in files if f.stem.replace("_history", "") in keep_methods
    ]

    # ---------------- LOSS PLOT ----------------
    fig, ax = plt.subplots(figsize=(9,5))

    for file in filtered_files:
        data = np.load(file)
        loss = data["loss"]
        steps = np.arange(len(loss)) * 10
        label = file.stem.replace("_history", "")
        ax.plot(steps, loss, linewidth=2, alpha=0.9, color=colors.get(label, None), label=label)

    ax.set_yscale("log")

    ax.axvline(200, linestyle="--", color="gray", alpha=0.5)
    ax.axvline(700, linestyle="--", color="gray", alpha=0.5)

    # horizontal labels above plot
    ax.text(200, 1.02, "TWD", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom")
    ax.text(700, 1.02, "Pruning", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"{mesh_name.capitalize()} – Loss")
    ax.set_xlim(-10, 1010)

    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")

    plt.tight_layout()
    plt.show()


    # ---------------- IOU PLOT ----------------
    fig, ax = plt.subplots(figsize=(9,5))

    for file in filtered_files:
        data = np.load(file)
        iou = data["iou"]
        steps = data["steps"]
        label = file.stem.replace("_history", "")
        ax.plot(steps, iou, marker="o", markersize=4, linewidth=2, alpha=0.9, label=label, color=colors.get(label, None))

    ax.axvline(200, linestyle="--", color="gray", alpha=0.5)
    ax.axvline(700, linestyle="--", color="gray", alpha=0.5)

    # horizontal labels above plot
    ax.text(200, 1.02, "TWD", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom")
    ax.text(700, 1.02, "Pruning", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("IoU")
    ax.set_title(f"{mesh_name.capitalize()} – IoU")
    ax.set_xlim(-10, 1010)

    ax.set_yscale("function", functions=(lambda x: x**4, lambda x: x**0.25))
    ax.set_ylim(0,1)

    # avoid overlapping ticks near zero
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")

    plt.tight_layout()
    plt.show()