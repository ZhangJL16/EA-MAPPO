"""Opening figure from the SAME existing T=12 certificate; no experiment.

Structural/style adaptation of the earlier evidence figure. All five numeric
source rows are retained, bounds remain bounds, and original exports remain intact.
Contract: equal two specific summaries do not imply equal finite-budget risk.
Archetype: schematic-led composite; Python, 183x88 mm, editable SVG/PDF.
"""
import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.certificate.read_bytes()
    row = next(x for x in json.loads(raw)["results"] if x["T"] == 12)
    lower, upper, gap = map(F, (row["lower"], row["upper"], row["gap"]))
    assert lower-upper == gap
    assert row["low_all_safe_terminal_states_allowed"] and row["high_witness_finishes_at_charger"]
    p, q = .05, .30
    assert 0 < p < 1 and 0 < q < 1
    kl = p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))
    coefficient = .10/kl
    assert abs(coefficient-.498691947) < 1e-8
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/"figure1_source.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("quantity", "capacity", "qualification", "exact", "decimal"))
        writer.writerow(("minimax_regret", 3, "lower_bound_T12", str(lower), float(lower)))
        writer.writerow(("minimax_regret", 4, "upper_bound_T12", str(upper), float(upper)))
        writer.writerow(("risk_gap", "3-minus-4", "lower_bound_T12", str(gap), float(gap)))
        for capacity in (3, 4):
            writer.writerow(("max_fixed_instance_C", capacity, "anchor",
                             "(1/10)/kl(1/20,3/10)", coefficient))
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 7.2, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "legend.frameon": False})
    width_mm, height_mm = 183, 88
    fig = plt.figure(figsize=(width_mm/25.4, height_mm/25.4), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.2, 1), left=.035, right=.975,
                           top=.80, bottom=.29, wspace=.23)
    a, b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    fig.suptitle("Same summaries. Different finite-budget learning risk.",
                 fontsize=10, fontweight="bold", y=.96)
    a.axis("off")
    a.set_title("a   Equal summaries; richer safe feedback", loc="left", fontsize=8, fontweight="bold", pad=14)
    a.add_patch(FancyBboxPatch((.01, .53), .98, .45, boxstyle="round,pad=.01",
                              facecolor="#F1F5F9", edgecolor="#D1D5DB", linewidth=.6))
    a.text(.045, .895, "Known-model gains agree", fontsize=8, fontweight="bold")
    a.text(.045, .80, "(q, A, B): (.05, .10, 19/60) at both capacities", fontsize=7.2)
    a.text(.045, .685, "Maximal fixed-instance coefficient agrees", fontsize=8, fontweight="bold")
    a.text(.045, .59, f"{coefficient:.6f} at both capacities", fontsize=7.2)
    for x, title, color in ((.01, "Capacity 3", "#596579"), (.55, "Capacity 4", "#16718A")):
        a.add_patch(FancyBboxPatch((x, .045), .44, .35, boxstyle="round,pad=.01",
                                  facecolor="white", edgecolor=color, linewidth=.8))
        a.text(x+.025, .31, title, fontsize=8, fontweight="bold", color=color)
        a.text(x+.025, .215, "dock, A, B", fontsize=7.2)
    a.text(.035, .115, "Joint route unavailable", fontsize=7.2, color="#596579")
    a.text(.575, .115, "+ joint AB / BA sortie", fontsize=7.2, color="#16718A")
    b.set_title("b   Certified risk bounds at T = 12", loc="left", fontsize=8, fontweight="bold", pad=14)
    lo, hi = float(lower), float(upper)
    for x, y, end, color, label, sign in ((lo, 1, .325, "#596579", "Capacity 3: lower bound", "≥"),
                                        (hi, 0, .055, "#16718A", "Capacity 4: upper bound", "≤")):
        b.scatter([x], [y], s=38, facecolors="white", edgecolors=color, linewidths=1.5, zorder=4)
        b.annotate("", xy=(end, y), xytext=(x, y), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4})
        b.text(.015, y+.24, label, fontsize=7.2, color=color)
        b.text(x, y-.18, f"{sign} {x:.6f}", fontsize=7.2, color=color, ha="center")
    b.annotate("", xy=(lo, .51), xytext=(hi, .51), arrowprops={"arrowstyle": "|-|", "lw": 1, "color": "#111827"})
    b.text((lo+hi)/2, .60, f"Gap ≥ {float(gap):.6f}", fontsize=7.2, ha="center")
    b.set_xlim(0, .34)
    b.set_ylim(-.40, 1.48)
    b.set_xticks((0, .1, .2, .3))
    b.set_yticks(())
    b.spines["left"].set_visible(False)
    b.set_xlabel("Finite-budget minimax regret (bounds)", fontsize=7.2)
    fig.text(.035, .105, "Equality concerns the maximum coefficient, not every instance or full asymptotic geometry.", fontsize=7.2, color="#475569")
    fig.text(.035, .055, r"Exact certificates, not learning curves. Known-model charger-terminal value: $12\rho_\theta$ at both capacities.", fontsize=7.2, color="#475569")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1, item.get_text()
            assert box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, item.get_text()
    stem = args.output_dir/"figure1_resource_separation"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    metadata = {"figure_version": 2, "archetype": "schematic-led composite",
                "backend": "python_matplotlib", "width_mm": width_mm, "height_mm": height_mm,
                "sampled_trajectories": 0, "new_horizons": [], "source_rows": 5,
                "certificate_sha256": hashlib.sha256(raw).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "bounds_not_exact_values": True, "clipping_preflight": "pass",
                "original_figure_preserved": True}
    (args.output_dir/"figure1_provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
