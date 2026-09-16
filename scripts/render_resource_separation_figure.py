"""Render the existing T=12 exact certificate; no learning or outcome sampling.

Figure contract: two specific equal summaries do not certify finite-budget
equivalence. Panel a records the anchor comparison and feasible experiment
enlargement; panel b displays certified lower/upper bounds, NOT exact risks.
Archetype: schematic-led composite. Export width: 183 mm; editable vector text.
Source: existing rational certificate. No rows or horizons are selected for an
empirical analysis: this is the theorem's explicitly identified anchor panel.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.certificate.read_bytes()
    certificate = json.loads(raw)
    row = next(r for r in certificate["results"] if r["T"] == 12)
    lower, upper, gap = map(Fraction, (row["lower"], row["upper"], row["gap"]))
    assert lower - upper == gap
    assert row["low_all_safe_terminal_states_allowed"]
    assert row["high_witness_finishes_at_charger"]
    assert certificate["sampled_trajectories"] == 0
    kl = .05 * math.log(.05 / .30) + .95 * math.log(.95 / .70)
    maximal_coefficient = .10 / kl
    assert abs(maximal_coefficient - .498691947) < 1e-8
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "figure1_source.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["quantity", "capacity", "qualification", "exact", "decimal"])
        writer.writerow(["minimax_regret", 3, "lower_bound_T12", str(lower), float(lower)])
        writer.writerow(["minimax_regret", 4, "upper_bound_T12", str(upper), float(upper)])
        writer.writerow(["risk_gap", "3-minus-4", "lower_bound_T12", str(gap), float(gap)])
        for capacity in (3, 4):
            writer.writerow(["max_fixed_instance_C", capacity, "anchor",
                             "(1/10)/kl(1/20,3/10)", maximal_coefficient])
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.2, "axes.titlesize": 8, "axes.labelsize": 7.2,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "legend.frameon": False,
    })
    fig = plt.figure(figsize=(183 / 25.4, 88 / 25.4), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.6, 1), left=.035,
                           right=.98, top=.80, bottom=.20, wspace=.23)
    a, b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    fig.suptitle("Two equal summaries, strictly different finite-budget risk",
                 fontsize=10, fontweight="bold", y=.96)
    a.axis("off")
    a.set_title("a   Resource-realizable nesting", loc="left", fontweight="bold", pad=14)
    cells = [
        ["Known-model gains (q, A, B)", "(.05, .10, 19/60)", "(.05, .10, 19/60)"],
        ["Terminal execution at T = 12", r"$12\rho_\theta$", r"$12\rho_\theta$"],
        ["Max fixed-instance coefficient", f"{maximal_coefficient:.6f}", f"{maximal_coefficient:.6f}"],
        ["Joint safe measurement sortie", "Not feasible", "Feasible"],
    ]
    table = a.table(cellText=cells, colLabels=["Specific comparison", "Capacity 3", "Capacity 4"],
                    cellLoc="center", colWidths=[.47, .265, .265], bbox=[0, .15, 1, .83])
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(.5)
        cell.set_facecolor("#F1F5F9" if r == 0 else "white")
        if r == 0:
            cell.get_text().set_fontweight("bold")
        if r == 4:
            cell.set_facecolor("#E8F1F7")
    a.text(.0, .035, r"Equality concerns $\max_\theta C_\theta$, not every $C_\theta$.",
           transform=a.transAxes, fontsize=7.2)
    b.set_title("b   Certified bounds at T = 12", loc="left", fontweight="bold", pad=14)
    low_color, high_color = "#596579", "#16718A"
    lo, hi = float(lower), float(upper)
    b.scatter([lo, hi], [1, 0], s=38, facecolors="white",
              edgecolors=[low_color, high_color], linewidths=1.5, zorder=4)
    b.annotate("", xy=(.325, 1), xytext=(lo, 1),
               arrowprops={"arrowstyle": "->", "color": low_color, "lw": 1.4})
    b.annotate("", xy=(.055, 0), xytext=(hi, 0),
               arrowprops={"arrowstyle": "->", "color": high_color, "lw": 1.4})
    b.text(.015, 1.24, "Capacity 3: lower bound", color=low_color, fontsize=7.2)
    b.text(.015, .24, "Capacity 4: upper bound", color=high_color, fontsize=7.2)
    b.text(lo, .82, f"≥ {lo:.6f}", ha="center", color=low_color, fontsize=7.2)
    b.text(hi, -.18, f"≤ {hi:.6f}", ha="center", color=high_color, fontsize=7.2)
    b.annotate("", xy=(lo, .51), xytext=(hi, .51),
               arrowprops={"arrowstyle": "|-|", "color": "#111827", "lw": 1})
    b.text((lo + hi) / 2, .59, f"Gap ≥ {float(gap):.6f}", ha="center", fontsize=7.2)
    b.set_xlim(0, .34)
    b.set_ylim(-.40, 1.48)
    b.set_xticks([0, .1, .2, .3])
    b.set_yticks([])
    b.spines["left"].set_visible(False)
    b.set_xlabel("Finite-budget minimax regret (bounds)")
    fig.text(.035, .075,
             "Exact anchor certificates, not sampled learning curves. Terminal execution equality: charger-terminal policies.",
             fontsize=7.2, color="#475569")
    fig.canvas.draw()
    # Deterministic clipping preflight. Review actual image separately for overlaps.
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1, item.get_text()
            assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, item.get_text()
    stem = args.output_dir / "figure1_resource_separation"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    metadata = {
        "sampled_trajectories": 0, "budget": 12,
        "certificate_sha256": hashlib.sha256(raw).hexdigest(),
        "bounds_not_exact_minimax_values": True,
        "width_mm": 183, "height_mm": 88,
        "backend": "python_matplotlib", "source_rows": 5,
        "no_new_horizons_evaluated": True,
        "clipping_preflight": "pass",
    }
    (args.output_dir / "figure1_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
