"""Read-only scientific analysis of the authorized single-seed T=4096 calibration.

Figure contract: mechanism validation, not an asymptotic-fit or algorithm-win claim.
Quantitative grid: hero four-cell normalized pseudo-regret + centered residual;
secondary all-method audit. One paired seed, no CI/p-value, all twenty trajectories.
Python-only, 183 mm width, editable SVG/PDF and 300 dpi PNG preview.
"""
import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_bundling_calibration as runner
from bundling_calibration_core import CELLS, METHODS, Plant, POSES, catalogue, solve_models


def mean_for_channel(channel):
    return {None: 0., "A": .30, "B": .05, "dock": .05}[channel]


def analyze(source):
    runner.validate(source)
    sealed = json.loads((source/"sealed_predictions.json").read_text())
    summaries, curves, all_data = [], [], {}
    for method in METHODS:
        for capacity, on in CELLS:
            key = f"B{capacity}_{'on' if on else 'off'}_{method}"
            path = source/f"{key}.checkpoint.json"
            data = json.loads(path.read_text())
            assert data["time"] == 4096 and len(data["trace"]) == 4096
            prediction = next(a for a in sealed["audits"] if a["capacity"] == capacity
                              and a["bundling"] == on and tuple(a["theta"]) == runner.TRUTH)
            C, gain = prediction["C"], prediction["gain"]
            routes = catalogue(capacity, on)
            costs = prediction["acquisition_costs"]
            plant = Plant(capacity, on)
            P, boundary = 0., 0.
            completed_cost, n_completed = 0., 0
            reasons = defaultdict(Counter)
            for t, row in enumerate(data["trace"], 1):
                assert runner.tuples(row["before"]) == plant.state
                channel = plant.step(row["action"])
                assert row["channel"] == channel
                assert runner.tuples(row["after"]) == plant.state
                assert tuple(row["pose"]) == POSES[plant.state[0]] and t == row["t"]
                delta = gain-mean_for_channel(channel)
                P += delta
                boundary += delta
                if plant.at_dock:
                    assert abs(boundary-costs[row["path"]]) < 1e-10
                    completed_cost += boundary
                    boundary = 0.
                    n_completed += 1
                    reasons[row["selection"]][row["path"]] += 1
                curves.append({"key": key, "method": method, "capacity": capacity,
                               "bundling": on, "t": t, "P": P, "C": C,
                               "normalized": P/math.log(t) if t > 1 else None,
                               "residual": P-C*math.log(t), "C_logT": C*math.log(t)})
            assert abs(P-data["pseudo_regret"]) < 1e-9
            assert abs(P-completed_cost-boundary) < 1e-9
            assert n_completed == sum(data["counts"])
            for ck in data["checkpoints"]:
                assert ck["t"] in (128, 4096)
                prefix = sum(gain-mean_for_channel(row["channel"]) for row in data["trace"][:ck["t"]])
                assert abs(prefix-ck["pseudo_regret"]) < 1e-9
            summary = {"key": key, "method": method, "capacity": capacity, "bundling": on,
                       "T": 4096, "P": P, "P_over_logT": P/math.log(4096), "C": C,
                       "residual": P-C*math.log(4096), "completed_cost": completed_cost,
                       "incomplete_path_cost": boundary, "path_counts": dict(zip((p.name for p in routes), data["counts"])),
                       "cost_by_path": {p.name: n*costs[p.name] for p, n in zip(routes, data["counts"])},
                       "completed_paths_by_reason": {why: dict(v) for why, v in reasons.items()},
                       "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            summaries.append(summary)
            all_data[key] = data
    contrasts = []
    for method in METHODS:
        group = [r for r in summaries if r["method"] == method]
        low_on, high_on, low_off, high_off = group
        # Negative control: compare observable continuation, not merely endpoints.
        a = all_data[low_off["key"]]["trace"]
        b = all_data[high_off["key"]]["trace"]
        for aa, bb in zip(a, b):
            for field in ("t", "path", "action", "pose", "channel", "reward", "selection"):
                assert aa[field] == bb[field]
            for field in ("before", "after"):
                assert aa[field][:2]+aa[field][3:] == bb[field][:2]+bb[field][3:]
        contrast = low_on["P"]-high_on["P"]-(low_off["P"]-high_off["P"])
        expected = low_on["C"]-high_on["C"]-low_off["C"]+high_off["C"]
        contrasts.append({"method": method, "low_minus_high_on": low_on["P"]-high_on["P"],
                          "low_minus_high_off": low_off["P"]-high_off["P"], "DID": contrast,
                          "sealed_DID_coefficient": expected, "sealed_DID_times_logT": expected*math.log(4096),
                          "negative_control_observable_identity": True})
    return summaries, curves, contrasts


def render(curves, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": .8,
        "legend.frameon": False})
    palette = ("#626970", "#367DA2", "#A0A0A0", "#8E6E99")
    styles = ("-", "-", "--", ":")
    labels = ("low/on", "high/on", "low/off", "high/off")
    groups = defaultdict(list)
    for row in curves:
        groups[(row["method"], row["capacity"], row["bundling"])].append(row)
    def panel(ax, method, metric):
        # t=1 normalized value undefined. No valid t>=2 observation excluded.
        for (capacity, on), color, style, label in zip(CELLS, palette, styles, labels):
            rows = groups[(method, capacity, on)]
            valid = [r for r in rows if r[metric] is not None]
            ax.plot([r["t"] for r in valid], [r[metric] for r in valid], color=color,
                    linestyle=style, linewidth=1, label=label)
            if metric == "normalized":
                ax.axhline(rows[0]["C"], color=color, linewidth=.6, linestyle=(0, (1, 3)), alpha=.7)
        ax.set_xscale("log")
        ax.set_xlabel("Primitive time T")
        ax.set_ylabel("P(T) / log T" if metric == "normalized" else "P(T) − C log T")
    figure, axes = plt.subplots(1, 2, figsize=(183/25.4, 82/25.4), layout="constrained")
    panel(axes[0], "resource_path", "normalized")
    panel(axes[1], "resource_path", "residual")
    axes[0].set_title("a  Resource-path learner", loc="left", fontweight="bold")
    axes[1].set_title("b  Unfitted prediction residual", loc="left", fontweight="bold")
    axes[0].legend(fontsize=7, ncol=2)
    figure.suptitle("One fixed paired seed; T ≤ 4096; dotted lines are sealed C predictions", fontsize=7)
    figure.savefig(destination/"calibration_primary.svg", bbox_inches="tight")
    figure.savefig(destination/"calibration_primary.pdf", bbox_inches="tight")
    figure.savefig(destination/"calibration_primary.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(183/25.4, 115/25.4), layout="constrained")
    for ax, method in zip(axes.flat, METHODS):
        panel(ax, method, "normalized")
        ax.set_title(method.replace("_", " "), fontsize=7)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", fontsize=7)
    figure.suptitle("All five predeclared methods; one seed; no confidence intervals", fontsize=7)
    figure.savefig(destination/"calibration_all_methods.svg", bbox_inches="tight")
    figure.savefig(destination/"calibration_all_methods.pdf", bbox_inches="tight")
    figure.savefig(destination/"calibration_all_methods.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true", help="Render saved analysis; never rerun learner or rewrite data")
    args = parser.parse_args()
    if args.render_only:
        runner.validate(args.source)
        destination = args.source/"analysis_4096"
        curves = json.loads((destination/"curves.json").read_text())
        render(curves, destination)
        render_receipt = {"analysis_data_sha256": hashlib.sha256((destination/"curves.json").read_bytes()).hexdigest(),
                          "render_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "render_only": True, "png_dpi": 600, "nominal_width_mm": 183,
                          "curve_rows": len(curves), "experiment_rerun": False}
        (destination/"render_receipt.json").write_text(json.dumps(render_receipt, indent=2)+"\n")
        print("Saved-analysis render only; no learner rerun or data modification")
        return
    summaries, curves, contrasts = analyze(args.source)
    destination = args.source/"analysis_4096"
    destination.mkdir(exist_ok=False)
    for name, rows in (("summary", summaries), ("curves", curves), ("contrasts", contrasts)):
        (destination/f"{name}.json").write_text(json.dumps(rows, indent=2)+"\n")
    with (destination/"curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader(); writer.writerows(curves)
    render(curves, destination)
    receipt = {"verification": "ANALYZED; single-seed descriptive calibration only", "n_seeds": 1,
               "trajectory_count": 20, "curve_rows": len(curves), "formal_tests": 0,
               "source_hashes": runner.sources(), "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "sealed_prediction_sha256": hashlib.sha256((args.source/"sealed_predictions.json").read_bytes()).hexdigest(),
               "task_throughput_analysis": False, "runtime_or_learner_modifications": False}
    (destination/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(contrasts, indent=2))


if __name__ == "__main__":
    main()
