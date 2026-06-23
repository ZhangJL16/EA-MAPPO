import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot curves from train_logs CSV files.")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("train_logs"),
        help="Root directory of training logs.",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="UAV3D",
        help="Map/environment name, e.g. UAV3D.",
    )
    parser.add_argument(
        "--algs",
        type=str,
        default="",
        help="Comma-separated algorithm names. Empty means all algorithms under the map.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="episode_reward",
        help="Metric column to plot.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="episode_reward,collision_count,episode_steps",
        help="Comma-separated metric columns to plot as subplots. Overrides --metric when provided.",
    )
    parser.add_argument(
        "--log-files",
        type=str,
        default="",
        help="Comma-separated CSV basenames to include, e.g. UAV3D_log_0.csv,UAV3D_log_1.csv.",
    )
    parser.add_argument(
        "--log-index",
        type=int,
        default=None,
        help="Only include files matching *_log_<index>.csv under each algorithm/map directory.",
    )
    parser.add_argument(
        "--latest-log",
        action="store_true",
        help="Only include the CSV with the largest *_log_<index>.csv suffix under each algorithm/map directory.",
    )
    parser.add_argument(
        "--event",
        type=str,
        default="TRAIN",
        choices=["TRAIN", "EVAL"],
        help="Which event rows to plot.",
    )
    parser.add_argument(
        "--x-key",
        type=str,
        default="timestep",
        help="Column used as x-axis.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling window size for smoothing. 1 disables smoothing.",
    )
    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="Draw each individual CSV run as a faint line.",
    )
    parser.add_argument(
        "--aggregate-all-logs",
        action="store_true",
        help="For each selected algorithm, aggregate all CSV logs under its map directory into one mean curve with std shading, and plot all selected algorithms on the same axes.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Custom plot title.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, defaults to plots/<map>_<event>_<metric>.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window after saving. Disabled by default to avoid blocking in terminal runs.",
    )
    parser.add_argument(
        "--band-alpha",
        type=float,
        default=0.30,
        help="Alpha for the std shading band. Larger values make the band more visible.",
    )
    parser.add_argument(
        "--band-smooth",
        type=int,
        default=0,
        help="Rolling window used to smooth the std band after aggregation. 1 disables band smoothing.",
    )
    return parser.parse_args()


def rolling_smooth(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def find_algorithms(log_root, map_name, algs_arg):
    if algs_arg.strip():
        return [alg.strip() for alg in algs_arg.split(",") if alg.strip()]

    algs = []
    if not log_root.exists():
        return algs
    for alg_dir in sorted(log_root.iterdir()):
        map_dir = alg_dir / map_name
        if alg_dir.is_dir() and map_dir.is_dir():
            algs.append(alg_dir.name)
    return algs


def load_curve(csv_path, event, x_key, metric, smooth):
    df = pd.read_csv(csv_path)
    if event not in set(df["event"].dropna().astype(str)):
        return None
    if x_key not in df.columns or metric not in df.columns:
        return None

    subset = df[df["event"].astype(str) == event].copy()
    subset = subset[[x_key, metric]].dropna()
    if subset.empty:
        return None

    subset[x_key] = pd.to_numeric(subset[x_key], errors="coerce")
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    subset = subset.dropna().sort_values(x_key)
    if subset.empty:
        return None

    # Smooth each run independently before any cross-run aggregation.
    subset[metric] = rolling_smooth(subset[metric], smooth)
    return subset


def aggregate_runs(curves, x_key, metric):
    # Curves are pre-smoothed per run, then aligned onto a shared x-axis by interpolation
    # before cross-run mean/std are computed.
    if not curves:
        return pd.DataFrame(columns=[x_key, "mean", "std", "count"])

    all_x = np.unique(
        np.concatenate([curve[x_key].to_numpy(dtype=float) for curve in curves])
    )
    aligned_values = []

    for curve in curves:
        curve = curve.drop_duplicates(subset=[x_key], keep="last").sort_values(x_key)
        run_x = curve[x_key].to_numpy(dtype=float)
        run_y = curve[metric].to_numpy(dtype=float)
        if run_x.size == 0:
            continue
        if run_x.size == 1:
            interp_y = np.full_like(all_x, np.nan, dtype=float)
            interp_y[np.isclose(all_x, run_x[0])] = run_y[0]
        else:
            interp_y = np.interp(all_x, run_x, run_y)
            outside = (all_x < run_x[0]) | (all_x > run_x[-1])
            interp_y[outside] = np.nan
        aligned_values.append(interp_y)

    if not aligned_values:
        return pd.DataFrame(columns=[x_key, "mean", "std", "count"])

    aligned = np.vstack(aligned_values)
    count = np.sum(~np.isnan(aligned), axis=0)
    mean = np.nanmean(aligned, axis=0)
    std = np.nanstd(aligned, axis=0)

    summary = pd.DataFrame(
        {
            x_key: all_x,
            "mean": mean,
            "std": std,
            "count": count,
        }
    )
    summary = summary[summary["count"] > 0].copy()
    return summary.sort_values(x_key)


def smooth_band(summary, window):
    if window <= 1 or summary.empty:
        return summary
    result = summary.copy()
    result["std"] = rolling_smooth(result["std"], window)
    return result


def parse_metrics(args):
    if args.metrics.strip():
        metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
        if metrics:
            return metrics
    return [args.metric]


def filter_csv_files(csv_files, log_files_arg, log_index):
    if log_files_arg.strip():
        allowed = {name.strip() for name in log_files_arg.split(",") if name.strip()}
        return [csv_path for csv_path in csv_files if csv_path.name in allowed]
    if log_index is not None:
        suffix = f"_log_{int(log_index)}.csv"
        return [csv_path for csv_path in csv_files if csv_path.name.endswith(suffix)]
    return csv_files


def pick_latest_log(csv_files):
    pattern = re.compile(r"_log_(\d+)\.csv$")
    indexed = []
    for csv_path in csv_files:
        match = pattern.search(csv_path.name)
        if match:
            indexed.append((int(match.group(1)), csv_path))
    if indexed:
        indexed.sort(key=lambda item: item[0])
        return [indexed[-1][1]]
    return csv_files[-1:] if csv_files else []


def main():
    args = parse_args()
    metrics = parse_metrics(args)
    algs = find_algorithms(args.log_root, args.map, args.algs)
    if not algs:
        raise SystemExit(f"No algorithms found under {args.log_root} for map {args.map}.")

    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(10, 5 * len(metrics)), squeeze=False
    )
    axes = axes.flatten()
    plotted_metrics = []
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color",
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    )
    linewidth_mean = 1.6
    alpha_mean = 1
    band_alpha = float(args.band_alpha)

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        plotted_any = False

        for alg_idx, alg in enumerate(algs):
            map_dir = args.log_root / alg / args.map
            csv_files = sorted(map_dir.glob("*.csv"))
            if not args.aggregate_all_logs:
                csv_files = filter_csv_files(csv_files, args.log_files, args.log_index)
                if args.latest_log:
                    csv_files = pick_latest_log(csv_files)
            if not csv_files:
                continue

            curves = []
            for csv_path in csv_files:
                curve = load_curve(
                    csv_path=csv_path,
                    event=args.event,
                    x_key=args.x_key,
                    metric=metric,
                    smooth=args.smooth,
                )
                if curve is None:
                    continue
                curves.append(curve)
                if args.show_runs:
                    ax.plot(
                        curve[args.x_key],
                        curve[metric],
                        alpha=0.2,
                        linewidth=1.0,
                    )

            if not curves:
                continue

            summary = aggregate_runs(curves, args.x_key, metric)
            summary = smooth_band(summary, args.band_smooth)
            line_color = color_cycle[alg_idx % len(color_cycle)]
            ax.plot(
                summary[args.x_key],
                summary["mean"],
                linewidth=linewidth_mean,
                color=line_color,
                alpha=alpha_mean,
                label=alg,
            )
            if len(curves) > 1:
                ax.fill_between(
                    summary[args.x_key],
                    summary["mean"] - summary["std"],
                    summary["mean"] + summary["std"],
                    color=line_color,
                    alpha=band_alpha,
                    linewidth=0.0,
                )
            plotted_any = True

        if not plotted_any:
            ax.set_visible(False)
            continue

        plotted_metrics.append(metric)
        ax.set_title(metric)
        ax.set_xlabel(args.x_key)
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

    if not plotted_metrics:
        raise SystemExit(
            f"No valid data found for event={args.event}, metrics={metrics}, x_key={args.x_key}."
        )

    figure_title = args.title or f"{args.map} {args.event}"
    fig.suptitle(figure_title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(top=0.90)

    output_path = args.save
    if output_path is None:
        output_dir = Path("plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_tag = "_".join(plotted_metrics)
        output_path = output_dir / f"{args.map}_{args.event}_{metrics_tag}.png"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
