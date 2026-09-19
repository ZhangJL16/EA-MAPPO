"""DEV figure contract: exact conditional risk versus actual planning work.

Quantitative grid, two roots x six paired conditions. Three separate x metrics:
model calls, expansions and measured latency. All 240 fresh-policy cells are
displayed. Legacy 131 cells belong to a separate audit and remain in source
data/results, not pooled into fresh-root evidence. No statistical error bars:
risk integrates feedback exactly, conditional on one internal algorithm seed.
Vector PDF/SVG with editable text + PNG preview, 183 x 218 mm report layout.
"""
import argparse
import csv
from fractions import Fraction as F
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(summary_path,output):
    data = json.loads(Path(summary_path).read_text())
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    allcells = data["cells"]
    fresh = [c for c in allcells if c["cohort"]=="fresh_dev_v2"]
    fields = ["cohort","group_id","condition","method","tier","scope","seed","status","horizon",
              "bayes_risk","worst_hypothesis_risk","expected_model_calls","expected_expansions","expected_latency"]
    with (output/"source_data.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fields,extrasaction="ignore"); writer.writeheader()
        for c in allcells:
            writer.writerow(c)
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"font.size":6,"axes.titlesize":6.5,
        "axes.labelsize":6,"xtick.labelsize":5.5,"ytick.labelsize":5.5,"legend.fontsize":6,
        "pdf.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False,
        "axes.linewidth":0.6,"legend.frameon":False})
    methods = data["plan"]["methods"]
    colors = dict(zip(methods,["#89939E","#BC9751","#557EAA","#5B9986","#9B78A3"]))
    markers = dict(zip(methods,["o","s","^","v","P"]))
    labels = dict(channel_cover="ChannelCover",posterior_sampling="PS (seed 17)",beam_bayes="Beam",
                  one_step_voi="OneStepVOI",belief_mcts="Belief-MCTS (seed 17)")
    roots = sorted({c["group_id"] for c in fresh})
    conditions = [s["id"] for s in data["plan"]["conditions"]]
    tiers = list(data["plan"]["tiers"])
    plotted = {}
    for metric,caption in (("expected_model_calls","Expected episode model-call charges"),
                           ("expected_expansions","Expected episode expansion charges"),
                           ("expected_latency","Integrated measured planning latency (seconds)")):
        fig,axes = plt.subplots(len(conditions),len(roots),figsize=(7.204724,8.582677),squeeze=False)
        count = 0
        for i,condition in enumerate(conditions):
            for j,root in enumerate(roots):
                ax = axes[i,j]
                group = [c for c in fresh if c["group_id"]==root and c["condition"]==condition]
                horizon = group[0]["horizon"]
                for method in methods:
                    points = sorted([c for c in group if c["method"]==method and c["scope"]=="episode"],key=lambda c:tiers.index(c["tier"]))
                    xs,ys = [],[]
                    for c in points:
                        if c["status"]!="exact_conditional_policy":
                            raise ValueError("cannot plot unresolved as exact")
                        x = float(F(str(c[metric]))); y = float(F(c["bayes_risk"]))/horizon
                        if x<=0:
                            raise ValueError("log-axis requires positive measured work")
                        xs.append(x); ys.append(y); count += 1
                    ax.plot(xs,ys,marker=markers[method],markersize=3,linewidth=.8,color=colors[method],label=labels[method])
                    for c in group:
                        if c["method"]==method and c["scope"]=="selection":
                            if c["status"]!="exact_conditional_policy":
                                raise ValueError("cannot plot unresolved selection-only result")
                            x = float(F(str(c[metric])))
                            if x<=0: raise ValueError("nonpositive log axis")
                            ax.scatter([x],[float(F(c["bayes_risk"]))/horizon],marker="D",s=14,
                                       facecolors="none",edgecolors=colors[method],linewidths=.7)
                            count += 1
                key = group[0]["instance_hash"]
                ref = data["references"][key]
                if ref["status"]=="exact":
                    ax.axhline(float(F(ref["bayes_risk"]))/horizon,color="#333333",linestyle="--",linewidth=.65)
                else:
                    ax.text(.98,.05,"Bayes unresolved",ha="right",va="bottom",transform=ax.transAxes,color="#555555",fontsize=5.5)
                ax.set_xscale("log"); ax.set_ylim(0,1.02); ax.set_yticks([0,.5,1])
                ax.set_title(f"{root.replace('hierarchy-','Root ')} · {condition.replace('_',' ')} · H={horizon}",pad=3)
                if j==0: ax.set_ylabel("Bayes risk / H")
                ax.grid(axis="y",alpha=.12,linewidth=.5)
        handles,legend_labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,loc="lower center",bbox_to_anchor=(.5,.02),ncol=3)
        fig.suptitle("Exact fixed-policy quality–compute study",fontsize=9,y=.993)
        fig.text(.5,.966,caption,ha="center",fontsize=7)
        fig.text(.5,.004,"Lines: increasing work tiers with episode allowance; open diamonds: per-selection allowance only. Dashed: exact Bayes.",
                 ha="center",fontsize=5.5)
        fig.subplots_adjust(left=.09,right=.985,top=.935,bottom=.095,hspace=.8,wspace=.25)
        stem = output/metric
        fig.savefig(stem.with_suffix(".svg"))
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"),dpi=300)
        plt.close(fig)
        plotted[metric] = count
    if set(plotted.values())!={len(fresh)}:
        raise AssertionError("plot row accounting mismatch")
    notes = dict(all_source_rows=len(allcells),fresh_plot_rows=len(fresh),legacy_separate_rows=len(allcells)-len(fresh),
                 plotted_counts=plotted,risk="exact expectation over environment feedback, conditional on seed",
                 latency="single measured node timings integrated over feedback; not deterministic/exact latency",
                 unit="two fresh root groups; six repeated axis conditions per root; no significance tests",
                 missing_reference="two unresolved deeper-hierarchy optima explicitly labeled; no substituted line",
                 exclusions="Legacy audit is not mixed with fresh DEV. All 371 rows exported to source CSV.",
                 line_semantics="Connect tier settings, not an optimal Pareto frontier or a monotonicity assertion.")
    (output/"QA.json").write_text(json.dumps(notes,indent=2)+"\n")
    print(json.dumps(notes))


if __name__=="__main__":
    p = argparse.ArgumentParser(); p.add_argument("--summary",required=True); p.add_argument("--output",required=True)
    a = p.parse_args(); plot(a.summary,a.output)
