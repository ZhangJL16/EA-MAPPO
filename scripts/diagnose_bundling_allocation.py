"""Audit existing observations only: no plant rollout, RNG draws or learner edits."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_bundling_calibration as runner
from bundling_calibration_core import CELLS, HYPOTHESES, Learner, catalogue


def diagnostic(source):
    runner.validate(source)
    sealed = json.loads((source/"sealed_predictions.json").read_text())
    outputs = []
    for capacity, on in CELLS:
        key = f"B{capacity}_{'on' if on else 'off'}_resource_path"
        path = source/f"{key}.checkpoint.json"
        data = json.loads(path.read_text())
        routes = catalogue(capacity, on)
        learner = Learner(routes, "resource_path")
        prediction = next(r for r in sealed["audits"] if r["capacity"] == capacity
                          and r["bundling"] == on and tuple(r["theta"]) == runner.TRUTH)
        true_index = HYPOTHESES.index(runner.TRUTH)
        optimal = next(i for i, p in enumerate(routes) if p.name == prediction["optimal_path"])
        decisions, current = [], None
        correct_quota = Counter()
        for row in data["trace"]:
            if current is None:
                t = row["t"]-1
                chosen, exploration, reason = learner.select(t)
                assert routes[chosen].name == row["path"] and reason == row["selection"]
                estimate = max(range(len(HYPOTHESES)), key=learner.likelihood.__getitem__)
                f = math.log(t+3)+2*math.log(math.log(t+3))+math.log(len(HYPOTHESES)+1)
                plausible = [i for i in range(len(HYPOTHESES)) if
                             learner.likelihood[estimate]-learner.likelihood[i] <= f]
                model, j = learner.models[estimate], sum(learner.Q)
                decision = {"t": t, "path": routes[chosen].name, "reason": reason,
                            "MLE": HYPOTHESES[estimate], "Q_before": learner.Q[:], "j": j,
                            "sqrt_j": math.sqrt(j), "f": f, "eta": (j+1)**(-1/8),
                            "plausible": [HYPOTHESES[i] for i in plausible],
                            "likelihood_before": learner.likelihood[:],
                            "LP_quota_targets": [(1+(j+1)**(-1/8))*a*f for a in model["allocation"]]}
                decisions.append(decision)
                if reason == "allocation" and estimate == true_index:
                    correct_quota[routes[chosen].name] += 1
                current = {"index": chosen, "exploration": exploration, "feedback": [], "offset": 0}
            route = routes[current["index"]]
            assert row["action"] == route.actions[current["offset"]]
            if row["channel"]:
                current["feedback"].append((row["channel"], row["reward"]))
            current["offset"] += 1
            if current["offset"] == route.length:
                learner.observe(current["index"], current["feedback"], current["exploration"])
                current = None
        for name in ("Q", "likelihood", "plays", "totals"):
            assert getattr(learner, name) == data["policy"][name]
        assert learner.plays == data["counts"]
        if current is not None:
            assert current["index"] == data["pending"]["index"]
            assert current["offset"] == data["pending"]["offset"]
        allocation_rows = []
        for i, p in enumerate(routes):
            if i == optimal:
                continue  # Optimal execution is O(T); lambda LP has no such coordinate.
            target = prediction["allocation"][p.name]
            actual = learner.plays[i]/math.log(data["time"])
            allocation_rows.append({"path": p.name, "lambda_star": target,
                                    "N_complete": learner.plays[i], "Q_exploration": learner.Q[i],
                                    "N_over_logT": actual, "absolute_difference": abs(actual-target),
                                    "c": prediction["acquisition_costs"][p.name]})
        j = sum(learner.Q)
        f_final = math.log(data["time"]+3)+2*math.log(math.log(data["time"]+3))+math.log(len(HYPOTHESES)+1)
        true_model = learner.models[true_index]
        expected_B_KL = sum(n*p.information(runner.TRUTH, (.30,.95)) for n,p in zip(learner.plays,routes))
        output = {"key": key, "T": data["time"], "allocation_rows": allocation_rows,
                  "L1_nonoptimal": sum(r["absolute_difference"] for r in allocation_rows),
                  "weighted_cost_excess_over_C": sum(r["N_over_logT"]*r["c"] for r in allocation_rows)-prediction["C"],
                  "Q": dict(zip((p.name for p in routes), learner.Q)), "j": j,
                  "sqrt_j": math.sqrt(j), "f_final": f_final,
                  "true_LP": dict(zip((p.name for p in routes), true_model["allocation"])),
                  "true_LP_finite_quota": {p.name: (1+(j+1)**(-1/8))*a*f_final for p,a in zip(routes,true_model["allocation"])},
                  "correct_MLE_LP_quota_plays": dict(correct_quota),
                  "decision_reason_counts": dict(Counter(d["reason"] for d in decisions)),
                  "final_likelihood": learner.likelihood,
                  "observed_logLR_true_vs_B_alt": learner.likelihood[true_index]-learner.likelihood[2],
                  "expected_KL_from_complete_counts_vs_B_alt": expected_B_KL,
                  "last_decision": decisions[-1], "decision_count": len(decisions),
                  "decision_reconstruction_matches_trace": True,
                  "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "decisions": decisions}
        outputs.append(output)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    outputs = diagnostic(args.source)
    dest = args.source/"allocation_diagnostic_4096"
    dest.mkdir(exist_ok=False)
    (dest/"diagnostic.json").write_text(json.dumps(outputs, indent=2)+"\n")
    (dest/"receipt.json").write_text(json.dumps({"status": "ANALYZED; transcript reconstruction only",
        "new_samples": 0, "learner_modified": False, "source_hashes": runner.sources(),
        "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+"\n")
    for output in outputs:
        print(json.dumps({k: v for k,v in output.items() if k not in ("decisions", "last_decision")}, indent=2))


if __name__ == "__main__":
    main()
