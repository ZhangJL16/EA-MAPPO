"""One short physical execution check of the new policy path."""

from dataclasses import replace
import json

from run import (Config, FROZEN, FrozenNavigator, PersistentUAVThroughput,
                 EstimateModel, choose, workload)


def main():
    frozen = json.loads(FROZEN.read_text())
    config = replace(Config(**frozen["regimes"][13]["config"]), cutoff=10.0)
    tasks = workload(1402609290, config, frozen["layout"])
    navigator = FrozenNavigator(config.capacity, option_step_limit=config.option_step_limit,
                                layout=frozen["layout"])
    try:
        environment = PersistentUAVThroughput(config, navigator, tasks)
        model = EstimateModel(**frozen["model"])
        decisions = 0
        while not environment.done:
            if environment.decision_required:
                action, _ = choose(environment.observe(), "C", model)
                decisions += 1
            else:
                action = None
            environment.step(action)
        summary = environment.summary()
        assert decisions >= 1 and summary["done"]
        assert abs(summary["simulation_time"] - config.cutoff) < 1e-6
        print(json.dumps(dict(smoke="pass", decisions=decisions,
                              policy_steps=navigator.policy_steps, terminal=environment.mode)))
    finally:
        navigator.close()


if __name__ == "__main__":
    main()
