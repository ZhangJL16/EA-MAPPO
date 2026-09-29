"""Tiny real navigation smoke for the new full-window decision path."""

from dataclasses import replace
import json

from run_full_window_pilot import (Config, FROZEN, FrozenNavigator,
                                   PersistentUAVThroughput, choose_full_recharge,
                                   workload)


def main():
    frozen = json.loads(FROZEN.read_text())
    config = replace(Config(**frozen["regimes"][0]["config"]), cutoff=10.0)
    tasks = workload(1302609290, config, frozen["layout"])
    nav = FrozenNavigator(config.capacity, option_step_limit=config.option_step_limit,
                          layout=frozen["layout"])
    try:
        env = PersistentUAVThroughput(config, nav, tasks)
        decisions = 0
        while not env.done:
            observation = env.observe()
            action = (choose_full_recharge(observation, "full_recharge_nearest", None)
                      if env.decision_required else None)
            decisions += int(action is not None)
            env.step(action)
        summary = env.summary()
        assert decisions >= 1 and summary["done"] and abs(summary["simulation_time"] - config.cutoff) < 1e-6
        print(json.dumps(dict(smoke="pass", decisions=decisions,
                              policy_steps=nav.policy_steps, terminal=env.mode)))
    finally:
        nav.close()


if __name__ == "__main__":
    main()
