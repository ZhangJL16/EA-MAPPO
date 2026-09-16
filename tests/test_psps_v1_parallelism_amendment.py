from scripts.amend_psps_v1_parallelism import NEW_EXECUTION, OLD_EXECUTION
from scripts.run_psps_v1_collection_3x8 import WORKERS_PER_WORLD, WORLD_PROCESSES
from scripts.validate_psps_v1_contract import static_spec


def test_execution_overlay_changes_only_outer_world_parallelism() -> None:
    assert OLD_EXECUTION == {
        "world_processes": 4,
        "workers_per_world": 8,
        "max_worlds_per_child": 1,
    }
    assert NEW_EXECUTION == {
        "world_processes": 3,
        "workers_per_world": 8,
        "max_worlds_per_child": 1,
    }
    assert WORLD_PROCESSES == 3
    assert WORKERS_PER_WORLD == 8
    assert static_spec()["execution"]["world_processes"] == 4
    assert static_spec()["execution"]["workers_per_world"] == 8
