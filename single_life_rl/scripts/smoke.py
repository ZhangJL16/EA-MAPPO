"""One physical midflight clone/resume check; no library/performance selection."""
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
from single_life_rl.scripts.io_utils import write_json


def run(output):
    repo=Path(__file__).resolve().parents[2]
    os.environ['PERSISTENT_UAV_LEGACY_ROOT']=str(repo)
    sys.path.insert(0,str(repo/'persistent_uav_throughput_v1'))
    from persistent_uav.navigation import FrozenNavigator
    from persistent_uav.storage import snapshot,restore
    frozen=json.loads((repo/'persistent_uav_throughput_v1/evidence/v1_2/calibration/frozen_regimes.json').read_text())
    nav=FrozenNavigator(1e9,layout=frozen['layout'])
    # Fixed engineering target, not one of the 200 sampled mission arms.
    nav.start_leg([1800.,1800.,100.]);nav.advance_flight(.2)
    with tempfile.TemporaryDirectory(prefix='single_life_smoke_') as folder:
        ref=snapshot(folder,dict(nav=nav))
        clone=restore(folder)['nav']
        result1=nav.advance_flight(.2);result2=clone.advance_flight(.2)
        assert result1==result2
        assert np.array_equal(nav.position,clone.position)
        assert nav.time==clone.time and nav.energy==clone.energy
        assert nav.reset_count==clone.reset_count==1
        assert nav.policy_steps==clone.policy_steps==2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure=plt.figure();plt.close(figure)
    record=dict(passed=True,nonzero_policy_steps=2,midflight_disk_resume_exact=True,
                physical_resets_per_copy=1,training_updates=0,matplotlib_available=True,snapshot_bytes=ref['bytes'])
    write_json(output,record);print(json.dumps(record))
    nav.close();clone.close()


if __name__=='__main__':
    run(Path(__file__).resolve().parents[1]/'evidence/engineering_smoke.json')
