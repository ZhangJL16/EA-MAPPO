"""Audit passive telemetry against the frozen B4 runtime after path relocation."""
import os,sys,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE/'repo';PROJECT=ROOT/'persistent_uav_throughput_v1'
os.environ['PERSISTENT_UAV_LEGACY_ROOT']=str(ROOT);sys.path.insert(0,str(PROJECT))
from persistent_uav.provenance import provenance,verify_provenance
from persistent_uav.storage import sha,write_json
from persistent_uav.evaluation import calibration_compatibility
from review_bundle.safety.collision.hocbf import _NATIVE_QP_FUNCTION
prior_path=HERE.parent/'consolidation_20260920/calibration_compatibility_zjl_exp.json';prior=json.loads(prior_path.read_text());old=prior['diagnostic_provenance'];current=provenance()
for k in old.keys()-{'sources'}:assert old[k]==current[k],k
added=set(current['sources'])-set(old['sources']);removed=set(old['sources'])-set(current['sources']);assert not removed
assert added=={str(PROJECT/'persistent_uav/diagnostics.py')}
changed={k:dict(before=v,after=current['sources'][k]) for k,v in old['sources'].items() if v!=current['sources'][k]}
assert set(changed)=={str(PROJECT/'persistent_uav/evaluation.py')}
# These hashes cover all estimator/scheduler/config/physics/workload source files.
assert all(current['sources'][k]==v for k,v in old['sources'].items() if k not in changed)
frozen_path=PROJECT/'evidence/v1_2/calibration/frozen_regimes.json';frozen=json.loads(frozen_path.read_text())
assert sha(frozen_path)==prior['calibration_sha256'];assert frozen['provenance']==prior['calibration_provenance'];assert len(frozen['regimes'])==27
assert frozen_path.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show','c158cc07:persistent_uav_throughput_v1/evidence/v1_2/calibration/frozen_regimes.json'])
assert _NATIVE_QP_FUNCTION is not None
assert sha(ROOT/'runtime_support/review_bundle/safety/collision/_qp_native.so')==prior['native']['arm64_binary_sha256']
assert sha(ROOT/'runtime_support/review_bundle/safety/collision/_qp_native.c')==prior['native']['source_sha256']
assert json.loads((HERE/'smoke/smoke.json').read_text())['passed']
assert 'Ran 41 tests' in (HERE/'tests.txt').read_text() and (HERE/'tests.txt').read_text().rstrip().endswith('OK')
record=dict(reason='User-authorized B5-only validation; passive per-decision predictions and failure-leg telemetry.',calibration_sha256=sha(frozen_path),calibration_provenance=frozen['provenance'],diagnostic_provenance=current,prior_compatibility_sha256=sha(prior_path),repository_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),changed_sources=changed,added_sources={k:current['sources'][k] for k in added},native=prior['native'],checks=dict(frozen_bytes_unchanged=True,scheduler_estimator_config_physics_unchanged=True,model_packages_unchanged=True,native_unchanged=True,tests_41_passed=True,real_smoke_passed=True),calibration_rerun=False,regimes_retuned=False,limits='Only 270 B5 validation jobs; zero added margin. First checkpoint then hand back. No B0-B3, evaluation, training, MPC/Oracle or kill test.')
path=HERE/'calibration_compatibility_b5.json';write_json(path,record);assert calibration_compatibility(frozen_path,frozen,path)==record
try:verify_provenance(old)
except RuntimeError:pass
else:raise AssertionError('Old B4 provenance incorrectly accepted')
print('PASS: frozen calibration, scheduler, estimator, physics, native/model/packages unchanged; only passive telemetry sources changed; stale provenance rejected.')
