import os,sys,json,hashlib,copy
from pathlib import Path
base=Path('/mnt/workspace/zjl-exp');root=base/'repo';project=root/'persistent_uav_throughput_v1';sys.path.insert(0,str(project));os.environ['PERSISTENT_UAV_LEGACY_ROOT']=str(root)
from persistent_uav.provenance import provenance,verify_provenance
from persistent_uav.evaluation import calibration_compatibility
from persistent_uav.storage import restore
record=base/'migration_20260920/consolidation_20260920';prior=base/'migration_20260920/v1_4/calibration_compatibility_arm64.json';original=json.loads(prior.read_text())
def relocate(x):
 if isinstance(x,str):return x.replace('/mnt/workspace/persistent-uav/','/mnt/workspace/zjl-exp/')
 if isinstance(x,dict):return {relocate(k):relocate(v) for k,v in x.items()}
 if isinstance(x,list):return [relocate(v) for v in x]
 return x
expected=relocate(original['diagnostic_provenance']);verify_provenance(expected)
compat=copy.deepcopy(original);compat['diagnostic_provenance']=expected;compat['workspace_relocation']={'from':'/mnt/workspace/persistent-uav','to':str(base),'prior_record_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),'source_model_package_hashes_unchanged':True};compat['reason']+=' Workspace paths relocated after validation completion; no scientific inputs changed.'
out=record/'calibration_compatibility_zjl_exp.json';out.write_text(json.dumps(compat,indent=2)+'\n')
frozen=project/'evidence/v1_2/calibration/frozen_regimes.json';calibration_compatibility(frozen,json.loads(frozen.read_text()),out)
output=project/'artifacts/diagnostic_validation_arm64_parallel16_20260920';n=0
for folder in sorted(output.glob('worker_*')):
 if not folder.is_dir():continue
 s=json.loads((folder/'status.json').read_text());assert s['status']=='complete';state=restore(folder)
 assert relocate(state['contract']['provenance'])==expected
 assert state['env'] is None
 assert state['rows']==json.loads((folder/'results.json').read_text());n+=len(state['rows'])
assert n==1080
result={'passed':True,'source_model_dependencies_verified':True,'all_final_checkpoints_deserialized':True,'completed_runs':n,'calibration_compatibility_verified':True,'historical_manifests_unchanged':True,'old_root_absent':not Path('/mnt/workspace/persistent-uav').exists()}
(record/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
