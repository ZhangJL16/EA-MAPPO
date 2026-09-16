#!/usr/bin/env python3
"""Create a gated PSPS CONFIRM child without accessing confirmation outcomes."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_conditional_history_collection import atomic_json
from scripts.validate_identification_contract import file_hash, object_hash, record_hash, seal_access_registry
from scripts.validate_psps_v1_contract import validate_contract


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--parent-contract",type=Path,required=True); parser.add_argument("--parent-receipt",type=Path,required=True); parser.add_argument("--dev-run-dir",type=Path,required=True); parser.add_argument("--dev-analysis-dir",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    parent_path=args.parent_contract.resolve(); parent=json.loads(parent_path.read_text()); receipt=json.loads(args.parent_receipt.resolve().read_text()); validation=validate_contract(parent,parent_path.parent,receipt)
    if not validation["PSPS_DEV_READY"] or validation["psps_dev_accessed"]!=96 or validation["psps_confirm_accessed"]!=0 or validation["old_pai_confirm_accessed"]!=0: raise ValueError("PSPS confirmation opening boundary is not eligible")
    results_path=args.dev_run_dir.resolve()/"results.json"; gates_path=args.dev_analysis_dir.resolve()/"gates.json"; analysis_path=args.dev_analysis_dir.resolve()/"analysis.json"; gates=json.loads(gates_path.read_text())
    if gates.get("gate_o",{}).get("passed") is not True or gates.get("gate_d",{}).get("passed") is not True or gates.get("confirmation_authorized") is not True or gates.get("method_train_authorized") is not False: raise ValueError("both PSPS DEV gates must pass before CONFIRM")
    results=json.loads(results_path.read_text())
    if results.get("completed")!=96: raise ValueError("PSPS DEV collection incomplete")
    child=copy.deepcopy(parent); child["schema_version"]="psps-v1-confirmation-child-v1"; child["status"]="FROZEN_PSPS_V1_CONFIRM_AWAITING_EXTERNAL_ANCHOR"
    authorization={"schema_version":"psps-v1-confirmation-authorization-v1","parent_contract_sha256":file_hash(parent_path),"parent_access_head":parent["access_registry"]["head_event_hash"],"dev_results_sha256":file_hash(results_path),"dev_gates_sha256":file_hash(gates_path),"dev_analysis_sha256":file_hash(analysis_path),"models_sha256":file_hash(args.dev_analysis_dir.resolve()/"models.npz"),"models_manifest_sha256":file_hash(args.dev_analysis_dir.resolve()/"models.json"),"support_sha256":file_hash(args.dev_analysis_dir.resolve()/"support.npz"),"support_manifest_sha256":file_hash(args.dev_analysis_dir.resolve()/"support.json"),"confirm_manifest_hash":parent["pre_h"]["world_manifest_hashes"]["PSPS_CONFIRM"],"gate_o_passed":True,"gate_d_passed":True,"old_pai_confirm_accessed":0,"refit":False,"automatic_training":False,"method_train_authorized":False}
    authorization["record_sha256"]=record_hash(authorization); digest=object_hash(authorization); child["content_registry"][digest]={"inline":authorization}; child["confirmation_authorization"]=authorization
    events=child["access_registry"]["events"]; events.append({"sequence":events[-1]["sequence"]+1,"event_id":"authorize-psps-v1-confirm","event_type":"STAGE_TRANSITION","stage":"PSPS_CONFIRM","record_hash":authorization["record_sha256"]}); seal_access_registry(child); atomic_json(args.output.resolve(),child)
    print(json.dumps({"confirmation_contract":str(args.output.resolve()),"authorization_record":authorization["record_sha256"],"head_event_hash":child["access_registry"]["head_event_hash"],"requires_external_anchor":True},sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
