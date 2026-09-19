from dataclasses import replace
from fractions import Fraction as F
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from fpl.problem import PublicProblem, Operation, UtilitySpec
from fpl.protocols import enumerate_protocols, PlanningLimit
from fpl.belief import PlannerState
from fpl.work import PlanningWorkBudget
from fpl.evaluation import save_new
from fpl_v2.decoder import ScalableFeasibilityMask, token, STOP, END
from fpl_v2.generator import generate, rng_for, structural_check, as_json, from_json, admit, FAMILIES
from fpl_v2.collection import stratum, retain, information_candidates
from fpl_v2.runtime import run

ROOT = Path(__file__).resolve().parents[1]
DESIGN = json.loads((ROOT/'configs/teacher_v2_design_v1.json').read_text())


def catalogue_masks(routes):
    masks = {(): {STOP}}
    for route in routes:
        path = route.operations
        for i, name in enumerate(path):
            masks.setdefault(path[:i], set()).add(token('operation', name))
        masks[path] = {END}
    return masks


class V2Tests(unittest.TestCase):
    def test_rng_and_pairing(self):
        a = rng_for(300001, 'cost')
        expected = a.getrandbits(256)
        for _ in range(100):
            rng_for(300001, 'topology').random()
        self.assertEqual(expected, rng_for(300001, 'cost').getrandbits(256))
        for family in FAMILIES:
            p = generate(300001, family)
            self.assertEqual(p, from_json(as_json(p)))
            q = generate(300001, family, 9, 3)
            self.assertEqual((p.operations, p.hypotheses, p.prior), (q.operations, q.hypotheses, q.prior))
            self.assertIsNone(structural_check(p))

    def test_mask_all_generated_prefixes(self):
        for family in FAMILIES:
            for seed in range(8):
                p = generate(800001+seed, family)
                for h in (0, 4, 9, 12):
                    expected = catalogue_masks(enumerate_protocols(p, h))
                    mask = ScalableFeasibilityMask(p)
                    with patch('fpl.protocols.enumerate_protocols', side_effect=AssertionError('catalogue forbidden')):
                        for prefix, targets in expected.items():
                            self.assertEqual(set(mask.legal_next(prefix, h)), targets)

    def test_joint_return_and_count_limits(self):
        # Both individual minima fit, but NO single return path fits both.
        p = PublicProblem('test','test','train',('q','x'), 'q',('c',),((F(1,2),),),(F(1),),
                          (Operation('out','q','x',1,1,('c',)),
                           Operation('fast','x','q',1,5),Operation('cheap','x','q',5,1)),3,3,1)
        self.assertEqual(ScalableFeasibilityMask(p).legal_next((),3),[STOP])
        p = replace(p, operations=(Operation('out','q','x',1,1,('c',)),
                                   Operation('return','x','q',1,1,('c',))))
        self.assertEqual(ScalableFeasibilityMask(p).legal_next((),3),[STOP])
        p = replace(p, max_measurements=2, per_channel_limit=2)
        mask = ScalableFeasibilityMask(p)
        self.assertIn(token('operation','out'),mask.legal_next((),3))
        self.assertEqual(mask.legal_next(('out','return'),3),[END])
        with self.assertRaises(ValueError):mask.legal_next(('out','return','out'),3)
        with self.assertRaises(PlanningLimit):ScalableFeasibilityMask(p,max_states=0).legal_next((),3)

    def test_v1_all_exact_prefixes(self):
        total = 0
        with zipfile.ZipFile(ROOT/'provenance/teacher_v1_complete/teacher_v1_complete_evidence.zip') as z:
            for name in z.namelist():
                if not name.endswith('.shard.json'):continue
                shard = json.loads(z.read(name))['payload']; p = from_json(shard['problem'])
                mask = ScalableFeasibilityMask(p)
                for r in shard['records']:
                    routes = [type('Route',(),{'operations':tuple(c['operations'])}) for c in r['label']['candidates'] if c['operations'] is not None]
                    for prefix, targets in catalogue_masks(routes).items():
                        self.assertEqual(set(mask.legal_next(prefix,r['state']['remaining'])), targets)
                        total += 1
        self.assertEqual(total,5383)

    def test_explorer_ignores_utility_and_catalogue(self):
        p = generate(300001,'adaptive_hierarchy')
        q = replace(p, operations=tuple(replace(o,utility=UtilitySpec(by_hypothesis=(F(999),)*4)) for o in p.operations))
        state = PlannerState(p.prior,12,5)
        with patch('fpl.protocols.enumerate_protocols',side_effect=AssertionError('no catalogue')):
            a = information_candidates(p,state,PlanningWorkBudget(100000,1000000))
            b = information_candidates(q,state,PlanningWorkBudget(100000,1000000))
        self.assertTrue(a)
        self.assertEqual(a,b)

    def test_stratification_and_metadata_retention(self):
        p=generate(300001,'adaptive_hierarchy'); state=PlannerState(p.prior,12,5)
        b=stratum(p,state,0,DESIGN)
        self.assertEqual(b[1:],(2,0,0))
        from fpl.teacher_data import state_key
        key=state_key(state)
        pool={key:dict(id=key,public_strata=[b,b[:-1]+(2,)],provenance=[])}
        rows=retain(pool,p,300001,DESIGN)
        self.assertEqual(rows[0]['retention_stratum'],b)

    def test_admission_freeze_and_resume_seal(self):
        excluded=json.loads((ROOT/'configs/teacher_v1/exclusions.json').read_text())
        v1=json.loads((ROOT/'configs/teacher_v1/admission.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)/'admission'; out=Path(d)/'data'
            with patch('fpl.teacher_data.exact_label',side_effect=AssertionError('no label admission')):
                manifest=admit(DESIGN,excluded,v1,base)
            self.assertEqual(len(manifest['groups']),32)
            self.assertEqual(sum(j['pilot'] for j in manifest['jobs']),32)
            self.assertEqual(len({g['clone_hash'] for g in manifest['groups']}),32)
            self.assertEqual(sum(g['fold']=='student_validation' for g in manifest['groups']),8)
            # Runner resume engineering smoke uses explicitly mocked tiny labels,
            # never saved as scientific evidence.
            fake=dict(records=[],sampling=[],pool_states=0,selected_before_labels=True)
            with patch('fpl_v2.runtime.collect',return_value=fake):
                self.assertEqual(run(DESIGN,base,out,max_new=1),1)
                first=next(out.glob('*.shard.json')); old=first.read_bytes()
                self.assertEqual(run(DESIGN,base,out,max_new=1,resume=True),1)
                self.assertEqual(first.read_bytes(),old)
                self.assertEqual(run(DESIGN,base,out,max_new=0,resume=True),0)
            job=manifest['jobs'][2]
            save_new(out/(Path(job['problem_file']).stem+'.started.json'),{'interrupted':True})
            with self.assertRaises(RuntimeError):run(DESIGN,base,out,max_new=1,resume=True)


if __name__=='__main__':unittest.main()
