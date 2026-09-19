import copy
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prefix_export',ROOT/'scripts/export_teacher_targets.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture():
    p=dict(reset='q',capacity=3,max_measurements=1,per_channel_limit=1,operations=[
        dict(name='start',source='q',target='x',energy=1,duration=1,channels=['c']),
        dict(name='a',source='x',target='q',energy=2,duration=1,channels=[]),
        dict(name='b',source='x',target='q',energy=1,duration=2,channels=[]),
        dict(name='dead',source='x',target='z',energy=1,duration=1,channels=[])])
    r=dict(state=dict(remaining=3),label=dict(status='exact',value='2',candidates=[
        dict(operations=None,q='0'),dict(operations=['start','a'],q='2'),dict(operations=['start','b'],q='2')]))
    return p,r


class PrefixTests(unittest.TestCase):
    def test_ties_viability_and_advantages(self):
        p,r=fixture()
        rows=mod.derive_prefixes(p,r)
        mod.verify_prefixes(p,r,rows)
        u=next(x for x in rows if x['prefix']==['start'])
        self.assertEqual(len(u['optimal_next']),2)
        self.assertEqual(set(u['advantage'].values()),{'0'})
        self.assertNotIn(mod.action_key('operation','dead'),u['legal_next'])
        self.assertNotIn(mod.STOP,u['legal_next'])
        self.assertEqual(u['position']['remaining_resource'],2)
        self.assertEqual(u['position']['pending_channel_counts'],{'c':1})

    def test_end_is_not_stop_and_pre_reload_debit(self):
        p,r=fixture()
        rows=mod.derive_prefixes(p,r)
        end=next(x for x in rows if x['prefix']==['start','a'])
        self.assertEqual(end['legal_next'],[mod.END])
        self.assertEqual(end['value_prefix'],'2')
        self.assertEqual(end['position']['remaining_resource'],0)
        self.assertEqual(end['position']['resource_after_return'],3)
        self.assertEqual(end['policy_weight'],'0')
        self.assertEqual(end['normalized_value_prefix'],'2/3')

    def test_state_total_weight_and_negative_alternative(self):
        p,r=fixture()
        r['label']['candidates'][2]['q']='-1'
        rows=mod.derive_prefixes(p,r)
        mod.verify_prefixes(p,r,rows)
        self.assertEqual(sum(F(x['policy_weight']) for x in rows),1)
        u=next(x for x in rows if x['prefix']==['start'])
        self.assertEqual(u['advantage'][mod.action_key('operation','b')],'-3')
        bad=copy.deepcopy(rows)
        bad[0]['q_next'][mod.STOP]='9'
        with self.assertRaises(ValueError):mod.verify_prefixes(p,r,bad)

    def test_stop_only_zero_horizon(self):
        p,r=fixture()
        r['state']['remaining']=0
        r['label'].update(value='0',candidates=[dict(operations=None,q='0')])
        rows=mod.derive_prefixes(p,r)
        mod.verify_prefixes(p,r,rows)
        self.assertEqual(rows[0]['optimal_next'],[mod.STOP])
        self.assertEqual(rows[0]['policy_weight'],'1')
        r['label']['status']='unresolved'
        self.assertEqual(mod.derive_prefixes(p,r),[])

    def test_group_holdout_never_splits_reskins(self):
        shards=[dict(metadata=dict(root_seed=i,clone_hash=str(i//2))) for i in range(16)]
        folds=mod.group_folds(shards,'fixed')
        self.assertEqual(list(folds.values()).count('student_validation'),2)
        self.assertEqual(folds,mod.group_folds(list(reversed(shards)),'fixed'))
        shards.append(dict(metadata=dict(root_seed=0,clone_hash='other')))
        with self.assertRaises(ValueError):mod.group_folds(shards,'fixed')

    def test_real_export_reproducible_without_solver(self):
        src=ROOT/'provenance/teacher_v1_complete'
        with tempfile.TemporaryDirectory() as d:
            args=(src/'teacher_v1_complete_evidence.zip',src/'audit.json',ROOT/'configs/teacher_prefix_export_v1.json')
            a=mod.export(*args,Path(d)/'a')
            b=mod.export(*args,Path(d)/'b')
            self.assertEqual(a,b)
            self.assertEqual(a['summary']['states'],273)
            self.assertEqual(a['solver_calls'],0)
            self.assertEqual(a['excluded_states'],0)
            with self.assertRaises(FileExistsError):mod.export(*args,Path(d)/'a')
            for name in a['files']:
                self.assertEqual((Path(d)/'a'/name).read_bytes(),(Path(d)/'b'/name).read_bytes())


if __name__=='__main__':unittest.main()
