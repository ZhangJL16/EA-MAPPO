import ast
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from bdr.core import Policy, coupled_policy, mean_signal, random_inputs, rollout, information_samples, objective
from bdr.run import train, load_checkpoint_policy, evaluate


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_signal_analytic(self):
        theta = torch.tensor([[[0., 0.], [1., 0.]]], dtype=torch.double)
        design = torch.tensor([[0., 1.]], dtype=torch.double)
        expected = math.log(.1 + 1 / 1.0001 + 1 / 2.0001)
        self.assertAlmostEqual(mean_signal(design, theta).item(), expected)

    def test_permutation_and_budget_boundary(self):
        h = torch.randn(3, 4, 3)
        for kind in ["pool", "fixed_attention", "budget_attention", "dad_native"]:
            p = coupled_policy(kind, 44)
            torch.testing.assert_close(p(h, 4), p(h[:, [3, 1, 0, 2]], 4))
            self.assertTrue(torch.isfinite(p(h[:, :0], 4)).all())
            if kind != "dad_native":
                self.assertFalse(torch.allclose(p(h, 4), p(h, 8)))

    def test_three_main_variants_start_equal(self):
        h = torch.randn(5, 3, 3)
        pool = coupled_policy("pool", 55)
        for kind in ["fixed_attention", "budget_attention"]:
            torch.testing.assert_close(pool(h, 4), coupled_policy(kind, 55)(h, 4))

    def test_query_intervention_keeps_emitter_budget(self):
        p = coupled_policy("budget_attention", 6)
        with torch.no_grad():
            p.query_slope.fill_(1)
        h = torch.randn(3, 4, 3)
        self.assertFalse(torch.allclose(p(h, 4), p(h, 4, query_override=0)))
        torch.testing.assert_close(p(h[:, :0], 4), p(h[:, :0], 4, query_override=0))

    def test_spce_independent_loop_and_gradient(self):
        p = coupled_policy("pool", 3).double()
        theta, noise, inner = (x.double() for x in random_inputs(77, 3, 3, 5))
        history = rollout(p, theta, noise)
        lower, upper = information_samples(history, theta, inner, chunk=2)
        brute = []
        for n in range(3):
            vals = []
            for particle in torch.cat([theta[n:n+1], inner[:, n]], 0):
                likelihood = history.new_tensor(0.)
                for t in range(3):
                    mu = torch.log(.1 + (1e-4 + (history[n,t,:2] - particle).square().sum(-1)).reciprocal().sum())
                    likelihood += -.5 * ((history[n,t,2] - mu) / .5).square() - math.log(.5 * math.sqrt(2 * math.pi))
                vals.append(likelihood)
            vals = torch.stack(vals)
            brute.append(vals[0] - vals.logsumexp(0) + math.log(6))
        torch.testing.assert_close(lower, torch.stack(brute))
        self.assertTrue((lower <= math.log(6) + 1e-12).all())
        (-lower.mean()).backward()
        analytic = p.emitter.bias.grad[0].item()
        original = p.emitter.bias[0].item()
        values = []
        for delta in [1e-6, -1e-6]:
            with torch.no_grad():
                p.emitter.bias[0] = original + delta
            values.append(-objective(p, (theta, noise, inner)).item())
        self.assertAlmostEqual(analytic, (values[0] - values[1]) / 2e-6, places=4)

    def test_noise_coupling_and_source_symmetry(self):
        a, b = random_inputs(1, 2, 4, 5), random_inputs(1, 2, 4, 5)
        for x, y in zip(a,b):
            self.assertTrue(torch.equal(x,y))
        p = coupled_policy("pool", 1)
        torch.testing.assert_close(rollout(p,a[0],a[1]), rollout(p,a[0].flip(1),a[1]))

    def test_resume_exact(self):
        config = dict(methods=["pool"], seeds=[123], steps=4, hidden=8, encoding=4,
            lr=.0003, batch=4, contrasts=3, train_horizons=[2,3], native_horizon=3,
            gradient_clip=10., checkpoint_every=2, hard_timeout_seconds=60)
        with tempfile.TemporaryDirectory() as d:
            full, resumed = Path(d)/"full", Path(d)/"resumed"
            train(config, full, "cpu")
            train(config, resumed, "cpu", max_updates=2)
            train(config, resumed, "cpu")
            x = torch.load(full/"pool_123.pt", weights_only=False)
            y = torch.load(resumed/"pool_123.pt", weights_only=False)
            self.assertEqual(x["log"],y["log"])
            for k,v in x["state_dict"].items():
                self.assertTrue(torch.equal(v, y["state_dict"][k]))
            loaded,_=load_checkpoint_policy(full/"pool_123.pt",config,"cpu")
            initial=coupled_policy("pool",123,8,4)
            h=torch.randn(4,2,3)
            self.assertFalse(torch.allclose(loaded(h,2),initial(h,2)))
            for k,v in loaded.state_dict().items():
                self.assertTrue(torch.equal(v,x["state_dict"][k]))

    def test_evaluation_uses_checkpoint_and_resumes(self):
        config=dict(methods=["pool"], seeds=[123], steps=2, hidden=8, encoding=4,
            lr=.003,batch=4,contrasts=3,train_horizons=[2],native_horizon=2,
            gradient_clip=10.,checkpoint_every=2,hard_timeout_seconds=60,
            eval_horizons=[2],eval_rollouts=4,eval_batch=4,eval_contrasts=3,eval_seed=5000)
        with tempfile.TemporaryDirectory() as d:
            train(config,d,"cpu")
            evaluate(config,d,"cpu")
            path=Path(d)/"evaluation.json"
            row=json.loads(path.read_text())[0]
            loaded,_=load_checkpoint_policy(Path(d)/"pool_123.pt",config,"cpu")
            inputs=random_inputs(7000,4,2,3)
            self.assertAlmostEqual(row["lower_mean"],objective(loaded,inputs).item(),places=5)
            original=(Path(d)/"eval_pool_123_normal_H2.json").read_bytes()
            evaluate(config,d,"cpu")
            self.assertEqual(original,(Path(d)/"eval_pool_123_normal_H2.json").read_bytes())

    @unittest.skipUnless(os.environ.get("DAD_REFERENCE"), "optional pinned author checkout")
    def test_actual_author_source(self):
        # Execute only reviewed pure Torch AST nodes; avoid obsolete Pyro/MLflow.
        root = Path(os.environ["DAD_REFERENCE"])
        scope = {"torch": torch, "nn": nn}
        definitions = []
        for filename, names in [("location_finding.py", {"EncoderNetwork", "EmitterNetwork"}),
                                ("neural/modules.py", {"SetEquivariantDesignNetwork"})]:
            tree = ast.parse((root/filename).read_text())
            definitions.extend(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name in names)
        exec(compile(ast.Module(body=definitions,type_ignores=[]), "author_ast", "exec"), scope)
        encoder = scope["EncoderNetwork"]((1,2),1,64,16)
        emitter = scope["EmitterNetwork"](16,(1,2))
        ours = coupled_policy("dad_native", 33)
        encoder.linear1.load_state_dict(ours.encoder[0].state_dict())
        encoder.output_layer.load_state_dict(ours.encoder[2].state_dict())
        emitter.linear.load_state_dict(ours.emitter.state_dict())
        author = scope["SetEquivariantDesignNetwork"](encoder, emitter, torch.ones(1,2)*.01)
        h = torch.randn(3,4,3)
        pairs = [(h[:,t,:2].unsqueeze(1), h[:,t,2:]) for t in range(4)]
        torch.testing.assert_close(author(*pairs).squeeze(1), ours(h,4))
        torch.testing.assert_close(author().expand(3,2), ours(h[:,:0],4))
        source = ast.parse((root/"location_finding.py").read_text())
        cls = next(x for x in source.body if isinstance(x, ast.ClassDef) and x.name == "HiddenObjects")
        fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "forward_map")
        exec(compile(ast.Module(body=[fn],type_ignores=[]), "author_forward", "exec"), scope)
        theta, _, _ = random_inputs(6,3,4,5)
        got = scope["forward_map"](SimpleNamespace(base_signal=.1,max_signal=1e-4), h[:,0,:2].unsqueeze(1), theta)
        torch.testing.assert_close(got.squeeze(-1),mean_signal(h[:,0,:2],theta))


if __name__ == "__main__":
    unittest.main()
