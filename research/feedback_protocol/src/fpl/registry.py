"""Conservative, name-invariant structural grouping for split enforcement."""
import hashlib
import json
from dataclasses import dataclass, field


def structural_key(problem):
    # Colored operation/node/channel incidence graph; ignore names, rewards,
    # durations, capacity, budget and public probabilities. WL collisions merge
    # extra families (conservative), never separate isomorphic clones.
    colors, edges = {}, []
    for n in problem.nodes:
        colors[("node",n)] = "reset" if n == problem.reset else "node"
    for c in problem.channels:
        colors[("channel",c)] = "channel"
    for op in problem.operations:
        key = ("op",op.name)
        colors[key] = "operation"
        edges.extend([(key,("node",op.source),"source"),(key,("node",op.target),"target")])
        edges.extend((key,("channel",c),"observation") for c in op.channels)
    for _ in range(len(colors)):
        nxt = {}
        for node, color in colors.items():
            neighbors = sorted((role, colors[b if a == node else a])
                               for a,b,role in edges if node in (a,b))
            nxt[node] = hashlib.sha256(json.dumps([color,neighbors]).encode()).hexdigest()
        colors = nxt
    return hashlib.sha256(json.dumps(sorted(colors.values())).encode()).hexdigest()


@dataclass
class Registry:
    entries: dict = field(default_factory=dict)
    structural_splits: dict = field(default_factory=dict)
    family_splits: dict = field(default_factory=dict)

    def register(self, problem):
        if problem.split_id not in ("train","dev","test","ood","debug-only"):
            raise ValueError("unknown split")
        key = structural_key(problem)
        for table, group in ((self.structural_splits,key),(self.family_splits,problem.family_id)):
            if group in table and table[group] != problem.split_id:
                raise ValueError("family/topology clone crosses splits")
        digest = problem.instance_hash
        self.entries[digest] = dict(family_id=problem.family_id, split_id=problem.split_id, structural_key=key)
        self.structural_splits[key] = problem.split_id
        self.family_splits[problem.family_id] = problem.split_id
        return digest

    def to_dict(self):
        return {"schema":"fpl-registry-v1", "entries":self.entries,
                "structural_splits":self.structural_splits,"family_splits":self.family_splits}

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != "fpl-registry-v1":
            raise ValueError("unsupported registry")
        obj = cls(data["entries"],data["structural_splits"],data["family_splits"])
        for row in obj.entries.values():
            if (obj.structural_splits.get(row["structural_key"]) != row["split_id"] or
                obj.family_splits.get(row["family_id"]) != row["split_id"]):
                raise ValueError("inconsistent registry")
        return obj


@dataclass
class RegistryV2:
    """Distribution membership is not split identity; structural clones are.

    WL hash is conservative: collisions may reject extra structures. It is
    not a proof of graph non-isomorphism. Reskins and capacity/budget variants
    stay together; template and distribution may occur in several IID splits.
    """
    entries: dict = field(default_factory=dict)
    clone_splits: dict = field(default_factory=dict)

    def register(self,problem,*,distribution_id,template_id,parameter_seed,
                 structure_seed,ood_axis=None,group_id=None):
        split = problem.split_id
        if split not in ("train","dev","test","ood","debug-only"):
            raise ValueError("unknown split")
        if (split=="ood") != (ood_axis is not None):
            raise ValueError("OOD split requires an explicit OOD axis, and vice versa")
        if ood_axis not in (None,"size","topology","cost_ratio","hypothesis_count"):
            raise ValueError("unknown OOD axis")
        key = structural_key(problem)
        if key in self.clone_splits and self.clone_splits[key]!=split:
            raise ValueError("structural clone crosses splits")
        row = dict(distribution_id=distribution_id,template_id=template_id,clone_hash=key,
                   parameter_seed=parameter_seed,structure_seed=structure_seed,ood_axis=ood_axis,
                   split=split,group_id=group_id or key,instance_hash=problem.instance_hash)
        if problem.instance_hash in self.entries and self.entries[problem.instance_hash]!=row:
            raise ValueError("conflicting identity")
        self.entries[problem.instance_hash] = row
        self.clone_splits[key] = split
        return problem.instance_hash

    def to_dict(self):
        return dict(schema="fpl-registry-v2",entries=self.entries,clone_splits=self.clone_splits)

    @classmethod
    def from_dict(cls,data):
        if data.get("schema")!="fpl-registry-v2":
            raise ValueError("unsupported registry")
        obj = cls(data["entries"],data["clone_splits"])
        for digest,row in obj.entries.items():
            if digest!=row["instance_hash"] or obj.clone_splits.get(row["clone_hash"])!=row["split"]:
                raise ValueError("inconsistent registry")
        return obj
