"""Prepared sensor-only two-option predictor; no training at import time."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


def fresh_jobs() -> list[dict]:
    """Disjoint worlds; strata-balanced400/100/100 train/calibration/test worlds."""
    jobs=[]
    for i in range(600):
        split='train' if i%6<4 else 'calibration' if i%6==4 else 'test'
        # Each split sees all SOC levels equally up to rounding: rotate by block.
        soc=(.1,.2,.4)[(i//6)%3]
        for seed in range(3):
            jobs.append({'job_id':i*3+seed,'world_seed':1090000001+i,'head_seed':seed,
                         'method':2,'soc':soc,'split':split})
    return jobs


def paired_labels(row: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Depletion is an observed failure at this budget, not infinite-energy truth.

    Do not relabel other budgets from battery-killed trajectories. A global
    emergency guard is censoring, whereas option deadlines define real failure.
    """
    if not row['paired']:return None
    outcomes=[row['control'],row['continue_then_return']]
    labels=[];mask=[]
    allowed={'returned','energy_exhausted','return_deadline','task_step_limit','episode_emergency_step_guard'}
    for r in outcomes:
        if r['termination'] not in allowed:raise ValueError('unknown outcome semantics')
        safe=bool(r['returned'] and r['collision_count']==0)
        if safe!=r['safe_return']:raise ValueError('unified collision label mismatch')
        labels.append(float(safe));mask.append(r['termination']!='episode_emergency_step_guard')
    return np.array(labels,np.float32),np.array(mask,bool)


class FrozenOptionFeatures(nn.Module):
    """Single LiDAR encoding plus known task context and live battery.

    Frozen2081 return features + frozen32 task goal/velocity embedding + task
    clock1 + log1p(synthetic battery)1 =2115. No simulator layout, duplicate
    LiDAR convolution, head-seed identity or rollout outcome is supplied.
    """
    def __init__(self, extractor: nn.Module):
        super().__init__();self.extractor=extractor
        self.extractor.eval()
        for p in self.extractor.parameters():p.requires_grad_(False)

    def forward(self, nav:torch.Tensor, rth:torch.Tensor, battery:torch.Tensor) -> torch.Tensor:
        if nav.ndim!=2 or nav.shape[1]!=2056 or nav.shape!=rth.shape or battery.shape!=(len(nav),1):
            raise ValueError('two2056 views and battery column required')
        if not all(torch.isfinite(x).all() for x in (nav,rth,battery)) or (battery<0).any():
            raise ValueError('finite views and nonnegative live battery required')
        if not torch.equal(nav[:,:3],rth[:,:3]) or not torch.equal(nav[:,7:2055],rth[:,7:2055]):
            raise ValueError('both goals must share identical velocity and sensor state')
        if not torch.all(rth[:,-1]==1):raise ValueError('return option must start with fresh clock')
        self.extractor.eval()
        with torch.no_grad():
            x=self.extractor(rth)
            task=self.extractor.spatial.goal_encoder(nav[:,:7])
        return torch.cat((x,task,nav[:,-1:],torch.log1p(battery)),dim=1)


class OptionFeasibilityHead(nn.Module):
    """One shared128→128 body and two probability logits, not new RL critics."""
    def __init__(self, dim:int=2115):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,2))

    def forward(self,x:torch.Tensor)->torch.Tensor:return self.net(x)


def option_loss(logits:torch.Tensor,labels:torch.Tensor,mask:torch.Tensor,weights:torch.Tensor)->torch.Tensor:
    if logits.shape!=labels.shape or mask.shape!=labels.shape or logits.ndim!=2 or logits.shape[1]!=2:
        raise ValueError('aligned batch-by-two outcomes required')
    if weights.shape!=(len(logits),) or torch.any(weights<0) or weights.sum()<=0:
        raise ValueError('nonnegative scene weights required')
    if not all(torch.isfinite(x).all() for x in (logits,labels,weights)) or torch.any((labels!=0)&(labels!=1)):
        raise ValueError('finite logits and binary labels required')
    if mask.dtype!=torch.bool:raise ValueError('boolean censoring mask required')
    if not mask.any():raise ValueError('no observed outcomes')
    loss=nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    return (loss*mask*weights[:,None]).sum()/(mask*weights[:,None]).sum()
