"""Independent posterior-sampling baseline over bounded generated protocols."""
import random
from fractions import Fraction as F
from .beam_bayes import beam_protocols
from ..utility import protocol_value


class PosteriorSampling:
    def __init__(self,problem,seed=0,width=8,max_expansions=2000):
        self.problem,self.rng = problem,random.Random(seed)
        self.width,self.max_expansions = width,max_expansions

    def select(self,state):
        if state.resource != self.problem.capacity:
            raise ValueError("selection only at reset")
        index = self.rng.choices(range(len(state.posterior)),weights=list(map(float,state.posterior)))[0]
        belief = tuple(F(int(i == index)) for i in range(len(state.posterior)))
        routes,n,pruned = beam_protocols(self.problem,state.remaining,belief,self.width,self.max_expansions)
        self.stats = {"expansions":n,"truncated":pruned}
        return max(routes,key=lambda r:protocol_value(self.problem,belief,r)/r.duration,default=None)
