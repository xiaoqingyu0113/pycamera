import theseus as th
import torch
import torch.nn as nn
from typing import Tuple, Sequence, Optional



class InvariantFactorGraph(nn.Module):
    '''
        invariant factor graph with no camera
    '''
    def __init__(self):
        super().__init__()


        # Add a theseus layer with a single cost function whose error depends on the NN
        self.objective = th.Objective()
        
    
    def add(self, cost_function: th.AutoDiffCostFunction):
        self.objective.add(cost_function)
     

    def forward(self, ival):
        self.optimizer = th.LevenbergMarquardt(self.objective)
        self.layer = th.TheseusLayer(self.optimizer)
        sol, info = self.layer.forward(ival)
        return sol, info
    
    def update(self,update_inputs):
        self.objective.update(update_inputs)

    def compute_loss(self):
        return self.objective.error_metric()