import torch


class Node:
    def __init__(self, name:str, value:torch.Tensor):
        self.name = name
        name_split = name.split('_')
        self.prefix = name_split[0]
        self.nid = int(name_split[1])
        self.value = value
        self.size = value.shape[0]
        
    def __repr__(self):
        return f'{self.name} = {self.value}'

class L(Node):
    def __init__(self, nid:int, value:torch.Tensor):
        super().__init__('L_'+str(nid), value)