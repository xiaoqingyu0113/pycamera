import torch
from lfg.model_traj import PhyTune
model = PhyTune()

model.load_state_dict(torch.load('logdir/traj_train/PhyTune/nonoise_consspin_nobounce/GT/run01/model_PhyTune.pth'))

for p in model.parameters():
    print(p)