from lfg.estimator import OptimLayer
from lfg.model_traj import PhyTune
from train_traj import TrajectoryDataset

import torch
from torch.utils.data import DataLoader

def test():
    model = PhyTune()
    est = OptimLayer(model, size=32)
    model.to('cuda:0')
    est.to('cuda:0')
    
    model.load_state_dict(torch.load('logdir/traj_train/PhyTune/nonoise_consspin_nobounce/GT/run01/model_PhyTune.pth'))

    dataset = TrajectoryDataset('data/synthetic/traj_nonoise_consspin_nobounce.csv', noise=10e-3)
    dataloader  = DataLoader(dataset, batch_size=4, shuffle=True)

    data = next(iter(dataloader))
    tN = data[:,:, 1:2]
    pN = data[:,:, 2:5]
    p0 = pN[:, 0:1, :]
    v0 = data[:, 0:1, 5:8]
    w0 = data[:, 0:1, 8:11]

    p0_est, v0_est, w0_est = est(data[:,:est.size,1:5], w0=w0)


    print(f"p0: {p0.cpu()}")
    print(f"p0_est: {p0_est.detach().cpu()}")
    print(f"v0: {v0.cpu()}")
    print(f"v0_est: {v0_est.detach().cpu()}")
    print(f"w0: {w0.cpu()}")
    print(f"w0_est: {w0_est.detach().cpu()}")
    

if __name__ == '__main__':
    test()
