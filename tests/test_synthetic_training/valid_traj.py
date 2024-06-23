import os
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'


from pathlib import Path
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


from draw_util import draw_util
import matplotlib.pyplot as plt

from common import load_csv, get_summary_writer, get_summary_writer_path

from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import hydra


from functools import partial
from synthetic.data_generator import generate_data


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def print_cfg(cfg):
    checklist = ['model', 'estimator', 'dataset', 'task']

    txt = ''
    for k in checklist:
        txt += '\n'
        txt += '#'*15 + f' {k} config ' + '#'*15 + '\n'
        cf_k = cfg[k]
        for kk, vv in cf_k.items():
            txt += f'{kk}: {vv}\n'
        txt += '#'*15 + f' {k} ' + '(END)' + '#'*15 + '\n'
    print(txt)


class TrajectoryDataset(Dataset):
    def __init__(self, csv_file, noise=0.0):
        self.data = torch.from_numpy(load_csv(csv_file)).to(DEVICE).float()
        self.noise = noise


    def __len__(self):
        return int(self.data[-1, 0]) + 1 

    def __getitem__(self, idx):
        idx_tensor = torch.tensor(idx).to(DEVICE)
        mask = torch.isin(self.data[:, 0].int(), idx_tensor.int())
        data_got = self.data[mask, :]
        data_got[:, 2:5] += torch.randn_like(data_got[:, 2:5])*self.noise
        return data_got

    


    @classmethod
    def get_dataloaders(cls, config):
        folder = Path(config.dataset.folder)
        dataset = cls(folder / config.dataset.trajectory_data, noise=config.model.add_noise)
        total_data_size = len(dataset)
        split_ratio = config.model.training_data_split
        train_data_size = int(total_data_size * split_ratio)
        test_data_size = total_data_size - train_data_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_data_size, test_data_size])
        
        train_loader = DataLoader(train_dataset, batch_size=config.dataset.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.dataset.batch_size, shuffle=False)
        return train_loader, test_loader


def get_model(cfg):
    from lfg import model_traj
    from lfg import estimator

    Model = getattr(model_traj, cfg.model.name)
    model = Model(*cfg.model.args, **cfg.model.kwargs)
    model = model.to(DEVICE)

    if cfg.estimator.name == 'GT':
        est = None
    elif cfg.estimator.name == 'SlideWindowEstimator':
        Estimator = getattr(estimator, cfg.estimator.name)
        est = Estimator(*cfg.estimator.args, **cfg.estimator.kwargs)
        est = est.to(DEVICE)
    else:
        Estimator = getattr(estimator, cfg.estimator.name)
        est = Estimator(model, **cfg.estimator.kwargs)
        est = est.to(DEVICE)

    autoregr = getattr(model_traj, f'autoregr_{cfg.model.name}')
    return model, est, autoregr

def compute_loss(model, est, autoregr, data, criterion,cfg):
    pN_est = autoregr(data, model, est, cfg)
    pN_gt = data[:, :,2:5]

    loss = 0.0

    if 'pos' in cfg.model.loss_type:
        loss += criterion(pN_est, pN_gt)

    if 'vel' in cfg.model.loss_type:
        d_pN_est = torch.diff(pN_est, dim=1)
        d_pN_gt = torch.diff(pN_gt, dim=1)
        loss += criterion(d_pN_est, d_pN_gt)*100.0
    
    if 'acc' in cfg.model.loss_type :
        dd_pN_est = torch.diff(torch.diff(pN_est, dim=1), dim=1)
        dd_pN_gt = torch.diff(torch.diff(pN_gt, dim=1), dim=1)
        loss += criterion(dd_pN_est, dd_pN_gt)*100.0
    

    return loss


def compute_valid_loss(model, est, autoregr, test_loader, criterion, cfg):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for data in test_loader:
            loss = compute_loss(model, est, autoregr, data, criterion, cfg)
            total_loss += loss.item()* data.shape[0]
    model.train()  # Set the model back to training mode
    return total_loss / len(test_loader)

def visualize_traj(model, est, autoregr, data, cfg):
    model.eval()
    single_data = data[:3,:,:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    with torch.no_grad():
        pN_est = autoregr(single_data, model, est, cfg)
        pN_gt = single_data[:, :, 2:5]
        pN_gt = pN_gt.cpu().numpy()
        pN_est = pN_est.cpu().numpy()

        for batch in range(single_data.shape[0]):
            ax.plot(pN_gt[batch, :, 0], pN_gt[batch, :, 1], pN_gt[batch, :, 2], label=f'gt_{batch}')
            ax.plot(pN_est[batch, :, 0], pN_est[batch, :, 1], pN_est[batch, :, 2], label=f'est_{batch}')

        ax.legend()
        draw_util.set_axes_equal(ax, zoomin=2.0)

    model.train()
    return fig

def augment_data( data):
        '''
        data = [batch, seq_len, 11]
        rotate the data along z axis randomly from 0 to 2pi
        use device = DEVICE
        '''
        theta = torch.rand(1,device=DEVICE )*2* torch.pi
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                [torch.sin(theta), torch.cos(theta), 0],
                                [0, 0, 1]], device=DEVICE).float()
        rot_mat = rot_mat.unsqueeze(0).repeat(data.shape[0], 1, 1)
        p_xyz = data[:,:,2:5]
        p_xyz = torch.matmul(p_xyz - p_xyz[:,0:1,:], rot_mat) + p_xyz[:,0:1,:]
        data[:,:,2:5] = p_xyz
        data[:,:,8:11] = torch.matmul(data[:,:,8:11], rot_mat)
        data[:,:,5:8] = torch.matmul(data[:,:,5:8], rot_mat)
        return data


def hessian_test(model:torch.nn.modules, est, autoregr, test_loader, criterion, cfg):
    # from torch.autograd.functional import hessian, jacobian
    from torch.autograd import grad

    def model_functional_wrapper(input):

        model.param1 = nn.Parameter(input[0].view(1,1))
        model.param2 = nn.Parameter(input[1].view(1,1))
        model.param3 = nn.Parameter(input[2:5].view(1,3))
        model.param4 = nn.Parameter(input[5])
        model.param5 = nn.Parameter(input[6])
        model.param6 = nn.Parameter(input[7])


        loss = compute_loss(model, est, autoregr, next(iter(test_loader)), criterion, cfg)
        return loss
    
    inputs = torch.cat([p.view(-1) for p in model.parameters()])
    loss = model_functional_wrapper(inputs)
    grads = grad(loss, model.parameters(), create_graph=True)
    
    print(grads)
    grads_flat = torch.cat([g.view(-1) for g in grads])
    n = grads_flat.size(0)

    # Initialize the Hessian matrix
    H = torch.zeros(n, n)

    for idx in range(n):
        grad2 = grad(grads_flat[idx], model.parameters(), retain_graph=True)
        grad2_flat = torch.cat([g.view(-1) for g in grad2])
        H[idx] = grad2_flat

    H = H.detach()
    print(H)
    eigens = torch.linalg.eigvals(H)
    print(f'eig(H): {eigens}')



def validate(cfg):

    cfg.model.continue_training=True
    # print out config info
    print_cfg(cfg)
    
    # get dataloaders
    train_loader, test_loader = TrajectoryDataset.get_dataloaders(cfg)

    # get model
    model, est, autoregr = get_model(cfg)
    model._init_gt() # set ground truth for model
    model.cuda()

    # load from save
    # model_path = get_summary_writer_path(cfg)/f'model_{cfg.model.name}.pth'
    # model.load_state_dict(torch.load(model_path))
    
    criterion = nn.MSELoss()

    data = next(iter(train_loader))
    if cfg.model.augment_data:
        data = augment_data(data)

    
    # hessian_test(model, est, autoregr, test_loader, criterion, cfg)
    valid_loss = compute_valid_loss(model, est, autoregr, test_loader, criterion, cfg)

    print(f'Validation Loss: {valid_loss}')

    from synthetic.predictor import predict_trajectory
    fig = visualize_traj(model, est, autoregr, data, cfg)
    plt.show()



@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg):
    validate(cfg)
    


if __name__ == '__main__':
    main()
