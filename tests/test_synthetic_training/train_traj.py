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

from common import load_csv, get_summary_writer

from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import hydra


from functools import partial
from synthetic.data_generator import generate_data


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def print_cfg(cfg):
    from collections import defaultdict
    from colorama import Fore, Style, init
    init(autoreset=True)
    def default_styles():
        return Fore.WHITE
    styles = defaultdict(default_styles)
    styles.update({'name': Fore.GREEN,
                   'lr_init': Fore.LIGHTBLUE_EX,
                   'continue_training': Fore.LIGHTRED_EX,})

    checklist = ['model', 'estimator', 'dataset', 'task']
    for k in checklist:
        print('\n')
        print('#'*15 + f' {k} config ' + '#'*15)
        cf_k = cfg[k]
        for kk, vv in cf_k.items():
            print(styles[kk] + f'{kk}: {vv}')
        print('#'*15 + f' {k} ' + '(END)' + '#'*15)

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

        # data_got = self._augment_data(data_got)

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
    loss = criterion(pN_est, pN_gt)
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
    single_data = data[0:1,:,:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    with torch.no_grad():
        pN_est = autoregr(single_data, model, est, cfg)
        pN_gt = single_data[:, :, 2:5]
        pN_gt = pN_gt.cpu().numpy().squeeze()
        pN_est_i = pN_est.cpu().numpy().squeeze()

        ax.plot(pN_gt[:, 0], pN_gt[:, 1], pN_gt[:, 2], label='gt')
        ax.plot(pN_est_i[:, 0], pN_est_i[:, 1], pN_est_i[:, 2], label='est')
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

def train_loop(cfg):

    # print out config info
    print_cfg(cfg)
    
    # get dataloaders
    train_loader, test_loader = TrajectoryDataset.get_dataloaders(cfg)

    # get summary writer
    tb_writer, initial_step = get_summary_writer(cfg)

    # get model
    model, est, autoregr = get_model(cfg)
    if cfg.model.continue_training:
        model_path = Path(tb_writer.get_logdir())/f'model_{cfg.model.name}.pth'
        model.load_state_dict(torch.load(model_path))
        print(f"model loaded from {model_path}")

        if cfg.estimator.name == 'SlideWindowEstimator':
            est_path = Path(tb_writer.get_logdir())/f'est_{cfg.estimator.name}.pth'
            est.load_state_dict(torch.load(est_path))
            print(f"est loaded from {est_path}")

    opt_params = list(model.parameters()) + list(est.parameters()) if cfg.estimator.name == 'SlideWindowEstimator' \
                else list(model.parameters())
    
    optimizer = torch.optim.Adam(opt_params, lr=cfg.model.lr_init, weight_decay=1e-3)
    criterion = nn.MSELoss()

    best_valid_loss = torch.inf
    for epoch in range(cfg.model.num_epochs):
        for i, data in enumerate(train_loader):
            data = augment_data(data)
            optimizer.zero_grad()
            loss = compute_loss(model,est, autoregr, data, criterion, cfg)
            loss.backward()
            optimizer.step()
            tb_writer.add_scalars('loss', {'training': loss.item()}, initial_step)
            initial_step += 1
            if i % 100 == 0:
                print(f'epoch: {epoch} iter: {i} training loss: {loss.item()}')


        if epoch % cfg.model.valid_interval == 0:
            valid_loss = compute_valid_loss(model,est, autoregr, test_loader, criterion, cfg)
            tb_writer.add_scalars('loss', {'validation': valid_loss}, initial_step)
            tb_writer.add_figure('visualization', visualize_traj(model, est, autoregr, next(iter(test_loader)), cfg), initial_step)
            print(f'epoch: {epoch} validation loss: {valid_loss}')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                model_path = Path(tb_writer.get_logdir())/f'model_{cfg.model.name}.pth'
                torch.save(model.state_dict(), model_path)
                print(f"model saved to {model_path}")

                if cfg.estimator.name == 'SlideWindowEstimator':
                    est_path = Path(tb_writer.get_logdir())/f'est_{cfg.estimator.name}.pth'
                    torch.save(est.state_dict(), est_path)
                    print(f"est saved to {est_path}")
    


@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg):
    if cfg.task.generate_data:
        generate_data(cfg)
    if cfg.task.train:
        train_loop(cfg)
    


if __name__ == '__main__':
    main()
