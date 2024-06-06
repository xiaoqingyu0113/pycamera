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
    checklist = ['model', 'dataset', 'task']
    for k in checklist:
        print('\n')
        print('#'*15 + f' {k} config ' + '#'*15)
        cf_k = cfg[k]
        for kk, vv in cf_k.items():
            print(f'{kk}: {vv}')
        print('#'*15 + f' {k} ' + '(END)' + '#'*15)

class TrajectoryDataset(Dataset):
    def __init__(self, csv_file):
        self.data = torch.from_numpy(load_csv(csv_file)).to(DEVICE).float()

    def __len__(self):
        return int(self.data[-1, 0]) + 1 

    def __getitem__(self, idx):
        idx_tensor = torch.tensor(idx).to(DEVICE)
        mask = torch.isin(self.data[:, 0].int(), idx_tensor.int())
        return self.data[mask, :]


    @classmethod
    def get_dataloaders(cls, config):
        folder = Path(config.dataset.folder)
        dataset = cls(folder / config.dataset.trajectory_data)
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
    Model = getattr(model_traj, cfg.model.name)
    model = Model(*cfg.model.args, **cfg.model.kwargs)
    model = model.to(DEVICE)
    autoregr = getattr(model_traj, f'autoregr_{cfg.model.name}')
    return model, autoregr

def compute_loss(model, autoregr, data, criterion,cfg):
    pN_est = autoregr(data, model, cfg)
    pN_gt = data[:, :,2:5]
    loss = criterion(pN_est, pN_gt)
    return loss

def compute_valid_loss(model, autoregr, test_loader, criterion, cfg):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for data in test_loader:
            loss = compute_loss(model, autoregr, data, criterion, cfg)
            total_loss += loss.item()* data.shape[0]
    model.train()  # Set the model back to training mode
    return total_loss / len(test_loader)

def visualize_traj(model, autoregr, data, cfg):
    model.eval()
    single_data = data[0:1,:,:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    with torch.no_grad():
        pN_est = autoregr(single_data, model, cfg)
        pN_gt = single_data[:, :, 2:5]
        pN_gt = pN_gt.cpu().numpy().squeeze()
        pN_est_i = pN_est.cpu().numpy().squeeze()

        ax.plot(pN_gt[:, 0], pN_gt[:, 1], pN_gt[:, 2], label='gt')
        ax.plot(pN_est_i[:, 0], pN_est_i[:, 1], pN_est_i[:, 2], label='est')
        ax.legend()
        draw_util.set_axes_equal(ax)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    model.train()
    return fig

def train_loop(cfg):
    print_cfg(cfg)
    
    # get dataloaders
    train_loader, test_loader = TrajectoryDataset.get_dataloaders(cfg)

    # get summary writer
    tb_writer, initial_step = get_summary_writer(cfg)

    # get model
    model, autoregr = get_model(cfg)
    if cfg.model.continue_training:
        model_path = Path(tb_writer.get_logdir())/f'model_{cfg.model.model_name}.pth'
        model.load_state_dict(torch.load(model_path))
        print(f"model loaded from {model_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr_init)
    criterion = nn.MSELoss()

    best_valid_loss = torch.inf
    for epoch in range(cfg.model.num_epochs):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            loss = compute_loss(model, autoregr, data, criterion, cfg)
            loss.backward()
            optimizer.step()
            tb_writer.add_scalars('loss', {'training': loss.item()}, initial_step)
            initial_step += 1
            if i % 100 == 0:
                print(f'epoch: {epoch} iter: {i} training loss: {loss.item()}')


        if epoch % cfg.model.valid_interval == 0:
            valid_loss = compute_valid_loss(model, autoregr, test_loader, criterion, cfg)
            tb_writer.add_scalars('loss', {'validation': valid_loss}, initial_step)
            tb_writer.add_figure('visualization', visualize_traj(model, autoregr, next(iter(test_loader)), cfg), initial_step)
            print(f'epoch: {epoch} validation loss: {valid_loss}')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                model_path = Path(tb_writer.get_logdir())/f'model_{cfg.model.name}.pth'
                torch.save(model.state_dict(), model_path)
                print(f"model saved to {model_path}")

    







@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg):

    if cfg.task.generate_data:
        generate_data(cfg)

    if cfg.task.train:
        train_loop(cfg)
    


if __name__ == '__main__':
    main()
