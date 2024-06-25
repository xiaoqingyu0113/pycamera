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

class Constraint:
    def small_param_thresh(model):
        threshold = 1e-20  # Define a threshold
        for name, param in model.named_parameters():
            with torch.no_grad():
                param.data = torch.where(param.data.abs() < threshold, torch.tensor(threshold, dtype=param.data.dtype), param.data)
    def positive_param(model):
        for name, param in model.named_parameters():
            if 'model' not in name:
                with torch.no_grad():
                    param.data = torch.where(param.data < 0.0, torch.tensor(0.0, dtype=param.data.dtype), param.data)


def train_loop(cfg):

    # print out config info
    print_cfg(cfg)
    
    # get dataloaders
    train_loader, test_loader = TrajectoryDataset.get_dataloaders(cfg)

    # get summary writer
    tb_writer, initial_step = get_summary_writer(cfg)
    tb_writer.add_text('config', f'```yaml\n{OmegaConf.to_yaml(cfg)}\n```',global_step=initial_step)

    # get model
    model, est, autoregr = get_model(cfg)
    if cfg.model.continue_training:
        model_path = Path(tb_writer.get_logdir())/f'model_{cfg.model.name}.pth'
        model.load_state_dict(torch.load(model_path))
        print(f"model loaded from {model_path}")

        if cfg.estimator.name != 'GT':
            est_path = Path(tb_writer.get_logdir())/f'est_{cfg.estimator.name}.pth'
            est.load_state_dict(torch.load(est_path))
            print(f"est loaded from {est_path}")


    opt_params = list(model.parameters()) + [p for n, p in est.named_parameters() if 'model' not in n] if cfg.estimator.name != 'GT' \
                else model.parameters()
    
    optimizer = torch.optim.Adam(opt_params, lr=cfg.model.lr_init, weight_decay=1e-3, eps=1e-5)
    criterion = nn.MSELoss()

    best_valid_loss = torch.inf
    for epoch in range(cfg.model.num_epochs):
        for i, data in enumerate(train_loader):
            if cfg.model.augment_data:
                data = augment_data(data)
            est_size = 0 if cfg.estimator.name == 'GT' else cfg.estimator.kwargs.size
            N_seq = max(int(data.shape[1] * cfg.model.seq_ratio), est_size)
            data = data[:, :N_seq, :] 

            optimizer.zero_grad()
            loss = compute_loss(model,est, autoregr, data, criterion, cfg)
            loss.backward()
            
            # Print gradients of each parameter
            print(f"params1, value: {model.param1}")
            print(f"params2, value: {model.param2}")
            print(f"params3, value: {model.param3}")

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Parameter: {name}, Value: {param.data}, Gradient: {param.grad}")
            #     else:
            #         print(f"Parameter: {name} has no gradient")
            # if cfg.estimator.name != 'GT':
            #     for name, param in est.named_parameters():
            #         if param.grad is not None:
            #             print(f"Parameter: {name}, Value: {param.data}, Gradient: {param.grad}")
            #         else:
            #             print(f"Parameter: {name} has no gradient")


            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, norm_type = 2.0, error_if_nonfinite=True)
            optimizer.step()
            Constraint.small_param_thresh(model)


            if cfg.estimator.name != 'GT':
                Constraint.positive_param(est)
            tb_writer.add_scalars('loss', {'training': loss.item()}, initial_step)
            initial_step += 1
            if i % 1 == 0:
                print(f'epoch: {epoch} iter: {i} training loss: {loss.item()}')


        if epoch % cfg.model.valid_interval == 0:
            valid_loss = compute_valid_loss(model,est, autoregr, test_loader, criterion, cfg)
            tb_writer.add_scalars('loss', {'validation': valid_loss}, initial_step)
            tb_writer.add_figure('plot_train', visualize_traj(model, est, autoregr, data, cfg), initial_step)
            tb_writer.add_figure('plot_validate', visualize_traj(model, est, autoregr, next(iter(test_loader)), cfg), initial_step)
            print(f'epoch: {epoch} validation loss: {valid_loss}')


            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                model_path = Path(tb_writer.get_logdir())/f'model_{cfg.model.name}.pth'
                torch.save(model.state_dict(), model_path)
                print(f"model saved to {model_path}")

                if cfg.estimator.name != 'GT':  
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
