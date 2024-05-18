from pathlib import Path
from .predictor import  predict_trajectory
import tqdm
import numpy as np
import csv
from pycamera import CameraParam

def random_xyz(xyz, d_xyz, mag=1.0):
    return xyz + np.array([np.random.uniform(-d_xyz[0], d_xyz[0]),
                          np.random.uniform(-d_xyz[1], d_xyz[1]),
                          np.random.uniform(-d_xyz[2], d_xyz[2])]) * mag

def get_random_initial_state(cfg):
    p0 = random_xyz(np.array(cfg.dataset.p0), cfg.dataset.dp0)
    v0 = random_xyz(np.array(cfg.dataset.v0), cfg.dataset.dv0)
    w0 = random_xyz(np.array(cfg.dataset.w0), cfg.dataset.dw0) * np.pi * 2
    return p0, v0, w0

def nonuniform_span(ts,te, num, max_dt):
    t = np.linspace(ts, te, num)
    dt = np.random.uniform(-max_dt, max_dt, num)
    return t + dt

def write_csv(file_name: str, data:np.ndarray):
    with open(file_name, mode='w') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)

def append_csv(file_name: str, data:np.ndarray):
    with open(file_name, mode='a') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)

def generate_synthetic_trajectory_data(cfg):
    file_path = Path(cfg.dataset.folder) / cfg.dataset.trajectory_data
    total_traj_num = cfg.dataset.total_traj_num
    seq_len = cfg.dataset.seq_len
    t_max = cfg.dataset.t_max
    for traj_idx in range(total_traj_num):
        p0, v0, w0 = get_random_initial_state(cfg)
        tspan = nonuniform_span(0, t_max, seq_len, t_max/seq_len*0.8)
        traj = predict_trajectory(p0, v0, w0, tspan)
        traj_idx_col = np.full((traj.shape[0],1), traj_idx)
        traj = np.hstack((traj_idx_col,tspan.reshape(-1,1), traj))
        if traj_idx == 0:
            write_csv(file_path, traj)
        else:
            append_csv(file_path, traj)
    print(f"Generated synthetic trajectory data in {file_path}")

def noise_uv(uv, dev_uv):
    return uv + np.random.normal(loc=0, scale=dev_uv, size=(2,))

def load_synthetic_data(file_path):
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    data = np.array(data, dtype=float)
    print(f"Loaded synthetic trajectory data from {file_path}")
    return data

def get_w0_from_traj_idx(data, traj_idx):
    traj = data[data[:,0] == traj_idx, :]
    w0 = traj[0, 8:11]
    return  w0
def generate_camera_data_with_noise(cfg):
    camera_ids = cfg.camera.cam_ids
    camera_param_dict = {camera_id: CameraParam.from_yaml(Path(cfg.camera.folder) / f'{camera_id}_calibration.yaml') for camera_id in cfg.camera.cam_ids}
    data = load_synthetic_data(Path(cfg.dataset.folder) / cfg.dataset.trajectory_data)
    with open(Path(cfg.dataset.folder) / cfg.dataset.camera_data, mode='w') as f:
        writer = csv.writer(f)
        # for data_idx, row in enumerate(data):
        for data_idx, row in enumerate(data):
            traj_idx = int(row[0])
            w0 = get_w0_from_traj_idx(data, traj_idx)
            t = row[1]
            xyz = row[2:5]
            camera_id = camera_ids[data_idx % len(camera_ids)]  # Select camera parameter based on traj_idx
            cm = camera_param_dict[camera_id]
            uv = cm.proj2img(xyz)
            uv = noise_uv(uv, cfg.dataset.uv_noise)
            writer.writerow([traj_idx, data_idx, t, camera_id, uv[0], uv[1],w0[0],w0[1],w0[2]])
    print(f"Generated camera data with noise in {cfg.dataset.folder}/{cfg.dataset.camera_data}")

def generate_data(cfg):
    generate_synthetic_trajectory_data(cfg)
    generate_camera_data_with_noise(cfg)


