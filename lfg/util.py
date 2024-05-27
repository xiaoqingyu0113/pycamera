import time
import contextlib
import numpy as np
from tensorboardX import SummaryWriter
from matplotlib.figure import Figure
import torch
from typing import Dict, List, Tuple
from pycamera import CameraParam, triangulate

def plot_to_tensorboard(writer:SummaryWriter, tag:str, figure:Figure, global_step:int):
    """
    Converts a matplotlib figure to a TensorBoard image and logs it.

    Parameters:
    - writer: The TensorBoard SummaryWriter instance.
    - tag: The tag associated with the image.
    - figure: The matplotlib figure to log.
    - global_step: The global step value to associate with this image.
    """
    # Draw the figure canvas
    figure.canvas.draw()

    # Convert the figure canvas to an RGB image
    width, height = figure.canvas.get_width_height()
    img = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)

    # Convert RGB to BGR format (which OpenCV uses)
    img = img[:, :, [2, 1, 0]]

    # Add an image channel dimension (C, H, W)
    img = np.moveaxis(img, -1, 0)

    # Convert to float and scale to [0, 1]
    img = img / 255.0

    # Log the image to TensorBoard
    writer.add_image(tag, img, global_step)


def get_uv_from_3d(y, cam_id_list, camera_param_dict):
    '''
        Get the uv coordinates from 3D positions
    '''
    uv_list = []
    for yi, cm in zip(y, cam_id_list):
        uv = camera_param_dict[cm].proj2img(yi)
        uv_list.append(uv)
        
    return torch.stack(uv_list, dim=0)


def compute_stamped_triangulations(data:np.ndarray, camera_param_dict:Dict[str, CameraParam]):
    '''
        Compute the 3D positions of the stamped points

        data: [seq_len, 4]

        output: [seq_len-1, 4]
    '''
    positions = []
    stamp = []
    for data_left, data_right in zip(data[0:-1], data[1:]):
        uv_left = data_left[4:6]
        uv_right = data_right[4:6]
        camid_left = str(int(data_left[3]))
        camid_right = str(int(data_right[3]))
        if camid_left != camid_right:
            p = triangulate(uv_left, uv_right, camera_param_dict[camid_left], camera_param_dict[camid_right])
            positions.append(p)
            stamp.append(data_right[2])
    positions = np.array(positions)
    stamp = np.array(stamp)
    return np.hstack((stamp.reshape(-1, 1), positions))


class KalmanFilter:
    def __init__(self, x0 = None,  q_noise = 0.010, r_noise=0.010, p0_noise=0.010, device='cpu'):
        """
        Initialize the Kalman Filter.
        Args:
        - H (torch.Tensor): Observation matrix.
        - Q (torch.Tensor): Process noise covariance matrix.
        - R (torch.Tensor): Measurement noise covariance matrix.
        - x0 (torch.Tensor): Initial state estimate.
        - P0 (torch.Tensor): Initial covariance estimate.
        """
        self.device = device    
        self.H = torch.tensor([[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0]], dtype=torch.float32, device=device)


        self.Q = torch.eye(6, dtype=torch.float32, device=device) * q_noise  # Process noise covariance
        self.R = torch.eye(3, dtype=torch.float32, device=device) * r_noise  # Measurement noise covariance

        self.x = x0  # State estimate
        self.P = torch.eye(6, dtype=torch.float32, device=device)  # Covariance estimate

    def predict(self, dt, g=9.81):
        """
        Predict the next state and estimate covariance.
        Args:
        - dt (float): Time step to the next prediction.
        - g (float): Acceleration due to gravity, positive downwards.
        """
        # Dynamic state transition matrix F
        F = torch.tensor([
            [1, 0, 0, dt,  0,  0],
            [0, 1, 0,  0, dt,  0],
            [0, 0, 1,  0,  0, dt],
            [0, 0, 0,  1,  0,  0],
            [0, 0, 0,  0,  1,  0],
            [0, 0, 0,  0,  0,  1]
        ], dtype=torch.float32, device=self.device)

        # Control input vector u influenced by gravity
        u = torch.tensor([0, 0, 0.5 * g * dt**2, 0, 0, g * dt], dtype=torch.float32, device=self.device)

        # Control input matrix G
        G = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32, device=self.device)


        # Predict next state
        self.x = F @ self.x + G * u
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        Update the state estimate from a new measurement.
        Args:
        - z (torch.Tensor): Measurement vector.
        """
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ torch.inverse(S)
 
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (torch.eye(self.H.size(1), dtype=torch.float32, device=self.device) - K @ self.H) @ self.P

    def smooth(self,meas: torch.Tensor, dt):
        if self.x is None:
            self.x = torch.cat((meas, torch.zeros(3, dtype=torch.float32, device=self.device)),dim=0) 
            return self.x
        else:
            self.predict(dt)
            self.update(meas)
            return self.x

if __name__ == '__main__':
    kf = KalmanFilter(q_noise=0.01, r_noise=0.01, p0_noise=0.001, device='cpu')
    t = torch.linspace(0, 1.0, 100) + 0.004 * torch.randn(100)
    v0 = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float32)
    pos = v0[None, :] * t[:, None] + 0.5 * torch.tensor([0.0, 3.0, -9.81], dtype=torch.float32)[None, :] * t[:, None]**2
    p  = torch.cat((t[:,None], pos + torch.rand_like(pos)*0.030), dim=1) 
    dt = torch.diff(t)
    xN = []
    for i in range(1,100):
        x = kf.smooth(p[i, 1:], dt[i-1])
        xN.append(x)

    xN = torch.stack(xN, dim=0)
    print(xN)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    p = p.numpy()
    xN = xN.numpy()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:,1], p[:,2], p[:,3], label='True')
    ax.plot(xN[:,0], xN[:,1], xN[:,2], label='Kalman')
    ax.legend()
    plt.show()