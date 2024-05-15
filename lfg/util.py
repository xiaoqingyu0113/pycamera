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