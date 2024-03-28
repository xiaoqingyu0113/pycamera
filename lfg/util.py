import time
import contextlib
import numpy as onp
from tensorboardX import SummaryWriter
from matplotlib.figure import Figure


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
    img = onp.frombuffer(figure.canvas.tostring_rgb(), dtype=onp.uint8).reshape(height, width, 3)

    # Convert RGB to BGR format (which OpenCV uses)
    img = img[:, :, [2, 1, 0]]

    # Add an image channel dimension (C, H, W)
    img = onp.moveaxis(img, -1, 0)

    # Convert to float and scale to [0, 1]
    img = img / 255.0

    # Log the image to TensorBoard
    writer.add_image(tag, img, global_step)