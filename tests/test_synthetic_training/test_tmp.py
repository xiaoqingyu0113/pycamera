import torch

from train_traj import TrajectoryDataset
import matplotlib.pyplot as plt
from draw_util import draw_util

dataset = TrajectoryDataset('data/synthetic/traj_consspin.csv', noise=0.01)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for _ in range(100):
    data = dataset[0]
    x = data[:,2:5].cpu().numpy()
    ax.plot(x[:,0], x[:,1], x[:,2])

draw_util.set_axes_equal(ax, zoomin=2.0) 
plt.show()