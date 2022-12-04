import numpy as np

from .core_ic import *
from .core_vm import *
from .scurve.hilbert import Hilbert
from .scurve.zigzag import ZigZag
from .scurve.zorder import ZOrder
from .tools import *


def plot_grid2d(x1_nodes, x2_nodes, x, fronts=(), path=None):
  import matplotlib.pyplot as plt

  x2_grid, x1_grid = np.meshgrid(x2_nodes, x1_nodes)
  plt.plot(x1_grid, x2_grid, '-', color='k', alpha=0.7)
  plt.plot(x1_grid.T, x2_grid.T, '-', color='k', alpha=0.7)
  plt.xticks(x1_nodes)
  plt.yticks(x2_nodes)
  plt.scatter(x[:, 0], x[:, 1], marker='.')
  plt.xlabel('x1')
  plt.ylabel('x2')

  colors = plt.cm.rainbow(np.linspace(0, 1, len(fronts)))
  np.random.seed(0)
  np.random.shuffle(colors)

  for i, (p1, p2, p3) in enumerate(fronts):
    plt.plot([x[p1][0], x[p2][0], x[p3][0]],
             [x[p1][1], x[p2][1], x[p3][1]],
             color=colors[i], linewidth=2)

  if path is not None:
    x_path = x[np.hstack((path[:, 0], path[-1][1]))]
    plt.plot(x_path[:, 0], x_path[:, 1], 'k')

  plt.grid()
  plt.show()


def build_path(problem, curve=None, order=3, random=False):
  from . import core_ic

  x_bounds = list(zip(*problem.variables_bounds))
  n_cells = 2 ** order

  x1_bounds = x_bounds[0]
  x1_nodes = np.linspace(x1_bounds[0], x1_bounds[1], n_cells + 1)
  x1_low = np.tile(x1_nodes[:-1], n_cells)
  x1_up = np.tile(x1_nodes[1:], n_cells)

  x2_bounds = x_bounds[1]
  x2_nodes = np.linspace(x2_bounds[0], x2_bounds[1], n_cells + 1)
  x2_low = np.repeat(x2_nodes[:-1], n_cells)
  x2_up = np.repeat(x2_nodes[1:], n_cells)

  x_low = np.vstack((x1_low, x2_low)).T
  x_up = np.vstack((x1_up, x2_up)).T

  if random:
    x = np.random.uniform(low=x_low, high=x_up, size=(n_cells**2, 2))
  else:
    x = np.random.uniform(low=(x_up + x_low) / 2,
                          high=(x_up + x_low) / 2, size=(n_cells**2, 2))

  if curve is None:
    return x

  x1_cell = np.tile(np.arange(n_cells), n_cells)
  x2_cell = np.repeat(np.arange(n_cells), n_cells)
  x_cell = np.vstack((x1_cell, x2_cell)).T

  curve = curve(2, order if curve is not ZigZag else 2**order)
  path_idx = np.array(list(map(curve.index, x_cell)))
  path = np.argsort(path_idx)
  path = np.vstack((path[:-1], path[1:])).T
  # plot_grid2d(x1_nodes, x2_nodes, x, path)
  return x, path


def generate_test_sample(x, s_size, seed=None):
  lb = x.min(axis=0)
  ub = x.max(axis=0)
  s_size = (int(x.shape[0] * s_size), x.shape[1])
  np.random.seed(seed=seed)
  test_x = np.random.uniform(low=lb, high=ub, size=s_size)
  return test_x
