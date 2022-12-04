from __future__ import division

import numpy as np


def random(dim, size):
  return np.vstack([np.random.rand(size) for i in range(dim)]).T

def n_cells(dim, size):
  return (int(np.ceil(np.power(size, 1.0 / dim))), ) * dim

def grid(dim, n_cells, random=True, allowed_area=0.99):
  # Fills from 0 to 1 centers of cells
  nodes_list = []
  if np.array(n_cells).ndim == 0:
    n_cells = [n_cells] * dim
  for i in range(dim):
    cell_size = 1.0 / (2.0 * n_cells[i])
    xi_nodes = np.linspace(cell_size, 1 - cell_size, n_cells[i])
    nodes_list.append(xi_nodes)
  x = np.vstack([grid.flatten() for grid in np.meshgrid(*nodes_list)]).T
  if random:
    for i in range(dim):
      x[:, i] += np.random.uniform(-allowed_area * cell_size, allowed_area * cell_size, x.shape[0])
  return x


def full_factorial(dim, n_cells, random=True):
  # Fills from 0 to 1 in regular grid
  if np.array(n_cells).ndim == 0:
    n_cells = [n_cells] * dim
  result = np.empty((np.product(n_cells), len(n_cells),), dtype=float)
  repeat_count = 1
  tile_count = result.shape[0]
  for i, n_cells_i in enumerate(n_cells):
    tile_count //= n_cells_i
    values = np.random.random(n_cells_i) if random else np.linspace(0, 1, n_cells_i)
    result[:, i] = np.tile(np.repeat(values, repeat_count, axis=0), tile_count)
    repeat_count *= n_cells_i
  return result
