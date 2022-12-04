import itertools

import numpy as np


def ext_fronts_to_path(fronts, ext_fronts):
  slices = np.split(ext_fronts, [np.where(ext_fronts[:, 2] == i)[0][0] + 1 for i in fronts[:, 2]])[:-1]
  return np.hstack([np.hstack((s[:, 0], s[-1][1:])) for s in slices])


def extend_fronts(x, y, fronts, n_parts, min_n_parts=2, w=None, klog=10.0, imag_w=0.8):
  assert n_parts > 1

  x = np.array(x, dtype=float)
  y = np.array(y, dtype=float).reshape((-1, 1))
  w = np.ones((y.shape[0], 1), dtype=float) if w is None else np.array(w, dtype=float).reshape(-1, 1)

  norm_x = (lambda dx: np.fabs(dx, out=dx).flatten()) if 1 == x.shape[1] else (lambda dx: np.hypot.reduce(dx, axis=2))
  dx_nrm2 = norm_x(np.diff(x[fronts], axis=1)).reshape((-1, 2))
  np.clip(dx_nrm2, dx_nrm2.max() * np.finfo(float).eps, np.inf, out=dx_nrm2)

  max_n_parts = n_parts
  const_weights = {}
  for n_parts in np.arange(min_n_parts, max_n_parts + 1):
    linear_weights = np.arange(1, n_parts, dtype=float) / n_parts
    linear_weights = np.vstack((1. - linear_weights, linear_weights))
    # Symlog weights distribution
    nonlinear_weights = 1. / (1. + np.exp(-klog * (linear_weights.T - 0.5)))
    const_weights[n_parts] = (nonlinear_weights, nonlinear_weights.prod(axis=1).reshape(-1, 1))

  x1D = np.zeros(fronts.size, dtype=float)
  # Add extra dx values to join all the triples into 1d sequence
  x1D[1:] = np.insert(dx_nrm2, np.arange(2, dx_nrm2.size, 2), 1.0)
  x1D = np.cumsum(x1D).reshape(fronts.shape)
  y1D = y[fronts]

  try:
    from da.p7core import gtapprox
    model = gtapprox.Builder().build(x1D.reshape(-1, 1), y1D.reshape(-1, y.shape[1]),
                          options={"GTApprox/Technique": "PLA", "GTApprox/LogLevel": "error", "GTApprox/InternalValidation": False})
  finally:
    pass

  def _read_weights(n_parts):
    return const_weights.get(n_parts, const_weights[max_n_parts])

  dx_nrm2_min = dx_nrm2.min()
  n_points_dx = (max_n_parts - 2) / max(dx_nrm2.ptp(), np.finfo(float).eps)
  dx_nrm2_limit = np.percentile(dx_nrm2, 90) / 2.0

  # first pass - calculate the number of points to add
  n_x_add = 0
  for dxi in dx_nrm2:
    n_x_add += _read_weights(int(2.5 + (dxi[0] - dx_nrm2_min) * n_points_dx))[0].shape[0]
    n_x_add += _read_weights(int(2.5 + (dxi[1] - dx_nrm2_min) * n_points_dx))[0].shape[0]

  new_x = np.empty((x.shape[0] + n_x_add, x.shape[1]))
  new_y = np.empty((y.shape[0] + n_x_add, y.shape[1]))
  new_w = np.empty((w.shape[0] + n_x_add, w.shape[1]))

  new_fronts = np.empty((len(fronts) + n_x_add, 3), dtype=int)

  new_x[:x.shape[0]] = x
  new_y[:y.shape[0]] = y
  new_w[:w.shape[0]] = w

  # Second pass - fill the data
  next_id = x.shape[0]
  next_front = 0
  for f_id, (i, j, k) in enumerate(fronts):
    path = [i]

    # Split ij front
    wij, wij_prod = _read_weights(int(2.5 + (dx_nrm2[f_id][0] - dx_nrm2_min) * n_points_dx))

    prev_id = next_id
    next_id += wij.shape[0]
    path.extend(np.arange(prev_id, next_id))

    np.dot(wij, np.vstack((x[i], x[j])), out=new_x[prev_id:next_id])
    distance_penalty = (w[i] + w[j]) * dx_nrm2[f_id][0] / dx_nrm2_limit
    new_w[prev_id:next_id] = np.dot(wij, np.vstack((w[i], w[j]))) * imag_w - wij_prod * distance_penalty
    new_y[prev_id:next_id, 0] = x1D[f_id][0] + dx_nrm2[f_id][0] * wij[:, 1]

    path.append(j)

    # Split jk front
    wij, wij_prod = _read_weights(int(2.5 + (dx_nrm2[f_id][1] - dx_nrm2_min) * n_points_dx))

    prev_id = next_id
    next_id += wij.shape[0]
    path.extend(np.arange(prev_id, next_id))

    np.dot(wij, np.vstack((x[j], x[k])), out=new_x[prev_id:next_id])
    distance_penalty = (w[j] + w[k]) * dx_nrm2[f_id][1] / dx_nrm2_limit
    new_w[prev_id:next_id] = np.dot(wij, np.vstack((w[j], w[k]))) * imag_w - wij_prod * distance_penalty
    new_y[prev_id:next_id, 0] = x1D[f_id][1] + dx_nrm2[f_id][1] * wij[:, 1]

    path.append(k)

    # Add all the possible fronts
    fronts_update = np.vstack((path[:-2], path[1:-1], path[2:])).T
    new_fronts[next_front:next_front + fronts_update.shape[0]] = fronts_update
    next_front += fronts_update.shape[0]

  new_y[x.shape[0]:] = model.calc(new_y[x.shape[0]:,0].reshape(-1, 1))
  np.clip(new_w, 0., np.inf, out=new_w)

  return new_x, new_y, new_w, new_fronts[:next_front]


def extend_path(path, x, y, n_parts=3, relative=True):
  if path.ndim == 1:
    path = np.vstack((path[:-1], path[1:])).T

  # relative = False: step size is fixed, relative = True: number of parts is fixed
  steps = np.linalg.norm(x[path[:, 1]] - x[path[:, 0]], axis=1).reshape(-1, 1)
  if relative:
    n_parts = np.array([[n_parts]] * path.shape[0])
  else:
    new_step = steps.min() / n_parts
    n_parts = steps // new_step
  dx = (x[path[:, 1]] - x[path[:, 0]]) / n_parts
  dy = (y[path[:, 1]] - y[path[:, 0]]) / n_parts
  n_parts = n_parts.astype(np.int)
  n_new_points = n_parts - 1

  new_path = np.empty((n_parts.sum(), 2), int)
  new_x = np.empty((n_new_points.sum(), x.shape[1]), float)
  new_y = np.empty((n_new_points.sum(), y.shape[1]), float)
  path_i = 0
  for i in range(path.shape[0]):
    current_i = path[i][0]
    last_i = path[i][1]
    next_x = x[current_i] + dx[i]
    next_y = y[current_i] + dy[i]
    for j in np.arange(n_new_points[i]):
      new_i = (n_new_points[: i]).sum() + j
      new_x[new_i] = next_x
      new_y[new_i] = next_y
      next_x += dx[i]
      next_y += dy[i]
      new_i += x.shape[0]
      new_path[path_i] = [current_i, new_i]
      path_i += 1
      current_i = new_i
    new_path[path_i] = [current_i, last_i]
    path_i += 1
  return np.vstack((x, new_x)), np.vstack((y, new_y)), np.insert(new_path[:, 1], 0, new_path[0][0])


def distance_path(x, path):
  distances = np.hypot.reduce(np.diff(x[path], axis=0), axis=1)
  return np.insert(np.cumsum(distances), 0, 0)


def build_nn_slice(x, central_point=0, n_steps=3):
  n_steps = min(n_steps, int((x.shape[0] - 1) / 2.0))
  idx = np.arange(x.shape[0], dtype=int)
  path = np.zeros(1 + n_steps * 2, dtype=int)
  unchecked = np.ones_like(idx, dtype=bool)

  path[n_steps] = central_point
  unchecked[central_point] = False

  for i in np.arange(n_steps):
    distances = np.hypot.reduce(x[path[n_steps - i]] - x[unchecked], axis=1)
    path[n_steps - i - 1] = idx[unchecked][np.argmin(distances)]
    unchecked[path[n_steps - i - 1]] = False

    distances = np.hypot.reduce(x[path[n_steps + i]] - x[unchecked], axis=1)
    path[n_steps + i + 1] = idx[unchecked][np.argmin(distances)]
    unchecked[path[n_steps + i + 1]] = False
  return path


def build_distance_table(x):
  n = x.shape[0]
  distances_table = np.zeros((n, n), dtype=float)
  for i in range(n - 1):
    # if i % 1000 == 0:
    #   print(i)
    distances = np.fabs(x[i + 1:, :] - x[i, :]).flatten() if 1 == x.shape[1] else np.hypot.reduce(x[i + 1:, :] - x[i, :], axis=1)
    np.round(distances, 10, out=distances)
    distances_table[i, i + 1:] = distances[:]
    distances_table[i + 1:, i] = distances[:]
  return distances_table


def build_nn_route(x):
  idx = np.arange(x.shape[0], dtype=int)
  path = np.zeros_like(idx, dtype=int)
  unchecked = np.ones_like(idx, dtype=bool)

  path[0] = 0
  unchecked[0] = False
  for i in np.arange(1, path.shape[0]):
    # if i % 1000 == 0:
    #   print(i)
    distances = np.hypot.reduce(x[path[i - 1]] - x[unchecked], axis=1)
    path[i] = idx[unchecked][np.argmin(distances)]
    unchecked[path[i]] = False
  return path


def get_triples_minobj(x, y, angle_percentile=90):
  x = np.array(x, copy=True)
  for i in np.arange(x.shape[1]):
    x[:, i] = (x[:, i] - x[:, i].min()) / x[:, i].ptp()

  result = np.zeros((0, 3), int)
  distances = np.hypot.reduce(x - x[np.argmin(y)], axis=1)
  percentiles = np.linspace(0, 100, 5 + 1)
  for i in np.arange(percentiles.size - 1):

    d_lower = np.percentile(distances, percentiles[i])
    d_upper = np.percentile(distances, percentiles[i + 1])

    mask = (distances > d_lower) & (distances <= d_upper)
    points = np.argwhere(mask).ravel()
    if points.size < 2:
      continue

    triples = np.array(list(itertools.combinations(points, 2)))
    triples = np.insert(triples, 1, np.argmin(y), axis=1)
    angle = calc_angle(x, triples)

    min_angle = np.percentile(angle, angle_percentile)
    result = np.vstack((result, triples[angle >= min_angle]))

  return np.vstack((result, result[:, [2, 1, 0]]))


def get_triples_iter(x, n_fronts, step_size=None, max_step=1.05, seed=0):
  check_point = int(10 * max(x.shape[0], n_fronts))
  if seed is not None:
    np.random.seed(seed)

  # Calculate max min distance to determine step size
  if step_size is None:
    max_min_distance = -np.inf
    for i in np.arange(x.shape[0]):
      # if i % 1000 == 0:
      #   print(i)
      min_distance = np.hypot.reduce(np.delete(x, i, axis=0) - x[i], axis=1).min()
      if max_min_distance < min_distance:
        max_min_distance = min_distance
    step_size = max_min_distance * max_step
    # print('Step size:', step_size)

  def _calc_angle(x, triples):
    # np.seterr(all='raise')
    v1 = np.diff((x[triples[:, 0]], x[triples[:, 1]]), axis=0)[0]
    v2 = np.diff((x[triples[:, 2]], x[triples[:, 1]]), axis=0)[0]
    # np.dot for each row
    dot = np.einsum('ij, ij->i', v1, v2)
    norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)

    cosine_angle = np.zeros_like(dot, float)
    valid_values = norm > 0
    cosine_angle[valid_values] = dot[valid_values] / norm[valid_values]

    # if some of points have same x and different f the angle is assumed 180
    if not np.all(valid_values):
      cosine_angle[~valid_values] = -1

    np.round(cosine_angle, 10, out=cosine_angle)
    return np.arccos(cosine_angle) / np.pi

  def _iterate_randomly(n_values):
    step = 0
    indices = np.arange(n_values)
    while True:
      np.random.shuffle(indices)
      for idx in indices:
        step += 1  # step should start from 1
        yield idx, step

  result = np.zeros((0, 3), int)
  random_idx = _iterate_randomly(x.shape[0])
  while len(result) < n_fronts:
    if len(result) % 1000 == 0:
      print(len(result))

    current_idx, step = next(random_idx)

    distances = np.hypot.reduce(x - x[current_idx], axis=1)
    np.round(distances, 10, out=distances)
    distances[current_idx] = np.inf

    closest_points = np.argwhere(distances <= step_size).ravel()
    if closest_points.size > 1:
      # Form candidate fronts starting from a random point
      np.random.shuffle(closest_points)
      candidate_fronts = np.empty((closest_points.size - 1, 3), dtype=int)
      candidate_fronts[:, 0] = closest_points[0]
      candidate_fronts[:, 1] = current_idx
      candidate_fronts[:, 2] = closest_points[1:]
      # Choose the front having maximum angle assumed by it's 3 points
      front = candidate_fronts[_calc_angle(x, candidate_fronts).argmax()]
      result = np.vstack((result, front))
    # Increase step_size to avoid infinite loops
    if step % check_point == 0:
      step_size *= 1.5
  return result


def get_triples(x, max_step=1.05, min_angle=0.0):
  # Normalize distances
  x = np.array(x, copy=True)
  for i in np.arange(x.shape[1]):
    x[:, i] = (x[:, i] - x[:, i].min()) / x[:, i].ptp()
  # Calculate max min distance to determine step size
  max_min_distance = -np.inf
  for i in np.arange(x.shape[0]):
    # if i % 1000 == 0:
    #   print(i)
    min_distance = np.hypot.reduce(np.delete(x, i, axis=0) - x[i], axis=1).min()
    if max_min_distance < min_distance:
      max_min_distance = min_distance
  step_size = max_min_distance * max_step
  # print('Step size:', step_size)
  result = np.zeros((0, 3), int)
  for i in np.arange(x.shape[0]):
    # if i % 1000 == 0:
    #   print(i)
    distances = np.hypot.reduce(x - x[i], axis=1)
    np.round(distances, 10, out=distances)
    distances[i] = np.inf
    closest_points = np.argwhere(distances <= step_size).ravel()
    if closest_points.size < 2:
      continue
    triples = np.array(list(itertools.combinations(closest_points, 2)))
    triples = np.insert(triples, 1, i, axis=1)
    angle = calc_angle(x, triples)
    result = np.vstack((result, triples[angle >= min_angle]))
  return np.vstack((result, result[:, [2, 1, 0]]))


def calc_angle(x, triples):
  # np.seterr(all='raise')
  v1 = np.diff((x[triples[:, 0]], x[triples[:, 1]]), axis=0)[0]
  v2 = np.diff((x[triples[:, 2]], x[triples[:, 1]]), axis=0)[0]
  # np.dot for each row
  dot = np.einsum('ij, ij->i', v1, v2)
  norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)

  cosine_angle = np.zeros_like(dot, float)
  valid_values = norm > 0
  cosine_angle[valid_values] = dot[valid_values] / norm[valid_values]

  # if some of points have same x and different f the angle is assumed 180
  if not np.all(valid_values):
    cosine_angle[~valid_values] = -1

  np.round(cosine_angle, 10, out=cosine_angle)
  return np.arccos(cosine_angle) / np.pi
