import re

import numpy as np


class Route(object):

  _CHANGES = {'negative': '-', 'neutral': '=', 'positive': '+', 'uncaught': '.'}

  def __init__(self, x, y, path):
    self.x = x
    self.y = y
    self.delta_x = np.linalg.norm(x[path[:, 1]] - x[path[:, 0]], axis=1).reshape(-1, 1)
    self.delta_y = y[path[:, 1]] - y[path[:, 0]]
    self.delta = self.delta_y / self.delta_x
    self.path = path
    self.path_points = np.append(path[0, 0], path[:, 1])
    self.abs_delta = np.absolute(self.delta)
    self.eps_max = self.abs_delta.max()
    self.eps_min = self.abs_delta.min()
    # Blocks for analysis
    caught_changes = [self._CHANGES[i] for i in ['negative', 'neutral', 'positive']]
    # caught_changes = [self._CHANGES[i] for i in ['negative', 'neutral', 'positive', 'uncaught']]
    self.blocks = np.array([i + j for i in caught_changes for j in caught_changes if i != j])
    self.pattern = re.compile(r"(.)\1+", re.DOTALL)
    self.ic_base = np.log(self.blocks.size)

  def analyze(self, eps_lb=None, eps_ub=None):
    eps_lb = self.eps_min if eps_lb is None else eps_lb
    eps_ub = self.eps_max if eps_ub is None else eps_ub
    # Fill symbol sequence
    sequence = np.empty((self.delta.size, 1), str)
    sequence[:] = self._CHANGES['neutral']
    sequence[self.delta < -eps_lb] = self._CHANGES['negative']
    sequence[self.delta > eps_lb] = self._CHANGES['positive']
    sequence[abs(self.delta) > eps_ub] = self._CHANGES['uncaught']
    sequence_str = sequence.astype('|S1').tostring().decode('utf-8')
    s_sequence = float(sequence.size)
    # Calculate information content
    ic = 0
    for b in self.blocks:
      bcount = sequence_str.count(b) / (s_sequence - 1)
      if bcount != 0:
        ic -= bcount * np.log(bcount) / self.ic_base
    # Calculate partial information content
    sequence_str = sequence_str.replace(self._CHANGES['neutral'], '')
    sequence_partial = self.pattern.sub(r"\1", sequence_str)
    icp = len(sequence_partial) / s_sequence
    return ic, icp

  def analyze_all(self, vary_min_value=True, eps_lb=None, eps_ub=None):
    eps_lb = self.eps_min if eps_lb is None else eps_lb
    eps_ub = self.eps_max if eps_ub is None else eps_ub
    eps = np.unique(self.abs_delta)
    if vary_min_value:
      eps_lb = eps
    else:
      eps_ub = eps
    analyze = np.vectorize(self.analyze)
    ic, icp = analyze(eps_lb=eps_lb, eps_ub=eps_ub)
    return eps, ic, icp

  def plot_path(self, label=''):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax_path = fig.add_subplot(111)
    plt.grid()
    # Path
    x_points = np.zeros(self.path_points.size)
    for i in range(self.path_points.size - 1):
      x_points[i + 1] += x_points[i] + self.delta_x[i]
    y_points = self.y[self.path_points].ravel()
    # y_points = np.array([sum(self.delta_y[:i]) for i in range(self.delta.size)])
    ax_path.set_xlabel('x\'', fontsize=12)
    ax_path.set_ylabel('f', fontsize=12)
    ax_path.plot(x_points, y_points, label=label)
    ax_path.set_xlim([-x_points[-1] * 0.01, x_points[-1] * 1.01])
    if label:
      plt.legend()
    plt.show()

  def plot_deltas(self, plot_points=False):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax_deltas = fig.add_subplot(111)
    plt.grid()
    ax_deltas.set_xlabel('step')
    ax_deltas.set_ylabel('df')
    ax_deltas.vlines(range(self.delta.size), [0], self.delta.flatten())
    ax_deltas.hlines([0], [0], [self.delta.size - 1])
    if plot_points:
      ax_deltas.scatter(range(self.delta.size), self.delta.flatten())
    # ax_deltas.set_xlim([- self.delta.size * 0.01, self.delta.size * 1.01])
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['right'].set_color('none')
    # ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    plt.show()

  def plot_surface(self, x_nodes, eps_lb=None, eps_ub=None, wireframe=False, scale='log'):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # Prepare sample
    eps_lb = self.eps_min if eps_lb is None else eps_lb
    eps_ub = self.eps_max if eps_ub is None else eps_ub
    lb = self.eps_min
    ub = self.eps_max
    if scale == 'linear':
      x1 = x2 = np.linspace(lb1, lb2, x_nodes)
    elif scale == 'log':
      lb1 = lb2 = self.eps_min if self.eps_min > 0 else self.abs_delta[self.abs_delta > 0].min()
      x1 = x2 = np.logspace(np.log10(eps_lb), np.log10(eps_ub), x_nodes)
    x1, x2 = np.meshgrid(x1, x2)
    x = np.vstack([x1.ravel(), x2.ravel()]).transpose()

    results = np.zeros((x.shape[0], 2))
    for i in range(x.shape[0]):
      results[i] = self.analyze(x[i, 0], x[i, 1])
    ic = results[:, 0]
    icp = results[:, 1]

    ic = np.reshape(ic, (-1, x1.shape[1]))
    icp = np.reshape(icp, (-1, x1.shape[1]))
    # Create figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Plot surface
    if wireframe:
      surf = ax.plot_wireframe(x1, x2, ic, rstride=1, cstride=1)
    else:
      surf = ax.plot_surface(x1, x2, ic, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


class Navigator(object):

  def __init__(self, route_type='nn', limited_step=True, start_point='min', seed=None):
    self.route_type = route_type
    self.start_point = start_point
    self.limited_step = limited_step
    self.seed = seed

  def get_route(self, x, y):
    if self.route_type == 'nn':
      path = self.__build_nn(x, y)
    elif self.route_type == 'full':
      path = self.__build_full(x)
    else:
      raise ValueError('Wrong route type value!')
    return Route(x=x, y=y, path=path)

  def get_route_extend(self, route, n_parts, relative=False):
    # relative = False: step size fixed, relative = True: number of parts fixed
    new_x, new_y, new_path = self.__extend(route.path, route.x, route.y, n_parts, relative)
    return Route(x=new_x, y=new_y, path=new_path)

  def get_route_recalc(self, route, model):
    return Route(x=route.x, y=model.calc(route.x), path=route.path)

  def __split_steps(self, path, x, y, n_parts, relative):
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
    return dx, dy, n_parts, n_new_points

  def __extend(self, path, x, y, n_parts, relative):
    n_steps = path.shape[0]
    dx, dy, n_parts, n_new_points = self.__split_steps(path, x, y, n_parts, relative)
    new_path = np.empty((n_parts.sum(), 2), int)
    new_x = np.empty((n_new_points.sum(), x.shape[1]), float)
    new_y = np.empty((n_new_points.sum(), y.shape[1]), float)
    path_i = 0
    for i in range(n_steps):
      current_i = path[i][0]
      last_i = path[i][1]
      next_x = x[current_i] + dx[i]
      next_y = y[current_i] + dy[i]
      for j in range(n_new_points[i]):
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
    return np.vstack((x, new_x)), np.vstack((y, new_y)), new_path

  def __get_first_i(self, y, s_size):
    if self.start_point == 'max':
      return np.argmax(y)
    elif self.start_point == 'min':
      return np.argmin(y)
    elif self.start_point == 'rand':
      np.random.seed(seed=self.seed)
      return np.random.choice(np.arange(s_size))
    else:
      raise ValueError('Wrong start point value!')

  def __build_nn(self, x, y):
    from sklearn.neighbors import NearestNeighbors
    s_size = x.shape[0]
    path = np.empty((0, 2), int)
    nn = NearestNeighbors(n_neighbors=s_size - 1, p=1).fit(x)
    nn_d, nn_i = nn.kneighbors()
    step_size = nn_d[:, 0].max() if self.limited_step else nn_d.max()
    current_i = self.__get_first_i(y=y, s_size=s_size)
    while True:
      # while path.shape[0] < s_size - 1:
      # Remove current index from nn_i matrix
      nn_i[nn_i == current_i] = -1
      # Find neighbors for current element
      current_nn_i, current_nn_d = nn_i[current_i], nn_d[current_i]
      # Remove passed elements
      passed_elements = current_nn_i == -1
      current_nn_i = current_nn_i[~passed_elements]
      current_nn_d = current_nn_d[~passed_elements]
      # Check distance
      if current_nn_d.size == 0 or current_nn_d[0] > step_size:
        steps = np.argwhere(np.logical_and(nn_d <= step_size, nn_i != -1))
        if steps.size:
          current_i = steps[0][0]
          continue
        else:
          break
      # Add next element to path
      next_i = current_nn_i[0]
      path = np.vstack((path, [current_i, next_i]))
      current_i = next_i
    # Add unpassed elements
    for i in np.argwhere(~np.any(nn_i == -1, axis=1)):
      current_i = i[0]
      current_nn_i = nn_i[current_i]
      next_i = current_nn_i[0]
      path = np.vstack((path, [current_i, next_i]))
    return path

  def __build_full(self, x):
    s_size = x.shape[0]
    # Calculate step size
    min_distances = np.zeros(s_size)
    for i in range(s_size):
      distances = np.fabs(x - x[i]).flatten() if 1 == x.shape[1] else np.hypot.reduce(x - x[i], axis=1)
      distances[i] = np.inf
      min_distances[i] = np.min(distances)
    step_size = min_distances.max()
    points = np.array(range(s_size), int)
    # Fill path
    path = np.empty((0, 2), int)
    for i in range(s_size - 1):
      x_i = x[i]
      x_j = x[i + 1:]
      dist_i = np.fabs(x_i - x_j).flatten() if 1 == x_j.shape[1] else np.hypot.reduce(x_i - x_j, axis=1)
      next_points = points[i + 1:][dist_i - step_size <= 1e-6]
      if next_points.size:
        steps = np.vstack(([i] * next_points.shape[0], next_points)).T
        path = np.vstack((path, steps))
    return path
