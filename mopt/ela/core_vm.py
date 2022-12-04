import numpy as np

from . import tools


class VM(object):

  def __init__(self, x, y, triples, penalty=1e2):
    self.x = x
    self.y = y
    self.triples = triples
    self.deltas_x = np.linalg.norm(np.diff(x[triples], axis=1), axis=2)
    self.deltas_y = np.diff(y[:, 0][triples], axis=1)

    self.deltas = np.zeros_like(self.deltas_x)
    valid_values = self.deltas_x > 0
    self.deltas[valid_values] = self.deltas_y[valid_values] / self.deltas_x[valid_values]
    # if some of points have same x and different f the angle is assumed 180
    if not np.all(valid_values):
      self.deltas[~valid_values] = penalty * np.abs(self.deltas).max() * np.sign(self.deltas_y[~valid_values])
    self.valid_values = valid_values
    self.penalty = penalty

    self.angles = tools.calc_angle(x, triples)
    self.abs_deltas = np.absolute(self.deltas)
    self.eps_max = self.abs_deltas.max()
    self.eps_min = self.abs_deltas.min()
    self._eps_min = self.abs_deltas[self.abs_deltas > 0].min() / 2.0

  def extended_copy(self, n_parts):
    assert n_parts > 1
    const_weights = tuple((float(n_parts - k) / n_parts, float(k) / n_parts) for k in range(1, n_parts))

    def weights(*args):
      for wi, wj in const_weights:
        yield wi, wj

    class Counter(object):

      def __init__(self, index):
        self.index = index - 1

      def __call__(self):
        self.index += 1
        return self.index

    x1D = np.zeros(self.triples.size, dtype=float)
    # Add extra dx values to join all the triples into 1d sequence
    x1D[1:] = np.insert(self.deltas_x, np.arange(2, self.deltas_x.size, 2), 1.0)
    x1D = np.cumsum(x1D).reshape(self.triples.shape)
    y1D = self.y[self.triples]

    next_idx = Counter(self.x.shape[0])

    triples = []
    new_x = []
    new_y = []

    for f_id, (i, j, k) in enumerate(self.triples):
      new_x1D = []
      path = [i]
      # Split ij front
      for wi, wj in weights():
        path.append(next_idx())
        new_x.append(self.x[i] * wi + self.x[j] * wj)
        new_y.append(self.y[i] * wi + self.y[j] * wj)
      path.append(j)
      # Split jk front
      for wj, wk in weights():
        path.append(next_idx())
        new_x.append(self.x[j] * wj + self.x[k] * wk)
        new_y.append(self.y[i] * wj + self.y[j] * wk)
      path.append(k)
      # Add all the possible triples
      triples.append(np.vstack((path[:-2], path[1:-1], path[2:])).T)

    new_x = np.vstack((self.x, new_x))
    new_y = np.vstack((self.y, new_y))

    triples = np.vstack(triples)
    return VM(new_x, new_y, np.vstack((triples, triples[:, [2, 1, 0]])))

  def update_copy(self, y):
    return VM(x=self.x, y=y, triples=self.triples, penalty=self.penalty)

  def plot(self, scale='linear', minobj=False, fontsize=12):
    import matplotlib.pyplot as plt
    plt.rc('figure', figsize=(6.4, 6.4))

    plt.plot(self.deltas[:, 0], self.deltas[:, 1], '.')
    if minobj:
      plt.plot([-np.abs(self.deltas).max(), 0], [np.abs(self.deltas).max(), 0], '-', c='k', alpha=0.5)
    else:
      plt.plot([-np.abs(self.deltas).max(), np.abs(self.deltas).max()],
               [np.abs(self.deltas).max(), -np.abs(self.deltas).max()], '-', c='k', alpha=0.5)
    # plt.xlim(-np.abs(self.deltas).max(), np.abs(self.deltas).max())
    # plt.ylim(-np.abs(self.deltas).max(), np.abs(self.deltas).max())
    plt.xlabel('$\delta_1$', fontsize=fontsize)
    plt.ylabel('$\delta_2$', fontsize=fontsize)
    plt.xscale(scale)  # symlog
    plt.yscale(scale)  # symlog
    plt.grid()
    plt.show()


class VMSegment(VM):

  def __init__(self, x, y, triples, penalty=1e2):
    super(VMSegment, self).__init__(x, y, triples, penalty)
    half_mask = self.deltas.sum(1) >= 0
    self.half_angles = self.angles[half_mask]
    self.half_deltas = self.deltas[half_mask]
    self.distances = np.linalg.norm(self.half_deltas, axis=1)
    self.sectors = np.angle(self.half_deltas[:, 0] + self.half_deltas[:, 1] * 1j, deg=True)

  def run_center(self, alpha):
    sector_mask = (self.sectors >= 45 - alpha / 2.0) * (self.sectors <= 45 + alpha / 2.0)
    # import matplotlib.pyplot as plt
    # plt.plot(self.half_deltas[sector_mask][:, 0], self.half_deltas[sector_mask][:, 1], 'o')
    return sum(sector_mask) / float(self.sectors.size)

  def run_center_all(self, n_sectors=361):
    alphas = np.linspace(0, 180, n_sectors)
    # run = np.vectorize(self.run_center)
    # return alphas, run(alpha=alphas)
    result = np.zeros((alphas.size, self.sectors.size))
    for i, alpha in enumerate(alphas):
      result[i] = (self.sectors >= 45 - alpha / 2.0) * (self.sectors <= 45 + alpha / 2.0)
    return alphas, np.sum(result, axis=1) / float(self.sectors.size)

  def run_linear(self, alpha):
    sector_mask = self.sectors + 45 <= alpha
    # import matplotlib.pyplot as plt
    # plt.plot(self.half_deltas[sector_mask][:, 0], self.half_deltas[sector_mask][:, 1], 'o')
    return sum(sector_mask) / float(self.sectors.size)

  def run_linear_all(self, n_sectors=361):
    alphas = np.linspace(0, 180, n_sectors)
    # run = np.vectorize(self.run_linear)
    # return alphas, run(alpha=alphas)
    result = np.zeros((alphas.size, self.sectors.size))
    for i, alpha in enumerate(alphas):
      result[i] = self.sectors + 45 <= alpha
    return alphas, np.sum(result, axis=1) / float(self.sectors.size)

  def run(self, alpha):
    sector_mask = self.sectors + 45 <= alpha
    # import matplotlib.pyplot as plt
    # plt.plot(self.half_deltas[sector_mask][:, 0], self.half_deltas[sector_mask][:, 1], 'o')
    return sum(self.half_angles[sector_mask]) / float(sum(self.half_angles))

  def run_all(self, n_sectors=361, sectors=None):
    alphas = np.linspace(0, 180, n_sectors) if sectors is None else sectors
    # run = np.vectorize(self.run_linear)
    # return alphas, run(alpha=alphas)
    result = []
    for i, alpha in enumerate(alphas):
      result.append(sum(self.half_angles[self.sectors + 45 <= alpha]))
    return alphas, np.array(result) / float(sum(self.half_angles))

  def plot_sectors(self, sectors, res):
    import matplotlib.pyplot as plt
    plt.plot(sectors, res)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.4)
    plt.axvline(x=45, color='k', linestyle='--', alpha=0.4)
    plt.axvline(x=90, color='k', linestyle='--', alpha=0.4)
    plt.axvline(x=135, color='k', linestyle='--', alpha=0.4)
    plt.axvline(x=180, color='k', linestyle='--', alpha=0.4)
    plt.xticks(range(0, 181, 45))
    plt.grid()


class VMIC(VM):
  '''
  +--------+--------+--------+--------+--------+
  |   i\j  | [=]  0 | [-] -1 | [+]  2 | [*] -5 |
  +--------+--------+--------+--------+--------+
  | [=]  0 |    0   |   -1   |    2   |   -5   |
  +--------+--------+--------+--------+--------+
  | [-] -1 |    1   |    0   |    3   |   -4   |
  +--------+--------+--------+--------+--------+
  | [+]  2 |   -2   |   -3   |    0   |   -7   |
  +--------+--------+--------+--------+--------+
  | [*] -5 |    5   |    4   |    7   |    0   |
  +--------+--------+--------+--------+--------+
  '''
  _CHANGES = {'negative': -1, 'neutral': 0, 'positive': 2, 'uncaught': -5}

  def __init__(self, x, y, triples, penalty=1e2):
    super(VMIC, self).__init__(x, y, triples, penalty)
    # Blocks for analysis
    caught_changes = [self._CHANGES[i] for i in ['negative', 'neutral', 'positive']]
    # caught_changes = [self._CHANGES[i] for i in ['negative', 'neutral', 'positive', 'uncaught']]
    self.blocks = np.array([j - i for i in caught_changes for j in caught_changes if i != j])
    self.ic_base = np.log(self.blocks.size)

  def run(self, eps1, eps2=None):
    sequence = np.zeros_like(self.deltas, int)
    if eps2 is None:
      eps2 = eps1
    sequence[:] = self._CHANGES['neutral']
    sequence[self.deltas[:, 0] > eps1, 0] = self._CHANGES['positive']
    sequence[self.deltas[:, 0] < -eps2, 0] = self._CHANGES['negative']
    sequence[self.deltas[:, 1] > eps2, 1] = self._CHANGES['positive']
    sequence[self.deltas[:, 1] < -eps1, 1] = self._CHANGES['negative']
    # Count blocks
    sequence_blocks = np.diff(sequence, axis=1).ravel()
    total_sum = np.sum(self.angles)
    # Calculate values
    ic = 0
    for block in self.blocks:
      prob_of_block = np.sum(self.angles[sequence_blocks == block]) / total_sum
      if prob_of_block != 0:
        ic -= prob_of_block * np.log(prob_of_block) / self.ic_base
    # Remove neutral changes
    mask = sequence_blocks == (self._CHANGES['positive'] - self._CHANGES['negative'])
    mask += sequence_blocks == (self._CHANGES['negative'] - self._CHANGES['positive'])
    # Count without blocks with repeated changes
    icp = np.sum(self.angles[mask]) / (total_sum + 1.0)
    return ic, icp


  def run_old(self, eps1, eps2=None):
    sequence = np.zeros_like(self.deltas, int)
    if eps2 is None:
      eps2 = eps1
    sequence[:] = self._CHANGES['neutral']
    sequence[self.deltas[:, 0] > eps1, 0] = self._CHANGES['positive']
    sequence[self.deltas[:, 0] < -eps2, 0] = self._CHANGES['negative']
    sequence[self.deltas[:, 1] > eps2, 1] = self._CHANGES['positive']
    sequence[self.deltas[:, 1] < -eps1, 1] = self._CHANGES['negative']
    # Count blocks
    sequence_blocks = np.diff(sequence, axis=1).ravel()
    blocks, counts = np.unique(sequence_blocks, return_counts=True)
    b_counts = dict(zip(blocks, counts))
    # Calculate values
    ic = 0
    for block in self.blocks:
      bcount = float(b_counts.get(block, 0)) / sequence_blocks.size
      if bcount != 0:
        ic -= bcount * np.log(bcount) / self.ic_base
    # Remove neutral changes
    mask = sequence_blocks == (self._CHANGES['positive'] - self._CHANGES['negative'])
    mask += sequence_blocks == (self._CHANGES['negative'] - self._CHANGES['positive'])
    # Count without blocks with repeated changes
    icp = np.sum(mask) / (sequence_blocks.size + 1.0)
    return ic, icp

  def run_all_2d(self, eps1=None, eps2=None):
    unique_points = np.unique(self.deltas, axis=0)
    if eps1 is None:
      eps1 = unique_points[:, 0]
    if eps2 is None:
      eps2 = unique_points[:, 1]
    eps = np.vstack(list(map(np.ravel, np.meshgrid(eps1, eps2)))).T
    eps1, eps2 = eps[:, 0], eps[:, 1]
    run = np.vectorize(self.run)
    ic, icp = run(eps1=eps1, eps2=eps2)
    return eps, ic.reshape(-1, 1), icp.reshape(-1, 1)

  def run_all_1d(self, eps=None):
    if eps is None:
      eps = np.unique(self.abs_deltas)
    run = np.vectorize(self.run)
    ic, icp = run(eps1=eps)
    return eps, ic, icp
