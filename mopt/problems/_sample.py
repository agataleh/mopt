import os

import numpy as np
import pandas as pd


class Sample(object):

  _DOE_METHODS = {
    'ff': 'generate_sample_ff',
    'grd': 'generate_sample_grid',
    'rnd': 'generate_sample_random',
    'lhs': 'generate_sample_lhs',
    'olhs': 'generate_sample_olhs',
  }

  def __init__(self, problem, tag='', doe=None, size=None, verbose=False, load_only=False, *args, **kwargs):
    self.problem = problem
    self.size_x = problem.size_x
    self.size_f = problem.size_f
    self.p_id = '%s_%d_%d' % (problem.NAME, problem.size_x, problem.size_f)

    self.tag = tag
    doe, size = getattr(problem, 'samples', {tag: (doe, size)}).get(tag, (doe, size))
    self.doe = doe or getattr(problem, 'doe', 'rnd')
    self.size = size or getattr(problem, 'size', None)
    self.s_id = f'{self.doe.lower()}_{self.size}' + (f' {tag}' if tag else '')

    self.path = os.path.join(os.path.dirname(__file__), 'samples', self.p_id, self.s_id + '.csv')
    self.x, self.f = None, None
    self.verbose=verbose

    if os.path.isfile(self.path):
      self.load()
    elif load_only:
      raise Exception('Can not find the sample file ' + self.path)
    else:
      self.generate(*args, **kwargs)
      self.save()

  @property
  def full_id(self):
    return self.p_id + ' ' + self.s_id

  def load(self, path=None):
    if path is None:
      path = self.path

    if self.verbose:
      print(f'Loading sample {path}')

    sample = np.loadtxt(path, dtype=float, skiprows=1, delimiter=',', ndmin=2)
    assert sample.shape[1] == (self.size_x + self.size_f), 'Invalid number of columns in file'
    self.x = sample[:, 0:self.size_x]
    self.f = sample[:, self.size_x:self.size_x + self.size_f]
    return self

  def generate(self, *args, **kwargs):
    try:
      self.problem.generate_sample_random(1)
    except NotImplementedError:
      sample = os.path.splitext(os.path.basename(self.path))[0]
      available_samples = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.dirname(self.path)))
      raise Exception('No such sample %s, available samples: %s' % (sample, ', '.join(available_samples)))

    if self.verbose:
      print(f'Generating {self.doe} sample of size {self.size}')

    self.x, self.f = getattr(self.problem, Sample._DOE_METHODS[self.doe])(self.size, *args, **kwargs)
    return self

  def save(self, path=None):
    if path is None:
      path = self.path

    if self.verbose:
      print(f'Saving sample {path}')

    try:
      os.makedirs(os.path.dirname(path))
    except:
      pass
    header = ','.join(self.problem.variables_names + self.problem.objectives_names)
    np.savetxt(path, np.hstack((self.x, self.f)), header=header, delimiter=',', comments='')
    return self

  def filter(self, l_bounds=(), u_bounds=(), idx=None, tag=None):
    if idx is None:
      idx = np.ones(self.x.shape[0], dtype=bool)
      for i, lb in enumerate(l_bounds):
        idx *= self.x[:, i] >= lb
      for i, ub in enumerate(u_bounds):
        idx *= self.x[:, i] <= ub

    self.x, self.f = self.x[idx], self.f[idx]
    self.size = self.x.shape[0]
    self.s_id += f' {tag}' if tag is not None else ''
    return self

  @staticmethod
  def clear_cache():
    import shutil
    path = os.path.join(os.path.dirname(__file__), 'samples')
    sample_problems = os.listdir(os.path.join(os.path.dirname(__file__), 'fs'))
    for s in os.listdir(path):
      if s.split('_')[0] + '.py' not in sample_problems:
        print('Deleting', os.path.join(path, s))
        shutil.rmtree(os.path.join(path, s))

  def plot():
    import tools
    tools.plot_sample_2d(self.x)

  def __repr__(self):
    return self.full_id
