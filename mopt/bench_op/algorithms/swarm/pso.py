import numpy as _np

from .. import _algorithm, _result
from .._options import Option as _Option
from .._options import Options as _Options
from . import pso_core as _pso_core


class Algorithm(_algorithm.Algorithm):

  VERSION = '0.1'

  def __init__(self):
    options = _Options(
        _Option(name='n_parts', type='int', default=15, description='Number of particles'),
        _Option(name='max_itr', type='int', default=100, description='Maximum iterations'),
        _Option(name='inertia_start', type='float', default=0.9, description='Initital inertia weight'),
        _Option(name='inertia_end', type='float', default=0.4, description='Ending inertia weight'),
        _Option(name='nostalgia', type='float', default=2.1, description='Nostalia weight'),
        _Option(name='societal', type='float', default=2.1, description='Societal weight'),
        _Option(name='topology', type='string', default='gbest', description='Neighborhood topology'),
        _Option(name='tol_thres', type='float', default=None, description='Tolerance stop criteria'),
        _Option(name='tol_win', type='int', default=5, description='Tolerance stagnation rate'),
    )
    super(Algorithm, self).__init__(options=options)

  def solve(self, problem, options={}, seed=0, initial_sample=None):
    """Solve optimization problem."""
    # Extract objective function
    def func(*x):
      outputs = problem.evaluate(x)
      f = outputs[0][0]
      return 10**30 if _np.isnan(f) else f
    # Configure algorithm
    save_options = self._options.get()
    # Set run options
    self._options.set(options)

    _np.random.seed(seed)

    # Prepare problem and options
    box_bounds = list(map(list, zip(*problem.variables_bounds)))
    keys_run = ['max_itr', 'tol_thres', 'tol_win']
    kwargs_run = dict([item for item in self._options.get().items() if item[0] in keys_run])
    kwargs_init = dict([item for item in self._options.get().items() if item[0] not in keys_run])
    # Solve problem
    pso = _pso_core.PSO(obj_func=func, box_bounds=box_bounds, **kwargs_init)
    result = pso.run(**kwargs_run)
    # Reset algorithm options
    self._options.set(save_options)
    # Return result
    return _result.Result(x=result[0], f=result[1], calls_count=problem.calls_count)

  def _expected_budget(self, problem=None, options={}):
    n_parts = options.setdefault('n_parts', self.options.get('n_parts'))
    max_itr = options.setdefault('max_itr', self.options.get('max_itr'))
    return n_parts * max_itr + n_parts + max_itr + 1
