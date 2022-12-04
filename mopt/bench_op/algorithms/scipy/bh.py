import numpy as _np
from scipy.optimize import basinhopping as _basinhopping

from .._options import Option as _Option
from .._options import Options as _Options
from . import _algorithm


class Algorithm(_algorithm.Algorithm):

  def __init__(self):
    options = _Options(
        _Option(name='x0', type='1-D array', default=None,
                description='Initial guess.'),
        _Option(name='niter', type='int', default=100,
                description='The number of basin hopping iterations.'),
        _Option(name='minimizer_kwargs', type='dict', default={'method': 'COBYLA', 'options': {'maxiter': 10}},
                description='Extra arguments.'),
        _Option(name='accept_test', type='callable', default=None,
                description='Define a test to define whether or not to accept the step.'),
    )
    super(Algorithm, self).__init__(options=options)
    self._algorithm = _basinhopping

  def _set_bounds(self, problem):
    bounds = problem.variables_bounds
    self._options.set({'accept_test': _MyBounds(bounds[0], bounds[1])})

  def _set_initials(self, problem):
    x0 = problem.initial_guess or [0] * problem.size_x
    self._options.set({'x0': x0})

  def _expected_budget(self, problem=None, options={}):
    niter = options.setdefault('niter', self.options.get('niter'))
    minimizer_kwargs = options.setdefault('minimizer_kwargs', self.options.get('minimizer_kwargs'))
    maxiter = minimizer_kwargs['options']['maxiter']
    return (niter + 1) * maxiter


class _MyBounds(object):

  def __init__(self, xmin, xmax):
    self.xmin = _np.array(xmin)
    self.xmax = _np.array(xmax)

  def __call__(self, **kwargs):
    x = kwargs["x_new"]
    tmax = bool(_np.all(x <= self.xmax))
    tmin = bool(_np.all(x >= self.xmin))
    return tmax and tmin
