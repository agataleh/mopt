import warnings as _warnings

import numpy as _np
from bayes_opt import BayesianOptimization as _BayesianOptimization
from pkg_resources import get_distribution as _get_distribution
from sklearn.gaussian_process.kernels import Matern as _Kernel

from .. import _algorithm, _result
from .._options import Option as _Option
from .._options import Options as _Options


class Algorithm(_algorithm.Algorithm):

  VERSION = _get_distribution('bayesian-optimization').version

  def __init__(self):
    options = _Options(
        _Option(name='init_points', type='int', default=5, description='Number of randomly chosen points to sample the target function before fitting the gp.'),
        _Option(name='n_iter', type='int', default=25, description='Total number of times the process is to repeated.'),
        _Option(name='acq', type='str', default='ei', description='Acquisition function:\
                                                                   upper confidence bound (ucb),\
                                                                   expected improvement (ei),\
                                                                   probability  of improvement (poi).'),
        _Option(name='kappa', type='float', default=2.576, description='For ucb'),
        _Option(name='xi', type='float', default=0.0, description='For ei and poi'),
        _Option(name='nu', type='float', default=2.5, description='Main parameter of Matern kernel ~(0, 10)'),
    )
    super(Algorithm, self).__init__(options=options)

  def solve(self, problem, initial_sample=None, options={}, verbose=0, seed=0, **kwargs):
    """Solve optimization problem."""

    def func(**kwargs):
      x = [value for (key, value) in sorted(kwargs.items())]
      f = problem.evaluate(x)[0][0]
      return -f if _np.isfinite(f) else -10**30

    pbounds = zip(*problem.variables_bounds)
    pbounds = dict(zip(problem.variables_names, pbounds))

    # _warnings.filterwarnings("ignore")
    bo = _BayesianOptimization(f=func, pbounds=pbounds, verbose=verbose, random_state=seed)

    if initial_sample is not None:
      init_x, init_f = initial_sample
      for i in range(init_x.shape[0]):
        if init_f is not None:
          bo.register(params=bo.space.array_to_params(init_x[i]), target=-float(init_f[i]))
        else:
          bo.probe(params=bo.space.array_to_params(init_x[i]), lazy=True)

    # Configure algorithm
    saved_options = self.options.get()
    # Set local options
    self.options.set(options)
    # Solve problem
    bo.maximize(
        init_points=self.options.get('init_points'),
        n_iter=self.options.get('n_iter'),
        acq=self.options.get('acq'),
        xi=self.options.get('xi'),
        kappa=self.options.get('kappa'),
        kernel=_Kernel(nu=self.options.get('nu')),
        normalize_y=True,
    )
    # Reset saved options
    self.options.set(saved_options)
    # _warnings.resetwarnings()

    x = [bo.max['params'][_] for _ in sorted(bo.max['params'].keys())]
    f = -bo.max['target']
    return _result.Result(x=x, f=f, calls_count=problem.calls_count)

  def _expected_budget(self, problem=None, options={}):
    init_points = options.setdefault('init_points', self.options.get('init_points'))
    n_iter = options.setdefault('n_iter', self.options.get('n_iter'))
    return init_points + n_iter + 1


"""
self._gp = GaussianProcessRegressor(
    kernel=Matern(nu=2.5),
    alpha=1e-6,
    normalize_y=True,
    n_restarts_optimizer=5,
    random_state=self._random_state,
)
# The parameter nu controlling the smoothness of the learned function. The smaller nu, the less smooth the approximated function is.
# For nu=inf, the kernel becomes equivalent to the RBF kernel and for nu=0.5 to the absolute exponential kernel.
# Important intermediate values are nu=1.5 (once differentiable functions) and nu=2.5 (twice differentiable functions).
# Note that values of nu not in [0.5, 1.5, 2.5, inf] incur a considerably higher computational cost (appr. 10 times higher)
# since they require to evaluate the modified Bessel function. Furthermore, in contrast to l, nu is kept fixed to its initial value and not optimized.


def _ucb(x, gp, kappa):
  mean, std = gp.predict(x, return_std=True)
  return mean + kappa * std


def _ei(x, gp, y_max, xi):
  mean, std = gp.predict(x, return_std=True)
  z = (mean - y_max - xi) / std
  return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)


def _poi(x, gp, y_max, xi):
  mean, std = gp.predict(x, return_std=True)
  z = (mean - y_max - xi) / std
  return norm.cdf(z)
"""
