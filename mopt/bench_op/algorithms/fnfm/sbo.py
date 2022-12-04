import numpy as _np

from .. import _algorithm, _result
from .._options import Option as _Option
from .._options import Options as _Options
from . import bayesian


class ScaledProblem(object):

  def __init__(self, scale, problem):
    self.scale = scale
    self.problem = problem

  def evaluate(self, x):
    return _np.array(self.problem.evaluate(x)) * self.scale

  @property
  def variables_bounds(self):
    return self.problem.variables_bounds

  @property
  def variables_names(self):
    return self.problem.variables_names

  @property
  def initial_guess(self):
    return self.problem.initial_guess

  @property
  def calls_count(self):
    return self.problem.calls_count


class Algorithm(_algorithm.Algorithm):

  VERSION = '0.0'

  def __init__(self):
    options = _Options(
        _Option(name='n_init', type='int', default=9, description='Number of points to sample the target function before fitting the gp.'),
        _Option(name='n_iter', type='int', default=16, description='Total number of times the process is to repeated.'),
        _Option(name='acquisition', type='str', default='ei', description='Acquisition function: upper confidence bound (ucb), expected improvement (ei), probability  of improvement (poi).'),
        _Option(name='coefficient', type='float', default=1.0, description='Main parameter: kappa for ucb, xi for ei and poi'),
    )
    super(Algorithm, self).__init__(options=options)

  def solve(self, problem, options={}, verbose=0, seed=0, initial_sample=None):
    """Solve optimization problem."""
    # Configure algorithm
    save_options = self._options.get()
    # Set local options
    self._options.set(options)

    algorithm = bayesian.Algorithm()
    # Extract options values
    n_init = self._options.get('n_init')
    n_iter = self._options.get('n_iter')
    acquisition = self._options.get('acquisition')
    coefficient = self._options.get('coefficient')
    kappa = algorithm.options.get('kappa')
    xi = algorithm.options.get('xi')
    # Generate initial sample
    if initial_sample is None:
      _np.random.seed(seed)
      initial_sample = problem.generate_sample_random(size=n_init)
    # Scale ojective function
    scale = 1.0 / _np.abs(initial_sample[1]).max()
    scaled_problem = ScaledProblem(scale, problem)
    initial_sample[1] *= scale
    # Preapre options and solve the problem
    options = {'init_points': 0, 'n_iter': n_iter, 'acq': acquisition, 'kappa': kappa, 'xi': xi}
    options['kappa' if acquisition == 'ucb' else 'xi'] = coefficient
    if verbose > 0: print('\nproblem: %-20s | n_iter: %2d/%-2d | seed: %-5d ' % (problem.NAME, n_init, n_iter, seed))
    result = algorithm.solve(problem=scaled_problem, initial_sample=initial_sample, options=options, verbose=verbose, seed=seed)
    result.optimal.f = problem.evaluate(result.optimal.x)
    if verbose > 0: print('kappa=%.3f' % options['kappa'], 'xi=%.3f' % options['xi'], result)
    # Reset saved options
    self._options.set(save_options)
    return result

  def _expected_budget(self, problem=None, options={}):
    return bayesian.Algorithm()._expected_budget(problem=problem, options=options)
