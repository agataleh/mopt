import numpy as _np
import scipy as _scipy

from .. import _algorithm, _result


class Algorithm(_algorithm.Algorithm):

  VERSION = _scipy.__version__

  def solve(self, problem, options={}, seed=0, initial_sample=None):
    """Solve optimization problem."""
    # Extract objective function
    def func(x):
      outputs = problem.evaluate(x)
      f = outputs[0][0]
      return 10**15 if _np.isnan(f) else f
    # Configure algorithm
    save_options = self._options.get()
    # Intitial guess
    self._set_initials(problem)
    # Variables bounds
    self._set_bounds(problem)
    # Set run options
    self._options.set(options)
    # Solve problem
    _np.random.seed(seed)
    result = self._algorithm(func, **self._options.get())
    # Reset algorithm options
    self._options.set(save_options)
    # Return result
    return _result.Result(x=result['x'], f=result['fun'], calls_count=problem.calls_count)

  def _set_bounds(self, bounds):
    raise Exception('Bounds setter not implemented')

  def _set_initials(self, initial_guess):
    raise Exception('Initials setter not implemented')
