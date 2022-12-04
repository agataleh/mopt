import os as _os

import numpy as _np
import rbfopt as _rbfopt

from .. import _algorithm, _result
from .._options import Option as _Option
from .._options import Options as _Options


class Algorithm(_algorithm.Algorithm):

  VERSION = _rbfopt.__version__

  def __init__(self):
    options = _Options(
        _Option(name='max_iterations', type='int', default=_np.inf,
                description='Maximum number of iterations.'),
        _Option(name='max_evaluations', type='int', default=300,
                description='Maximum number of function evaluations in accurate mode.'),
        _Option(name='rbf', type='str', default='auto',
                description='Radial basis function used by the method.'),
        _Option(name='eps_zero', type='float', default=1.0e-15,
                description='Any value smaller than this will be considered zero.'),
        _Option(name='eps_impr', type='float', default=1.0e-4,
                description='Any improvement in the objective function by less than this amount in absolute and relative terms, will be ignored.'),
        _Option(name='save_state_interval', type='int', default=100000,
                description='Number of iterations after which the state of the algorithm should be dumped to file (rbfopt_algorithm_state.dat).'),
    )
    super(Algorithm, self).__init__(options=options)

  def solve(self, problem, options={}, verbose=False):
    """Solve optimization problem."""

    class RbfoptProblem(_rbfopt.RbfoptBlackBox):

      def get_dimension(self):
        return problem.size_x

      def get_var_lower(self):
        return _np.array(problem.variables_bounds[0])

      def get_var_upper(self):
        return _np.array(problem.variables_bounds[1])

      def get_var_type(self):
        return _np.array(['R'] * self.get_dimension())

      def evaluate(self, x):
        f = problem.evaluate(x)[0][0]
        # Does not support NaN values, 10**30 raises numpy error https://github.com/numpy/numpy/issues/2554
        return 10**15 if _np.isnan(f) else f

      def evaluate_noisy(self, x):
        raise NotImplementedError('evaluate_noisy not available')

      def has_evaluate_noisy(self):
        return False

    initials = _np.atleast_2d(problem.initial_guess)

    # Configure algorithm
    save_options = self._options.get()
    # Set local options
    self._options.set(options)
    # Solve problem
    rbfopt_problem = RbfoptProblem()
    settings = _rbfopt.RbfoptSettings(**self._options.get())
    algorithm = _rbfopt.RbfoptAlgorithm(settings=settings, black_box=rbfopt_problem, init_node_pos=initials)
    if not verbose:
      algorithm.set_output_stream(open(_os.devnull, 'w'))
    f, x, itercount, evalcount, fast_evalcount = algorithm.optimize()
    # Reset saved options
    self._options.set(save_options)

    return _result.Result(x=x, f=f, calls_count=problem.calls_count)

  def _expected_budget(self, problem=None, options={}):
    return options.setdefault('max_evaluations', self.options.get('max_evaluations'))
