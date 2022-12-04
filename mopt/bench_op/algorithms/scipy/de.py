from scipy.optimize import differential_evolution as _differential_evolution

from .._options import Option as _Option
from .._options import Options as _Options
from . import _algorithm


class Algorithm(_algorithm.Algorithm):

  def __init__(self):
    options = _Options(
        _Option(name='bounds', type='sequence', default='None',
                description='Bounds for variables.'),
        _Option(name='maxiter', type='int', default=100,
                description='The maximum number of generations'),
        _Option(name='popsize', type='int', default=15,
                description='The population has popsize * len(x) individuals.'),
        _Option(name='tol', type='float', default=0.0,
                description='Relative tolerance for convergence.'),
        _Option(name='atol', type='float', default=0.0,
                description='Abslute tolerance for convergence.'),
        _Option(name='polish', type='bool', default=False,
                description='If True, then L-BFGS-B method is used to polish the best population member.'),
        _Option(name='init', type='str', default='random',
                description='String latinhypercube, random or array specifying the initial population.'),
    )
    super(Algorithm, self).__init__(options=options)
    self._algorithm = _differential_evolution

  def _set_bounds(self, problem):
    bounds = problem.variables_bounds
    self._options.set({'bounds': list(map(list, zip(*bounds)))})

  def _set_initials(self, problem):
    pass

  def _expected_budget(self, problem=None, options={}):
    maxiter = options.setdefault('maxiter', self.options.get('maxiter'))
    popsize = options.setdefault('popsize', self.options.get('popsize'))
    return (maxiter + 1) * popsize * problem.size_x
