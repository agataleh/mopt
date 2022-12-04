import inspect
import os

import numpy as np

from .._problem import GenericProblem


def random_plane(initial_guess, optimal_point, bounds, seed=0, return_calculator=False,
                 initial_guess_crange=(0, 1), optimal_point_crange=(-1, 0), random_points_crange=(-1, 1)):
  """
  Build plane thorught the given points with coefficients a in form
  a[0] * x[0] + ... + a[d + ] * x[d + 1] = 1.0, where d is the number of inputs
  Values for initial point, optimal point and randomly generated points can be specified by
  *_crange parameter.

  By default the plane is configured to split the initial and optimial points for c(x) >= 0 constrain.
  To achieve that initial point is set with the constraint value more or equal to 0 (valid point for c(x) >= 0)
  and the optimal point is set with the constraint value less than 0 (invalid point for c(x) >= 0).
  Remaining points that are generated to calculate plane coefficients can have any value within range (-1, 1).

  One can configure a random plane so that both initial and optimal points are valid like that:
  initial_guess_crange=(0, 1)
  optimal_point_crange=(0, 1)
  random_points_crange=(-1, 0)
  """
  assert np.isfinite(np.array(initial_guess, dtype=float)).all(), 'Initial guess should be set'
  np.random.seed(seed)
  dim = len(initial_guess) + 1
  bounds = np.array(bounds)
  points = np.random.random((dim, dim))
  # First point is initial point with random positive output
  points[0, :dim - 1] = initial_guess
  points[0, -1] = np.min(initial_guess_crange) + points[0, -1] * np.ptp(initial_guess_crange)
  # Second point is known optimal point with random negative output
  points[1, :dim - 1] = optimal_point
  points[1, -1] = np.min(optimal_point_crange) + points[1, -1] * np.ptp(optimal_point_crange)
  # Remaining points are random points in given bounds
  # with random output in range [-1, 1]
  points[2:, :dim - 1] = bounds[0] + points[2:, :dim - 1] * bounds.ptp(0)
  points[2:, -1] = np.min(random_points_crange) + points[2:, -1] * np.ptp(random_points_crange)
  coeff = np.dot(np.linalg.inv(points), np.ones((dim, 1))).flatten()

  def calc_plane_values(x):
    return (1 - np.dot(coeff[:-1], x.T)) / coeff[-1]

  return (coeff, calc_plane_values) if return_calculator else coeff


class LinConstrainedProblem(GenericProblem):

  def __init__(self, problem, coefficients=None):
    if coefficients is None:
      self.coefficients = [0] * problem.size_x + [1]
    self.coefficients = np.atleast_2d(coefficients)
    assert self.coefficients.shape[1] == problem.size_x + 1, "Wrong number of coefficients for linear constraint"

    self.problem = problem
    self.dimension = problem.size_x
    self.optimal = problem.optimal

  def prepare_problem(self):
    self.NAME += '.' + self.problem.NAME

    for ig, lb, ub in zip(self.problem.initial_guess, *self.problem.variables_bounds):
      self.add_variable(bounds=(lb, ub), initial_guess=ig)

    for i in range(len(self.coefficients)):
      self.add_constraint(bounds=(0, None))

    for obj in self.problem.objectives_names:
      self.add_objective(name=obj)

  def define_objectives(self, x):
    return self.problem.define_objectives(x)

  def define_constraints(self, x):
    return [(1 - np.dot(coeff[:-1], x.T)) / coeff[-1] for coeff in self.coefficients]


_lists = set([])
_all = set([])


def _exestr(class_name, func_name):
  if class_name in globals():
    return '%s.append(%s)' % (class_name, func_name)
  else:
    _lists.add(class_name)
    return '%s = [%s]' % (class_name, func_name)


__all__ = [_[:-3] for _ in os.listdir(os.path.dirname(__file__)) if _.endswith('.py') and not _.startswith('_')]
for _name in __all__:
  exec('from . import %s' % _name)
  exec('_all.add(%s)' % _name)
  # Classify problem by multimodality (_u, _m, _x)
  exec('from .._functions import %s as _function' % _name)
  _class_name = '_%s' % _function.func_type
  exec(_exestr(class_name=_class_name, func_name=_name))
  # Classify problem by dimention (_d2, _d3, ..., _dn)
  _args = inspect.getargspec(_function).args
  _class_name = '_d%s' % ('n' if 'x' in _args else str(len(_args)))
  exec(_exestr(class_name=_class_name, func_name=_name))

# Sort by name
for _ in _lists:
  exec('%s.sort(key=lambda problem: problem.Problem().NAME)' % _)
_all = sorted(_all, key=lambda problem: problem.Problem().NAME)
