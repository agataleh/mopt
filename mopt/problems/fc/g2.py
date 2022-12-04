import numpy as np

from .._functions import g2
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal(f=[[-0.8036]])  # for d=20

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(0, 10), initial_guess=9)
    self.add_constraint(bounds=(None, 0))
    self.add_constraint(bounds=(None, 0))
    self.add_objective()

  def define_objectives(self, x):
    return [g2(x)]

  def define_constraints(self, x):
    """
    c1(x): -np.prod(x) + 0.75 <= 0
    c2(x): np.sum(x) - 7.5 * len(x) <= 0
    """
    return [-np.prod(x) + 0.75, np.sum(x) - 0.75 * len(x)]
