from .._functions import deceptive
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self._alpha = 0.5
    self.dimension = dimension or 2
    self.optimal = Optimal(x=[[self._alpha] * self.dimension], f=[[-1]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(0, 1), initial_guess=0.8)
    self.add_objective()

  def define_objectives(self, x):
    return [deceptive(x, self._alpha)]
