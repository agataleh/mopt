from .._functions import powell
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 4
    self.optimal = Optimal(x=[[0.0] * self.dimension], f=[[0]])
    if self.dimension % 4 != 0:
      raise ValueError("The problem dimension should be multiple of 4")

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-4, 5), initial_guess=4)
    self.add_objective()

  def define_objectives(self, x):
    return [powell(x)]
