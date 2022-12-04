from .._functions import whitley
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal(x=[[1.0] * self.dimension], f=[[0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-10.24, 10.24), initial_guess=10)
    self.add_objective()

  def define_objectives(self, x):
    return [whitley(x)]
