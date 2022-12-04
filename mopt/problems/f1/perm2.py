from .._functions import perm2
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal(x=[1.0 / (i + 1) for i in range(self.dimension)], f=[[0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-self.dimension, self.dimension))
    self.add_objective()

  def define_objectives(self, x):
    return [perm2(x)]
