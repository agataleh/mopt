from .._functions import xinsheyang2
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal(x=[[0.0] * self.dimension], f=[[0.0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-6.28, 6.28), initial_guess=3)
    self.add_objective()

  def define_objectives(self, x):
    return [xinsheyang2(x)]
