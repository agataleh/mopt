from .._functions import goldsteinprice
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = 2
    self.optimal = Optimal(x=[[0, -1]], f=[[3]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-2, 2), initial_guess=2)
    self.add_objective()

  def define_objectives(self, x):
    return [goldsteinprice(*x)]
