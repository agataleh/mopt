from .._functions import levy3
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal(x=[[1] * self.dimension], f=[[0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-20, 20), initial_guess=10)
    self.add_objective()

  def define_objectives(self, x):
    return [levy3(x)]
