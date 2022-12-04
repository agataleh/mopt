from .._functions import qing
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal(x=[[i**0.5 for i in range(1, self.dimension + 1)]], f=[[0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-100, 100), initial_guess=80)
    self.add_objective()

  def define_objectives(self, x):
    return [qing(x)]
