from .._functions import fon
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal()

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-4, 4), initial_guess=4)
    self.add_objective()
    self.add_objective()

  def define_objectives(self, x):
    return fon(x)
