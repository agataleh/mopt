from .._functions import shubert4
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal(f=[[-29.017 if dimension == 2 else None]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-5.12, 5.12), initial_guess=3)
    self.add_objective()

  def define_objectives(self, x):
    return [shubert4(x)]
