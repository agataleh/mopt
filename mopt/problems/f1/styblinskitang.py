from .._functions import styblinskitang
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal(x=[[-2.903534] * self.dimension], f=[[-39.1662 * self.dimension]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-5, 5), initial_guess=5)
    self.add_objective()

  def define_objectives(self, x):
    return [styblinskitang(x)]
