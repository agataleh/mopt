from .._functions import trid
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal(x=[[(i + 1) * (self.dimension - i) for i in range(self.dimension)]],
                           f=[[-self.dimension * (self.dimension + 4) * (self.dimension - 1) / 6.0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-self.dimension**2, self.dimension**2), initial_guess=-self.dimension**2)
    self.add_objective()

  def define_objectives(self, x):
    return [trid(x)]
