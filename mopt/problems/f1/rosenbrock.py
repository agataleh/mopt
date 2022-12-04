from .._functions import rosenbrock
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal(x=[[1.0] * self.dimension], f=[[0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-2.048, 2.048), initial_guess=-2)
    self.add_objective()

  def define_objectives(self, x):
    return [rosenbrock(x)]
