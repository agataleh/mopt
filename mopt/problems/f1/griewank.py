from .._functions import griewank
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal(x=[[0.0] * self.dimension], f=[[0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-600, 600), initial_guess=500)
    self.add_objective()

  def define_objectives(self, x):
    return [griewank(x)]
