from .._functions import amgm
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal(x=[[0.0] * self.dimension], f=[[0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(0, 10), initial_guess=i % 10)
    self.add_objective()

  def define_objectives(self, x):
    return [amgm(x)]
