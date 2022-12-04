from .._functions import beale
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = 2
    self.optimal = Optimal(x=[[3.0, 0.5]], f=[[0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-10, 10), initial_guess=8)
    self.add_objective()

  def define_objectives(self, x):
    return [beale(*x)]
