from .._functions import brent
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = 2
    self.optimal = Optimal(x=[[-10.0, -10.0]], f=[[0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-100, 100), initial_guess=80)
    self.add_objective()

  def define_objectives(self, x):
    return [brent(*x)]
