from .._functions import bukin04
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = 2
    self.optimal = Optimal(x=[[-10.0, 0.0]], f=[[0]])

  def prepare_problem(self):
    self.add_variable(bounds=(-15, -5), initial_guess=-6)
    self.add_variable(bounds=(-3, 3), initial_guess=2)
    self.add_objective()

  def define_objectives(self, x):
    return [bukin04(*x)]
