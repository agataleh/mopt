from .._functions import wolfe
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = 3
    self.optimal = Optimal(x=[[0, 0, 0]], f=[[0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(0, 2), initial_guess=2)
    self.add_objective()

  def define_objectives(self, x):
    return [wolfe(*x)]
