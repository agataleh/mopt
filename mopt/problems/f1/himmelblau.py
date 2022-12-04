from .._functions import himmelblau
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = 2
    self.optimal = Optimal(x=[[3, 2]], f=[[0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-6, 6), initial_guess=6)
    self.add_objective()

  def define_objectives(self, x):
    return [himmelblau(*x)]
