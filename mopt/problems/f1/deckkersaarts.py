from .._functions import deckkersaarts
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = 2
    self.optimal = Optimal(x=[[0, 15]], f=[[-24777]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-20, 20), initial_guess=18)
    self.add_objective()

  def define_objectives(self, x):
    return [deckkersaarts(*x)]
