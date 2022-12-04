from .._functions import levy5
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = 2
    self.optimal = Optimal(x=[[-1.3086, -1.4248]], f=[[-176.1376]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-2, 2), initial_guess=1)
    self.add_objective()

  def define_objectives(self, x):
    return [levy5(*x)]
