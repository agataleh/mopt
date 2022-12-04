from .._functions import hosaki
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = 2
    self.optimal = Optimal(x=[[4, 2]], f=[[-2.34589]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(0, 10), initial_guess=10)
    self.add_objective()

  def define_objectives(self, x):
    return [hosaki(*x)]
