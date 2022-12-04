from .._functions import branin
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = 2
    self.optimal = Optimal(f=[[0.3978]])

  def prepare_problem(self):
    self.add_variable(bounds=(-5, 10), initial_guess=5)
    self.add_variable(bounds=(0, 15), initial_guess=15)
    self.add_objective()

  def define_objectives(self, x):
    return [branin(*x)]
