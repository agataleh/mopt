from .._functions import elatar
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = 2
    self.optimal = Optimal(x=[[3.409186, -2.171433]], f=[[1.7127803540]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-100, 100), initial_guess=80)
    self.add_objective()

  def define_objectives(self, x):
    return [elatar(*x)]
