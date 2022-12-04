from .._functions import sixhumpcamel
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = 2
    self.optimal = Optimal(x=[[0.0898, -0.7126], [-0.0898, 0.7126]], f=[[-1.03164]])

  def prepare_problem(self):
    self.add_variable(bounds=(-3, 3), initial_guess=3)
    self.add_variable(bounds=(-2, 2), initial_guess=2)
    self.add_objective()

  def define_objectives(self, x):
    return [sixhumpcamel(*x)]
