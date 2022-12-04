from .._functions import gulf
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = 3
    self.optimal = Optimal(x=[[50, 25, 1.5]], f=[[0]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(0.1, 100), initial_guess=80)
    self.add_objective()

  def define_objectives(self, x):
    return [gulf(*x)]
