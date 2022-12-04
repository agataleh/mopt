from .._functions import michalewicz
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    self.dimension = dimension or 2
    self.optimal = Optimal(f=[[{2: -1.8014, 5: -4.6877, 10: -9.6602}.setdefault(self.dimension, float('nan'))]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(0, 3.1415))
    self.add_objective()

  def define_objectives(self, x):
    return [michalewicz(x)]
