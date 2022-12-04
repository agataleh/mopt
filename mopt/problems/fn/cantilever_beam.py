from .._problem import GenericProblem
from .._optimal import Optimal


def volume(h, h1, b1, b2, l=36):
  """
  Cantilevered Beam Problem. The volume of the beam function.

  l is the length of the beam.

  3.0 <= h <= 7.0
  0.1 <= h1 <= 1.0
  2.0 <= b1 <= 12.0
  0.1 <= b2 <= 2.0

  f(7.0, 0.1, 9.48482, 0.1) = 92.77
  h = 7.0
  h1 = 0.1
  b1 = 9.48482
  b2 = 0.1
  """
  f = (2 * h1 * b1 + (h - 2 * h1) * b2) * l
  return f


def stress(h, h1, b1, b2, l=36, p=1000):
  """
  Cantilevered Beam Problem. The maximum bending stress at the root of the beam.

  l is the length of the beam.
  p is the applied transverse point load

  c(x) <= 5000
  """
  i = inertia(h, h1, b1, b2)
  c = p * l * h / (2 * i)
  return c


def deflection(h, h1, b1, b2, l=36, p=1000, e=10e6):
  """
  Cantilevered Beam Problem. The maximum deflection at the tip of the beam.

  l is the length of the beam.
  p is the applied transverse point load
  e is the modulus of the material.

  c(x) <= 0.1
  """
  i = inertia(h, h1, b1, b2)
  c = p * l**3 / (3 * e * i)
  return c


def inertia(h, h1, b1, b2):
  """
  Cantilevered Beam Problem. The the second area moment of inertia of the beam cross section.
  """
  return (1.0 / 12.0) * b2 * (h - 2 * h1)**3 + 2 * ((1.0 / 12.0) * b1 * h1**3 + b1 * h1 * ((h - h1)**2) / 4)


class Problem(GenericProblem):
  NAME = 'cantilever_beam'

  def __init__(self, dimension=None):
    self.objective_calls_count = 0
    self.constraint_calls_count = 0
    self.dimension = dimension or 4
    self.optimal = Optimal(x=[[7.0, 0.1, 9.48482, 0.1]], f=[[92.77]], c=[[4999.9, 0.0617]])

  def prepare_problem(self):
    self.add_variable(bounds=(3.0, 7.0), name='h')
    self.add_variable(bounds=(0.1, 1.0), name='h1')
    self.add_variable(bounds=(2.0, 12.0), name='b1')
    self.add_variable(bounds=(0.1, 2.0), name='b2')

    self.add_constraint(bounds=(None, 5000), name='stress')
    self.add_constraint(bounds=(None, 0.10), name='deflection')

    self.add_objective(name='volume')

  def define_objectives(self, x):
    self.objective_calls_count += 1
    h, h1, b1, b2 = x
    return [volume(h, h1, b1, b2)]

  def define_constraints(self, x):
    self.constraint_calls_count += 1
    h, h1, b1, b2 = x
    c0 = stress(h, h1, b1, b2)
    c1 = deflection(h, h1, b1, b2)
    return [c0, c1]
