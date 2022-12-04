from .._functions import g7
from .._optimal import Optimal
from .._problem import GenericProblem


class Problem(GenericProblem):

  def __init__(self, dimension=None):
    if dimension not in [None, 10]:
      raise ValueError("The problem dimension should be 10")
    self.dimension = 10
    self.optimal = Optimal(f=[[24.3062091]], x=[[2.171996, 2.363683, 8.773926, 5.095984, 0.9906548, 1.430574, 1.321644, 9.828726, 8.280092, 8.375927]])

  def prepare_problem(self):
    for i in range(self.dimension):
      self.add_variable(bounds=(-10, 10), initial_guess=9)
    for i in range(8):
      self.add_constraint(bounds=(None, 0))
    self.add_objective()

  def define_objectives(self, x):
    return [g7(*x)]

  def define_constraints(self, x):
    """
    c1(x): 4*x[0] + 5*x[1] - 3*x[6] + 9*x[7] - 105 <=0
    c2(x): 10*x[0] - 8*x[1] - 17*x[6] + 2*x[7] <=0
    c3(x): - 8*x[0] + 2*x[1] + 5*x[8] - 2*x[9] - 12 <=0
    c4(x): 3*(x[0] - 2)**2 + 4*(x[1] - 3)**2 + 2*x[2]**2 - 7*x[3] - 120 <=0
    c5(x): 5*x[0]**2 + 8*x[1] + (x[2] - 6)**2 - 2*x[3] - 40 <=0
    c6(x): 0.5*(x[0] - 8)**2 + 2*(x[1] - 4)**2 + 3*x[4]**2 - x[5] - 30 <=0
    c7(x): x[0]**2 + 2*(x[1] - 2)**2 - 2*x[0]*x[1] + 14*x[4] - 6*x[5] <=0
    c8(x): - 3*x[0] + 6*x[1] + 12*(x[8] - 8)**2 - 7*x[9] <=0
    """
    return [
        4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7] - 105,
        10 * x[0] - 8 * x[1] - 17 * x[6] + 2 * x[7],
        - 8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12,
        3 * (x[0] - 2)**2 + 4 * (x[1] - 3)**2 + 2 * x[2]**2 - 7 * x[3] - 120,
        5 * x[0]**2 + 8 * x[1] + (x[2] - 6)**2 - 2 * x[3] - 40,
        0.5 * (x[0] - 8)**2 + 2 * (x[1] - 4)**2 + 3 * x[4]**2 - x[5] - 30,
        x[0]**2 + 2 * (x[1] - 2)**2 - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5],
        - 3 * x[0] + 6 * x[1] + 12 * (x[8] - 8)**2 - 7 * x[9],
    ]
