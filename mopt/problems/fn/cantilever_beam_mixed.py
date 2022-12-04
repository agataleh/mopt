from .._optimal import Optimal
from .._problem import GenericProblem
from . import cantilever_beam


class Problem(GenericProblem):
  NAME = 'cantilever_beam'

  def __init__(self, dimension=None):
    self.objective_calls_count = 0
    self.constraint_calls_count = 0
    self.dimension = dimension or 4
    self.optimal = Optimal(x=[[7.0, 0.1, 9.48482, 0.1]], f=[[92.77]], c=[[4999.9, 0.0617]])
    self._h1_values = {1: 0.1, 2: 0.25, 3: 0.35, 4: 0.5, 5: 0.65, 6: 0.75, 7: 0.9, 8: 1.0}

  def prepare_problem(self):
    self.add_variable(bounds=(3.0, 7.0), name='h')
    self.add_variable(bounds=(1, 8), name='h1'), #hints={"@GTOpt/VariableType": "Integer"})
    self.add_variable(bounds=(2.0, 12.0), name='b1')
    self.add_variable(bounds=(0.1, 2.0), name='b2')

    self.add_constraint(bounds=(None, 5000), name='stress')
    self.add_constraint(bounds=(None, 0.10), name='deflection')

    self.add_objective(name='volume')

  def define_objectives(self, x):
    self.objective_calls_count += 1
    h, h1, b1, b2 = x
    h1 = self._h1_values[round(h1)]
    return [cantilever_beam.volume(h, h1, b1, b2)]

  def define_constraints(self, x):
    self.constraint_calls_count += 1
    h, h1, b1, b2 = x
    h1 = self._h1_values[round(h1)]
    c0 = cantilever_beam.stress(h, h1, b1, b2)
    c1 = cantilever_beam.deflection(h, h1, b1, b2)
    return [c0, c1]
