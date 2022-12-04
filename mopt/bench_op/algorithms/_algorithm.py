class Algorithm(object):

  VERSION = '-'

  def __init__(self, options):
    self._options = options
    self.NAME = self.__module__.split('algorithms.')[-1]

  @property
  def options(self):
    """Optimizer options."""
    return self._options

  def solve(self, problem, options={}):
    """Solve optimization problem."""
    raise Exception('Solve method not implemented')
