import numpy as np
from six import with_metaclass
import copy


class _Objective(object):

  def __init__(self, name):
    if not name:
      raise ValueError('Objective must have a name!')
    self.name = name


class _Constraint(object):

  def __init__(self, name, bounds):
    if not name:
      raise ValueError('Constraint must have a name!')
    if not bounds:
      raise ValueError('Constraint must define bounds!')
    if not isinstance(bounds, (list, tuple, np.array)) or len(bounds) != 2:
      raise ValueError('Wrong constraint bounds structure!')

    self.name = name
    self.lower_bound = -np.inf
    self.upper_bound = np.inf

    if bounds[0] is not None:
      self.lower_bound = max(self.lower_bound, float(bounds[0]))
    if bounds[1] is not None:
      self.upper_bound = min(self.upper_bound, float(bounds[1]))
    if self.lower_bound > self.upper_bound:
      raise ValueError('Invalid constraint bounds %s, lower > upper' % str(bounds))


class _Variable(object):

  def __init__(self, name, bounds, initial_guess):
    if not name:
      raise ValueError('Variable must have a name!')
    if not bounds:
      raise ValueError('Variable must define bounds!')

    self.name = name
    self.lower_bound = -np.inf
    self.upper_bound = np.inf
    self.initial_guess = None

    if bounds[0] is not None:
      self.lower_bound = max(self.lower_bound, float(bounds[0]))
    if bounds[1] is not None:
      self.upper_bound = min(self.upper_bound, float(bounds[1]))
    if self.lower_bound > self.upper_bound:
      raise ValueError('Invalid variable bounds %s, lower > upper' % str(bounds))

    if initial_guess is not None:
      self.initial_guess = float(initial_guess)
      is_below = bounds[0] is not None and self.initial_guess < bounds[0]
      is_above = bounds[0] is not None and self.initial_guess < bounds[0]
      if is_below or is_above:
        raise ValueError("Initial guess of variable '%s' = %s doesn't satisfy bounds %s" %
                         (self.name, self.initial_guess, str(bounds)))


class _GenericProblemInitializer(type):

  def __call__(cls, *args, **kwargs):
    new_problem = type.__call__(cls, *args, **kwargs)  # overwritten
    new_problem._initialize()  # fixed method
    new_problem.prepare_problem()  # overwritten
    new_problem.shift = [0] * new_problem.size_x
    new_problem.rotation['matrix'] = [[np.nan] * new_problem.size_x] * new_problem.size_x
    new_problem.step = [0] * new_problem.size_x
    return new_problem


class GenericProblem(with_metaclass(_GenericProblemInitializer, object)):

  def _initialize(self):
    self.NAME = self.__module__.split('.')[-1]

    self.calls_count = 0
    self._save_history = True
    self._variables = []
    self._objectives = []
    self._constraints = []
    self._history_cache = []

    self.scale = (0, 1)
    self.noise = 0
    self.shift = []
    self.rotation = {'angle': 0, 'matrix': [[]]}
    self.step = []

  # USER SIDE METHODS

  def prepare_problem(self):
    raise Exception("Prepare problem was not implemented")

  def define_objectives(self, x):
    raise Exception("Objectives was not defined")

  def define_constraints(self, x):
    raise Exception("Constraints was not defined")

  # EVALUATION METHODS

  def _evaluate_objectives(self, x):
    return np.array([self.define_objectives(xi) for xi in x])

  def _evaluate_constraints(self, x):
    return np.array([self.define_constraints(xi) for xi in x])

  def print_iters(self, step=1):
    self.i_print = lambda n, i: [print(_, end='\'') for _ in range(n + 1, n + i + 1) if _ % step == 0]
    return self

  def evaluate(self, x):
    x = np.atleast_2d(x)
    if x.shape[1] != self.size_x:
      raise Exception('Wrong dimensionality of input points: ' + str(x))

    design_points = np.zeros((x.shape[0], self.size_x + self.size_full))
    design_points[:, :self.size_x] = x
    if self.size_f > 0:
      design_points[:, self.size_x: self.size_x + self.size_f] = self._evaluate_objectives(x)
    if self.size_c > 0:
      design_points[:, self.size_x + self.size_f:] = self._evaluate_constraints(x)

    getattr(self, 'i_print', lambda *_: _)(self.calls_count, design_points.shape[0])

    self.calls_count += design_points.shape[0]
    if self._save_history:
      self._history_cache.extend(design_points.tolist())

    return design_points[:, self.size_x:]

  # ADD METHODS

  def add_variable(self, bounds, initial_guess=None, name=None):
    if not name:
      name = 'x%d' % (len(self._variables) + 1)
    self._variables.append(_Variable(name, bounds, initial_guess))

  def add_objective(self, name=None):
    if not name:
      name = 'f%d' % (len(self._objectives) + 1)
    self._objectives.append(_Objective(name))

  def add_constraint(self, bounds, name=None):
    if not name:
      name = 'c%d' % (len(self._constraints) + 1)
    self._constraints.append(_Constraint(name, bounds))

  # SET METHODS

  def set_variable_initial_guess(self, index, initial_guess):
    if (index < 0) or (index >= len(self._variables)):
      raise ValueError('Wrong variable index')
    var = self._variables[index]
    if initial_guess is not None:
      var.initial_guess = float(initial_guess)
    else:
      var.initial_guess = None

  def set_variable_bounds(self, index, bounds):
    if (index < 0) or (index >= len(self._variables)):
      raise ValueError('Wrong variable index')
    if not bounds:
      raise ValueError('Variable must define bounds!')

    var = self._variables[index]
    lower_bound = -np.inf
    upper_bound = np.inf
    if bounds[0] is not None:
      lower_bound = max(lower_bound, float(bounds[0]))
    if bounds[1] is not None:
      upper_bound = min(upper_bound, float(bounds[1]))
    if lower_bound > upper_bound:
      raise ValueError('Invalid constraint bounds %s, lower > upper' % str(bounds))
    var.lower_bound = lower_bound
    var.upper_bound = upper_bound

  def set_variables_bounds(self, bounds):
    for i in range(self.size_x):
      self.set_variable_bounds(i, bounds)
    return self

  def set_constraint_bounds(self, index, bounds):
    if (index < 0) or (index >= len(self._constraints)):
      raise ValueError('Wrong constraint index')
    if not bounds:
      raise ValueError('Constraint must define bounds!')

    con = self._constraints[index]
    lower_bound = -np.inf
    upper_bound = np.inf
    if bounds[0] is not None:
      lower_bound = max(lower_bound, float(bounds[0]))
    if bounds[1] is not None:
      upper_bound = min(upper_bound, float(bounds[1]))
    if lower_bound > upper_bound:
      raise ValueError('Invalid constraint bounds %s, lower > upper' % str(bounds))
    con.lower_bound = lower_bound
    con.upper_bound = upper_bound

  # SIZE METHODS

  @property
  def size_x(self):
    return len(self._variables)

  @property
  def size_c(self):
    return len(self._constraints)

  @property
  def size_f(self):
    return len(self._objectives)

  @property
  def size_full(self):
    return self.size_f + self.size_c

  # NAMES METHODS

  @property
  def variables_names(self):
    return [var.name for var in self._variables]

  @property
  def constraints_names(self):
    return [con.name for con in self._constraints]

  @property
  def objectives_names(self):
    return [obj.name for obj in self._objectives]

  # BOUNDS AND INITIAL GUESS METHODS

  @property
  def variables_bounds(self):
    lower = []
    upper = []
    for x in self._variables:
      lower.append(x.lower_bound)
      upper.append(x.upper_bound)
    return lower, upper

  @property
  def constraints_bounds(self):
    lower = []
    upper = []
    for c in self._constraints:
      lower.append(c.lower_bound)
      upper.append(c.upper_bound)
    return lower, upper

  @property
  def initial_guess(self):
    guess = []
    for x in self._variables:
      if x.initial_guess is not None:
        guess.append(x.initial_guess)
      else:
        # return None
        guess.append(None)
    return guess

  # HISTORY METHODS

  @property
  def history(self):
    return self._history_cache

  def enable_history(self):
    self._save_history = True

  def disable_history(self):
    self._save_history = False

  def clear_history(self):
    self._history_cache = []

  def modify(self, step=0, scale=(0, 1), initial_guess='default', noise=0, shift=0, rotation=0, seed=None):
    problem = copy.deepcopy(self)
    # Reset history
    problem.clear_history()
    problem.calls_count = 0

    # Set random initial guess
    if initial_guess != 'default':
      if seed is not None:
        np.random.seed(seed)
      for i in range(problem.size_x):
        if initial_guess is None:
          problem.set_variable_initial_guess(i, None)
        elif initial_guess == 'random':
          lb, ub = problem.variables_bounds
          problem.set_variable_initial_guess(i, np.random.uniform(lb[i], ub[i]))
        else:
          problem.set_variable_initial_guess(i, initial_guess[i])

    # Modify ojective function
    define_objectives = problem.define_objectives

    # Add noise to ojective function
    if noise != 0:
      problem.noise = noise
      def noise_wrapper(function):
        return lambda x: np.array(function(x)) * np.random.uniform(1, 1 + noise)
      define_objectives = noise_wrapper(define_objectives)

    # Shift ojective function
    if shift != 0:
      if np.array(shift).ndim == 0:
        if seed is not None:
          np.random.seed(seed)
        # Normalize shift since the area out of domain depends on dimensionality
        shift /= problem.size_x
        # x_domain = np.diff(problem.variables_bounds, axis=0).ravel()
        # shift = np.random.uniform(low=-shift * x_domain, high=shift * x_domain, size=problem.size_x)
        shift = shift * np.random.choice((-1, 1), problem.size_x) * np.diff(problem.variables_bounds, axis=0).ravel()
      if len(shift) != problem.size_x:
        raise AttributeError("Wrong value of shift.")

      problem.shift = shift
      def shift_wrapper(function):
        return lambda x: function(x + shift)
      define_objectives = shift_wrapper(define_objectives)

    # Rotate ojective function
    if rotation != 0:
      if isinstance(rotation, dict):
        if 'angle' not in rotation or 'matrix' not in rotation:
          raise AttributeError("Wrong value of rotation.")
        if np.array(rotation['matrix']).shape != (problem.size_x, problem.size_x):
          raise AttributeError("Wrong value of rotation matrix.")
        problem.rotation = rotation
      elif np.array(rotation).ndim == 0:
        if seed is not None:
          np.random.seed(seed)
        # Vectors of the rotation plane
        v1 = np.random.rand(problem.size_x)
        v2 = np.random.rand(problem.size_x)
        # Gram-Schmidt orthogonalization
        n1 = v1 / np.linalg.norm(v1)
        v2 = v2 - np.dot(n1, v2) * n1
        n2 = v2 / np.linalg.norm(v2)
        # Rotation matrix preparation
        # Normalize angle since the area out of domain depends on dimensionality
        angle = rotation / problem.size_x
        identity = np.identity(problem.size_x)
        generator2 = (np.outer(n2, n1) - np.outer(n1, n2)) * np.sin(angle)
        generator3 = (np.outer(n1, n1) + np.outer(n2, n2)) * (np.cos(angle) - 1)
        rot_matrix = identity + generator2 + generator3
        problem.rotation = {'angle': angle, 'matrix': rot_matrix.tolist()}

      x_center = np.mean(problem.variables_bounds, axis=0)

      def rotation_wrapper(function):
        return lambda x: function(np.dot(x - x_center / 2.0, problem.rotation['matrix']) + x_center / 2.0)
      define_objectives = rotation_wrapper(define_objectives)

    # Scale ojective function to mean 0 and std 1 (if current scale differs)
    if scale != (0, 1):
      '''
      x, f = problem.generate_sample(100)
      np.mean(f), np.std(f) # (18.33453708854788, 4.702735441066875)

      xs, fs = problem.modify(scale=[np.mean(f), np.std(f)]).generate_sample(100)
      np.mean(fs), np.std(fs) # (3.61932706027801e-16, 1.0)

      xss, fss = problem.modify(scale=[np.mean(f), np.std(f)]).modify(scale=[-np.mean(f) / np.std(f), 1.0 / np.std(f)]).generate_sample(100)
      np.mean(fss), np.std(fss) # (18.33453708854788, 4.702735441066875)
      '''
      problem.scale = scale
      def scale_wrapper(function):
        return lambda x: np.array(function(x) - scale[0]) / scale[1]
      define_objectives = scale_wrapper(define_objectives)

    # Set step for problem inputs
    if isinstance(step, (list, tuple, np.ndarray)):
      assert len(step) == problem.size_x, "Resolution should be an array of length size_x or double"
    else:
      step = [step] * problem.size_x
    if np.any(np.array(step) > 0):
      problem.step = step
      def step_wrapper(function):
        def stepped_function(x):
          for i, res in enumerate(problem.step):
            x[i] = res * np.round(x[i] / res) if res > 0 else x[i]
          return function(x)
        return stepped_function
      define_objectives = step_wrapper(define_objectives)

    problem.define_objectives = define_objectives
    return problem

  def generate_sample_ff(self, size=None, n_cells=None, random=True, seed=None):
    from . import _doe
    assert size is not None or n_cells is not None
    if n_cells is None and size is not None:
      n_cells = _doe.n_cells(dim=self.size_x, size=size)
    if seed is not None:
      np.random.seed(seed)
    lb, ub = np.array(self.variables_bounds)
    x = lb + _doe.full_factorial(dim=self.size_x, n_cells=n_cells, random=random) * (ub - lb)
    return x, self.evaluate(x)

  def generate_sample_grid(self, size=None, n_cells=None, random=True, allowed_area=0.99, seed=None, partial=False):
    from . import _doe
    if n_cells is None and size is not None:
      n_cells = _doe.n_cells(dim=self.size_x, size=size)
    if seed is not None:
      np.random.seed(seed)
    lb, ub = np.array(self.variables_bounds)
    x = lb + _doe.grid(dim=self.size_x, n_cells=n_cells, random=random, allowed_area=allowed_area) * (ub - lb)
    if partial and size is not None and size < len(x):
      x = x[np.random.choice(np.arange(len(x)), size, replace=False)]
    return x, self.evaluate(x)

  def generate_sample_random(self, size, seed=None):
    from . import _doe
    if seed is not None:
      np.random.seed(seed)
    lb, ub = np.array(self.variables_bounds)
    x = lb + _doe.random(dim=self.size_x, size=size) * (ub - lb)
    return x, self.evaluate(x)

  def generate_sample_lhs(self, size, seed=None, centered=False):
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=self.size_x, seed=seed, centered=centered)
    x = qmc.scale(sampler.random(size), *self.variables_bounds)
    return x, self.evaluate(x)

  def plot(self, x1_nodes=50, x2_nodes=50, wireframe=False, file_name=None, x1_bounds=None, x2_bounds=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import Axes3D

    if self.size_x != 2:
      raise Exception('For 2d problems only!')

    # Prepare sample
    x1_bounds = list(zip(*self.variables_bounds))[0] if x1_bounds is None else x1_bounds
    x2_bounds = list(zip(*self.variables_bounds))[1] if x2_bounds is None else x2_bounds
    lb1, ub1 = x1_bounds
    lb2, ub2 = x2_bounds
    s1 = np.complex(0, x1_nodes)
    s2 = np.complex(0, x2_nodes)
    x1, x2 = np.mgrid[lb1: ub1: s1, lb2: ub2: s2]
    x = np.vstack([x1.ravel(), x2.ravel()]).transpose()
    y = self.evaluate(x)
    y = np.reshape(y, (-1, x1.shape[1]))
    # Create figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f$')
    ax.set_title(self.NAME)
    # Plot surface
    if wireframe:
      surf = ax.plot_wireframe(x1, x2, y, rstride=1, cstride=1)
    else:
      surf = ax.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap=cm.coolwarm,
                             linewidth=0, antialiased=False)
    if file_name is None:
      plt.show()
    else:
      plt.savefig(file_name)
      plt.close()
