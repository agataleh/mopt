import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PerformanceProfile(object):

  def __init__(self, df, problem_fields=['p_name', 'size_x'], solver_fields=['a_name', 'version', 'tag'], penalty=1.05, tol=1e-6, use_optimal=True):
    """
    Generate data for performance profiles.

    Parameters
    ----------
    problem_fields : list of string
      Fields describing optimization problem.
    solver_fields : list of string
      Fields describing optimization algorithm.
    penalty : float
      Penalty value for unfeasible runs - algorithm didnt solve the problem.
    use_optimal : bool
      Whether or not to use the known optimal objective value to normalize the results.
      If there is no known solution available the minimum reached value by all tested algorithms is used.
    """
    self.problem_fields = problem_fields
    self.solver_fields = solver_fields
    self.penalty = penalty
    self.df = self._normalize(df.copy(deep=False), tol=tol, use_optimal=use_optimal)
    self.df = self.df.sort_values(['p_name', 'size_x', 'calls_count_mean', 'a_name'])

  def _calc(self, df, metric, norm_type):
    """
    Generate data for performance profiles.

    Parameters
    ----------
    df : pandas DataFrame
        Data containing benchmark results.
    metric : string
        Label that indicates the performance measure.
    norm_type : 'min', 'max', 'none'
        Indicating if the assigned performance measure is the
        value divided by the smallest value (standard), or the
        inverse of this operation. Or without normalization!

    Returns
    -------
    x : numpy.ndarray
        Unique performance metric values where the solvers have an increased
        number of solved problems.
    y : numpy.ndarray
        Number of problems each solver solved within the ratio of x.
    solvers : string
        Unique solver names
    data : pandas DataFrame
        Dataframe with normalized performance measure values.
    """
    if not isinstance(self.problem_fields, (tuple, list)):
      raise TypeError("'problem' should be the lists of relating to the problem fields")
    if not isinstance(self.solver_fields, (tuple, list)):
      raise TypeError("'solver' should be the lists of relating to the solver fields")
    if len(set(self.solver_fields) & set(self.problem_fields)) != 0:
      raise ValueError('Solver and problem definitions share fields: ',
                       list(set(self.solver_fields) & set(self.problem_fields)))

    data = df.sort_values(by=list(self.problem_fields + self.solver_fields)).copy()
    # Merging columns if more than one solver characteristic is selected
    for i, column in enumerate(self.solver_fields):
      if i > 0:
        data[self.solver_fields[0]] = data[self.solver_fields[0]].map(str).str.cat(data[column].map(str), sep=' ')
    # Finding the unique solvers
    solvers = data[self.solver_fields[0]].unique()
    # Generating df containing all unique problems
    grouped_by_problem = data.groupby(self.problem_fields)
    metric_max = data[metric].max()
    metric_min = data[metric].min()
    # dividing by the minimum value
    for i, (problem, group) in enumerate(grouped_by_problem):
      # Checking if all problems have an equal number of solvers
      if i == 0:
        group_length = len(group)
      elif group_length != len(group):
        raise ValueError('Problem group lengths not equal! Problem group:', problem)
      try:
        # Normalizing and penalizing infeasible designs
        feasible = group['feas'].values.astype(bool) if 'feas' != metric else np.ones_like(group['feas'].values)
        feasible_min = group.loc[feasible][metric].min()
        if norm_type == 'min':
          data.loc[group.loc[feasible].index, metric] = group.loc[feasible][metric] / feasible_min
          data.loc[group.loc[~feasible].index, metric] = (metric_max / metric_min) * self.penalty
        elif norm_type == 'max':
          data.loc[group.loc[feasible].index, metric] = feasible_min / group.loc[feasible][metric]
          data.loc[group.loc[~feasible].index, metric] = (metric_min / metric_max) / self.penalty
        else:
          data.loc[group.loc[feasible].index, metric] = group.loc[feasible][metric]
          data.loc[group.loc[~feasible].index, metric] = metric_max * self.penalty
      except KeyError:
        if norm_type == 'min':
          data.set_value(group.index, metric, group[metric] / group[metric].min())
        elif norm_type == 'max':
          data.set_value(group.index, metric, group[metric].min() / group[metric])
        else:
          data.set_value(group.index, metric, group[metric])
    # Generate array for plot
    if (len(data) // len(solvers)) != len(grouped_by_problem):
      import warnings
      warnings.warn('Combination of problem and solver characteristic cause, possibly unwanted, aggregation of problems.')
    # Grouping by unique solver
    grouped_by_solver = data.groupby(self.solver_fields)
    # Finding the unique tau values
    x = np.sort(data[metric].unique()).astype(float)
    # Finding the fraction of problems that each solver solved within tau
    n_problems = float(len(grouped_by_problem))
    y = np.zeros((len(x), len(grouped_by_solver)))
    for i, x_value in enumerate(x):
      for solver_id, (solver, group) in enumerate(grouped_by_solver):
        y[i, solver_id] = len(group.loc[group[metric] <= x_value]) / n_problems
    return x, y, solvers, data

  def _normalize(self, df, tol, use_optimal=True):
    """
    Calculate normalized budgets and objectives values.

    Parameters
    ----------
    df : pandas DataFrame
        Data containing benchmark results.
    tol : float
        Tolerance for normalized objective values.
    use_optimal : bool
        Whether or not to use the known optimal objective value to normalize the results.
        If there is no known solution available the minimum reached value by all tested algorithms is used.

    Returns
    -------
    df : pandas DataFrame
        Data containing benchmark results and additional columns with
        normalized budgets and objectives values
    """
    # calculate normalized metric values for each problem instance
    df['objectives'] = ''
    df['objectives_norm'] = ''
    df['calls_count_dim'] = ''
    df['calls_count_norm'] = ''
    # Iterate problems to normalize metric values
    for values in df.loc[:, self.problem_fields].drop_duplicates().values:
      mask = np.prod([df[self.problem_fields[i]] == values[i] for i in range(len(self.problem_fields))], axis=0)
      mask = mask.astype(bool)
      # Define the best known value
      f_mean = df.loc[mask].f_mean.values.astype(float)
      f_best = df.loc[mask].f_best.values.astype(float)
      f_best = f_best.min() if use_optimal and np.isfinite(f_best).all() else f_mean.min()
      # Normalize the best reached values
      f = np.fabs((f_best - f_mean)) / np.fmax(np.fabs(f_best), 1)
      f_norm = (f_mean - f_best) / (f_mean.max() - f_best) if f_mean.max() > f_best else np.zeros_like(f_mean)
      # Ignore minor differences
      f[f < tol] = 0
      f_norm[f_norm < tol] = 0
      df.loc[mask, 'objectives'] = f
      df.loc[mask, 'objectives_norm'] = f_norm
      # Normalize the number of objective evaluations
      b_mean = df.loc[mask].calls_count_mean.values
      df.loc[mask, 'calls_count_dim'] = b_mean / values[self.problem_fields.index('size_x')]
      df.loc[mask, 'calls_count_norm'] = (b_mean - b_mean.min()) / b_mean.ptp() if b_mean.ptp() > 0 else np.zeros_like(b_mean)
    return df

  def _estimate_f(self, f, b, b_target):
    if b_target > b.max():
        # Target budget is more than any budget used (for lost info see DataProfiles)
      f_value = f[np.argmax(b)]
      feasible = True
    elif b_target < b.min():
      # Target budget is less than any budget used
      f_value = f.max() * self.penalty  # np.nan
      feasible = False
    else:
      # Linear interpolation
      f_value = np.interp(b_target, b, f)
      # f_value = f[b <= b_target][-1]
      feasible = True
    return f_value, feasible

  def _estimate_b(self, f, b, f_target):
    if f_target >= f.max():
      feasible = True
      # Target point is worse than all reached f
      b_value = b.min()
    elif f_target < f.min():
      feasible = False
      # Target point is better than all reached f
      b_value = b.max() * self.penalty  # np.nan
    else:
      feasible = True
      # Linear interpolation
      b_right = b[f <= f_target].min()
      if b_right == b.min():
        b_value = b_right
      else:
        b_left = b[(f > f_target) & (b <= b_right)].max()
        f_right = f[b == b_right][0]
        f_left = f[b == b_left][0]
        # Transpose axes and interpolate b(f)
        b_value = np.interp(f_target, [f_right, f_left], [b_right, b_left])
    return b_value, feasible

  def arrange(self, f_target=None, b_target=None, f_metric='objectives', b_metric='calls_count_dim', norm_type='none'):
    data = []
    key_fields = np.append(self.problem_fields, self.solver_fields).tolist()

    for key, group in self.df.groupby(key_fields):
      f, b = group[[f_metric, b_metric]].values.astype(float).T
      if f_target is not None and b_target is None:
        b_value, feasible = self._estimate_b(f, b, f_target)
        f_value = f_target
      elif b_target is not None and f_target is None:
        f_value, feasible = self._estimate_f(f, b, b_target)
        b_value = b_target
      elif b_target is not None and f_target is not None:
        feasible = ~np.any((f <= f_target) & (b <= b_target))
        f_value = f_target
        b_value = b_target
      else:
        raise ValueError('At least one *_target should be specified')
      data.append(np.hstack((key, f_value, b_value, feasible)))

    df = pd.DataFrame(data, columns=np.hstack((key_fields, 'objectives', 'calls_count_dim', 'feas')))
    df = df.astype({'objectives': float, 'calls_count_dim': float})
    # Bug in pandas, astype converts all boolean objects to True
    df['feas'] = df['feas'].astype(str) == 'True'

    if f_target is not None and b_target is None:
      x, y, solvers, data = self._calc(df, metric='calls_count_dim', norm_type=norm_type)
    elif f_target is None and b_target is not None:
      x, y, solvers, data = self._calc(df, metric='objectives', norm_type=norm_type)
    elif b_target is not None and f_target is not None:
      x, y, solvers, data = self._calc(df, metric='feas', norm_type=norm_type)
    # Remove nan values if penalty is set to np.nan (will not continue plot up to 100 probability)
    return np.hstack((x[0], x[np.isfinite(x)])), np.vstack((np.zeros((1, solvers.size)), y[np.isfinite(x)])), solvers
    # return np.hstack((x[0], x, x[-1] * self.penalty)), np.vstack((np.zeros((1, solvers.size)), y, y[-1])), solvers

  def plot(self, x, y, solvers, labels=[], colors=None, markers=None, scale='linear'):
    """
    Draw performance profiles.

    Parameters
    ----------
    x : numpy.ndarray
        Unique performance metric values where the solvers have an increased
        number of solved problems.
    y : numpy.ndarray
        Number of problems each solver solved within the ratio of x.
    solvers : string
        Unique solver names.
    colors : str or list
        Unique solver colors.
    markers : str or list
        Unique solver markers.
    scale : 'linear', 'log' or 'symlog'
       Scale for x axis representing performance metric values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, solver in enumerate(solvers):
      ax.step(x, y[:, i],
              label=labels[i] if labels else solver,
              marker=markers[i] if markers else None,
              c=colors[i] if colors else None,
              where='post', alpha=0.8)

    plt.legend() # loc=4
    ax.set_xlabel('Performance rate')
    ax.set_ylabel('Fraction of problems')
    ax.set_xscale(scale)

  def plot_full(self, f_nodes=10, f_max=10, b_nodes=10, colors=None):
    f_nodes = 10 if f_nodes is None else f_nodes
    f_max = self.df['objectives'].max() if f_max is None else f_max
    f_values = np.linspace(self.df['objectives'].min(), f_max, f_nodes)
    f_values.sort()

    b = self.df['calls_count_dim']
    b_values = b.drop_duplicates().values.astype(float) if b_nodes is None else np.linspace(b.min(), b.max(), b_nodes)
    b_values.sort()

    f_values = np.hstack((-0.01, f_values))
    b_values = np.hstack((b_values.min() - 1, b_values))

    f_values = f_values.reshape(-1, 1)
    b_values = b_values.reshape(-1, 1)
    b_data = np.zeros((0, 1))
    f_data = np.zeros((0, 1))
    p_data = np.zeros((0, self.arrange(b_target=-1, f_target=-1)[-1].size))

    # for each b generate pp and use interp1d to calculate f values
    for b in b_values:
      x, y, solvers = self.arrange(b_target=b)
      p_values = np.hstack([np.interp(f_values, x, y[:, i]) for i, _ in enumerate(solvers)])

      b_data = np.vstack((b_data, np.full(f_values.shape, b)))
      f_data = np.vstack((f_data, f_values))
      p_data = np.vstack((p_data, p_values))

    shape = [b_values.size, f_values.size]
    b_data = b_data.reshape(shape)
    f_data = f_data.reshape(shape)
    p_data = p_data.reshape(shape + [-1])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Computational budget')
    ax.set_ylabel('Reached precision')
    ax.set_zlabel('Fraction of problems')
    for i, solver in enumerate(solvers):
      surf = ax.plot_wireframe(b_data, f_data, p_data[:, :, i], label=solver,
                               color=colors[i] if colors else None, alpha=0.8)
    plt.legend()
