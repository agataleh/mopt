import os

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ._benchmark import Base, AlgorithmData, BlankResultData, ProblemData, SampleData


class Database(object):

  def __init__(self, data_file, clear=False):
    if clear and os.path.isfile(data_file):
      os.system(f'rm \"{data_file}\"')
    self.data_file = data_file
    self.engine = create_engine('sqlite:///' + self.data_file)
    self.session_maker = sessionmaker(bind=self.engine)
    self.session = self.session_maker()
    Base.metadata.create_all(self.engine)

  def add(self, benchmark_result):
    """Add new benchmark result to database"""
    self._add_unique_records(benchmark_result)
    # Add benchmark result
    self.session.add(benchmark_result)
    self.session.commit()

  def _add_unique_records(self, benchmark_result):
    # Avoid algorithms duplication
    if getattr(benchmark_result, 'algorithm', None):
      existing_algorithm = self._get_instance(instance=benchmark_result.algorithm)
      if existing_algorithm is not None:
        benchmark_result.algorithm = existing_algorithm
      self.session.add(benchmark_result.algorithm)
    # Avoid problems duplication
    if getattr(benchmark_result, 'problem', None):
      existing_problem = self._get_instance(instance=benchmark_result.problem)
      if existing_problem is not None:
        benchmark_result.problem = existing_problem
      self.session.add(benchmark_result.problem)
    # Avoid samples duplication
    if getattr(benchmark_result, 'sample', None):
      if getattr(benchmark_result.sample, 'problem', None):
        existing_problem = self._get_instance(instance=benchmark_result.sample.problem)
        if existing_problem is not None:
          benchmark_result.sample.problem = existing_problem
          benchmark_result.sample.problem_id = existing_problem.id
      existing_sample = self._get_instance(instance=benchmark_result.sample)
      if existing_sample is not None:
        benchmark_result.sample = existing_sample
      self.session.add(benchmark_result.sample)

  def _get_instance(self, instance):
    """Find in db and return Algorithm, Sample or Problem instance"""
    kwargs = {}
    for column in instance.__table__.columns:
      if column not in list(instance.__table__.primary_key.columns) and hasattr(instance, column.key):
        kwargs[column.key] = getattr(instance, column.key)
    return self.session.query(instance.__class__).filter_by(**kwargs).first()

  def select(self, query_data=BlankResultData, **kwargs):
    """Handling conflicting fields of 3 tables: results, problems, algorithms"""
    # Example: db.select(p_name='f1.amgm', a_name='swarm.pso', n_runs=3)
    query = self.session.query(query_data)
    for key, val in kwargs.items():
      # Rename conflicting fields
      if key == 'p_name':
        query = query.filter(query_data.problem.has(p_name=val))
      elif key == 's_name':
        query = query.filter(query_data.sample.has(s_name=val))
      elif key == 'a_name':
        query = query.filter(query_data.algorithm.has(a_name=val))
      # Query a table with specified field
      elif key in ProblemData.__dict__.keys():
        query = query.filter(query_data.problem.has(getattr(ProblemData, key) == val))
      elif key in SampleData.__dict__.keys():
        query = query.filter(query_data.sample.has(getattr(SampleData, key) == val))
      elif key in AlgorithmData.__dict__.keys():
        query = query.filter(query_data.algorithm.has(getattr(AlgorithmData, key) == val))
      else:
        query = query.filter(getattr(query_data, key) == val)
    return query.all()

  def select_df(self, query_data=BlankResultData, *args, **kwargs):
    """Return all the data in pandas.DataFrame format"""
    # Example: db.select_df('id', 'sample', 'f', algorithm='swarm.pso', size_x=2)
    # Example: db.select_df('id', 'problem', 'f', algorithm='swarm.pso', size_x=2)
    df = pd.DataFrame.from_records([res.dict() for res in self.session.query(query_data).all()])
    for key, value in kwargs.items():
      df = df[df[key] == value]
    if args:
      df = df[list(args)]
    return df

  def values(self, *args, **kwargs):
    """ Get unique values of specified fields """
    # Example: db.values('calls_count_mean', sample='f1c0.perm', size_x=10)
    # Example: db.values('calls_count_mean', problem='f1.perm', size_x=10)
    return self.select_df(*args, **kwargs).drop_duplicates().values.tolist()
