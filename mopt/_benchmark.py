import copy
import json
import os
import pickle
import sys
import time
from datetime import timedelta

import numpy as np
from sqlalchemy import Column, ForeignKey, types
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator

Base = declarative_base()


class JSONString(TypeDecorator):
  impl = types.String

  def process_bind_param(self, value, dialect):
    return json.dumps(value) if value is not None else None

  process_literal_param = process_bind_param

  def process_result_value(self, value, dialect):
    return json.loads(value) if value is not None else None


class AlgorithmData(Base):
  __tablename__ = 'algorithm'
  id = Column(types.Integer, primary_key=True)
  a_tag = Column(types.String)
  a_name = Column(types.String)
  version = Column(types.String)

  def __init__(self, algorithm, tag=''):
    self.a_tag = getattr(algorithm, 'tag', '') or tag
    self.a_name = algorithm.NAME
    self.version = algorithm.VERSION

  def copy(self):
    exec('import algorithms.%s as algorithm' % self.a_name.lower())
    return algorithm.Algorithm()

  def __str__(self):
    return '<%20s    >' % self.a_name

  def __repr__(self):
    return self.__str__()


class ProblemData(Base):
  __tablename__ = 'problem'
  id = Column(types.Integer, primary_key=True)
  # Problem name
  p_tag = Column(types.String)
  p_name = Column(types.String)
  # Problem attributes
  size_x = Column(types.Integer)
  size_f = Column(types.Integer)
  size_c = Column(types.Integer)
  bounds = Column(JSONString())
  # Modification paramters
  scale = Column(JSONString())
  noise = Column(types.Integer)
  shift = Column(JSONString())
  rotation = Column(JSONString())
  step = Column(JSONString())
  # Known optimal solution
  x_best = Column(JSONString())
  f_best = Column(JSONString())

  # Store constraint-specific information in the same table.
  # Identity is needed to request problems of specific type
  # (unconstrained, linear constrained and etc)
  constraints_type = Column(types.String)
  __mapper_args__ = {
      'polymorphic_on': constraints_type,
      'polymorphic_identity': 'none'
  }

  def __init__(self, problem, tag=''):
    # Problem name
    self.p_tag = getattr(problem, 'tag', '') or tag
    self.p_name = problem.NAME
    # Problem attributes
    self.size_x = problem.size_x
    self.size_f = problem.size_f
    self.size_c = problem.size_c
    self.bounds = problem.variables_bounds
    # Modification paramters
    self.scale = problem.scale
    self.noise = problem.noise
    self.shift = problem.shift
    self.rotation = problem.rotation
    self.step = problem.step
    # Known optimal solution
    self.x_best = problem.optimal.x[0].tolist()
    self.f_best = problem.optimal.f[0][0].tolist()

  def copy(self):
    import importlib
    problem = importlib.import_module(f'.problems.f1.{self.p_name.lower()}', package='mopt')
    # TODO should we apply modifications (noise, shift, scale, ...) ?
    return problem.Problem(self.size_x)

  def __str__(self):
    return '<%20s %2d >' % (self.p_name, self.size_x)

  def __repr__(self):
    return self.__str__()


class LinConProblemData(ProblemData):

  # Use it to store constraint-specific information in separate table
  # __tablename__ = 'lincon_problem'
  # id = Column(types.Integer, ForeignKey('problem.id'), primary_key=True)
  coeff = Column(JSONString())

  __mapper_args__ = {'polymorphic_identity': 'linear'}

  def __init__(self, problem, coeff, tag=''):
    super().__init__(problem=problem, tag=tag)
    self.coeff = np.atleast_2d(coeff).tolist()

  def copy(self):
    import importlib
    from .problems.fc import LinConstrainedProblem
    p_class = importlib.import_module(f'.problems.f1.{self.p_name.lower()}', package='mopt')
    # TODO should we apply modifications (noise, shift, scale, ...) ?
    base_problem = p_class.Problem(self.size_x)
    return LinConstrainedProblem(base_problem, self.coeff)


class SampleData(Base):
  __tablename__ = 'sample'
  id = Column(types.Integer, primary_key=True)
  s_tag = Column(types.String)
  # Sample paramters
  doe = Column(types.String)
  size = Column(types.Integer)
  path = Column(types.String)
  # Original problem
  problem_id = Column(types.Integer, ForeignKey('problem.id'))
  problem = relationship(ProblemData)
  # Features
  f_range = Column(JSONString())
  features = Column(JSONString())
  features_tag = Column(types.String)

  def __init__(self, sample):
    if sample is None:
      return
    self.s_tag = sample.tag
    # Sample paramters
    self.doe = sample.doe
    self.size = sample.size
    self.path = sample.path
    # Original problem
    self.problem = ProblemData(sample.problem)
    # Features
    self.f_range = sample.f.min(), sample.f.max()
    self.features = None
    self.features_tag = '_'

  def calc_features(self, func, features_tag='_', triples_tag='vm', load_triples=True, load_features=False, log=False):
    sample = self.sample
    self.features_tag = features_tag

    features_path = self.path + f'.{self.features_tag}.features'
    if os.path.isfile(features_path) and load_features:
      if log:
        print(f'Loading features {features_path}')
      with open(features_path, 'rb') as fp:
        self.features = pickle.load(fp)
    else:
      triples_path = self.path + f'.{triples_tag}.triples'
      if os.path.isfile(triples_path) and load_triples:
        if log:
          print(f'Loading triples {triples_path}')
        triples = np.loadtxt(triples_path).astype(int)
      else:
        from . import ela
        if triples_tag == 'vm':
          triples = ela.get_triples(sample.x)
        elif triples_tag == 'minobj':
          triples = ela.get_triples_minobj(sample.x, sample.y)
        np.savetxt(triples_path, triples)
      self.features = func(sample.x, sample.f, triples)
      with open(features_path, 'wb') as fp:
        pickle.dump(self.features, fp)
    return self.features

  @property
  def sample(self):
    from . import Sample
    return Sample(self.problem.copy(), tag=self.s_tag, doe=self.doe, size=self.size, log=False, load_only=True)

  def __str__(self):
    if self.problem is None:
      return 'empty sample'
    return '%s: %d R^%d->R^%d' % (self.problem.p_name, self.size, self.problem.size_x, self.problem.size_f)

  def __repr__(self):
    return self.__str__()


class BlankResultData(Base):

  __tablename__ = 'blank_result'
  id = Column(types.Integer, primary_key=True)
  tag = Column(types.String)
  # Sample
  sample_id = Column(types.Integer, ForeignKey('sample.id'))
  sample = relationship(SampleData)
  # Problem
  problem_id = Column(types.Integer, ForeignKey('problem.id'))
  problem = relationship(ProblemData)
  # Algorithm
  algorithm_id = Column(types.Integer, ForeignKey('algorithm.id'))
  algorithm = relationship(AlgorithmData)
  # Algorithm settings
  options = Column(JSONString())
  extra_settings = Column(JSONString())
  # Results of run
  time = Column(JSONString())
  results = Column(JSONString())

  type = Column(types.String)
  __mapper_args__ = {'polymorphic_on': type}

  def __init__(self, sample, problem, algorithm, tag=''):
    self.tag = tag
    self.sample = SampleData(sample)
    self.problem = ProblemData(problem)
    self.algorithm = AlgorithmData(algorithm)
    # Algorithm settings
    self.options = {}
    self.extra_settings = {}
    #
    self.time = []
    self.result = []

  def config(self, options, **extra_settings):
    # Update dictionaries the way it is recognized by SQL (for mutable fields)
    self.options = dict(self.options, **options)
    self.extra_settings = dict(self.extra_settings, **extra_settings)

  def add(self, result, t=None):
    self.result.append(result)
    self.time.append(t)

  def dict(self):
    """
    Get all the information about benchmark run in dict format
    """
    attributes = {}
    attributes.update(self.algorithm.__dict__)
    attributes.update(self.problem.__dict__)
    attributes.update(self.sample.__dict__)
    # Merge algorithms and problems tables
    for key in set(self.algorithm.__dict__) & set(self.problem.__dict__):
      if key[0] != '_':
        attributes['a_' + key] = self.algorithm.__dict__[key]
        attributes['p_' + key] = self.problem.__dict__[key]
        if key in self.sample.__dict__ and '_id' not in key:
          attributes['s_' + key] = self.sample.__dict__[key]
      del attributes[key]

    # Merge results tables
    for key, field in vars(self.__class__).items():
      value = getattr(self, key)
      if isinstance(value, ProblemData) or isinstance(value, AlgorithmData) or isinstance(value, SampleData):
        continue
      if key[0] == '_' or '_id' in key:
        continue
      elif key in self.__dict__ or isinstance(field, property):
        attributes[key] = value
    return attributes


class Benchmark(object):

  def __init__(self, algorithm, problem, sample, tag=''):
    self.sample = sample
    self.problem = problem
    self.algorithm = algorithm
    self.tag = tag

  def run(self, options={}, **kwargs):
    result = BlankResultData(self.sample, self.problem, self.algorithm, self.tag)
    result.config(options, **kwargs)
    return result
