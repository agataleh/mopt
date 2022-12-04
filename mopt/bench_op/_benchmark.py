import copy
import json
import os
import pickle
import platform
import sys
import time
from datetime import timedelta

import numpy as np
from sqlalchemy import Column, ForeignKey, types
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator

from .._benchmark import (AlgorithmData, Base, JSONString, ProblemData,
                          SampleData)


class Timer(object):
  """Class to simplify time measuring routine"""

  def __init__(self):
    self.T = {}
    self.t = 0
    self._is_run = {}

  def start(self, tag=None):
    if self._is_run.get(tag, False):
      self.T[tag] = time.time() - self.T[tag]
      self.t = self.T[tag]
      print(f'WARN: timer {tag} was reset at {self.str(tag)}')
    self.T[tag] = time.time()
    self.t = self.T[tag]
    self._is_run[tag] = True

  def stop(self, tag=None):
    if not self._is_run.get(tag, False) or tag not in self.T:
      print(f'WARN: timer {tag} was not started')
      return
    self.T[tag] = time.time() - self.T[tag]
    self.t = self.T[tag]
    self._is_run[tag] = False
    return self.str(tag=tag)

  def str(self, tag=None, seconds=None):
    def time_str(seconds): return '<%s> (%g sec)' % (timedelta(seconds=seconds), seconds)
    if seconds is not None:
      return time_str(seconds)
    if tag not in self.T:
      print(f'WARN: timer {tag} was not set')
      return {tag: time_str(seconds) for tag, seconds in self.T.items()}
    else:
      return time_str(self.T[tag])


class ResultData(Base):

  __tablename__ = 'op_results'

  id = Column(types.Integer, primary_key=True)
  tag = Column(types.String)
  platform = Column(types.String)
  # Initial sample
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
  # Run settings
  seed = Column(types.Integer)
  n_runs = Column(types.Integer)
  initial_guess = Column(JSONString())
  # Results of run (arrays)
  x = Column(JSONString())
  f = Column(JSONString())
  calls_count = Column(JSONString())
  time = Column(JSONString())
  histories = Column(JSONString())
  statuses = Column(JSONString())

  def __init__(self, algorithm, problem, sample, tag):
    self.tag = tag
    self.platform = platform.node()
    self.sample = SampleData(sample)
    self.problem = ProblemData(problem)
    self.algorithm = AlgorithmData(algorithm)
    # Algorithm settings
    self.options = algorithm.options.get()
    self.extra_settings = {}
    # Run settings
    self.seed = None
    self.n_runs = 1
    self.initial_guess = []
    # Results of run
    self.x = []
    self.f = []
    self.calls_count = []
    self.time = []
    self.histories = []
    self.statuses = []

  def config(self, options, n_runs, seed, **extra_settings):
    # Update dictionaries the way it is recognized by SQL (for mutable fields)
    self.options = dict(self.options, **options)
    self.n_runs = n_runs
    self.seed = seed
    self.extra_settings = dict(self.extra_settings, **extra_settings)

  def add(self, initial_guess, result, t=None, history=None):
    self.initial_guess.append(initial_guess)
    # Update results of run
    self.x.append(result.optimal.x[0].tolist() if result.optimal.x.size else [None] * self.problem.size_x)
    self.f.append(result.optimal.f[0][0].tolist() if result.optimal.f.size else None)
    self.calls_count.append(result.calls_count)
    self.time.append(t)
    self.histories.append(history)
    self.statuses.append(result.status)

  @property
  def x_dist(self):
    x = np.array(self.x, dtype=float)
    return np.hypot.reduce(x[np.all(np.isfinite(x), axis=1)] - np.array(self.problem.x_best), axis=1)

  @property
  def x_dist_mean(self):
    return np.mean(self.x_dist)

  @property
  def f_mean(self):
    return np.nanmean(np.array(self.f, dtype=float))

  @property
  def f_std(self):
    return np.nanstd(np.array(self.f, dtype=float))

  @property
  def calls_count_mean(self):
    return np.mean(self.calls_count)

  @property
  def time_mean(self):
    return np.mean(self.time)

  @property
  def histories_best(self):
    histories = np.array(self.histories, dtype=object)
    if not histories.size or histories.ndim < 2:
      return None
    best_f = self._append_initial_history(best_f=histories[:, :, -1].T, full_history=False)
    return np.array([best_f[:i].min(0) for i in range(1, len(best_f) + 1)])

  def _append_initial_history(self, best_f, full_history):
    s_size = self.sample.size
    if s_size is not None:
      initial_best_f = np.zeros((s_size if full_history else 1, 1))
      initial_best_f[:] = self.sample.f_range[0]
      best_f = np.vstack([initial_best_f, best_f])
    return best_f

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
    for key, field in vars(ResultData).items():
      value = getattr(self, key)
      if isinstance(value, ProblemData) or isinstance(value, AlgorithmData) or isinstance(value, SampleData):
        continue
      if key[0] == '_' or '_id' in key:
        continue
      elif key in self.__dict__ or isinstance(field, property):
        attributes[key] = value
    return attributes

  def __str__(self):
    settings = 'RES:%15s|%25s\t||\t' % (self.algorithm.a_name, self.problem.p_name + '_' + str(self.problem.size_x))
    results = 'n = %-6d| f = %-6f | f* = %-6f | t = %s' % (self.calls_count_mean, self.f_mean,
                                                           self.problem.f_best, self.time_mean)
    return settings + results

  def __repr__(self):
    return self.__str__()

  # OUTDATED METHOD
  # def copy(self):
  #   result = ResultData(algorithm=self.algorithm.copy(), problem=self.problem.copy(), tag=self.tag)
  #   # Algorithm settings
  #   result.options = self.options
  #   result.extra_settings = self.extra_settings
  #   # Run settings
  #   result.n_runs = self.n_runs
  #   result.initial_guess = self.initial_guess
  #   result.seed = self.seed
  #   # Results of run
  #   result.x = self.x
  #   result.f = self.f
  #   result.calls_count = self.calls_count
  #   result.time = self.time
  #   result.histories = self.histories
  #   return result


class Benchmark(object):

  def __init__(self, algorithm, problem, sample=None, tag='', logging=False):
    self.tag = tag

    self.algorithm = algorithm
    self.problem = problem
    self.sample = sample

    self.logging = logging
    self.timer = Timer()

  def get_sample(self, sample, seed=None):
    if sample is None:
      return None
    elif not hasattr(sample, 'x') or not hasattr(sample, 'f'):
      from mopt.problems import Sample
      doe = sample.split('_')[0]
      size = int(sample.split('_')[1])
      sample = Sample(self.problem, tag=f'seed={seed}', doe=doe, size=size, seed=seed, log=self.logging)
    return sample.x, sample.f

  def run(self, options={}, n_runs=1, initial_guess=None, initial_sample_code=None, seed=None, save_history=True, **kwargs):
    data = ResultData(algorithm=self.algorithm, problem=self.problem, sample=self.sample, tag=self.tag)
    # Configure benchmark result, kwargs contain algorithm-specific settings (e.g. expensive for p7core)
    data.config(options=options, n_runs=n_runs, seed=seed, **kwargs)
    # Prepare problems list using specified seed
    np.random.seed(seed)
    problems = [self.problem.modify(initial_guess=initial_guess) for i in range(n_runs)]
    samples = [self.get_sample(initial_sample_code or self.sample, seed=i) for i in range(n_runs)]
    # Solve all the problems
    for i in range(n_runs):
      self._write_log(i, n_runs)
      self.timer.start(tag=self.tag)
      result = self.algorithm.solve(problem=problems[i], initial_sample=samples[i], options=options, seed=i, **kwargs)
      self.timer.stop(tag=self.tag)
      self._write_log(i, n_runs, result)
      data.add(result=result, initial_guess=problems[i].initial_guess, t=self.timer.T[self.tag], history=problems[i].history if save_history else problems[i].clear_history())
    return data

  def _write_log(self, i, n_runs, result=None):
    if not self.logging and i != n_runs - 1:
      return
    elif result is None:
      if i == 0 and n_runs > 1:
        print
      print('%3d|%-10s|%15s|%25s  ||  ' % (i, self.tag, self.algorithm.NAME, self.problem.NAME + '_' + str(self.problem.size_x)), end=' ')
      sys.stdout.flush()
    else:
      print('c = %-6d| f = %-6g | f* = %-6g' % (result.calls_count, result.optimal.f[0][0], self.problem.optimal.f[0][0]), end=' ')
      print('| t = %s' % self.timer.str(tag=self.tag) if result is not None else '')
      sys.stdout.flush()
