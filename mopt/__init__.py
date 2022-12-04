import time
from datetime import timedelta

from ._benchmark import (AlgorithmData, Benchmark, BlankResultData,
                         LinConProblemData, ProblemData, SampleData)
from ._database import Database
from .problems import Sample

from._utils import *


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


timer = Timer()
