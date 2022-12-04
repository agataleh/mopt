import numpy as np


class Optimal(object):

  def __init__(self, x=np.nan, f=np.nan, c=np.nan):
    self.x = np.atleast_2d(x).astype(float)
    self.f = np.atleast_2d(f).astype(float)
    self.c = np.atleast_2d(c).astype(float)
    self.count = self.x.shape[0]


class Result(object):

  def __init__(self, calls_count, status='Success', **kwargs):
    self.optimal = Optimal(**kwargs)
    self.calls_count = calls_count
    self.status = status

  def __repr__(self):
    return 'x: %s; f: %s; calls: %d' % (self.optimal.x, self.optimal.f, self.calls_count)
