import numpy as np


class Optimal(object):

  def __init__(self, x=np.nan, f=np.nan, c=np.nan):
    self.x = np.atleast_2d(x).astype(float)
    self.f = np.atleast_2d(f).astype(float)
    self.c = np.atleast_2d(c).astype(float)
    self.count = self.f.shape[0]

  def __repr__(self):
    return 'x: %s; f: %s; c: %s; count: %d' % (np.array_str(self.x), np.array_str(self.f), np.array_str(self.c), self.count)
