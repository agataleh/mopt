import pandas as pd

from .._database import Database as DB
from ._benchmark import ResultData


class Database(DB):

  def select(self, **kwargs):
    return super().select(query_data=ResultData, **kwargs)

  def select_df(self, *args, **kwargs):
    return super().select_df(query_data=ResultData, *args, **kwargs)
