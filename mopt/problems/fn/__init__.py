import inspect as _inspect
import os as _os

_lists = set([])
_all = set([])


def _exestr(class_name, func_name):
  if class_name in globals():
    return '%s.append(%s)' % (class_name, func_name)
  else:
    _lists.add(class_name)
    return '%s = [%s]' % (class_name, func_name)


__all__ = [_[:-3] for _ in _os.listdir(_os.path.dirname(__file__)) if _.endswith('.py') and not _.startswith('_')]
for _name in __all__:
  exec('from . import %s' % _name)
  exec('_all.add(%s)' % _name)


# Sort by name
for _ in _lists:
  exec('%s.sort(key=lambda problem: problem.Problem().NAME)' % _)
_all = sorted(_all, key=lambda problem: problem.Problem().NAME)
