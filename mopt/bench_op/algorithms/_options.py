class Option(object):

  def __init__(self, name, type, default, description, **kwargs):
    self.name = name
    self.type = type
    self.default = default
    self.description = description
    self.info = kwargs


class Options(object):
  """General options interface."""

  def __init__(self, *args):
    self._options_list = args
    self._data = dict((opt.name, opt.default) for opt in self._options_list)
    self._names = tuple(sorted(self._data.keys()))
    self._default_values = tuple([self._data[i] for i in self._names])
    self._user_defined_data = {}

  def __set_option(self, option, value):
    if option not in self._names:
      raise Exception("Unknown option name '%s'" % option)
    if value is None:
      self._data[option] = self._default_values[self._names.index(option)]
      if self._user_defined_data.get(option) is not None:
        del self._user_defined_data[option]
    # elif type(value) != type(self._data[option]):
    #   raise Exception("Wrong option %s value type" % option)
    else:
      self._data[option] = value
      self._user_defined_data[option] = value

  def set(self, option, value=None):
    """Set option values."""
    if isinstance(option, dict):
      for key in option:
        self.__set_option(key, option[key])
    else:
      self.__set_option(option, value)

  def get(self, name=None):
    """Get current value of an option or all options."""
    if name is None:
      return dict((opt, self._data.get(opt)) for opt in self._names)
    elif name not in self._names:
      raise Exception("Unknown option name '%s'" % name)
    else:
      return self._data.get(name)

  def info(self, name=None):
    """Get options list."""
    if name is not None:
      return self.__info.get(name)
    else:
      info = {}
      for opt in self._options_list:
        info.setdefault(opt.name, {})
        info[opt.name]['Type'] = opt.type
        info[opt.name]['Default'] = opt.default
        info[opt.name]['Description'] = opt.description
        info[opt.name].update(opt.info)

  @property
  def list(self):
    return self._names

  def __str__(self):
    tab = max(len('%s' % opt) for opt in self._names)
    return '\n'.join('%%-%ds: %%s' % tab % (opt, self._data.get(opt)) for opt in self._names)

  @property
  def values(self):
    """Non-default option values."""
    return dict((opt, self._user_defined_data.get(opt)) for opt in self._names if self._user_defined_data.get(opt) is not None)

  def reset(self):
    """Reset all options to their default values."""
    for opt in self._names:
      self.__set_option(opt, None)
