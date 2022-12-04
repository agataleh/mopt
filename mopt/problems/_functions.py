import inspect

import numpy as np


def _type(func_type):
  '''
  To classify functions:
    n - unknown,
    u - unimodal,
    m - multimodal,
    x - extra multimodal (crazy trigonometric)
  '''
  def wrapper(func):
    func.func_type = func_type
    return func
  return wrapper


# UNCONSTRAINED ############################################################

@_type('u')
def fon(x):
  """
  Fonseca-Fleming's Problem (https://al-roomi.org/benchmarks/multi-objective/general-list/321-fonseca-fleming-s-function-fon)

  -4 <= xi <= 4

  """
  n = 1.0 / np.sqrt(len(x))
  f1 = 1 - np.exp(-np.sum([(xi - n) ** 2 for xi in x]))
  f2 = 1 - np.exp(-np.sum([(xi + n) ** 2 for xi in x]))
  return f1, f2


@_type('u')
def branin(x1, x2):
  """
  Branin Function (http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Branin01)

  -5 <= x1 <= 10
  0 <= x2 <= 15

  f(-pi, 12.275) = f(pi, 2.275) = f(9.4247, 2.475) = 0.3978
  """
  return (x2 - (1.275 / np.pi**2) * x1**2 + (5.0 / np.pi) * x1 - 6)**2 + (10 - (5.0 / (4.0 * np.pi))) * np.cos(x1) + 10


@_type('m')
def easom(x):
  """
  Easom Function (http://infinity77.net/global_optimization/test_functions_nd_E.html#go_benchmark.Easom)

  -100 <= xi <= 100

  f(0) = 0
  """
  # It seems to be a wrong formula, see original at https://www.sfu.ca/~ssurjano/easom.html
  a = 20.0
  b = 0.2
  c = 2.0 * np.pi
  x = np.array(x)
  sum1 = a / np.exp(b * np.sqrt(np.mean(np.square(x))))
  sum2 = np.exp(np.mean(np.cos(c * x)))
  return a - sum1 + np.e - sum2


@_type('u')
def ellipsoid(x):
  """
  Rotated Ellipsoid Function

  -6 <= xi <= 6

  f(0) = 0
  """
  x = np.array([x], dtype=float, copy=False)
  # return (np.add.accumulate(x, axis=1)**2).sum(axis=1).reshape((-1, 1))
  return (np.add.accumulate(x, axis=1)**2).sum(axis=1)[0]


@_type('u')
def zerosum(x):
  """
  ZeroSum Function (http://infinity77.net/global_optimization/test_functions_nd_Z.html#go_benchmark.ZeroSum)

  -10 <= xi <= 10

  f(sum(x) == 0) = 0
  """
  if np.abs(np.sum(x)) < 3e-16:
    return 0.0
  return 1.0 + (10000.0 * np.abs(np.sum(x))) ** 0.5


@_type('m')
def wolfe(x1, x2, x3):
  """
  Gulf Function (http://infinity77.net/global_optimization/test_functions_nd_W.html#go_benchmark.Wolfe)

  0 <= xi <= 2

  f(0, 0, 0) = 0
  """
  return 4.0 / 3.0 * (x1 ** 2 + x2 ** 2 - x1 * x2) ** 0.75 + x3


@_type('x')
def whitley(x):
  """
  Whitley Function (http://infinity77.net/global_optimization/test_functions_nd_W.html#go_benchmark.Whitley)

  -10.24 <= xi <= 10.24

  f(1, ..., 1) = 0
  """
  xi = np.atleast_1d(x)
  xj = np.atleast_2d(x).T
  temp = 100.0 * ((xi ** 2.0) - xj) + (1.0 - xj) ** 2.0
  inner = (temp ** 2.0 / 4000.0) - np.cos(temp) + 1.0
  return np.sum(np.sum(inner, axis=0))


@_type('x')
def weierstrass(x):
  """
  Weierstrass Function (http://infinity77.net/global_optimization/test_functions_nd_W.html#go_benchmark.Weierstrass)

  -0.5 <= xi <= 0.5

  f(0, ..., 0) = 0
  """
  kmax = 20
  a, b = 0.5, 3.0
  x = np.atleast_1d(x)
  k = np.atleast_2d(np.arange(kmax + 1.0)).T
  t1 = a ** k * np.cos(2 * np.pi * b ** k * (x + 0.5))
  t2 = (len(x)) * np.sum(a ** k.T * np.cos(np.pi * b ** k.T))
  return np.sum(np.sum(t1, axis=0)) - t2


@_type('u')
def katsuura(x):
  """
  Katsuura Function (http://infinity77.net/global_optimization/test_functions_nd_K.html#go_benchmark.Katsuura)

  0 <= xi <= 100

  f(0.0, ..., 0.0) = 1.0
  """
  d = 32
  x = np.atleast_1d(x)
  k = np.atleast_2d(np.arange(1, d + 1)).T
  i = np.arange(0.0, (len(x)) * 1.0)
  inner = np.floor(2 ** k * x) * (2. ** (-k))
  return np.prod(np.sum(inner, axis=0) * (i + 1) + 1)


@_type('x')
def keane(x1, x2):
  """
  Keane Function (http://infinity77.net/global_optimization/test_functions_nd_K.html#go_benchmark.Keane)

  0 <= xi <= 10

  f(0.0, 1.39325) = -0.6737
  """
  delimiter = np.sqrt(x1 ** 2 + x2 ** 2)
  if delimiter == 0:
    return np.nan
  return -((np.sin(x1 - x2)) ** 2 * (np.sin(x1 + x2)) ** 2) / delimiter


@_type('m')
def hosaki(x1, x2):
  """
  Hosaki Function (http://infinity77.net/global_optimization/test_functions_nd_H.html#go_benchmark.Hosaki)

  0 <= xi <= 10

  f(4, 2) = -2.34589
  """
  return (1 - 8 * x1 + 7 * x1 ** 2 - (7.0 / 3.0) * x1 ** 3 + 0.25 * x1 ** 4) * x2 ** 2 * np.exp(-x2)


@_type('m')
def himmelblau(x1, x2):
  """
  HimmelBlau Function (http://infinity77.net/global_optimization/test_functions_nd_H.html#go_benchmark.HimmelBlau)

  -6 <= xi <= 6

  f(0, 0) = 0
  """
  return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2


@_type('m')
def gulf(x1, x2, x3):
  """
  Gulf Function (http://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.Sargan)

  0.1 <= xi <= 100

  f(50, 25, 1.5) = 0
  """
  m = 99.0
  i = np.arange(1., m + 1)
  u = 25 + (-50 * np.log(i / 100.0)) ** (2 / 3.0)
  vec = (np.exp(-((np.abs(u - x2)) ** x3 / x1)) - i / 100.0)
  return sum(vec ** 2)


@_type('u')
def sargan(x):
  """
  Sargan Function (http://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.Sargan)

  -100 <= xi <= 100

  f(0, ..., 0) = 0
  """
  result = 0
  for i in np.arange(len(x)):
    result += len(x) * (x[i] ** 2 + 0.4 * np.sum([x[i] * x[j] for j in np.arange(len(x)) if i != j]))
  return result


@_type('u')
def sodp(x):
  """
  Sum Of Different Powers Function (http://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.Sodp)

  -1 <= xi <= 1

  f(0, ..., 0) = 0
  """
  return np.sum([(np.abs(x[i])) ** (i + 2) for i in np.arange(len(x))])


@_type('u')
def step(x):
  """
  Step Function (http://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.Step)

  -100 <= xi <= 100

  f(0, ..., 0) = 0
  """
  return np.sum([(np.floor(xi)) ** 2 for xi in x])


@_type('m')
def multimodal(x):
  """
  MultiModal Function (http://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.MultiModal)

  -10 <= xi <= 10

  f(0, ..., 0) = 0
  """
  return np.sum(np.abs(x)) * np.prod(np.abs(x))


@_type('m')
def mishra7(x):
  """
  Exponential Function (http://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Mishra07)

  -10 <= xi <= 10

  f(sqrt(n), ..., sqrt(n)) = 0
  """
  return (np.prod(x) - np.prod(np.arange(1, len(x) + 1))) ** 2.0


@_type('u')
def exponential(x):
  """
  Exponential Function (http://infinity77.net/global_optimization/test_functions_nd_E.html#go_benchmark.Exponential)

  -1 <= xi <= 1

  f(0, ..., 0) = -1
  """
  return - np.exp(-0.5 * np.sum([xi ** 2 for xi in x]))


@_type('u')
def exp2(x1, x2):
  """
  Exp2 Function (http://infinity77.net/global_optimization/test_functions_nd_E.html#go_benchmark.Exp2)

  0 <= xi <= 20

  f(1.0, 10.0) = 0
  """
  def sum_e(i):
    t = - 0.1 * i
    return np.exp(t * x1) - 5 * np.exp(t * x2) - np.exp(t) + 5 * np.exp(-i)
  return np.sum([sum_e(i) ** 2 for i in range(10)])


@_type('u')
def elatar(x1, x2):
  """
  El-Attar-Vidyasagar-Dutta Function (http://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.DeckkersAarts)

  -100 <= xi <= 100

  f(3.409186, -2.171433) = 1.712780354
  """
  return (x1 ** 2 + x2 - 10) ** 2 + (x1 + x2 ** 2 - 7) ** 2 + (x1 ** 2 + x2 ** 3 - 1) ** 2


@_type('u')
def dixonprice(x):
  """
  DixonPrice Function (http://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.DixonPrice)

  -10 <= xi <= 10

  f(x1, ..., xn) = 0, xi = 2.0 ** ((2.0 - 2.0 ** i) / 2.0 ** i)
  """
  return (x[0] - 1.0) ** 2 + np.sum([(i + 1.0) * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, len(x))])


@_type('u')
def deckkersaarts(x1, x2):
  """
  Deckkers-Aarts Function (http://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.DeckkersAarts)

  -20 <= xi <= 20

  f(0, +-15) = -24777
  """
  return 1e5 * x1 ** 2 + x2 ** 2 - (x1 ** 2 + x2 ** 2) ** 2 + 1e-5 * (x1 ** 2 + x2 ** 2) ** 4


@_type('m')
def deceptive(x, a=0.3):
  """
  Deceptive Function (http://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.Deceptive)

  0 <= xi <= 1

  f(a1, ..., aN) = -1
  """
  if (np.array(x) < 0).any() | (1 < np.array(x)).any():
    return np.nan
  a = [a] * len(x)

  def g(i):
    if 0 <= x[i] <= (4.0 / 5.0) * a[i]:
      return - x[i] / a[i] + (4.0 / 5.0)
    elif (4.0 / 5.0) * a[i] <= x[i] <= a[i]:
      return 5.0 * x[i] / a[i] - 4.0
    elif a[i] <= x[i] <= (1.0 + 4.0 * a[i]) / 5.0:
      return 5.0 * (x[i] - a[i]) / (a[i] - 1)
    elif (1.0 + 4.0 * a[i]) / 5.0 <= x[i] <= 1:
      return (x[i] - 1) / (1 - a[i])
  return - np.mean([g(i) for i in range(len(x))]) ** 2


@_type('u')
def decanomial(x1, x2):
  """
  Decanomial Function (http://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.Decanomial)

  -10 <= xi <= 10

  f(2, -3) = 0
  """
  s1 = abs(x2 ** 4 + 12.0 * x2 ** 3 + 54.0 * x2 ** 2 + 108.0 * x2 + 81.0)
  s2 = abs(x1 ** 10 - 20.0 * x1 ** 9 + 180.0 * x1 ** 8 - 960.0 * x1 ** 7 + 3360.0 * x1 ** 6 - 8064.0 * x1 ** 5 +
           13340.0 * x1 ** 4 - 15360.0 * x1 ** 3 + 11520.0 * x1 ** 2 - 5120.0 * x1 + 2624.0)
  return 0.001 * (s1 + s2) ** 2


@_type('u')
def cube(x1, x2):
  """
  Cube Function (http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.Cube)

  -10 <= xi <= 10

  f(1, 1) = 0
  """
  return 100 * (x2 - x1 ** 3) ** 2 + (1 - x1) ** 2


@_type('u')
def cigar(x):
  """
  Cigar Function (http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.Cigar)

  -100 <= xi <= 100

  f(0, ..., 0) = 0
  """
  return x[0] ** 2 + 1e6 * np.sum([x[i] ** 2 for i in range(1, len(x))])


@_type('u')
def bukin06(x1, x2):
  """
  Bukin06 Function (http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bukin06)

  -15 <= x1 <= -5
  -3 <= x2 <= 3

  f(-10, 1) = 0
  """
  return 100.0 * np.sqrt(abs(x2 - 0.01 * x1 ** 2)) + 0.01 * abs(x1 + 10)


@_type('u')
def bukin04(x1, x2):
  """
  Bukin04 Function (http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bukin04)

  -15 <= x1 <= -5
  -3 <= x2 <= 3

  f(-10, 0) = 0
  """
  return 100.0 * x2 ** 2 + 0.01 * abs(x1 + 10.0)


@_type('u')
def bukin02(x1, x2):
  """
  Bukin02 Function (http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bukin02)

  -15 <= x1 <= -5
  -3 <= x2 <= 3

  f(-10, 0) = 0
  """
  return 100.0 * (x2 ** 2 - 0.01 * x1 ** 2 + 1.0) + 0.01 * (x1 + 10.0) ** 2


@_type('u')
def brown(x):
  """
  Brown Function (http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Brown)

  -1 <= xi <= 4

  f(0, ..., 0) = 0
  """
  return np.sum([(x[i] ** 2) ** (x[i + 1] ** 2 + 1) + (x[i + 1] ** 2) ** (x[i] ** 2 + 1) for i in range(len(x) - 1)])


@_type('u')
def brent(x1, x2):
  """
  Brent Function (http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Brent)

  -10 <= xi <= 10

  f(-10, -10) = 0
  """
  return (x1 + 10.0) ** 2 + (x2 + 10.0) ** 2 + np.exp(- x1 ** 2 - x2 ** 2)


@_type('u')
def beale(x1, x2):
  """
  Beale Function (http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Beale)

  -10 <= xi <= 10

  f(3.0, 0.5) = 0
  """
  return (x1 * x2 - x1 + 1.5) ** 2 + (x1 * x2 ** 2 - x1 + 2.25) ** 2 + (x1 * x2 ** 3 - x1 + 2.652) ** 2


@_type('u')
def amgm(x):
  """
  AMGM Function (http://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.AMGM)
  Arithmetic Mean - Geometric Mean Equality

  0 <= xi <= 10

  f(0, ..., 0) = 0
  """
  power = 1.0 / len(x)
  if np.any(np.array(x) < 0):
    return np.nan
  return (power * np.sum(x) - np.prod(x) ** power) ** 2


@_type('x')
def ackley(x):
  """
  Ackley Function (http://benchmarkfcns.xyz/benchmarkfcns/ackleyfcn.html)

  -2 <= xi <= 2

  f(0, ..., 0) = 0
  """
  firstSum = 0.0
  secondSum = 0.0
  for xi in x:
    firstSum += xi**2
    secondSum += np.cos(2.0 * np.pi * xi)
  return - 20.0 * np.exp(-0.2 * np.sqrt(firstSum / len(x))) - np.exp(secondSum / len(x)) + 20 + np.e


@_type('m')
def goldsteinprice(x1, x2):
  """
  Goldstein-Price's Function (https://www.sfu.ca/~ssurjano/goldpr.html)

  -2 <= x1 <= 2
  -2 <= x2 <= 2

  f(0, -1) = 3
  """
  f1 = (x1 + x2 + 1)**2
  f2 = 19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
  f3 = (2 * x1 - 3 * x2)**2
  f4 = 18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
  return (1 + f1 * f2) * (30 + f3 * f4)


@_type('u')
def rosenbrock(x):
  """
  Rosenbrock's Valley Function (https://en.wikipedia.org/wiki/Rosenbrock_function)

  -2.048 <= xi <= 2.048

  f(1, ..., 1) = 0
  """
  result = 0
  for i in range(len(x) - 1):
    result += 100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2
  return result


@_type('m')
def sixhumpcamel(x1, x2):
  """
  Six Hump Camel Back Function (https://www.sfu.ca/~ssurjano/camel6.html)

  -3 <= x1 <= 3
  -2 <= x2 <= 2

  f(0.0898, -0.7126) = f(-0.0898, 0.7126) = -1.03164
  """
  f1 = (4 - 2.1 * x1**2 + x1**4 / 3.0) * x1**2
  f2 = x1 * x2
  f3 = (-4 + 4 * x2**2) * x2**2
  return f1 + f2 + f3


@_type('x')
def griewank(x):
  """
  Griewank Function (https://www.sfu.ca/~ssurjano/griewank.html)

  -6 <= xi <= 6

  f(0, ..., 0) = 0
  """
  return 1.0 + np.sum([x[i]**2 for i in range(len(x))]) / 4000.0 - np.prod([np.cos(x[i] / np.sqrt(i + 1)) for i in range(len(x))])


@_type('x')
def levy3(x):
  """
  Levy 3 Function (http://infinity77.net/global_optimization/test_functions_nd_L.html#go_benchmark.Levy03)

  -10 <= xi <= 10

  f(1, ..., 1) = 0
  """
  x = [1 + (xi - 1) / 4 for xi in x]
  return np.sin(np.pi * x[0])**2 + (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2) + \
      np.sum([(x[i] - 1)**2 * (1 + 10 * np.sin(np.pi * x[i] + 1)**2) for i in range(len(x) - 1)])


@_type('x')
def levy5(x1, x2):
  """
  Levy 5 Function (http://infinity77.net/global_optimization/test_functions_nd_L.html#go_benchmark.Levy05)

  -2 <= xi <= 2

  f(-1.3086, -1.4248) = -176.1375
  """
  return np.sum([i * np.cos(x1 * (i - 1) + i) for i in range(1, 6)]) * \
      np.sum([j * np.cos(x2 * (j + 1) + j) for j in range(1, 6)]) + \
      (x1 + 1.42513)**2 + (x2 + 0.80032)**2


@_type('x')
def rastrigin(x):
  """
  Rastrigin Function (http://www.sfu.ca/~ssurjano/rastr.html)

  -5.12 <= xi <= 5.12

  f(0, ..., 0) = 0
  """
  return 10 * len(x) + np.sum([x[i] * x[i] - 10.0 * np.cos(2.0 * x[i] * np.pi) for i in range(len(x))])


@_type('u')
def perm(x):
  """
  Very sensitive to x
  Perm Function (http://www.sfu.ca/~ssurjano/permdb.html,
                 http://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.PermFunction01)

  -d <= xi <= d

  f(1, 2, .., dim) = 0
  """
  b = 0.5
  result = 0
  for i in range(1, len(x) + 1):
    inner_sum = np.sum([(j**i + b) * ((x[j - 1] / j)**i - 1.0) for j in range(1, len(x) + 1)])
    result += inner_sum**2
  return result


@_type('u')
def perm2(x):
  """
  Perm Function (http://www.sfu.ca/~ssurjano/perm0db.html)

  -d <= xi <= d

  f(1, 1/2, .., 1/dim) = 0
  """
  b = 10
  result = 0
  for i in range(1, len(x) + 1):
    inner_sum = np.sum([(j + b) * (x[j - 1]**i - 1.0 / j**i) for j in range(1, len(x) + 1)])
    result += inner_sum**2
  return result


@_type('u')
def powellsum(x):
  """
  Sum of Different Powers Function (http://benchmarkfcns.xyz/benchmarkfcns/powellsumfcn.html)

  -5 <= xi <= 5

  f(0, ..., 0) = 0
  """
  return np.sum([np.abs(x[i])**(i + 2) for i in range(len(x))])


@_type('u')
def powell(xx, x1=None, x2=None, x3=None): # dummy variables for parsing
  """
  Powell function (http://www.sfu.ca/~ssurjano/powell.html)

  -4 <= xi <= 5

  f(0, ..., 0) = 0
  """
  if len(xx) % 4 != 0:
    raise ValueError("The problem dimension should be multiple of 4")

  sum = 0
  for i in range(len(xx) // 4):
    i *= 4
    sum += (xx[i] + 10 * xx[i + 1])**2
    sum += 5 * (xx[i + 2] - xx[i + 3])**2
    sum += (xx[i + 1] - 2 * xx[i + 2])**4
    sum += 10 * (xx[i] - xx[i + 3])**4
  return sum


@_type('u')
def trid(x):
  """
  Trid Function (https://www.sfu.ca/~ssurjano/trid.html)

  -dim**2 <= xi <= dim**2

  dim = 6: f(6, 10, 12, 12, 10, 6) = -50
  """
  return np.sum([(x[i] - 1)**2 for i in range(len(x))]) - np.sum([x[i] * x[i - 1] for i in range(1, len(x))])


@_type('u')
def zakharov(x):
  """
  Zakharov Function (http://www.sfu.ca/~ssurjano/zakharov.html)

  -5 <= xi <= 10

  f(0, ..., 0) = 0
  """
  sum1 = np.sum([x[i]**2 for i in range(len(x))])
  sum2 = np.sum([0.5 * (i + 1) * x[i] for i in range(len(x))])
  return sum1 + sum2**2 + sum2**4


@_type('x')
def michalewicz(x):
  """
  Michalewicz Function (http://www.sfu.ca/~ssurjano/michal.html)

  0 <= xi <= pi

  dim = 2: f(2.20, 1.57) = -1.8013
  dim = 5: f = -4.687658
  dim = 10: f = -9.66015
  """

  return -np.sum([np.sin(x[i]) * (np.sin((i + 1) * x[i]**2 / np.pi)**20) for i in range(len(x))])


@_type('m')
def styblinskitang(x):
  """
  Styblinski-Tang Function (http://www.sfu.ca/~ssurjano/stybtang.html)

  -5 <= xi <= 5

  f(-2.903534, ..., -2.903534) = -39.166 * d
  """
  return 0.5 * np.sum([x[i]**4 - 16 * x[i]**2 + 5 * x[i] for i in range(len(x))])


@_type('x')
def shubert4(x):
  """
  Shubert 4 Function (http://benchmarkfcns.xyz/benchmarkfcns/shubert4fcn.html)

  -2 <= xi <= 2

  f = -29.017
  """
  return np.sum([np.sum([j * np.cos((j + 1) * x[i] + j) for j in range(1, 6)]) for i in range(len(x))])


@_type('x')
def shubert(x):
  """
  Shubert Function (http://benchmarkfcns.xyz/benchmarkfcns/shubertfcn.html)

  -2 <= xi <= 2

  f = -186.7309
  """
  return np.prod([np.sum([j * np.cos((j + 1) * x[i] + j) for j in range(1, 6)]) for i in range(len(x))])


@_type('x')
def salomon(x):
  """
  Salomon Function (http://benchmarkfcns.xyz/benchmarkfcns/salomonfcn.html)

  -2 <= xi <= 2

  f(0, ..., 0) = 0
  """
  s = np.sqrt(np.sum([xi**2 for xi in x]))
  return 1 - np.cos(2 * np.pi * s) + 0.1 * s


@_type('x')
def bohachevsky(x):
  """
  Bohachevsky Function (http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bohachevsky)

  -2 <= xi <= 2

  f(0, ..., 0) = 0
  """
  s = 0
  for i in range(len(x) - 1):
    s += x[i]**2 + 2 * x[i + 1]**2 - 0.3 * np.cos(3 * np.pi * x[i]) - 0.4 * np.cos(4 * np.pi * x[i + 1]) + 0.7
  return s


@_type('u')
def csendes(x):
  """
  Csendes Function (http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.Csendes)

  -1 <= xi <= 1

  f(0, ..., 0) = 0
  """
  s = 0
  for i in range(len(x)):
    if x[i] == 0:
      return float('nan')
    s += x[i]**6 * (2 + np.sin(1.0 / x[i]))
  return s


@_type('m')
def quintic(x):
  """
  Quintic Function (http://infinity77.net/global_optimization/test_functions_nd_Q.html#go_benchmark.Quintic)

  -10 <= xi <= 10

  f(-1, ..., -1) = 0
  """
  s = 0
  for xi in x:
    s += np.abs(xi**5 - 3 * xi**4 + 4 * xi**3 + 2 * xi**2 - 10 * xi - 4)
  return s


@_type('x')
def xinsheyang4(x):
  """
  Xin-She Yang 4 Function (http://benchmarkfcns.xyz/benchmarkfcns/xinsheyangn4fcn.html)

  -6 <= xi <= 6

  f(0, ..., 0) = -1
  """
  sum1 = 0
  sum2 = 0
  sum3 = 0
  for xi in x:
    sum1 += np.sin(xi)**2
    sum2 += xi**2
    sum3 += np.sin(np.sqrt(np.abs(xi)))**2
  return (sum1 - np.exp(-sum2)) * np.exp(-sum3)


@_type('x')
def xinsheyang2(x):
  """
  Xin-She Yang 2 Function (http://benchmarkfcns.xyz/benchmarkfcns/xinsheyangn2fcn.html)

  -2pi <= xi <= 2pi

  f(0, ..., 0) = 0
  """
  sum1 = 0
  sum2 = 0
  for xi in x:
    sum1 += np.abs(xi)
    sum2 += np.sin(xi**2)
  return sum1 * np.exp(-sum2)


@_type('u')
def qing(x):
  """
  Qing Function (http://benchmarkfcns.xyz/benchmarkfcns/qingfcn.html)

  -500 <= xi <= 500

  f(+-sqrt(i), ..., +-sqrt(i)) = 0
  """
  return np.sum([(x[i]**2 - i - 1)**2 for i in range(len(x))])


# CONSTRAINED #############################################################
@_type('n')
def steppedbeam(x):
  """
  https://arxiv.org/pdf/1211.6663.pdf
  """
  return 0


@_type('n')
def g2(x):
  """
  G2 Problem (http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2502.htm)

  0 <= xi <= 10

  c1(x): -np.prod(x) + 0.75 <= 0
  c2(x): np.sum(x) - 7.5 * len(x) <= 0
  """
  a = 0
  b = 1
  c = 0
  for i in range(len(x)):
    a += np.cos(x[i])**4
    b *= np.cos(x[i])**2
    c += (i + 1) * x[i]**2
  return -np.abs((a - 2 * b) / np.sqrt(c)) if c != 0 else np.nan


@_type('n')
def g3(x):
  """
  G3 Problem (http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2613.htm)

  0 <= xi <= 1

  c(x): np.sum([i**2 for i in x]) == 1
  """
  return - np.prod(x) * np.sqrt(len(x))**len(x)


@_type('n')
def g6(x):
  """
  G6 Problem (http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2533.htm)

  0 <= xi <= 100

  c1(x): (x[0] - 5)**2 + (x[1] - 5)**2 <= 100
  c2(x): (x[0] - 6)**2 + (x[1] - 5)**2 <= 82.81

  f(14.095, 0.84296) = -6961.81381
  """
  return (x[0] - 10)**3 + (x[1] - 20)**3


@_type('n')
def g7(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
  """
  G7 Problem (http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2424.htm)

  -10 <= xi <= 10

  c1(x): 4*x[0] + 5*x[1] - 3*x[6] + 9*x[7] - 105 <=0
  c2(x): 10*x[0] - 8*x[1] - 17*x[6] + 2*x[7] <=0
  c3(x): - 8*x[0] + 2*x[1] + 5*x[8] - 2*x[9] - 12 <=0
  c4(x): 3*(x[0] - 2)**2 + 4*(x[1] - 3)**2 + 2*x[2]**2 - 7*x[3] - 120 <=0
  c5(x): 5*x[0]**2 + 8*x[1] + (x[2] - 6)**2 - 2*x[3] - 40 <=0
  c6(x): 0.5*(x[0] - 8)**2 + 2*(x[1] - 4)**2 + 3*x[4]**2 - x[5] - 30 <=0
  c7(x): x[0]**2 + 2*(x[1] - 2)**2 - 2*x[0]*x[1] + 14*x[4] - 6*x[5] <=0
  c8(x): - 3*x[0] + 6*x[1] + 12*(x[8] - 8)**2 - 7*x[9] <=0

  f(2.171996, 2.363683, 8.773926, 5.095984, 0.9906548, 1.430574, 1.321644, 9.828726, 8.280092, 8.375927) = 24.3062091
  """
  return x1**2 + x2**2 + x1 * x2 - 14.0 * x1 - 16.0 * x2 + \
      (x3 - 10.0)**2 + 4.0 * (x4 - 5.0)**2 + (x5 - 3.0)**2 + 2.0 * (x6 - 1.0)**2 + \
      5.0 * x7**2 + 7.0 * (x8 - 11.0)**2 + 2.0 * (x9 - 10.0)**2 + (x10 - 7.0)**2 + 45.0


@_type('n')
def rosenbrock_constraint(x):
  """
  Rosenbrock Function constrained to a disk
  (https://www.mathworks.com/help/optim/ug/example-nonlinear-constrained-minimization.html)

  c(x) <= 1
  """
  return np.sum([x[i]**2 for i in range(len(x))])
