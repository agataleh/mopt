# https://github.com/fmfn/BayesianOptimization
# https://arxiv.org/pdf/1012.2599v1.pdf

# acq     :ei       Acquisition function: upper confidence bound (ucb), expected improvement (ei), probability  of improvement (poi).
# kappa   :2.576    For ucb
# xi      :0.0      For ei and poi
#
# Acquisition Function "Upper Confidence Bound"
# Prefer exploitation (kappa=1.0)
# Prefer exploration (kappa=10)
#
# Acquisition Function "Expected Improvement"
# Prefer exploitation (xi=0.0)
# Prefer exploration (xi=0.1)
#
# Acquisition Function "Probability of Improvement"
# Prefer exploitation (xi=0.0)
# Prefer exploration (xi=0.1)

# def utility(self, x, gp, y_max):
#     if self.kind == 'ucb':
#         return self._ucb(x, gp, self.kappa)
#     if self.kind == 'ei':
#         return self._ei(x, gp, y_max, self.xi)
#     if self.kind == 'poi':
#         return self._poi(x, gp, y_max, self.xi)
#
# def _ucb(x, gp, kappa):
#     mean, std = gp.predict(x, return_std=True)
#     return mean + kappa * std
#
# def _ei(x, gp, y_max, xi):
#     mean, std = gp.predict(x, return_std=True)
#     z = (mean - y_max - xi)/std
#     return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)
#
# def _poi(x, gp, y_max, xi):
#     mean, std = gp.predict(x, return_std=True)
#     z = (mean - y_max - xi)/std
#     return norm.cdf(z)
