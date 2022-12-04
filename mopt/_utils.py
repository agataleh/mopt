import numpy as np
import pandas as pd

from . import Sample, ela, problems
from ._benchmark import SampleData


def unfold_random_runs(df, columns=None, unfold_columns=None, index_columns=None):
  if columns is None:
    columns = ['p_name', 'size_x', 'f_best', 'a_name', 'version', 'tag']
  if unfold_columns is None:
    unfold_columns = {'f': 'f_mean', 'calls_count': 'calls_count_mean', 'time': 'time_mean'}
  if index_columns is None:
    index_columns = 'p_name'
  result = pd.DataFrame(columns=columns + list(unfold_columns.values()))
  for index, row in df.iterrows():
    size = len(row[list(unfold_columns)[0]])
    problem_df = pd.DataFrame(index=range(size), columns=result.columns)
    for column in columns:
      problem_df[column] = [row[column]] * size
    for source_column, target_column in unfold_columns.items():
      problem_df[target_column] = row[source_column]
    problem_df[index_columns] = [row[index_columns] + f'seed_{i}' for i in range(size)]
    result = result.append(problem_df, ignore_index=True)
  return result


def features_icp(x, f, triples, n=16):
  vm_ic = ela.core_vm.VMIC(x, f, triples)
  eps, ic, icp = vm_ic.run_all_1d(np.linspace(vm_ic._eps_min, vm_ic.eps_max, n // 2))
  return ic, icp


def features_seg(x, f, triples, n=None):
  vm = ela.VMSegment(x, f, triples)
  if n is None:  # 7 output features
    gamma, seg_p = vm.run_all(sectors=[0, 30, 45, 60, 90, 120, 135, 150, 180])
  else:
    gamma, seg_p = vm.run_all(n_sectors=n + 2)
  return seg_p[1:-1].tolist()


def features_gic(x, f, triples, n=16):
  vm_ic = ela.core_vm.VMIC(x, f, triples)
  eps_values = np.linspace(vm_ic._eps_min, vm_ic.eps_max, int(np.sqrt(n)))
  return vm_ic.run_all_2d(eps_values, eps_values)[1]


def features_ic(x, f, triples, n=8):
  vm_ic = ela.core_vm.VMIC(x, f, triples)
  eps_values = np.linspace(vm_ic._eps_min, vm_ic.eps_max, n // 2)
  ic_max_eps1 = [vm_ic.run_all_2d(eps_i, np.geomspace(vm_ic._eps_min, vm_ic.eps_max, 100))[1].max() for eps_i in eps_values]
  ic_max_eps2 = [vm_ic.run_all_2d(np.geomspace(vm_ic._eps_min, vm_ic.eps_max, 100), eps_i)[1].max() for eps_i in eps_values]
  return np.hstack((ic_max_eps1, ic_max_eps2)).tolist()


def features_segic(x, f, triples):
  # return features_seg(x, f, triples, n=11), features_ic(x, f, triples, n=16) # features_old analogue
  return features_seg(x, f, triples, n=None), features_ic(x, f, triples, n=8)


def features_old(x, f, triples):
  vm = ela.VMSegment(x, f, triples)
  features1 = vm.run_all(13)[1][1:-1]

  vm_ic = ela.core_vm.VMIC(x, f, triples)
  eps_values = np.linspace(vm_ic._eps_min, vm_ic.eps_max, 8)
  ic1 = [vm_ic.run_all_2d(e, np.geomspace(vm_ic._eps_min, vm_ic.eps_max, 100))[1].max() for e in eps_values]
  ic2 = [vm_ic.run_all_2d(np.geomspace(vm_ic._eps_min, vm_ic.eps_max, 100), e)[1].max() for e in eps_values]
  features2 = np.hstack((ic1, ic2))

  return features1.tolist(), features2.tolist()


def recalc_features(p_samples, calculator, samples_tag='seed=%d', features_tag='', triples_tag='vm', load_features=True, log=False):
  features = []
  for i in np.arange(sum([_.size for _ in p_samples.values()])):
    for (p_name, size_x), samples in p_samples.items():
      if i in samples.tolist():
        seed = samples.tolist().index(i)
        break
    pp = eval(f'problems.f1.{p_name}.Problem({size_x})')
    sample_data = SampleData(Sample(pp, tag=samples_tag % seed, doe='grd', size=36, log=log, seed=seed))
    sample_features = sample_data.calc_features(calculator, features_tag=features_tag, triples_tag=triples_tag,
                                                load_triples=True, load_features=load_features, log=log)
    features.append(np.hstack(sample_features))
  return np.vstack(features)


def rrms(model, X, y):
  return np.sqrt(np.mean((model.predict(X) - y) ** 2)) / np.std(y)


def fold(sample, c_columns=[0], b_columns=[1], i_columns=[2], e_columns=3):
  c_nodes = np.unique(sample[:, c_columns], axis=0).astype(int)
  b_nodes = np.unique(sample[:, b_columns], axis=0).astype(int)
  i_nodes = np.unique(sample[:, i_columns], axis=0).astype(int)

  # Use only for non-index columns
  # c_idx = np.where(np.all(sample[:, c_columns] == c_nodes[:, None], axis=-1).T)[1]
  # b_idx = np.where(np.all(sample[:, b_columns] == b_nodes[:, None], axis=-1).T)[1]
  # i_idx = np.where(np.all(sample[:, i_columns] == i_nodes[:, None], axis=-1).T)[1]

  # Columns already contain required indices
  c_idx = sample[:, c_columns].ravel().astype(int)
  b_idx = sample[:, b_columns].ravel().astype(int)
  i_idx = sample[:, i_columns].ravel().astype(int)

  e_nodes = np.empty((c_nodes.shape[0], b_nodes.shape[0], i_nodes.shape[0]))
  assert e_nodes.size == sample[:, e_columns].size
  e_nodes.reshape(-1)[np.ravel_multi_index((c_idx, b_idx, i_idx), dims=e_nodes.shape)] = sample[:, e_columns]
  return c_nodes, b_nodes, i_nodes, e_nodes


def unfold(c_nodes, b_nodes, i_nodes, e_nodes):
  c_sample = c_nodes.repeat(b_nodes.shape[0] * i_nodes.shape[0])
  b_sample = np.tile(b_nodes.repeat(i_nodes.shape[0]), c_nodes.shape[0])
  i_sample = np.tile(i_nodes.flatten(), b_nodes.shape[0] * c_nodes.shape[0])
  e_sample = np.ascontiguousarray(e_nodes).reshape(-1)
  # return np.vstack([c_sample, b_sample, i_sample, e_sample]).T
  return c_sample, b_sample, i_sample, e_sample


def test_folding(sample=None):
  if sample is not None:
    assert np.all(sample == unfold(*fold(sample)))
    return

  x1 = np.linspace(0.1, 0.6, 3)
  x2 = np.linspace(1, 6, 4)
  x3 = np.linspace(10, 60, 5)
  x = np.vstack([_.flatten() for _ in np.meshgrid(x1, x2, x3)]).T
  y = np.linspace(1000, 1000 * x.shape[0], x.shape[0]).reshape(-1, 1)
  sample = np.hstack([x, y])
  np.random.shuffle(sample)

  c_nodes, b_nodes, i_nodes, e_nodes = fold(sample)
  assert np.all(c_nodes == np.vstack([_.T.flatten() for _ in np.meshgrid(x1)]).T)
  assert np.all(b_nodes == np.vstack([_.T.flatten() for _ in np.meshgrid(x2)]).T)
  assert np.all(i_nodes == np.vstack([_.T.flatten() for _ in np.meshgrid(x3)]).T)
  assert np.all(np.sort(e_nodes.reshape(-1, 1), axis=0) == y)

  new_sample = unfold(c_nodes, b_nodes, i_nodes, e_nodes)
  assert np.all(sample[np.argsort(sample[:, -1])] == new_sample[np.argsort(new_sample[:, -1])])
