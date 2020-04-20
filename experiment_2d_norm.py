from norm_utils import get_convex_hull_dataset_from_name, plot_convex_dataset, plot_contours, plot_contours_heatmap
import pickle, numpy as np, tensorflow as tf, argparse, os, multiprocessing as mp
import ast, itertools, json, time

import matplotlib
import matplotlib.pyplot as plt

plt.switch_backend('agg')
matplotlib.rcParams.update({'font.size': 16})

from data import StaticDataset, AttrDict, make_session
from pathos.multiprocessing import ProcessPool as Pool
import ast, itertools, json, time, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from metrics_tf1 import euclidean_metric, mahalanobis_metric, widenorm_metric, deepnorm_metric, mlp_nonmetric,\
  max_relu_pairwise_activation, max_pool_pairwise_activation
from copy import deepcopy

# Global variables
sess, tr, te = None, None, None

def reset_graph():
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()

def make_graph_fn(network, data_dims=128, learning_rate=1e-3, grad_clip=5.0):
  def graph_fn():
    reset_graph()
    diff = tf.placeholder(tf.float32, [None, data_dims], 'xs')
    d = tf.placeholder(tf.float32, [None], 'ds')

    hx, hy, p_diff = network(diff)

    loss = tf.reduce_mean(tf.squared_difference(p_diff, d))
    loss_max = tf.reduce_max(tf.squared_difference(p_diff, d))
    ploss = tf.reduce_mean(tf.abs(p_diff - d) / d)  # Maximum of the abs of % (1=100%) error
    ploss_max = tf.reduce_max(tf.abs(p_diff - d) / d)  # Maximum of the abs of % (1=100%) error
    opt = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*opt.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
    ts = opt.apply_gradients(zip(gradients, variables))
    init = tf.global_variables_initializer()

    return AttrDict(locals())

  return graph_fn

def make_network(emb_dim, m='euclidean',
                 w_posdef_constraint_fn=tf.square,
                 sym=True):
  def network(diff):
    hx, hy = tf.zeros_like(diff), diff
    metric = m.split('_')

    if metric[0] == 'euclidean':
      return hx, hy, euclidean_metric(hx, hy)

    elif metric[0] == 'mahalanobis':
      size = int(metric[1])
      return hx, hy, mahalanobis_metric(hx, hy, size)

    elif metric[0] == 'widenorm':
      num_components, component_size, concave_activation_size, mode, _ = metric[1:]
      return hx, hy, widenorm_metric(hx, hy, int(num_components), int(component_size),
                                     int(concave_activation_size), mode, sym)

    elif metric[0] == 'deepnorm':
      layers, activation, concave_activation_size, mode, _ = metric[1:]
      layer_size, num_layers = layers.split('x')
      layers = [int(layer_size)] * int(num_layers)
      if activation == 'relu':
        activation = tf.nn.relu
      elif activation == 'maxrelu':
        activation = max_relu_pairwise_activation
      elif activation == 'maxpool':
        activation = max_pool_pairwise_activation
      else:
        raise NotImplementedError
      return hx, hy, deepnorm_metric(hx, hy, layers, activation, int(concave_activation_size),
                                     mode, sym)

    elif metric[0] == 'mlp':
      layers, mode = metric[1:]
      layer_size, num_layers = layers.split('x')
      layers = [int(layer_size)] * int(num_layers)
      return hx, hy, mlp_nonmetric(hx, hy, layers, mode)

    else:
      raise ValueError('Unsupported metric')

  return network


def eval_graph(g, dataset, emb_weight=None):
  losses = []
  losses_max = []
  plosses = []
  plosses_max = []
  for x, d in dataset.run_epoch(batch_size=5000):
    feed_dict = {g.diff: x, g.d: d}
    loss_mean, loss_max, ploss_mean, ploss_max = sess.run([g.loss, g.loss_max, g.ploss, g.ploss_max],
                                                          feed_dict=feed_dict)
    losses.append(loss_mean)
    losses_max.append(loss_max)
    plosses.append(ploss_mean)
    plosses_max.append(ploss_max)
  return {"mean_squared_error": float(np.mean(losses)),
          "max_squared_error": float(np.max(losses_max)),
          "mean_percentage_error": float(np.mean(plosses)),
          "max_percentage_error": float(np.max(plosses_max))}


def train_epoch(g, tr, batch_size=100, emb_weight=None):
  losses = []
  losses_max = []
  plosses = []
  plosses_max = []
  for x, d in tr.run_epoch(batch_size=batch_size):
    feed_dict = {g.diff: x, g.d: d, }
    # tr_loss_, _ = sess.run([g.loss, g.ts], feed_dict)
    loss_mean, loss_max, ploss_mean, ploss_max, _ = sess.run([g.loss, g.loss_max, g.ploss, g.ploss_max, g.ts],
                                                             feed_dict=feed_dict)
    losses.append(loss_mean)
    losses_max.append(loss_max)
    plosses.append(ploss_mean)
    plosses_max.append(ploss_max)
  return {"mean_squared_error": float(np.mean(losses)),
          "max_squared_error": float(np.max(losses_max)),
          "mean_percentage_error": float(np.mean(plosses)),
          "max_percentage_error": float(np.max(plosses_max))}


def experiment(graph_fn, tr, te, n_epochs=100, batch_size=100, verbose_every=0):
  g = graph_fn()
  global sess
  sess = make_session()
  sess.run(g.init)
  tr.reset()
  te.reset()
  tr_results = eval_graph(g, tr, )
  tr_losses = {}
  for key in tr_results:
    tr_losses[key] = [tr_results[key]]

  te_losses = {}
  te_results = eval_graph(g, te, )
  for key in te_results:
    te_losses[key] = [te_results[key]]

  if verbose_every:
    print("Test MSE at start of training: {}".format(te_losses["mean_squared_error"][-1]))

  for epoch in range(n_epochs):
    tr_loss = train_epoch(g, tr, batch_size)
    te_loss = eval_graph(g, te)

    for key in tr_loss:
      tr_losses[key].append(tr_loss[key])
    for key in te_loss:
      te_losses[key].append(te_loss[key])

    if verbose_every and epoch % verbose_every == 0:
      print(
        "Epoch {} MSE TR: {:.5f} | TE: {:.5f} | MPE TR: {:.5f} | TE: {:.5f} | MaxSE TR: {:.5f} | TE: {:.5f} | MaxPE TR: {:.5f} | TE: {:.5f}".format(
          epoch,
          tr_loss['mean_squared_error'], te_loss['mean_squared_error'],
          tr_loss['mean_percentage_error'], te_loss['mean_percentage_error'],
          tr_loss['max_squared_error'], te_loss['max_squared_error'],
          tr_loss['max_percentage_error'], te_loss['max_percentage_error']))

  if verbose_every:
    print(
      "Final {} MSE TR: {:.5f} | TE: {:.5f} | MPE TR: {:.5f} | TE: {:.5f} | MaxSE TR: {:.5f} | TE: {:.5f} | MaxPE TR: {:.5f} | TE: {:.5f}".format(
        epoch,
        tr_loss['mean_squared_error'], te_loss['mean_squared_error'],
        tr_loss['mean_percentage_error'], te_loss['mean_percentage_error'],
        tr_loss['max_squared_error'], te_loss['max_squared_error'],
        tr_loss['max_percentage_error'], te_loss['max_percentage_error']))

  return tr_losses, te_losses, g

def main():
  g_dict = {}  
  hulls_dict = {}
  train_data_dict = {}
  training_sizes = [16, 128]

  num_points = 4096

  archs = []
  # Do a sweep over the architecture variants to generate the arch strings:
  # Wide Norm variants
  wn_widths = [2, 10, 50]
  wn_comps = [2, 10, 50]
  for wn_width in wn_widths:
    for wn_comp in wn_comps:
      archs.append('widenorm_{}_{}_0_avg_'.format(wn_width, wn_comp))
  
  # Mahalanobis
  m_widths = [2, 10, 50]
  for m_width in m_widths:
      archs.append('mahalanobis_{}'.format(m_width))

  # Deep Norm variants
  dn_widths = [10, 50, 250]
  dn_depths = [2, 3, 4, 5]
  for dn_width in dn_widths:
    for dn_depth in dn_depths:
      archs.append('deepnorm_{}x{}_maxrelu_5_maxavg_'.format(dn_width, dn_depth))

  # ReLU MLP
  relu_widths = [10, 50, 250]
  relu_depths = [2, 3, 4, 5]
  for relu_width in relu_widths:
    for relu_depth in relu_depths:
      archs.append('mlp_{}x{}_subtract'.format(relu_width, relu_depth))
  
  hulls = ['asym_hull','sym_hull','square','diamond',]
  output_folder = '2D_metrics'
  
  data_dims = 2
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  lr = 0.001
  n_epochs = 5000
  verbose_every = 1000
  batch_size = 128

  results = []
  for hull_name in hulls:
    g_dict[hull_name] = {}

    # Generate the base dataset
    xs, ys, hull = get_convex_hull_dataset_from_name(hull_name, num_points, dims=data_dims)

    hulls_dict[hull_name] = hull

    train_data_dict[hull_name] = {"xs": xs, "ys": ys}

    is_sym = not "asym" in hull_name

    for num_train in training_sizes:
      g_dict[hull_name][num_train] = {}

      print("{} {}".format(hull_name, num_train))

      # Pick a subset of for the training size
      samples = np.random.permutation(num_points)[:num_train]
      perturbations = np.random.random(size=(num_points)) * 0.3 + 0.85  # Range of perturbations = [0.85, 1.15]
      pxs = (xs / (np.expand_dims(ys, 1)) * np.expand_dims(perturbations, 1))[
        samples]  # Normalize xs first by its norm under target convex hull then multiply perturbations
      pys = (perturbations)[samples]

      # test xs set 1: Set of vectors that have norm = 1  (interpolation)
      txs1 = xs / (np.expand_dims(ys, 1))
      tys1 = ys / ys  # Vector of 1's

      # test xs set 2: Set of vectors that have norm = 2  (extrapolation)
      txs2 = txs1 * 2.0
      tys2 = tys1 * 2.0

      # test xs set 3: Set of vectors that have norm = 0.5  (extrapolation)
      txs3 = txs1 * 0.5
      tys3 = tys1 * 0.5

      # Plot the current training data only if data_dims = 2
      if data_dims == 2:
        fig_name = "{}/hull-{}_train-{}".format(output_folder, hull_name, num_train)
        plot_convex_dataset(pxs, pys, hull, extra_contours=[], name=fig_name)

      train_data_dict[hull_name][num_train] = {"pxs": pxs, "pys": pys}
      tr, te, te2, te3 = StaticDataset(X=pxs, D=pys), StaticDataset(X=txs1, D=tys1), \
                         StaticDataset(X=txs2, D=tys2), StaticDataset(X=txs3, D=tys3)

      for arch in archs:
        # Visualize the contours
        print("{} {} {}".format(hull_name, num_train, arch))

        # Train the network
        network_fn = make_network(data_dims, m=arch, sym=is_sym)

        g = make_graph_fn(network_fn, data_dims=data_dims, learning_rate=lr)
        trl, tel, g = experiment(g, tr, te, n_epochs=n_epochs, batch_size=batch_size, verbose_every=verbose_every)

        tel2 = eval_graph(g, te2, )
        tel3 = eval_graph(g, te3, )

        tel_extra = {"test_contour=2": tel2, "test_contour=0.5": tel3}

        # Save results
        config = {"data_dim": data_dims,
                  "post_emb_type": arch,
                  "pre_emb": [],
                  "lr": lr,
                  "n_epochs": n_epochs,
                  "batch_size": batch_size,
                  "num_train": num_train,
                  "verbose": verbose_every,
                  "hull_name": hull_name,
                  "use_sym": is_sym
                  }

        results.append((config, trl, tel, tel_extra))

        # Save to g_dict
        g_dict[hull_name][num_train][arch] = g

        # Only plot if data_dims = 2
        if data_dims == 2:
          norm_func = lambda xs: sess.run(g.p_diff, feed_dict={g.diff: xs})
          fig_name = "{}/hull-{}_train-{}_arch-{}".format(output_folder, hull_name, num_train, arch)
          plot_contours_heatmap(xs, norm_func, hull, contour_list=[0.5, 1.0, 1.5], name=fig_name)

        print()

  # Write results
  with open(os.path.join(output_folder, 'results.txt'), 'w') as f:
    json.dump(results, f)


if __name__ == "__main__":
  main()
