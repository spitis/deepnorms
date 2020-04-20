import pickle, numpy as np, tensorflow as tf, argparse, os, re
from pathos.multiprocessing import ProcessPool as Pool
import ast, itertools, json, time, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data import load_dataset, AttrDict, make_session, make_emb_dict
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
    x = tf.placeholder(tf.float32, [None, data_dims], 'xs')
    y = tf.placeholder(tf.float32, [None, data_dims], 'ys')
    d = tf.placeholder(tf.float32, [None], 'ds')

    xemb, yemb, p = network(x, y)

    loss = tf.reduce_mean(tf.squared_difference(p, d))
    opt = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*opt.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
    ts = opt.apply_gradients(zip(gradients, variables))
    init = tf.global_variables_initializer()

    return AttrDict(locals())
  return graph_fn

def make_network(pre_emb=[800,800,800,800], m='euclidean'):
  def network(x, y):
    hx, hy = x, y
    for i, layer_size in enumerate(pre_emb):
      if i + 1 < len(pre_emb):
        H = tf.layers.Dense(layer_size, tf.nn.relu)
      else:
        H = tf.layers.Dense(layer_size)
      hx = H.apply(hx)
      hy = H.apply(hy)
    
    metric = m.split('_')

    if metric[0] == 'euclidean':
      return hx, hy, euclidean_metric(hx, hy)
    
    elif metric[0] == 'mahalanobis':
      size = int(metric[1])
      return hx, hy, mahalanobis_metric(hx, hy, size)
    
    elif metric[0] == 'widenorm':
      num_components, component_size, concave_activation_size, mode, symmetric = metric[1:]
      return hx, hy, widenorm_metric(hx, hy, int(num_components), int(component_size), 
                                     int(concave_activation_size), mode, bool(symmetric))
    
    elif metric[0] == 'deepnorm':
      layers, activation, concave_activation_size, mode, symmetric = metric[1:]
      layer_size, num_layers = layers.split('x')
      layers = [int(layer_size)]*int(num_layers)
      if activation=='relu':
        activation=tf.nn.relu
      elif activation=='maxrelu':
        activation=max_relu_pairwise_activation
      elif activation=='maxpool':
        activation=max_pool_pairwise_activation
      else:
        raise NotImplementedError
      return hx, hy, deepnorm_metric(hx, hy, layers, activation, int(concave_activation_size),
                             mode, bool(symmetric))

    elif metric[0] == 'mlp':
      layers, mode = metric[1:]
      layer_size, num_layers = layers.split('x')
      layers = [int(layer_size)]*int(num_layers)
      return hx, hy, mlp_nonmetric(hx, hy, layers, mode)
    
    else:
      raise ValueError('Unsupported metric')
  
  return network

def eval_graph(g, dataset):
  losses = []
  for x, y, d in dataset.run_epoch(batch_size=5000):
    losses.append(sess.run(g.loss, {g.x:x, g.y:y, g.d:d}))
  return float(np.mean(losses))

def train_epoch(g, batch_size=100):
  losses = []
  for x, y, d in tr.run_epoch(batch_size=batch_size):
    tr_loss_, _ = sess.run([g.loss, g.ts], {g.x: x, g.y: y, g.d: d})
    losses.append(tr_loss_)
  return float(np.mean(losses))

def experiment(graph_fn, n_epochs=100, batch_size=100, verbose_every=0):
  g = graph_fn()
  global sess
  sess = make_session()
  sess.run(g.init)
  
  tr.reset()
  te.reset()
  tr_losses = [eval_graph(g, tr)]
  te_losses = [eval_graph(g, te)]
  
  if verbose_every:
    print("Test loss at start of training: {}".format(te_losses[-1]))
  
  for epoch in range(n_epochs):
    tr_loss = train_epoch(g, batch_size)
    te_loss = eval_graph(g, te)
    tr_losses.append(tr_loss)
    te_losses.append(te_loss)
    if verbose_every and epoch % verbose_every == 0:
      print("Epoch {} TR|TE Loss:  {}  |  {}".format(epoch, tr_loss, te_loss))
  
  if verbose_every:
    print("Final TR|TE Loss:  {}  |  {}".format(tr_losses[-1], te_losses[-1]))
    
  return tr_losses, te_losses

def read_config_file(filename):
  configs = []
  with open(filename, 'r') as f:
    argdict = dict((l.split('=') for l in f.readlines() if l))
  for k, v in argdict.items():
    argdict[k] = ast.literal_eval(v.strip())
  for pre_emb, post_emb_type, lr, graph_id, embdict, datapickle, seed in itertools.product(argdict['pre_embs'], 
    argdict['post_emb_types'], argdict['lrs'], argdict['graph_ids'], argdict['embdicts'], argdict['datapickle'], argdict['seed']):
    configs.append(AttrDict({'pre_emb': pre_emb, 'post_emb_type': post_emb_type, 'lr':lr, 'graph_id': graph_id, 
      'embdict': embdict.replace('GRAPHID', graph_id), 'datapickle': datapickle.replace('GRAPHID', graph_id), 'seed': seed}))
  return configs

def run_config(config):
  network_fn = make_network(config.pre_emb, config.post_emb_type)
  global tr, te
  tr, te = load_dataset(data_pickle=config.datapickle, emb_dict_pickle=config.embdict, cutoff=config.datasize, test_size=30000, seed=config.seed)
  trl, tel = experiment(make_graph_fn(network_fn, data_dims=config.data_dims, learning_rate=config.lr), n_epochs=config.n_epochs, batch_size=config.batch_size, verbose_every=config.verbose)
  return (config, trl, tel)

def config_in(config, config_list):
  """drops verbose key, then compares"""
  config = deepcopy(config)
  config.pop('verbose', None)
  return config in config_list


def main(args):
  with open(os.path.join(args.folder, 'args.txt'), 'w') as f:
    for k, v in vars(args).items():
      f.write('{}={}\n'.format(k, v))
  configs = read_config_file(os.path.join(args.folder, 'configs.txt'))
  for config in configs:
    config.datasize = args.size
    config.n_epochs = args.n_epochs
    config.batch_size = args.batch_size
    config.verbose = args.verbose
    m = re.search('-(.+?)_emb', config.embdict)
    config.data_dims = 32 + int(m.group(1))

  old_results = []
  results_path = os.path.join(args.folder, 'results.txt')
  if os.path.exists(results_path):
    shutil.copyfile(results_path, results_path + '.bak')
    with open(results_path, 'r') as f:
      old_results = json.load(f)
      old_configs = deepcopy([c for c, _, _ in old_results])
      for c in old_configs:
        c.pop('verbose', None)
      print('Found old results!')
    
      print('Config file has {} configs, and old results have {} total configs...'.format(len(configs), len(old_results)))
      new_configs = [c for c in configs if not config_in(c, old_configs)]
      print('There are {} new configs!'.format(len(new_configs)))
      print('Running only the new configs...')
      configs = new_configs

  timestart = time.time()
  print("*** ARGS ***")
  print(args)
  print("************")
  print('Running experiment for {} configs... the time is now {}...'.format(len(configs), time.localtime(timestart)))
  results = []
  if args.processes == 1:
    for config in configs:
      print('\nCurrent config:', config)
      network_fn = make_network(config.pre_emb, config.post_emb_type)
      trl, tel = experiment(make_graph_fn(network_fn, data_dims=config.data_dims, learning_rate=config.lr), n_epochs=args.n_epochs, batch_size=args.batch_size, verbose_every=args.verbose)
      results.append((config, trl, tel))
      print('Final results: train {}; test {}'.format(trl[-1], tel[-1]))
  else:
    results = old_results
    for i in range(2000):
      p = args.processes
      with Pool(p) as pool:
        cs = configs[i*p:(i+1)*p]
        if cs:
          results += pool.map(run_config, cs)
          print("Done {} configs!".format(cs))
        else:
          print("Done all configs!")
          break
          
  timeend = time.time()
  with open(results_path, 'w') as f:
    json.dump(results, f)
  with open(os.path.join(args.folder, 'time.txt'), 'w') as f:
    f.write("Time start: {}, or {}\n".format(timestart, time.localtime(timestart)))
    f.write("Time end: {}, or {}\n".format(timestart, time.localtime(timeend)))

  print("Done!")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run Shortest Path Experiment")
  parser.add_argument('-f', '--folder',     required=True, type=str, help='folder containg config / where to write results')
  parser.add_argument('-s', '--size',    default=10000, type=int, help='number of datapoints to use for both training')
  parser.add_argument('-b', '--batch_size', default=100, type=int, help='batch_size'  )
  parser.add_argument('-v', '--verbose',    default=20, type=int, help='verbose every n epochs'  )
  parser.add_argument('-n', '--n_epochs',    default=100, type=int, help='epochs to run'  )
  parser.add_argument('-p', '--processes',    default=4, type=int, help='num_procs'  )
  args = parser.parse_args()
  main(args)
