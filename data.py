import pickle, numpy as np, tensorflow as tf, os, multiprocessing as mp
import networkx as nx, tqdm
from collections import deque

def make_emb_dict(graph_id='sf', emb_id='emb'):
  file_name='{}.emb'.format(graph_id)
  out_file = '{}_{}_dict.pickle'.format(graph_id, emb_id)
  with open(file_name, 'r') as f:
    line_splits = [l.split(' ') for l in f.read().split('\n') if l]
  embs = {}
  for line_split in line_splits[1:]:
    i = int(line_split[0])
    emb = np.array(line_split[1:]).astype(np.float32)
    embs[i] = emb
  
  with open(out_file, 'wb') as f:
    pickle.dump(embs, f)

def make_connected_edgelist(graph_id='sf'):
  if 'ASYM' in graph_id:
    raise NotImplementedError
  G = nx.read_edgelist('{}.edgelist'.format(graph_id), nodetype=int, 
                     data=(('weight',float),), create_using=nx.DiGraph())
  G = G.to_undirected()
  Gs = list(sorted(nx.connected_component_subgraphs(G), key = lambda g: g.number_of_nodes(), reverse=True))
  G = Gs[0]
  print('num nodes: {}; num edges: {}'.format(G.number_of_nodes(), G. number_of_edges()))

  nx.write_weighted_edgelist(G, '{}connected.edgelist'.format(graph_id))

def make_lm_embeddings(graph_id, NOISE_FEATS, N_LANDMARKS=32, LM_NOISE=0.2, OPTIMIZATION_TRIES=64):
  symmetric = True
  if 'ASYM' in graph_id:
    symmetric=False

  G = nx.read_edgelist('{}{}.edgelist'.format(graph_id, 'connected' if symmetric else ''), 
    nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
  if symmetric:
    G = G.to_undirected()
  else:
    Grev = G.reverse()
  nodes = np.array(G.nodes())

  def keywithmaxval(d):
    """ a) create a list of the dict's keys and values; 
      b) return the key with the max value"""  
    v=list(d.values())
    k=list(d.keys())
    return k[v.index(max(v))]

  # Collect landmarks
  landmarks = deque(np.random.choice(nodes, (1,), replace=False), N_LANDMARKS)
  for i in tqdm.tqdm(range(OPTIMIZATION_TRIES)):
    dists = nx.multi_source_dijkstra_path_length(G, landmarks)
    landmarks.append(keywithmaxval(dists))
  landmark_dists = [nx.single_source_dijkstra_path_length(G, l) for l in landmarks]
  if not symmetric:
    landmark_dists += [nx.single_source_dijkstra_path_length(Grev, l) for l in landmarks]
    N_LANDMARKS = 2 * N_LANDMARKS

  # Get landmark stats for normalization
  lm_dists = [list(landmark_dists[i].values()) for i in range(N_LANDMARKS)]
  lm_dists = np.array(lm_dists)
  lm_mean = np.mean(lm_dists)
  lm_std = np.std(lm_dists)

  # Collect embeddings based on landmarks, normalizing each
  embs = {}
  for i in nodes:
    if LM_NOISE:
      lm_noise = np.random.normal(scale=LM_NOISE, size=[N_LANDMARKS])
    else:
      lm_noise = 0.
    lm_feats = (np.array([landmark_dists[lm][i] for lm in range(N_LANDMARKS)]) - lm_mean) / lm_std + lm_noise
    noise_feats = np.random.normal(size=[NOISE_FEATS])
    embs[i] = np.concatenate([lm_feats, noise_feats])

  if not symmetric:
    N_LANDMARKS = N_LANDMARKS // 2
    
  # Save embs to disk
  with open('{}_lm_{}n{}-{}_emb_dict.pickle'.format(graph_id, N_LANDMARKS, LM_NOISE, NOISE_FEATS), 'wb') as f:
    pickle.dump(embs, f)

def make_bulk_lm_embeddings(graph_id_list, num_distractors_list, landmark_noises_list, n_landmarks=32):
  for graph_id in graph_id_list:
    for num_distractors in num_distractors_list:
      for landmark_noise in landmark_noises_list:
        make_lm_embeddings(graph_id, num_distractors, n_landmarks, landmark_noise, OPTIMIZATION_TRIES=2*n_landmarks)
        
def convert_XYints_to_XYembs(X, Y, emb_dict):
  X_ = []
  for x in X:
    X_.append(emb_dict[x])
  X = np.array(X_)

  Y_ = []
  for y in Y:
    Y_.append(emb_dict[y])
  Y = np.array(Y_)

  return X, Y

class AttrDict(dict):
  __setattr__ = dict.__setitem__

  def __getattr__(self, key):
    try:
      return dict.__getitem__(self, key)
    except KeyError:
      raise AttributeError

class StaticDataset():

  def __init__(self, **kwargs):
    self.data = kwargs
    size = None
    for k, v in kwargs.items():
      assert isinstance(v, np.ndarray), 'Dataset values must be numpy arrays of shape 2'
      if size is None:
        size = len(v)
      else:
        assert len(v) == size, 'All data arrays must be same length'
      assert not hasattr(self, k), 'Invalid key. Object cannot already have this atttribute'
      setattr(self, k, v)

    self.size = size
    self.reset()
    
  def reset(self):
    self.cursor = 0
    self.epoch = -1
    self.new_epoch()
    
  def new_epoch(self):
    self.epoch += 1
    self.shuffle()
    self.cursor = 0

  def shuffle(self):
    shuffled = np.random.permutation(range(self.size))
    for k, v in self.data.items():
      self.data[k] = v[shuffled]
  
  def next_batch(self, batch_size, partial_ok=True):
    res = []
    data_left = self.size - self.cursor
    if (data_left <= 0) or (data_left < batch_size and not partial_ok):
      return None

    for _, v in self.data.items():
      res.append(v[self.cursor:self.cursor+batch_size])
    self.cursor += batch_size
    return res

  def run_epoch(self, batch_size, partial_ok=True):
    while True:
      batch = self.next_batch(batch_size, partial_ok)
      if batch is None:
        self.new_epoch()
        break
      yield batch

def make_session(num_cpu=None, make_default=False, graph=None, cpu_only=False):
    """
    Returns a session that will use <num_cpu> CPU's only

    :param num_cpu: (int) number of CPUs to use for TensorFlow
    :param make_default: (bool) if this should return an InteractiveSession or a normal Session
    :param graph: (TensorFlow Graph) the graph of the session
    :return: (TensorFlow session)
    """
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', mp.cpu_count()))
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    # Prevent tensorflow from taking all the gpu memory
    if not cpu_only:
      tf_config.gpu_options.allow_growth = True
    if make_default:
        return tf.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.Session(config=tf_config, graph=graph)

def load_dataset(data_pickle='sf_1M.pickle', emb_dict_pickle='sf_emb_dict.pickle', cutoff=None, test_size = 30000, seed = 0):
  with open(data_pickle, 'rb') as f:
    X, Y, D = pickle.load(f)
  with open(emb_dict_pickle, 'rb') as f:
    embs = pickle.load(f)
  X, Y = convert_XYints_to_XYembs(X, Y, embs)

  assert cutoff + test_size <= len(X)

  np.random.seed(seed)
  training_indices = np.random.choice(150000-test_size, cutoff, replace=False)

  xtr = X[training_indices]
  ytr = Y[training_indices]
  dtr = D[training_indices]

  xte = X[-test_size:]
  yte = Y[-test_size:]
  dte = D[-test_size:]

  return StaticDataset(X=xtr, Y=ytr, D=dtr), StaticDataset(X=xte, Y=yte, D=dte)