# Tensorflow implementations of Deep Norm, Wide Norm, Neural Metrics
import tensorflow as tf, numpy as np
from tensorflow.python.framework import function

# ACTIVATIONS

@function.Defun(tf.float32, tf.float32)
def norm_grad(x, dy):
  return tf.expand_dims(dy, 1)*x/(tf.linalg.norm(x, axis=1, keepdims=True)+1.0e-19, 1)

@function.Defun(tf.float32, grad_func=norm_grad)
def norm(x):
  return tf.linalg.norm(x, axis=1)

def max_relu_pairwise_activation(layer):
  max_component = tf.squeeze(tf.layers.max_pooling1d((0.+tf.expand_dims(layer, -1))/1.0, 2, 2, 'same'), -1)
  weights = tf.nn.softplus(tf.Variable(tf.zeros((1, layer.get_shape()[-1]))))
  relu_component = tf.nn.relu(layer*weights)
  relu_component = tf.squeeze(tf.layers.average_pooling1d((0.+tf.expand_dims(relu_component, -1))/1.0, 2, 2, 'valid'), -1)

  return tf.concat((max_component, relu_component), -1)

def max_pool_pairwise_activation(layer):
  return tf.squeeze(tf.layers.max_pooling1d(tf.expand_dims(layer, -1), 2, 2, 'same'), -1)

def max_avg_global_activation(layer):
  alpha = tf.nn.sigmoid(tf.Variable(-1.))
  return alpha*tf.reduce_max(layer, axis=-1) + (1-alpha)*tf.reduce_mean(layer, axis=-1)

def concave_activation(h, concave_activation_size=None):
  if not concave_activation_size:
    return h

  assert concave_activation_size > 1

  bs_nonzero = tf.nn.softplus(tf.Variable(np.random.normal(-1., 1e-3, size=[1]*(len(h.get_shape())-1) + [h.get_shape()[-1], concave_activation_size-1]).astype(np.float32)))
  bs_zero = tf.constant(np.zeros(shape=[1]*(len(h.get_shape())-1) + [h.get_shape()[-1], 1], dtype=np.float32))
  bs = tf.concat((bs_nonzero, bs_zero), axis=-1)
  ms = 2*tf.nn.sigmoid(tf.Variable(np.random.normal(0., 1e-3, size=[1]*(len(h.get_shape())-1) + [h.get_shape()[-1], concave_activation_size]).astype(np.float32)))

  h = tf.expand_dims(h, -1)

  h = h * ms + bs
  return tf.reduce_min(h, axis=-1)

def reduce_metric(h, mode):
  if mode == 'avg':
    d = tf.reduce_mean(h, axis=1) # [batch]
  elif mode == 'max':
    d = tf.reduce_max(h, axis=1) # [batch]
  elif mode =='maxavg':
    d = max_avg_global_activation(h)
  else:
    raise NotImplementedError
  return d

# METRICS

def euclidean_metric(hx, hy):
  """hx, hy are tf.tensor embeddings, returns tf.tensor euclidean distance"""
  return tf.norm(hx - hy, axis = -1)

def mahalanobis_metric(hx, hy, size):
  """hx, hy are tf.tensor embeddings, size is integer, returns tf.tensor mahalanobis distance"""
  return tf.norm(tf.layers.dense(hx - hy, size, use_bias=False), axis=-1)

def widenorm_metric(hx, hy, num_components, component_size, concave_activation_size=None, mode='avg', symmetric=True):
  """ Implements the widenorm """
  h = hx - hy
  kernel_constraint = None
  if not symmetric:
    h = tf.concat((tf.nn.relu(h), tf.nn.relu(-h)), -1)
    kernel_constraint = tf.nn.relu

  W_components = [tf.layers.Dense(component_size, use_bias=False,
    kernel_constraint=kernel_constraint) for _ in range(num_components)]

  components = [W(h) for W in W_components] # list of [batch, latent_dims]

  h = tf.stack(components, axis=2) # [batch, latent_dims, num_components]
  h = norm(h)

  h = concave_activation(h, concave_activation_size)

  return reduce_metric(h, mode)

def deepnorm_metric(hx, hy, layers, activation=tf.nn.relu, concave_activation_size=None, mode='avg', symmetric=True):
  """ Implements the deepnorm metric """
  h = hx - hy

  U = tf.layers.Dense(layers[0], use_bias=False)
  h1 = activation(U(h))
  h2 = activation(U(-h))

  for layer_size in layers[1:]:
    W = tf.layers.Dense(layer_size, use_bias=False, kernel_constraint=tf.nn.relu)
    U = tf.layers.Dense(layer_size, use_bias=False)

    h1 = activation(W(h1) + U(h))
    h2 = activation(W(h2) + U(-h))

  if symmetric:
    h = h1 + h2
  else:
    h = h1

  h = concave_activation(h, concave_activation_size)

  return reduce_metric(h, mode)

def mlp_nonmetric(hx, hy, layers, mode='concat'):
  if mode == 'subtract':
    h = hy - hx
  elif mode == 'concat':
    h = tf.concat((hx, hy), 1)
  elif mode == 'add':
    h = hx + hy
  elif mode == 'concatsub':
    h = tf.concat((hy-hx, hx-hy), axis=-1)
  elif mode == 'mult':
    h = hx*hy
  elif mode == 'div':
    h = hx/hy
  else:
    raise ValueError('mode={} is not supported'.format(mode))

  for layer_size in layers:
    h = tf.layers.dense(h, layer_size, activation=tf.nn.relu)
  return tf.squeeze(tf.layers.dense(h, 1, tf.nn.softplus))
