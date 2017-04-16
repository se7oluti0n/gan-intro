import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

seed = 42
np.random.seed(seed)

class DataDistribution(object):
  def __init__(self):
    self.mu = 4
    self.sigma = 0.5
  
  def sample(self, N):
    samples = np.random.normal(self.mu, self.sigma, N)
    samples.sort()
    return samples
  
class GeneratorDistribution(object):
  def __init__(self, range):
    self.range = range
  
  def sample(self, N):
    return np.linspace(-self.range, self.range, N) + \
        np.random.random(N) * 0.01
  
def linear(input, output_dim, scope=None, stddev=0.01):
  norm = tf.random_normal_initializer(stddev=stddev)
  const = tf.constant_initializer(0.0)

  with tf.variable_scope(scope or 'linear'):
    w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
    b = tf.get_variable('b', [output_dim], initializer=const)

    return tf.matmul(input, w) + b

def generator(input, h_dim):
  h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
  h1 = linear(h0, 1, 'g1')

  return h1

def discriminiator(input, h_dim, minibatch_layer=True):
  h0 = tf.tanh(linear(input, h_dim * 2, 'd0'))
  h1 = tf.tanh(linear(h0, h_dim * 2, 'd1'))
  if minibatch_layer:
    h2 = minibatch(h1)
  else:
    h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))
  
  h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
  return h3

def minibatch(input, num_kernels=5, kernel_dim=3):
  x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
  activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
  diffs = activation[...,None] - tf.transpose(activation, [1, 2, 0])[None, ...]
  abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
  minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
  return tf.concat([input, minibatch_features], 1)

def optimizer(loss, vars):
  initial_learning_rate = 0.01
  decay = 0.95
  decay_step = 150

  batch = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    initial_learning_rate,
    batch,
    decay_step,
    decay,
    True
  )

  opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    loss,
    global_step=batch,
    var_list=vars
  )

  return opt

def _samples(session, x, z, D1, G, num_points=10000, num_bins=100, srange=8):
  xs = np.linspace(-srange, srange, num_points)[:, None]
  bins = np.linspace(-srange, srange, num_bins)

  db = np.zeros((num_points, 1))
  for i in range(num_points // batch_size):
    piece = xs[batch_size * i: batch_size * (i+1)]
    feed_dict = {x: piece}
    mmm = session.run([D1], feed_dict)
    db[batch_size * i: batch_size * (i+1)] = mmm[0]

  # data distribution
  d = data.sample(num_points)
  pd, _ = np.histogram(d, bins=bins, density=True)

  # Generated samples
  zs = np.linspace(-srange, srange, num_points)[:, None]
  gb = np.zeros((num_points, 1))
  for i in range(num_points // batch_size):
    [gb[batch_size * i : batch_size * (i+1)]] = session.run([G], {
        z: zs[batch_size * i : batch_size * (i+1)]
    })

  print "GB shape: ", gb.shape
  pg, _ = np.histogram(gb, bins=bins, density=True)

  return db, pd, pg

# Define Generator

hidden_size = 4
batch_size = 12

with tf.variable_scope('D_pre'):
  pre_input = tf.placeholder(tf.float32, shape=(batch_size, 1))
  pre_labels = tf.placeholder(tf.float32, shape=(batch_size, 1))

  D_pre = discriminiator(pre_input, hidden_size, True)
  pre_loss = tf.reduce_mean(tf.square(D_pre - pre_labels))
  

with tf.variable_scope('G'):
  # input noise
  z = tf.placeholder(tf.float32, shape=(batch_size, 1))
  G = generator(z, hidden_size)

# Define Discriminator
with tf.variable_scope('Disc') as scope:
  # input data
  x = tf.placeholder(tf.float32, shape=(batch_size, 1))
  D1 = discriminiator(x, hidden_size)
  scope.reuse_variables()

  D2 = discriminiator(G, hidden_size)

lossD = tf.reduce_mean(- tf.log(D1) - tf.log(1 - D2))
lossG = tf.reduce_mean(tf.log(D2))

all_vars = tf.trainable_variables()
vars_d = [v for v in all_vars if v.name.startswith('Disc')]
vars_g = [v for v in all_vars if v.name.startswith('G')]
d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')

pre_opt = optimizer(pre_loss, d_pre_params)

opt_d = optimizer(lossD, vars_d)
opt_g = optimizer(lossG, vars_g)

data = DataDistribution()
gen = GeneratorDistribution(range=8)
num_steps = 12000

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)

  num_pretrain_steps = 6000
  for step in xrange(num_pretrain_steps):
    d = (np.random.random(batch_size) - 0.5) * 10.0
    labels = norm.pdf(d, loc=data.mu, scale=data.sigma)
    pretrain_loss, _ = sess.run([pre_loss, pre_opt],{
      pre_input: np.reshape(d, (batch_size, 1)),
      pre_labels: np.reshape(labels, (batch_size, 1))
    })
  weightsD = sess.run(d_pre_params)

  for i, v in enumerate(vars_d):
    sess.run(v.assign(weightsD[i]))

  for step in range(num_steps):
    dx = data.sample(batch_size)
    dz = gen.sample(batch_size)

    loss_d, _ = sess.run([lossD, opt_d], {
      x: np.reshape(dx, (batch_size, 1)), 
      z: np.reshape(dz, (batch_size, 1))
    })

    # 
    dz = gen.sample(batch_size)
    loss_g, _ = sess.run([lossG, opt_g],{
      z: np.reshape(dz, (batch_size, 1))
    })

    if step % 10 == 0:
      print('{}: {}\t{}'.format(step, loss_d, loss_g))

  db, pd, pg = _samples(sess, x, z, D1, G)
  db_x = np.linspace(-8, 8, len(db))
  p_x = np.linspace(-8, 8, len(pd))
  f, ax = plt.subplots(1)
  ax.plot(db_x, db, label='decision boundary')
  ax.set_ylim(0, 1)
  plt.plot(p_x, pd, label='real data')
  plt.plot(p_x, pg, label='generated data')
  plt.title('1D Generative Adversarial Network')
  plt.xlabel('Data values')
  plt.ylabel('Probability density')
  plt.legend()
  plt.show()
