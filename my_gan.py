import numpy as np
import tensorflow as tf
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
  
def linear(input, output_dim, scope=None, stddev=1.0):
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
  initial_learning_rate = 0.005
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

# Define Generator

hidden_size = 4
batch_size = 12

with tf.variable_scope('G'):
  # input noise
  z = tf.placeholder(tf.float32, shape=(None, 1))
  G = generator(z, hidden_size)

# Define Discriminator
with tf.variable_scope('D') as scope:
  # input data
  x = tf.placeholder(tf.float32, shape=(None, 1))
  D1 = discriminiator(x, hidden_size)
  scope.reuse_variables()

  D2 = discriminiator(G, hidden_size)

lossD = tf.reduce_mean(- tf.log(D1) - tf.log(1 - D2))
lossG = tf.reduce_mean(tf.log(D2))

all_vars = tf.trainable_variables()
vars_d = [v for v in all_vars if v.name.startswith('D')]
vars_g = [v for v in all_vars if v.name.startswith('G')]

opt_d = optimizer(lossD, vars_d)
opt_g = optimizer(lossG, vars_g)

data = DataDistribution()
gen = GeneratorDistribution(range=8)
num_steps = 1200

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)

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

    if step % 10:
      print('{}: {}\t{}'.format(step, loss_d, loss_g))
