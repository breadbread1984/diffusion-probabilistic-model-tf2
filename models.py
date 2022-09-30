#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;

def MultiScaleConv(shape = (1024,1024), n_colors = 3, n_hidden = 20, n_scales = 1, **kwargs):
  inputs = tf.keras.Input((shape[0], shape[1], n_colors)); # inputs.shape = (batch, h, w, 3)
  imgs_accum = None;
  for scale in range(n_scales - 1, -1, -1):
    # from n_scales - 1 to 0, from [h/2**s x w/2**s] to [h x w]
    downsampled = tf.keras.layers.Lambda(lambda x, s: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1] // 2**s, 2**s, tf.shape(x)[2] // 2**s, 2**s, x.shape[-1])), arguments = {'s': scale})(inputs); # downswampled.shape = (batch, h/scale**2, scale**2, w/scale**2, scale**2, 3)
    downsampled = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis = [2,4]))(downsampled); # downsampled.shape = (batch, h/scale**2, w/scale**2, 3)
    downsampled = tf.keras.layers.Conv2D(filters = n_hidden, kernel_size = (3,3), padding = 'same', activation = tf.keras.layers.LeakyReLU(), kernel_initializer = tf.keras.initializers.RandomNormal(stddev = np.sqrt(1./n_hidden)/3**2), bias_initializer = tf.keras.initializers.Constant())(downsampled); # output.shape = (batch, h/scale**2, w/scale**2, n_hidden)
    imgs_accum = tf.keras.layers.Add()([imgs_accum, downsampled]) if imgs_accum is not None else downsampled;
    if scale > 0:
      # scale feature map of hxw to 2*hx2*w
      imgs_accum = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, axis = 2), axis = 4))(imgs_accum); # imgs_accum.shape = (batch, h, 1, w, 1, c)
      imgs_accum = tf.keras.layers.Concatenate(axis = 4)([imgs_accum, imgs_accum]); # imgs_accum.shape = (batch, h, 1, w, 2, c)
      imgs_accum = tf.keras.layers.Concatenate(axis = 2)([imgs_accum, imgs_accum]); # imgs_accum.shape = (batch, h, 2, w, 2, c)
      imgs_accum = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2], tf.shape(x)[3] * tf.shape(x)[4], x.shape[-1])))(imgs_accum); # imgs_accum.shape = (batch, 2*h, 2*w, c)
  imgs_accum = tf.keras.layers.Lambda(lambda x, s: x / s, arguments = {'s': n_scales})(imgs_accum);
  return tf.keras.Model(inputs = inputs, outputs = imgs_accum);

def MultiLayerConvolution(n_layers = 4, **kwargs):
  inputs = tf.keras.Input((None, None, kwargs.get('n_colors', 3))); # inputs.shape = (batch, h, w, 3)
  results = inputs;
  for i in range(n_layers):
    results = MultiScaleConv(**kwargs)(inputs);
  return tf.keras.Model(inputs = inputs, outputs = results);

def MLPConvolution(n_temporal_basis = 10, n_layers_dense_lower = 4, n_hidden_dense_lower = 500, n_hidden_dense_lower_output = 2, n_layers_dense_upper = 2, n_hidden_dense_upper = 20, **kwargs):
  inputs = tf.keras.Input((kwargs.get('shape')[0], kwargs.get('shape')[1], kwargs.get('n_colors', 3))); # inputs.shape = (batch, h, w, 3)
  conv_results = MultiLayerConvolution(**kwargs)(inputs); # conv_results.shape = (batch, h, w, n_hidden)
  conv_results = tf.keras.layers.Lambda(lambda x, s: x / s, arguments = {'s': np.sqrt(kwargs.get('n_hidden'))})(conv_results); # conv_results.shape = (batch, h, w, n_hidden)
  dense_results = tf.keras.layers.Flatten()(inputs); # dense_results.shape = (batch, h*w*3)
  for i in range(n_layers_dense_lower):
    if i != n_layers_dense_lower - 1:
      dense_results = tf.keras.layers.Dense(n_hidden_dense_lower, kernel_initializer = tf.keras.initializers.Orthogonal(), bias_initializer = tf.keras.initializers.Constant(), activation = tf.keras.layers.LeakyReLU())(dense_results); # dense_results.shape = (batch, n_hidden_dense_lower)
    else:
      dense_results = tf.keras.layers.Dense(n_hidden_dense_lower_output * kwargs.get('shape')[0] * kwargs.get('shape')[1], kernel_initializer = tf.keras.initializers.Orthogonal(), bias_initializer = tf.keras.initializers.Constant(), activation = tf.keras.layers.LeakyReLU())(dense_results); # dense_results.shape = (batch, n_hidden_dense_lower_output * shape[0] * shape[1])
  dense_results = tf.keras.layers.Lambda(lambda x, h, w, c: tf.reshape(x, (-1, h, w, c)), arguments = {'h': kwargs.get('shape')[0], 'w': kwargs.get('shape')[1], 'c': n_hidden_dense_lower_output})(dense_results); # dense_results.shape = (batch, shape[0], shape[1], n_hidden_dense_lower_output)
  dense_results = tf.keras.layers.Lambda(lambda x, s: x / s, arguments = {'s': np.sqrt(n_hidden_dense_lower_output)})(dense_results);
  dense_results = tf.keras.layers.Concatenate(axis = -1)([conv_results, dense_results]); # dense_results.shape = (batch, shape[0], shape[1], n_hidden + n_hidden_dense_lower_out)
  for i in range(n_layers_dense_upper):
    if i != n_layers_dense_upper - 1:
      dense_results = tf.keras.layers.Dense(n_hidden_dense_upper, kernel_initializer = tf.keras.initializers.Orthogonal(), bias_initializer = tf.keras.initializers.Constant(), activation = tf.keras.layers.LeakyReLU())(dense_results);
    else:
      dense_results = tf.keras.layers.Dense(kwargs.get('n_colors') * n_temporal_basis * 2, kernel_initializer = tf.keras.initializers.Orthogonal(), bias_initializer = tf.keras.initializers.Constant())(dense_results);
  return tf.keras.Model(inputs = inputs, outputs = dense_results);

class BetaForward(tf.keras.layers.Layer):
  def __init__(self, trajectory_length = 1000, n_basis = 10, step1_beta = 1e-3, **kwargs):
    self.trajectory_length = 1000;
    self.n_basis = 10;
    self.step1_beta = 1e-3;
    super(BetaForward, self).__init__(**kwargs);
  def build(self, input_shape):
    self.beta_perturb_coefficients = self.add_weight(shape = (self.n_basis,1), dtype = tf.float32, initializer = tf.keras.initializers.Constant(), trainable = True);
    self.temporal_basis = self.add_weight(shape = (self.n_basis, self.trajectory_length), dtype = tf.float32, trainable = False);
    self.temporal_basis.assign(self.generate_temporal_basis());
  def generate_temporal_basis(self,):
    # sample n_basis evenly scattered numbers between [-1,1]
    # every basis is a vector having a single peak at the corresponding sampled number
    temporal_basis = tf.zeros((self.trajectory_length, self.n_basis));
    xx = tf.linspace(-1, 1, self.trajectory_length); # xx.shape = (trajectory_length,)
    x_centers = tf.linspace(-1, 1, self.n_basis); # x_centers.shape = (n_basis,)
    width = (x_centers[1] - x_centers[0]) / 2;
    temporal_basis = tf.stack([tf.math.exp(-(xx - x_centers[ii])**2 / (2 * width**2)) for ii in range(self.n_basis)], axis = 1); # temporal_basis.shape = (trajectory_length, n_basis)
    temporal_basis /= tf.math.reduce_sum(temporal_basis, axis = 1, keepdims = True); # temporal_basis.shape = (trajectory_length, n_basis)
    return tf.cast(tf.transpose(temporal_basis), dtype = tf.float32); # shape = (n_basis, trajectory_length)
  def call(self, inputs):
    # 1) get beta vector for diffusion
    # beta_perturb = weighted sum of temporal_basis
    beta_perturb = tf.squeeze(tf.linalg.matmul(self.temporal_basis, self.beta_perturb_coefficients, transpose_a = True), axis = -1); # beta_perturb.shape = (trajectory_length,)
    # beta_baseline = [1/1000,...,1/2]
    beta_baseline = 1 / tf.linspace(self.trajectory_length, 2, self.trajectory_length); # beta_baseline.shape = (trajectory_length,)
    beta_baseline_offset = tf.math.log(beta_baseline / (1 - beta_baseline)); # beta_baseline.shape = (trajectory_length,)
    beta = tf.math.sigmoid(beta_perturb + tf.cast(beta_baseline_offset, dtype = tf.float32)); # beta.shape = (trajectory_length,)
    min_beta = tf.concat([tf.ones((1,)) * (1e-6 + self.step1_beta),tf.ones((self.trajectory_length - 1,)) * 1e-6], axis = 0); # min_beta.shape = (trajectory_length,)
    beta = min_beta + beta * (1. - min_beta - 1e-5); # beta.shape = (trajectory_length,)
    # 2) get beta weight vector
    t_weight = tf.math.maximum(1. - (tf.expand_dims(inputs, axis = -1) - tf.tile(tf.expand_dims(tf.range(self.trajectory_length, dtype = tf.float32), axis = 0), (tf.shape(inputs)[0], 1))), 0); # diff.shape = (batch, trajectory_length,)
    return tf.squeeze(tf.linalg.matmul(t_weight, tf.expand_dims(beta, axis = -1)), axis = -1); # shape = (batch,)
  def get_config(self):
    config = super(Beta, self).get_config();
    config['trajectory_length'] = self.trajectory_length;
    config['n_basis'] = self.n_basis;
    config['step1_beta'] = self.step1_beta;
  @classmethod
  def from_config(cls, config):
    return cls(**config);

if __name__ == "__main__":
  beta_forward = BetaForward();
  b = beta_forward(tf.random.normal(shape = (4,), dtype = tf.float32));
  print(b.shape)
  inputs = np.random.normal(size = (10, 64, 64, 3));
  model = MLPConvolution(shape = (64, 64), n_temporal_basis = 10, n_layers_dense_lower = 4, n_hidden_dense_lower = 500, n_hidden_dense_lower_output = 2, n_layers_dense_upper = 2, n_hidden_dense_upper = 20, n_layers = 4, n_colors = 3, n_hidden = 20, n_scales = 1);
  outputs = model(inputs);
  print(outputs.shape);
  model.save('model.h5');

