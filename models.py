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
      dense_results = tf.keras.layers.Dense(kwargs.get('n_colors') * 2 * n_temporal_basis, kernel_initializer = tf.keras.initializers.Orthogonal(), bias_initializer = tf.keras.initializers.Constant())(dense_results);
  # NOTE: dense_results.shape = (batch, height, width, n_colors * 2 * n_temporal_basis)
  return tf.keras.Model(inputs = inputs, outputs = dense_results);

def Decoder(trajectory_length = 1000, **kwargs):
  inputs = tf.keras.Input((kwargs.get('shape')[0], kwargs.get('shape')[1], kwargs.get('n_colors'))); # inputs.shape = (batch, height, width, n_colors)
  beta = tf.keras.Input((trajectory_length,)); # beta.shape = (batch, trajectory_length)
  results = MLPConvolution(**kwargs)(inputs); # results.shape = (batch, height, width, n_colors * 2 * n_temporal_basis)
  results = tf.keras.layers.Reshape((kwargs.get('shape')[0], kwargs.get('shape')[1], kwargs.get('n_colors'), 2, kwargs.get('n_temporal_basis')))(results);
  def generate_temporal_basis(trajectory_length, n_basis):
    # sample n_basis soft one-hot basises
    temporal_basis = tf.zeros((trajectory_length, n_basis));
    xx = tf.linspace(-1, 1, trajectory_length); # xx.shape = (trajectory_length,)
    x_centers = tf.linspace(-1, 1, n_basis); # x_centers.shape = (n_basis,)
    width = (x_centers[1] - x_centers[0]) / 2;
    temporal_basis = tf.stack([tf.math.exp(-(xx - x_centers[ii])**2 / (2 * width**2)) for ii in range(n_basis)], axis = 1); # temporal_basis.shape = (trajectory_length, n_basis)
    temporal_basis /= tf.math.reduce_sum(temporal_basis, axis = 1, keepdims = True); # temporal_basis.shape = (trajectory_length, n_basis)
    return tf.cast(tf.transpose(temporal_basis), dtype = tf.float32); # shape = (n_basis, trajectory_length)
  # conv_mlp_outputs{batch, height, width, n_colors, 2, n_basis} * temporal_basis{n_basis, trajectory_length} = results.shape{batch, height, width, n_colors, 2, trajectory_length}
  temporal_basis = tf.keras.layers.Lambda(lambda x, t, b: generate_temporal_basis(t, b), arguments = {'t': trajectory_length, 'b': kwargs.get('n_temporal_basis')})(inputs); # temporal_basis.shape = (n_basis, trajectory_length)
  results = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([results, temporal_basis]); # results.shape = (batch, height, width, n_colors, 2, trajectory_length)
  mu_coeff = tf.keras.layers.Lambda(lambda x: x[...,0,:])(results); # mu.shape = (batch, height, width, n_colors, trajectory_length)
  beta_coeff = tf.keras.layers.Lambda(lambda x: x[...,1,:])(results); # beta.shape = (batch, height, width, n_colors, trajectory_length)
  # beta and mean for reverse(decoder) are perturbation of beta and mean for forward(encoder) process
  # beta_reverse = sigmoid(beta_coeff + log(beta / (1 - beta)))
  # mean = x * sqrt(1 - beta) + mu_coeff * sqrt(beta)
  # std = sqrt(beta_reverse)
  beta_reverse = tf.keras.layers.Lambda(lambda x, t: tf.math.sigmoid(x[0] / tf.math.sqrt(tf.cast(t, dtype = tf.float32)) + tf.reshape(tf.math.log(x[1] / (1 - x[1])), (-1, 1, 1, 1, t))), arguments = {'t': trajectory_length})([beta_coeff, beta]); # beta_reverse.shape = (batch, height, width, n_colors, trajectory_length)
  mean = tf.keras.layers.Lambda(lambda x, t: tf.expand_dims(x[0], axis = -1) * tf.math.sqrt(1 - tf.reshape(x[1], (-1, 1, 1, 1, t))) + x[2] * tf.math.sqrt(tf.reshape(x[1], (-1, 1, 1, 1, t))), arguments = {'t': trajectory_length})([inputs, beta, mu_coeff]); # mean.shape = (batch, height, width, n_colors, trajectory_length)
  std = tf.keras.layers.Lambda(lambda x: tf.math.sqrt(x))(beta_reverse); # std.shape = (batch, height, width, n_colors, trajectory_length)
  samples = tfp.layers.DistributionLambda(lambda x: tfp.distributions.Independent(tfp.distributions.Normal(loc = x[0], scale = x[1])))([mean, std]);
  return tf.keras.Model(inputs = (inputs, beta), outputs = samples);

def Encoder(trajectory_length = 1000, **kwargs):
  inputs = tf.keras.Input((kwargs.get('shape')[0], kwargs.get('shape')[1], kwargs.get('n_colors', 3))); # inputs.shape = (batch, height, width, n_colors)
  beta = tf.keras.Input((trajectory_length,)); # beta.shape = (batch, trajectory_length)
  # calculate smoothed betas
  t = tf.keras.layers.Lambda(lambda x, t: tf.range(1, t + 1, dtype = tf.float32), arguments = {'t': trajectory_length})(inputs); # t.shape = (trajectory_length,)
  diff = tf.keras.layers.Lambda(lambda x, t: tf.math.abs(tf.expand_dims(x, axis = 1) - tf.expand_dims(tf.range(t, dtype = tf.float32), axis = 0)), arguments = {'t': trajectory_length})(t); # diff.shape = (trajectory_length, trajectory_length)
  soft_picker = tf.keras.layers.Lambda(lambda x: tf.math.maximum(1 - diff, 0.))(diff); # soft_picker.shape = (trajectory_length, trajectory_length), each row is soft picker
  smoothed_beta = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.linalg.matmul(x[0], x[1], transpose_b = True)))([soft_picker, beta]); # smoothed_beta.shape = (batch, trajectory_length)
  # 
  #smoothed_alpha = tf.keras.layers.Lambda(lambda x: 1 - x)(smoothed_beta); # smoothed_alpha.shape = (batch, trajectory_length)
  alpha = tf.keras.layers.Lambda(lambda x: 1 - x)(beta); # alpha.shape = (batch, trajectory_length)
  alpha_cumprod = tf.keras.layers.Lambda(lambda x: tf.math.cumprod(x, axis = -1))(alpha); # alpha_cumprod.shape = (batch, trajectory_length)
  smoothed_alpha_cumprod = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.linalg.matmul(x[0], x[1], transpose_b = True)))([soft_picker, alpha_cumprod]); # smoothed_alpha_cumprod.shape = (batch, trajectory_length)
  mean = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[0], axis = -1) * tf.math.sqrt(tf.reshape(x[1], (tf.shape(x[1])[0], 1, 1, 1, tf.shape(x[1])[-1]))))([inputs, smoothed_alpha_cumprod]); # mean.shape = (batch, height, width, n_colors, trajectory_length)
  std = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.math.sqrt(1 - x[1]), (tf.shape(x[1])[0], 1, 1, 1, tf.shape(x[1])[-1])), (1, tf.shape(x[0])[1], tf.shape(x[0])[2], tf.shape(x[0])[3], 1)))([inputs, smoothed_alpha_cumprod]); # std.shape = (batch, height, width, n_colors, trajectory_length)
  samples = tfp.layers.DistributionLambda(lambda x: tfp.distributions.Independent(tfp.distributions.Normal(loc = x[0], scale = x[1])))([mean, std]); # sample.shape = (batch, height, width, n_colors, trajectory_length)
  return tf.keras.Model(inputs = (inputs, beta), outputs = samples);

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
    # sample n_basis soft one-hot basises
    temporal_basis = tf.zeros((self.trajectory_length, self.n_basis));
    xx = tf.linspace(-1, 1, self.trajectory_length); # xx.shape = (trajectory_length,)
    x_centers = tf.linspace(-1, 1, self.n_basis); # x_centers.shape = (n_basis,)
    width = (x_centers[1] - x_centers[0]) / 2;
    temporal_basis = tf.stack([tf.math.exp(-(xx - x_centers[ii])**2 / (2 * width**2)) for ii in range(self.n_basis)], axis = 1); # temporal_basis.shape = (trajectory_length, n_basis)
    temporal_basis /= tf.math.reduce_sum(temporal_basis, axis = 1, keepdims = True); # temporal_basis.shape = (trajectory_length, n_basis)
    return tf.cast(tf.transpose(temporal_basis), dtype = tf.float32); # shape = (n_basis, trajectory_length)
  def call(self, inputs):
    # 1) calculate a array of beta as candidates
    # beta_perturb = weighted sum of temporal_basis
    beta_perturb = tf.squeeze(tf.linalg.matmul(self.temporal_basis, self.beta_perturb_coefficients, transpose_a = True), axis = -1); # beta_perturb.shape = (trajectory_length,)
    # NOTE: beta = beta_baseline = [1/1000,...,1/2] if beta_perturb = 0, because sigmoid(log(x/(1-x))) = x
    # beta_perturb is a learnable perturbation to beta steps
    beta_baseline = 1. / tf.linspace(self.trajectory_length, 2, self.trajectory_length); # beta_baseline.shape = (trajectory_length,)
    beta_baseline = tf.math.log(beta_baseline / (1 - beta_baseline)); # beta_baseline.shape = (trajectory_length,)
    beta = tf.math.sigmoid(beta_perturb + tf.cast(beta_baseline, dtype = tf.float32)); # beta.shape = (trajectory_length,)
    min_beta = tf.concat([tf.ones((1,)) * (1e-6 + self.step1_beta),tf.ones((self.trajectory_length - 1,)) * 1e-6], axis = 0); # min_beta.shape = (trajectory_length,)
    beta = min_beta + beta * (1. - min_beta - 1e-5); # beta.shape = (trajectory_length,)
    return beta;
  def get_config(self):
    config = super(Beta, self).get_config();
    config['trajectory_length'] = self.trajectory_length;
    config['n_basis'] = self.n_basis;
    config['step1_beta'] = self.step1_beta;
  @classmethod
  def from_config(cls, config):
    return cls(**config);

if __name__ == "__main__":
  encoder = Encoder(shape = (64, 64), n_temporal_basis = 10, n_layers_dense_lower = 4, n_hidden_dense_lower = 500, n_hidden_dense_lower_output = 2, n_layers_dense_upper = 2, n_hidden_dense_upper = 20, n_layers = 4, n_colors = 3, n_hidden = 20, n_scales = 1);
  decoder = Decoder(shape = (64, 64), n_temporal_basis = 10, n_layers_dense_lower = 4, n_hidden_dense_lower = 500, n_hidden_dense_lower_output = 2, n_layers_dense_upper = 2, n_hidden_dense_upper = 20, n_layers = 4, n_colors = 3, n_hidden = 20, n_scales = 1);
  inputs = np.random.normal(size = (10, 64, 64, 3));
  beta = np.random.normal(size = (1000,));
  sample = encoder([inputs, beta]);
  print(sample.shape);
  encoder.save('encoder.h5');
  sample = decoder([inputs, beta]);
  print(sample.shape);
  decoder.save('decoder.h5');
  beta_forward = BetaForward();
  beta = beta_forward([]);
  print(beta.shape);
