# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Network architectures used in the StyleGAN paper."""

import numpy as np
import tensorflow as tf
import math
import dnnlib
import dnnlib.tflib as tflib

# NOTE: Do not import any application-specific modules here!
# Specify all network parameters as kwargs.

#----------------------------------------------------------------------------
# Primitive ops for manipulating 4D activation tensors.
# The gradients of these are not necessary efficient or even meaningful.

def _blur2d(x, f=[1,2,1], normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[1]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0,0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16 :(
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, 1, stride, stride]
    x = tf.nn.depthwise_conv2d_native(x, f, strides=strides, padding='SAME', data_format='NCHW')
    x = tf.cast(x, orig_dtype)
    return x

def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = x.shape
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = tf.tile(x, [1, 1, 1, factor, 1, factor])
    x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x

def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2:# and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, 1, factor, factor]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW')

#----------------------------------------------------------------------------
# High-level ops for manipulating 4D activation tensors.
# The gradients of these are meant to be as efficient as possible.

def blur2d(x, f=[1,2,1], normalize=True):
    with tf.variable_scope('Blur2D'):
        @tf.custom_gradient
        def func(x):
            y = _blur2d(x, f, normalize)
            @tf.custom_gradient
            def grad(dy):
                dx = _blur2d(dy, f, normalize, flip=True)
                return dx, lambda ddx: _blur2d(ddx, f, normalize)
            return y, grad
        return func(x)

def upscale2d(x, factor=2):
    with tf.variable_scope('Upscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _upscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, factor, gain=factor**2)
                return dx, lambda ddx: _upscale2d(ddx, factor)
            return y, grad
        return func(x)

def downscale2d(x, factor=2):
    with tf.variable_scope('Downscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _downscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _upscale2d(dy, factor, gain=1/factor**2)
                return dx, lambda ddx: _downscale2d(ddx, factor)
            return y, grad
        return func(x)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, lrmul=1, name_extra=''):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable('weight' + name_extra, shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, **kwargs):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Fused convolution + scaling.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel, fused_scale='auto', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) * 2 >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return conv2d(upscale2d(x), fmaps, kernel, **kwargs)

    # Fused => perform both ops simultaneously using tf.nn.conv2d_transpose().
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)
    w = tf.transpose(w, [0, 1, 3, 2]) # [kernel, kernel, fmaps_out, fmaps_in]
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

def conv2d_downscale2d(x, fmaps, kernel, fused_scale='auto', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return downscale2d(conv2d(x, fmaps, kernel, **kwargs))

    # Fused => perform both ops simultaneously using tf.nn.conv2d().
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')


def depthwise_conv2d(x, kernel, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    orig_dtype = x.dtype
    w = get_weight([kernel, kernel, x.shape[1].value, 1], **kwargs)
    w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    x = tf.cast(x, tf.float32)
    x = tf.nn.depthwise_conv2d_native(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')
    return tf.cast(x, orig_dtype)

def bottleneck_dw_conv2d(x, fmaps, kernel, bottleneck_div, **kwargs):
    bottleneck_fmaps = fmaps // bottleneck_div
    with tf.variable_scope('bottleneck_down'):
        x = conv2d(x, bottleneck_fmaps, kernel=1, **kwargs)
    with tf.variable_scope('dw_conv'):
        x = depthwise_conv2d(x, kernel=kernel, **kwargs)
    with tf.variable_scope('bottleneck_up'):
        x = conv2d(x, fmaps, kernel=1, **kwargs)
    return x

def flatten(x) :
    return tf.layers.flatten(x)

def hw_flatten(x) :
    batch_size, num_channels, h, w  = x.shape.as_list()
    return tf.reshape(x, [-1, h * w, num_channels])

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def add_timing_signal_2d(x, min_timescale=1.0, max_timescale=1.0e4):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase in one of the positional dimensions.
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  The use of relative position is possible because sin(a+b) and cos(a+b) can be
  experessed in terms of b, sin(a) and cos(a).
  x is a Tensor with n "positional" dimensions, e.g. one dimension for a
  sequence or two dimensions for an image
  We use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels // (n * 2). For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
  Args:
    x: a Tensor with shape [batch, d1 ... dn, channels]
    min_timescale: a float
    max_timescale: a float
  Returns:
    a Tensor the same shape as x.
  """
  with tf.variable_scope('timing_signals'):
      num_dims = len(x.get_shape().as_list()) - 2
      channels = shape_list(x)[-1]
      num_timescales = channels // (num_dims * 2)
      log_timescale_increment = (
          math.log(float(max_timescale) / float(min_timescale)) /
          (tf.to_float(num_timescales) - 1))
      inv_timescales = min_timescale * tf.exp(
          tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
      for dim in range(num_dims):
        length = shape_list(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in range(1 + dim):
          signal = tf.expand_dims(signal, 0)
        for _ in range(num_dims - 1 - dim):
          signal = tf.expand_dims(signal, -2)
        signal = tf.cast(signal, x.dtype)
        x += signal
      return x

def sn_non_local_block_sim(x, n_heads=1, use_sn=True, scope='attention', use_timing=True, **kwargs):
  with tf.variable_scope(scope):
    batch_size, num_channels, h, w = x.get_shape().as_list()
    bs, ncs, hs, ws = shape_list(x)
    lns = hs * ws
    dns = lns // 4
    location_num = h * w
    downsampled_num = location_num // 4

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, n_heads), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    if use_timing:
        with tf.variable_scope('timing'):
            #We need to rearrange channels so they correlate with timing signal frequencies
            tp_in = conv2d(x, num_channels, kernel=1, **kwargs)
            tp_in = add_timing_signal_2d(tp_in)
    else:
        tp_in = x

    # theta path
    with tf.variable_scope('theta'):
        theta = conv2d(tp_in, (num_channels // 8), kernel=1, **kwargs)
        theta = tf.transpose(theta, [0,2,3,1])
        theta = tf.reshape(
            theta, [-1, location_num, num_channels // 8])
        if n_heads > 1:
            theta = split_heads(theta)

    # phi path
    with tf.variable_scope('phi'):
        phi = conv2d(tp_in, (num_channels // 8), kernel=1, **kwargs)
        phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2, data_format='channels_first')
        phi = tf.transpose(phi, [0, 2, 3, 1])

        phi = tf.reshape(
            phi, [-1, downsampled_num, num_channels // 8])
        if n_heads > 1:
            phi = split_heads(phi)

    attn = tf.matmul(tf.cast(theta, tf.float32), tf.cast(phi, tf.float32), transpose_b=True)
    attn = tf.nn.softmax(attn)


    # g path
    with tf.variable_scope('g'):
        g = conv2d(x, (num_channels // 2), kernel=1, **kwargs)
        g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2, data_format='channels_first')
        g = tf.transpose(g, [0, 2, 3, 1])
        g = tf.reshape(
          g, [-1, downsampled_num, num_channels // 2])
        if n_heads > 1:
            g = split_heads(g)
    with tf.variable_scope('attn_g'):
        attn_g = tf.matmul(attn, tf.cast(g, tf.float32))
        attn_g = tf.cast(attn_g, x.dtype)
        if n_heads > 1:
            attn_g = merge_heads(attn_g)
        attn_g = tf.reshape(attn_g, [-1, h, w, num_channels // 2])
        attn_g = tf.transpose(attn_g, [0, 3, 1, 2])
        attn_g = conv2d(attn_g, num_channels, kernel=1, **kwargs)

    sigma = tf.get_variable(
        'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
    return x + tf.cast(sigma, x.dtype) * attn_g

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x, lrmul=1):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. More efficient than tf.nn.leaky_relu() and supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.variable_scope('LeakyReLU'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        @tf.custom_gradient
        def func(x):
            y = tf.maximum(x, x * alpha)
            @tf.custom_gradient
            def grad(dy):
                dx = tf.where(y >= 0, dy, dy * alpha)
                return dx, lambda ddx: tf.where(y >= 0, ddx, ddx * alpha)
            return y, grad
        return func(x)

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

#----------------------------------------------------------------------------
# Instance normalization.

def instance_norm(x, epsilon=1e-8):
    assert len(x.shape) == 4 # NCHW
    with tf.variable_scope('InstanceNorm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=[2,3], keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2,3], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
        return x

#----------------------------------------------------------------------------
# Add coordinates (as in coord-conv).


def add_coords(x, with_r = False):
    with tf.name_scope('coord'):
        batch_size = tf.shape(x)[0]
        x_dim = tf.shape(x)[2]
        y_dim = tf.shape(x)[3]

        xx_indices = tf.tile(
            tf.expand_dims(tf.expand_dims(tf.range(x_dim), 0), 0),
            [batch_size, y_dim, 1])
        xx_indices = tf.expand_dims(xx_indices, 1)

        yy_indices = tf.tile(
            tf.expand_dims(tf.reshape(tf.range(y_dim), (y_dim, 1)), 0),
            [batch_size, 1, x_dim])
        yy_indices = tf.expand_dims(yy_indices, 1)

        xx_indices = tf.divide(xx_indices, x_dim - 1)
        yy_indices = tf.divide(yy_indices, y_dim - 1)

        xx_indices = tf.cast(tf.subtract(tf.multiply(xx_indices, 2.), 1.),
                             dtype=x.dtype)
        yy_indices = tf.cast(tf.subtract(tf.multiply(yy_indices, 2.), 1.),
                             dtype=x.dtype)

        processed_tensor = tf.concat([x, xx_indices, yy_indices], axis=1)

        if with_r:
            rr = tf.sqrt(tf.add(tf.square(xx_indices - 0.5),
                                tf.square(yy_indices - 0.5)))
            processed_tensor = tf.concat([processed_tensor, rr], axis=1)

        return processed_tensor

def depthwise_gated_self_modulation(x, kernel=13, depthwise_bottleneck_div=16, **kwargs):
    with tf.variable_scope('Modulator'):
        with tf.variable_scope('mod_conv'):
            mod = bottleneck_dw_conv2d(x, x.shape[1]*2, kernel=kernel, bottleneck_div=depthwise_bottleneck_div, **kwargs)
        with tf.variable_scope('mod_bias'):
            mod = apply_bias(mod)
        with tf.variable_scope('mod_act'):
            mod = leaky_relu(mod)
        mod = tf.reshape(mod, [-1, 2, x.shape[1], x.shape[2], x.shape[3]])
        return x * (mod[:,0] + 1) + mod[:,1]

#----------------------------------------------------------------------------
# Style modulation.

def style_mod(x, dlatent, **kwargs):
    with tf.variable_scope('StyleMod'):
        style = apply_bias(dense(dlatent, fmaps=x.shape[1]*2, gain=1, **kwargs))
        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        return x * (style[:,0] + 1) + style[:,1]

#----------------------------------------------------------------------------
# Noise input.

def apply_noise(x, noise_var=None, randomize_noise=True, noise_channels=2, noise_kernel=1):
    assert len(x.shape) == 4 # NCHW
    with tf.variable_scope('Noise'):
        if noise_var is None or randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], noise_channels, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_var, x.dtype)
        weight = tf.get_variable('weight', shape=[noise_kernel, noise_kernel, noise_channels, x.shape[1].value], initializer=tf.initializers.zeros())
        weight = tf.cast(weight, x.dtype)
        return x + tf.nn.conv2d(noise, weight, [1,1,1,1], 'SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Minibatch standard deviation.

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3]])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MncHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
        y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

#----------------------------------------------------------------------------
# Style-based generator used in the StyleGAN paper.
# Composed of two sub-networks (G_mapping and G_synthesis) that are defined below.

def G_style(
    latents_in,                                     # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                                      # Second input: Conditioning labels [minibatch, label_size].
    truncation_psi          = 0.7,                  # Style strength multiplier for the truncation trick. None = disable.
    truncation_cutoff       = 8,                    # Number of layers for which to apply the truncation trick. None = disable.
    truncation_psi_val      = None,                 # Value for truncation_psi to use during validation.
    truncation_cutoff_val   = None,                 # Value for truncation_cutoff to use during validation.
    dlatent_avg_beta        = 0.995,                # Decay for tracking the moving average of W during training. None = disable.
    style_mixing_prob       = 0.9,                  # Probability of mixing styles during training. None = disable.
    is_training             = False,                # Network is under training? Enables and disables specific features.
    is_validation           = False,                # Network is under validation? Chooses which value to use for truncation_psi.
    is_template_graph       = False,                # True = template graph constructed by the Network class, False = actual evaluation.
    dtype                   = 'float16',    # Data type to use for activations and outputs.
    components              = dnnlib.EasyDict(),    # Container for sub-networks. Retained between calls.
    **kwargs):                                      # Arguments for sub-networks (G_mapping and G_synthesis).

    # Validate arguments.
    assert not is_training or not is_validation
    assert isinstance(components, dnnlib.EasyDict)
    if is_validation:
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training or (truncation_cutoff is not None and not tflib.is_tf_expression(truncation_cutoff) and truncation_cutoff <= 0):
        truncation_cutoff = None
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    if not is_training or (style_mixing_prob is not None and not tflib.is_tf_expression(style_mixing_prob) and style_mixing_prob <= 0):
        style_mixing_prob = None

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network('G_synthesis', func_name=G_synthesis, **kwargs)
    num_layers = components.synthesis.input_shape[1]
    dlatent_size = components.synthesis.input_shape[2]
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_mapping', func_name=G_mapping, dlatent_broadcast=num_layers, **kwargs)

    # Setup variables.
    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)
    dlatent_avg = tf.get_variable('dlatent_avg', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False)

    # Evaluate mapping network.
    dlatents = components.mapping.get_output_for(latents_in, labels_in, **kwargs)

    # Update moving average of W.
    if dlatent_avg_beta is not None:
        with tf.variable_scope('DlatentAvg'):
            batch_avg = tf.reduce_mean(tf.cast(dlatents[:, 0], tf.float32), axis=0)
            update_op = tf.assign(dlatent_avg, tflib.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    # Perform style mixing regularization.
    if style_mixing_prob is not None:
        with tf.name_scope('StyleMix'):
            latents2 = tf.random_normal(tf.shape(latents_in))
            dlatents2 = components.mapping.get_output_for(latents2, labels_in, **kwargs)
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = num_layers - tf.cast(lod_in, tf.int32) * 2
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
                lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)

    # Apply truncation trick.
    if truncation_psi is not None and truncation_cutoff is not None:
        with tf.variable_scope('Truncation'):
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_idx.shape, dtype=np.float16)
            coefs = tf.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
            dlatents = tf.cast(dlatents, tf.float32)
            coefs = tf.cast(coefs, tf.float32)
            dlatent_avg = tf.cast(dlatent_avg, tf.float32)
            dlatents = tflib.lerp(dlatent_avg, dlatents, coefs)
            dlatents = tf.cast(dlatents, dtype)

    # Evaluate synthesis network.
    with tf.control_dependencies([tf.assign(components.synthesis.find_var('lod'), lod_in)]):
        images_out = components.synthesis.get_output_for(dlatents, force_clean_graph=is_template_graph, **kwargs)
    return tf.identity(images_out, name='images_out')

#----------------------------------------------------------------------------
# Mapping network used in the StyleGAN paper.

def G_mapping(
    latents_in,                             # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                              # Second input: Conditioning labels [minibatch, label_size].
    latent_size             = 512,          # Latent vector (Z) dimensionality.
    label_size              = 0,            # Label dimensionality, 0 if no labels.
    dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
    dlatent_broadcast       = None,         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
    mapping_layers          = 8,            # Number of mapping layers.
    mapping_fmaps           = 512,          # Number of activations in the mapping layers.
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
    mapping_nonlinearity    = 'lrelu',      # Activation function: 'relu', 'lrelu'.
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_wscale              = True,         # Enable equalized learning rate?
    normalize_latents       = True,         # Normalize latent vectors (Z) before feeding them to the mapping layers?
    dtype                   = 'float16',    # Data type to use for activations and outputs.
    **_kwargs):                             # Ignore unrecognized keyword args.

    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[mapping_nonlinearity]

    # Inputs.
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    # Embed labels and concatenate them with latents.
    if label_size:
        with tf.variable_scope('LabelConcat'):
            w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
            y = tf.matmul(labels_in, tf.cast(w, dtype))
            x = tf.concat([x, y], axis=1)

    # Normalize latents.
    if normalize_latents:
        x = pixel_norm(x, epsilon=pixelnorm_epsilon)

    # Mapping layers.
    for layer_idx in range(mapping_layers):
        with tf.variable_scope('Dense%d' % layer_idx):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            x = dense(x, fmaps=fmaps, gain=gain, use_wscale=use_wscale, lrmul=mapping_lrmul)
            x = apply_bias(x, lrmul=mapping_lrmul)
            x = act(x)

    # Broadcast.
    if dlatent_broadcast is not None:
        with tf.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')

#----------------------------------------------------------------------------
# Synthesis network used in the StyleGAN paper.

def G_synthesis(
    dlatents_in,                        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
    num_channels        = 3,            # Number of output color channels.
    resolution          = 1024,         # Output resolution.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    use_styles          = True,         # Enable style inputs?
    const_input_layer   = True,         # First layer is a learned constant?
    use_noise           = True,         # Enable noise inputs?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu'
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
    use_instance_norm   = True,         # Enable instance normalization?
    use_attention       = False,
    use_coords          = False,
    use_timing_signals  = False,
    noise_channels      = 1,
    self_mod_stages     = [],
    self_mod_kernels    = {7:13,8:7},
    self_mod_divs       = {7:16,8:8},
    dtype               = 'float16',    # Data type to use for activations and outputs.
    fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
    blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
    structure           = 'auto',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    force_clean_graph   = False,        # True = construct a clean graph that looks nice in TensorBoard, False = default behavior.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def blur(x): return blur2d(x, blur_filter) if blur_filter else x
    if is_template_graph: force_clean_graph = True
    if force_clean_graph: randomize_noise = False
    if structure == 'auto': structure = 'linear' if force_clean_graph else 'recursive'
    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[nonlinearity]
    num_layers = resolution_log2 * 2 - 2
    num_styles = num_layers if use_styles else 1
    images_out = None

    # Primary inputs.
    dlatents_in.set_shape([None, num_styles, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)

    # Noise inputs.
    noise_inputs = []
    if use_noise:
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, noise_channels, 2**res, 2**res]
            noise_inputs.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))

    # Things to do at the end of each layer.
    def layer_epilogue(x, layer_idx):
        if use_noise:
            x = apply_noise(x, noise_inputs[layer_idx], randomize_noise=randomize_noise, noise_channels=noise_channels)
        x = apply_bias(x)
        x = act(x)
        if use_pixel_norm:
            x = pixel_norm(x, epsilon=pixelnorm_epsilon)
        if use_instance_norm:
            x = instance_norm(x)
        if use_styles:
            x = style_mod(x, dlatents_in[:, layer_idx], use_wscale=use_wscale)
        return x

    # Early layers.
    with tf.variable_scope('4x4'):
        if const_input_layer:
            with tf.variable_scope('Const'):
                x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.ones())
                x = layer_epilogue(tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1]), 0)
        else:
            with tf.variable_scope('Dense'):
                x = dense(dlatents_in[:, 0], fmaps=nf(1)*16, gain=gain/4, use_wscale=use_wscale) # tweak gain to match the official implementation of Progressing GAN
                x = layer_epilogue(tf.reshape(x, [-1, nf(1), 4, 4]), 0)
        with tf.variable_scope('Conv'):
            x = layer_epilogue(conv2d(x, fmaps=nf(1), kernel=3, gain=gain, use_wscale=use_wscale), 1)

    # Building blocks for remaining layers.
    def block(res, x): # res = 3..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if use_attention and (2**res) == 64:
                x = sn_non_local_block_sim(x, gain=gain, use_wscale=use_wscale, use_timing=use_timing_signals)
            with tf.variable_scope('Conv0_up'):
                if use_coords:
                    x = add_coords(x)
                x = layer_epilogue(blur(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)), res*2-4)
            with tf.variable_scope('Conv1'):
                if use_coords:
                    x = add_coords(x)
                x = layer_epilogue(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale), res*2-3)
            if res in self_mod_stages:
                x = depthwise_gated_self_modulation(x, kernel=self_mod_kernels[res], depthwise_bottleneck_div=self_mod_divs[res], gain=gain, use_wscale=use_wscale)
            return x
    def torgb(res, x): # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

    # Fixed structure: simple and efficient, but does not support progressive growing.
    if structure == 'fixed':
        for res in range(3, resolution_log2 + 1):
            x = block(res, x)
        images_out = torgb(resolution_log2, x)

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        images_out = torgb(2, x)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(res, x)
            img = torgb(res, x)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = tflib.lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
        def grow(x, res, lod):
            y = block(res, x)
            img = lambda: upscale2d(torgb(res, y), 2**lod)
            img = cset(img, (lod_in > lod), lambda: upscale2d(tflib.lerp(torgb(res, y), upscale2d(torgb(res - 1, x)), lod_in - lod), 2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()
        images_out = grow(x, 3, resolution_log2 - 3)

    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')

#----------------------------------------------------------------------------
# Discriminator used in the StyleGAN paper.

def D_basic(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu',
    use_wscale          = True,         # Enable equalized learning rate?
    use_attention       = False,
    use_coords          = False,
    use_timing_signals  = False,
    mbstd_group_size    = 8,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float16',    # Data type to use for activations and outputs.
    fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
    blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
    structure           = 'auto',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def blur(x): return blur2d(x, blur_filter) if blur_filter else x
    if structure == 'auto': structure = 'linear' if is_template_graph else 'recursive'
    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[nonlinearity]

    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)
    scores_out = None

    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, gain=gain, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):

            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    if use_coords:
                        x = add_coords(x)
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Conv1_down'):
                    if use_coords:
                        x = add_coords(x)
                    x = act(apply_bias(conv2d_downscale2d(blur(x), fmaps=nf(res-2), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)))
                if use_attention and (2**res) == 64:
                    x = sn_non_local_block_sim(x, gain=gain, use_wscale=use_wscale, use_timing=use_timing_signals)
            else: # 4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
                with tf.variable_scope('Conv'):
                    if use_coords:
                        x = add_coords(x)
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=nf(res-2), gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=max(label_size, 1), gain=1, use_wscale=use_wscale))
            return x

    # Fixed structure: simple and efficient, but does not support progressive growing.
    if structure == 'fixed':
        x = fromrgb(images_in, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            x = block(x, res)
        scores_out = block(x, 2)

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = tflib.lerp_clip(x, y, lod_in - lod)
        scores_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: tflib.lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        scores_out = grow(2, resolution_log2 - 2)

    # Label conditioning from "Which Training Methods for GANs do actually Converge?"
    if label_size:
        with tf.variable_scope('LabelSwitch'):
            scores_out = tf.reduce_sum(scores_out * labels_in, axis=1, keepdims=True)

    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out

#----------------------------------------------------------------------------
