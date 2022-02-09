# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Feature loss based on 19 layer VGG network.


The network layers in the feature loss is weighted as described in
'Stereo Magnification: Learning View Synthesis using Multiplane Images',
Tinghui Zhou, Richard Tucker, Flynn, Graham Fyffe, Noah Snavely, SIGGRAPH 2018.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import tensorflow.compat.v1 as tf


def _build_net(layer_type: str,
               input_tensor: tf.Tensor,
               weight_bias: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
               name: Optional[str] = None) -> Callable[[Any], Any]:
  """Build a layer of the VGG network.

  Args:
    layer_type: A string, type of this layer.
    input_tensor: A tensor.
    weight_bias: A tuple of weight and bias.
    name: A string, name of this layer.

  Returns:
    A callable function of the tensorflow layer.

  Raises:
    ValueError: If layer_type is not conv or pool.
  """

  if layer_type == 'conv':
    return tf.nn.relu(
        tf.nn.conv2d(
            input_tensor,
            weight_bias[0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name=name) + weight_bias[1])
  elif layer_type == 'pool':
    return tf.nn.avg_pool(
        input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  else:
    raise ValueError('Unsupported layer %s' % layer_type)


def _get_weight_and_bias(vgg_layers: np.ndarray,
                         index: int) -> Tuple[tf.Tensor, tf.Tensor]:
  """Get the weight and bias of a specific layer from the VGG pretrained model.

  Args:
    vgg_layers: An array, the VGG pretrained model.
    index: An integer, index of the layer.

  Returns:
    weights: A tensor.
    bias: A tensor.
  """

  weights = vgg_layers[index][0][0][2][0][0]
  weights = tf.constant(weights)
  bias = vgg_layers[index][0][0][2][0][1]
  bias = tf.constant(np.reshape(bias, (bias.size)))

  return weights, bias


def _build_vgg19(image: tf.Tensor, model_filepath: str) -> Dict[str, tf.Tensor]:
  """Builds the VGG network given the model weights.

  The weights are loaded only for the first time this code is invoked.

  Args:
    image: A tensor, input image.
    model_filepath: A string, path to the VGG pretrained model.

  Returns:
    net: A dict mapping a layer name to a tensor.
  """

  with tf.variable_scope('vgg', reuse=True):
    net = {}
    if not hasattr(_build_vgg19, 'vgg_rawnet'):
      with tf.io.gfile.GFile(model_filepath, 'rb') as f:
        _build_vgg19.vgg_rawnet = sio.loadmat(f)
    vgg_layers = _build_vgg19.vgg_rawnet['layers'][0]
    imagenet_mean = tf.constant([123.6800, 116.7790, 103.9390],
                                shape=[1, 1, 1, 3])
    net['input'] = image - imagenet_mean
    net['conv1_1'] = _build_net(
        'conv',
        net['input'],
        _get_weight_and_bias(vgg_layers, 0),
        name='vgg_conv1_1')
    net['conv1_2'] = _build_net(
        'conv',
        net['conv1_1'],
        _get_weight_and_bias(vgg_layers, 2),
        name='vgg_conv1_2')
    net['pool1'] = _build_net('pool', net['conv1_2'])
    net['conv2_1'] = _build_net(
        'conv',
        net['pool1'],
        _get_weight_and_bias(vgg_layers, 5),
        name='vgg_conv2_1')
    net['conv2_2'] = _build_net(
        'conv',
        net['conv2_1'],
        _get_weight_and_bias(vgg_layers, 7),
        name='vgg_conv2_2')
    net['pool2'] = _build_net('pool', net['conv2_2'])
    net['conv3_1'] = _build_net(
        'conv',
        net['pool2'],
        _get_weight_and_bias(vgg_layers, 10),
        name='vgg_conv3_1')
    net['conv3_2'] = _build_net(
        'conv',
        net['conv3_1'],
        _get_weight_and_bias(vgg_layers, 12),
        name='vgg_conv3_2')
    net['conv3_3'] = _build_net(
        'conv',
        net['conv3_2'],
        _get_weight_and_bias(vgg_layers, 14),
        name='vgg_conv3_3')
    net['conv3_4'] = _build_net(
        'conv',
        net['conv3_3'],
        _get_weight_and_bias(vgg_layers, 16),
        name='vgg_conv3_4')
    net['pool3'] = _build_net('pool', net['conv3_4'])
    net['conv4_1'] = _build_net(
        'conv',
        net['pool3'],
        _get_weight_and_bias(vgg_layers, 19),
        name='vgg_conv4_1')
    net['conv4_2'] = _build_net(
        'conv',
        net['conv4_1'],
        _get_weight_and_bias(vgg_layers, 21),
        name='vgg_conv4_2')
    net['conv4_3'] = _build_net(
        'conv',
        net['conv4_2'],
        _get_weight_and_bias(vgg_layers, 23),
        name='vgg_conv4_3')
    net['conv4_4'] = _build_net(
        'conv',
        net['conv4_3'],
        _get_weight_and_bias(vgg_layers, 25),
        name='vgg_conv4_4')
    net['pool4'] = _build_net('pool', net['conv4_4'])
    net['conv5_1'] = _build_net(
        'conv',
        net['pool4'],
        _get_weight_and_bias(vgg_layers, 28),
        name='vgg_conv5_1')
    net['conv5_2'] = _build_net(
        'conv',
        net['conv5_1'],
        _get_weight_and_bias(vgg_layers, 30),
        name='vgg_conv5_2')

  return net


def _compute_error(fake: tf.Tensor,
                   real: tf.Tensor,
                   mask: Optional[tf.Tensor] = None) -> tf.Tensor:
  """Computes the L1 loss and reweights by the mask."""
  if mask is None:
    return tf.reduce_mean(tf.abs(fake - real))
  else:
    # Resizes mask to the same size as the input.
    size = (tf.shape(fake)[1], tf.shape(fake)[2])
    resized_mask = tf.image.resize(
        mask, size, method=tf.image.ResizeMethod.BILINEAR)
    return tf.reduce_mean(tf.abs(fake - real) * resized_mask)


# Normalized VGG loss (from
# https://github.com/CQFIO/PhotographicImageSynthesis)
def vgg_loss(image: tf.Tensor,
             reference: tf.Tensor,
             vgg_model_file: str,
             weights: Optional[Sequence[float]] = None,
             mask: Optional[tf.Tensor] = None) -> tf.Tensor:
  """Computes the VGG loss for an image pair.

  The VGG loss is the average feature vector difference between the two images.

  The input images must be in [0, 1] range in (B, H, W, 3) RGB format and
  the recommendation seems to be to have them in gamma space.

  The pretrained weights are publicly available in
    http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat

  Args:
    image: A tensor, typically the prediction from a network.
    reference: A tensor, the image to compare against, i.e. the golden image.
    vgg_model_file: A string, filename for the VGG 19 network weights in MATLAB
      format.
    weights: A list of float, optional weights for the layers. The defaults are
      from Qifeng Chen and Vladlen Koltun, "Photographic image synthesis with
      cascaded refinement networks," ICCV 2017.
    mask: An optional image-shape and single-channel tensor, the mask values are
      per-pixel weights to be applied on the losses. The mask will be resized to
      the same spatial resolution with the feature maps before been applied to
      the losses. When the mask value is zero, pixels near the boundary of the
      mask can still influence the loss if they fall into the receptive field of
      the VGG convolutional layers.

  Returns:
    vgg_loss: The linear combination of losses from five VGG layers.
  """

  if not weights:
    weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]

  vgg_ref = _build_vgg19(reference * 255.0, vgg_model_file)
  vgg_img = _build_vgg19(image * 255.0, vgg_model_file)
  p1 = _compute_error(vgg_ref['conv1_2'], vgg_img['conv1_2'], mask) * weights[0]
  p2 = _compute_error(vgg_ref['conv2_2'], vgg_img['conv2_2'], mask) * weights[1]
  p3 = _compute_error(vgg_ref['conv3_2'], vgg_img['conv3_2'], mask) * weights[2]
  p4 = _compute_error(vgg_ref['conv4_2'], vgg_img['conv4_2'], mask) * weights[3]
  p5 = _compute_error(vgg_ref['conv5_2'], vgg_img['conv5_2'], mask) * weights[4]

  final_loss = p1 + p2 + p3 + p4 + p5

  # Scale to range [0..1].
  final_loss /= 255.0

  return final_loss


def _compute_gram_matrix(input_features: tf.Tensor,
                         mask: tf.Tensor) -> tf.Tensor:
  """Computes Gram matrix of `input_features`.

  Gram matrix described in https://en.wikipedia.org/wiki/Gramian_matrix.

  Args:
    input_features: A tf.Tensor of shape (B, H, W, C) representing a feature map
      obtained by a convolutional layer of a VGG network.
    mask: A tf.Tensor of shape (B, H, W, 1) representing the per-pixel weights
      to be applied on the `input_features`. The mask will be resized to the
      same spatial resolution as the `input_featues`. When the mask value is
      zero, pixels near the boundary of the mask can still influence the loss if
      they fall into the receptive field of the VGG convolutional layers.

  Returns:
    A tf.Tensor of shape (B, C, C) representing the gram matrix of the masked
    `input_features`.
  """
  _, h, w, c = tuple([
      i if (isinstance(i, int) or i is None) else i.value
      for i in input_features.shape
  ])
  if mask is None:
    reshaped_features = tf.reshape(input_features, (-1, h * w, c))
  else:
    # Resize mask to match the shape of `input_features`
    resized_mask = tf.image.resize(
        mask, (h, w), method=tf.image.ResizeMethod.BILINEAR)
    reshaped_features = tf.reshape(input_features * resized_mask,
                                   (-1, h * w, c))
  return tf.matmul(
      reshaped_features, reshaped_features, transpose_a=True) / float(h * w)


def style_loss(image: tf.Tensor,
               reference: tf.Tensor,
               vgg_model_file: str,
               weights: Optional[Sequence[float]] = None,
               mask: Optional[tf.Tensor] = None) -> tf.Tensor:
  """Computes style loss as used in `A Neural Algorithm of Artistic Style`.

  Based on the work in https://github.com/cysmith/neural-style-tf. Weights are
  first initilaized to the inverse of the number of elements in each VGG layer
  considerd. After 1.5M iterations, they are rescaled to normalize the
  contribution of the Style loss to be equal to other losses (L1/VGG). This is
  based on the works of image inpainting (https://arxiv.org/abs/1804.07723)
  and frame prediction (https://arxiv.org/abs/1811.00684).

  The style loss is the average gram matrix difference between `image` and
  `reference`. The gram matrix is the inner product of a feature map of shape
  (B, H*W, C) with itself. Results in a symmetric gram matrix shaped (B, C, C).

  The input images must be in [0, 1] range in (B, H, W, 3) RGB format and
  the recommendation seems to be to have them in gamma space.

  The pretrained weights are publicly available in
    http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat

  Args:
    image: A tensor, typically the prediction from a network.
    reference: A tensor, the image to compare against, i.e. the golden image.
    vgg_model_file: A string, filename for the VGG 19 network weights in MATLAB
      format.
    weights: A list of float, optional weights for the layers. The defaults are
      from Qifeng Chen and Vladlen Koltun, "Photographic image synthesis with
      cascaded refinement networks," ICCV 2017.
    mask: An optional image-shape and single-channel tensor, the mask values are
      per-pixel weights to be applied on the losses. The mask will be resized to
      the same spatial resolution with the feature maps before been applied to
      the losses. When the mask value is zero, pixels near the boundary of the
      mask can still influence the loss if they fall into the receptive field of
      the VGG convolutional layers.

  Returns:
    Style loss, a linear combination of gram matrix L2 differences of from five
    VGG layer features.
  """

  if not weights:
    weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]

  vgg_ref = _build_vgg19(reference * 255.0, vgg_model_file)
  vgg_img = _build_vgg19(image * 255.0, vgg_model_file)

  p1 = tf.reduce_mean(
      tf.squared_difference(
          _compute_gram_matrix(vgg_ref['conv1_2'] / 255.0, mask),
          _compute_gram_matrix(vgg_img['conv1_2'] / 255.0, mask))) * weights[0]
  p2 = tf.reduce_mean(
      tf.squared_difference(
          _compute_gram_matrix(vgg_ref['conv2_2'] / 255.0, mask),
          _compute_gram_matrix(vgg_img['conv2_2'] / 255.0, mask))) * weights[1]
  p3 = tf.reduce_mean(
      tf.squared_difference(
          _compute_gram_matrix(vgg_ref['conv3_2'] / 255.0, mask),
          _compute_gram_matrix(vgg_img['conv3_2'] / 255.0, mask))) * weights[2]
  p4 = tf.reduce_mean(
      tf.squared_difference(
          _compute_gram_matrix(vgg_ref['conv4_2'] / 255.0, mask),
          _compute_gram_matrix(vgg_img['conv4_2'] / 255.0, mask))) * weights[3]
  p5 = tf.reduce_mean(
      tf.squared_difference(
          _compute_gram_matrix(vgg_ref['conv5_2'] / 255.0, mask),
          _compute_gram_matrix(vgg_img['conv5_2'] / 255.0, mask))) * weights[4]

  final_loss = p1 + p2 + p3 + p4 + p5

  return final_loss
