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
"""TF2 layer for estimating optical flow by a residual flow pyramid.

This approach of estimating optical flow between two images can be traced back
to [1], but is also used by later neural optical flow computation methods such
as SpyNet [2] and PWC-Net [3].

The basic idea is that the optical flow is first estimated in a coarse
resolution, then the flow is upsampled to warp the higher resolution image and
then a residual correction is computed and added to the estimated flow. This
process is repeated in a pyramid on coarse to fine order to successively
increase the resolution of both optical flow and the warped image.

In here, the optical flow predictor is used as an internal component for the
film_net frame interpolator, to warp the two input images into the inbetween,
target frame.

[1] F. Glazer, Hierarchical motion detection. PhD thesis, 1987.
[2] A. Ranjan and M. J. Black, Optical Flow Estimation using a Spatial Pyramid
    Network. 2016
[3] D. Sun X. Yang, M-Y. Liu and J. Kautz, PWC-Net: CNNs for Optical Flow Using
    Pyramid, Warping, and Cost Volume, 2017
"""

from typing import List

from . import options
from . import util
import tensorflow as tf


def _relu(x: tf.Tensor) -> tf.Tensor:
  return tf.nn.leaky_relu(x, alpha=0.2)


class FlowEstimator(tf.keras.layers.Layer):
  """Small-receptive field predictor for computing the flow between two images.

  This is used to compute the residual flow fields in PyramidFlowEstimator.

  Note that while the number of 3x3 convolutions & filters to apply is
  configurable, two extra 1x1 convolutions are appended to extract the flow in
  the end.

  Attributes:
    name: The name of the layer
    num_convs: Number of 3x3 convolutions to apply
    num_filters: Number of filters in each 3x3 convolution
  """

  def __init__(self, name: str, num_convs: int, num_filters: int):
    super(FlowEstimator, self).__init__(name=name)
    def conv(filters, size, name, activation=_relu):
      return tf.keras.layers.Conv2D(
          name=name,
          filters=filters,
          kernel_size=size,
          padding='same',
          activation=activation)

    self._convs = []
    for i in range(num_convs):
      self._convs.append(conv(filters=num_filters, size=3, name=f'conv_{i}'))
    self._convs.append(conv(filters=num_filters/2, size=1, name=f'conv_{i+1}'))
    # For the final convolution, we want no activation at all to predict the
    # optical flow vector values. We have done extensive testing on explicitly
    # bounding these values using sigmoid, but it turned out that having no
    # activation gives better results.
    self._convs.append(
        conv(filters=2, size=1, name=f'conv_{i+2}', activation=None))

  def call(self, features_a: tf.Tensor, features_b: tf.Tensor) -> tf.Tensor:
    """Estimates optical flow between two images.

    Args:
      features_a: per pixel feature vectors for image A (B x H x W x C)
      features_b: per pixel feature vectors for image B (B x H x W x C)

    Returns:
      A tensor with optical flow from A to B
    """
    net = tf.concat([features_a, features_b], axis=-1)
    for conv in self._convs:
      net = conv(net)
    return net


class PyramidFlowEstimator(tf.keras.layers.Layer):
  """Predicts optical flow by coarse-to-fine refinement.

  Attributes:
    name: The name of the layer
    config: Options for the film_net frame interpolator
  """

  def __init__(self, name: str, config: options.Options):
    super(PyramidFlowEstimator, self).__init__(name=name)
    self._predictors = []
    for i in range(config.specialized_levels):
      self._predictors.append(
          FlowEstimator(
              name=f'flow_predictor_{i}',
              num_convs=config.flow_convs[i],
              num_filters=config.flow_filters[i]))
    shared_predictor = FlowEstimator(
        name='flow_predictor_shared',
        num_convs=config.flow_convs[-1],
        num_filters=config.flow_filters[-1])
    for i in range(config.specialized_levels, config.pyramid_levels):
      self._predictors.append(shared_predictor)

  def call(self, feature_pyramid_a: List[tf.Tensor],
           feature_pyramid_b: List[tf.Tensor]) -> List[tf.Tensor]:
    """Estimates residual flow pyramids between two image pyramids.

    Each image pyramid is represented as a list of tensors in fine-to-coarse
    order. Each individual image is represented as a tensor where each pixel is
    a vector of image features.

    util.flow_pyramid_synthesis can be used to convert the residual flow
    pyramid returned by this method into a flow pyramid, where each level
    encodes the flow instead of a residual correction.

    Args:
      feature_pyramid_a: image pyramid as a list in fine-to-coarse order
      feature_pyramid_b: image pyramid as a list in fine-to-coarse order

    Returns:
      List of flow tensors, in fine-to-coarse order, each level encoding the
      difference against the bilinearly upsampled version from the coarser
      level. The coarsest flow tensor, e.g. the last element in the array is the
      'DC-term', e.g. not a residual (alternatively you can think of it being a
      residual against zero).
    """
    levels = len(feature_pyramid_a)
    v = self._predictors[-1](feature_pyramid_a[-1], feature_pyramid_b[-1])
    residuals = [v]
    for i in reversed(range(0, levels-1)):
      # Upsamples the flow to match the current pyramid level. Also, scales the
      # magnitude by two to reflect the new size.
      level_size = tf.shape(feature_pyramid_a[i])[1:3]
      v = tf.image.resize(images=2*v, size=level_size)
      # Warp feature_pyramid_b[i] image based on the current flow estimate.
      warped = util.warp(feature_pyramid_b[i], v)
      # Estimate the residual flow between pyramid_a[i] and warped image:
      v_residual = self._predictors[i](feature_pyramid_a[i], warped)
      residuals.append(v_residual)
      v = v_residual + v
    # Use reversed() to return in the 'standard' finest-first-order:
    return list(reversed(residuals))
