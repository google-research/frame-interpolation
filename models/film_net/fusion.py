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
"""The final fusion stage for the film_net frame interpolator.

The inputs to this module are the warped input images, image features and
flow fields, all aligned to the target frame (often midway point between the
two original inputs). The output is the final image. FILM has no explicit
occlusion handling -- instead using the abovementioned information this module
automatically decides how to best blend the inputs together to produce content
in areas where the pixels can only be borrowed from one of the inputs.

Similarly, this module also decides on how much to blend in each input in case
of fractional timestep that is not at the halfway point. For example, if the two
inputs images are at t=0 and t=1, and we were to synthesize a frame at t=0.1,
it often makes most sense to favor the first input. However, this is not
always the case -- in particular in occluded pixels.

The architecture of the Fusion module follows U-net [1] architecture's decoder
side, e.g. each pyramid level consists of concatenation with upsampled coarser
level output, and two 3x3 convolutions.

The upsampling is implemented as 'resize convolution', e.g. nearest neighbor
upsampling followed by 2x2 convolution as explained in [2]. The classic U-net
uses max-pooling which has a tendency to create checkerboard artifacts.

[1] Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image
    Segmentation, 2015, https://arxiv.org/pdf/1505.04597.pdf
[2] https://distill.pub/2016/deconv-checkerboard/
"""

from typing import List

from . import options
import tensorflow as tf


def _relu(x: tf.Tensor) -> tf.Tensor:
  return tf.nn.leaky_relu(x, alpha=0.2)


_NUMBER_OF_COLOR_CHANNELS = 3


class Fusion(tf.keras.layers.Layer):
  """The decoder."""

  def __init__(self, name: str, config: options.Options):
    super().__init__(name=name)

    # Each item 'convs[i]' will contain the list of convolutions to be applied
    # for pyramid level 'i'.
    self.convs: List[List[tf.keras.layers.Layer]] = []

    # Store the levels, so we can verify right number of levels in call().
    self.levels = config.fusion_pyramid_levels

    # Create the convolutions. Roughly following the feature extractor, we
    # double the number of filters when the resolution halves, but only up to
    # the specialized_levels, after which we use the same number of filters on
    # all levels.
    #
    # We create the convs in fine-to-coarse order, so that the array index
    # for the convs will correspond to our normal indexing (0=finest level).
    for i in range(config.fusion_pyramid_levels - 1):
      m = config.specialized_levels
      k = config.filters
      num_filters = (k << i) if i < m else (k << m)

      convs: List[tf.keras.layers.Layer] = []
      convs.append(
          tf.keras.layers.Conv2D(
              filters=num_filters, kernel_size=[2, 2], padding='same'))
      convs.append(
          tf.keras.layers.Conv2D(
              filters=num_filters,
              kernel_size=[3, 3],
              padding='same',
              activation=_relu))
      convs.append(
          tf.keras.layers.Conv2D(
              filters=num_filters,
              kernel_size=[3, 3],
              padding='same',
              activation=_relu))
      self.convs.append(convs)

    # The final convolution that outputs RGB:
    self.output_conv = tf.keras.layers.Conv2D(
        filters=_NUMBER_OF_COLOR_CHANNELS, kernel_size=1)

  def call(self, pyramid: List[tf.Tensor]) -> tf.Tensor:
    """Runs the fusion module.

    Args:
      pyramid: The input feature pyramid as list of tensors. Each tensor being
        in (B x H x W x C) format, with finest level tensor first.

    Returns:
      A batch of RGB images.
    Raises:
      ValueError, if len(pyramid) != config.fusion_pyramid_levels as provided in
        the constructor.
    """
    if len(pyramid) != self.levels:
      raise ValueError(
          'Fusion called with different number of pyramid levels '
          f'{len(pyramid)} than it was configured for, {self.levels}.')

    # As a slight difference to a conventional decoder (e.g. U-net), we don't
    # apply any extra convolutions to the coarsest level, but just pass it
    # to finer levels for concatenation. This choice has not been thoroughly
    # evaluated, but is motivated by the educated guess that the fusion part
    # probably does not need large spatial context, because at this point the
    # features are spatially aligned by the preceding warp.
    net = pyramid[-1]

    # Loop starting from the 2nd coarsest level:
    for i in reversed(range(0, self.levels - 1)):
      # Resize the tensor from coarser level to match for concatenation.
      level_size = tf.shape(pyramid[i])[1:3]
      net = tf.image.resize(net, level_size,
                            tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      net = self.convs[i][0](net)
      net = tf.concat([pyramid[i], net], axis=-1)
      net = self.convs[i][1](net)
      net = self.convs[i][2](net)
    net = self.output_conv(net)
    return net
