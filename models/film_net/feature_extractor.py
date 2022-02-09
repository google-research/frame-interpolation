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
"""TF2 layer for extracting image features for the film_net interpolator.

The feature extractor implemented here converts an image pyramid into a pyramid
of deep features. The feature pyramid serves a similar purpose as U-Net
architecture's encoder, but we use a special cascaded architecture described in
Multi-view Image Fusion [1].

For comprehensiveness, below is a short description of the idea. While the
description is a bit involved, the cascaded feature pyramid can be used just
like any image feature pyramid.

Why cascaded architeture?
=========================
To understand the concept it is worth reviewing a traditional feature pyramid
first: *A traditional feature pyramid* as in U-net or in many optical flow
networks is built by alternating between convolutions and pooling, starting
from the input image.

It is well known that early features of such architecture correspond to low
level concepts such as edges in the image whereas later layers extract
semantically higher level concepts such as object classes etc. In other words,
the meaning of the filters in each resolution level is different. For problems
such as semantic segmentation and many others this is a desirable property.

However, the asymmetric features preclude sharing weights across resolution
levels in the feature extractor itself and in any subsequent neural networks
that follow. This can be a downside, since optical flow prediction, for
instance is symmetric across resolution levels. The cascaded feature
architecture addresses this shortcoming.

How is it built?
================
The *cascaded* feature pyramid contains feature vectors that have constant
length and meaning on each resolution level, except few of the finest ones. The
advantage of this is that the subsequent optical flow layer can learn
synergically from many resolutions. This means that coarse level prediction can
benefit from finer resolution training examples, which can be useful with
moderately sized datasets to avoid overfitting.

The cascaded feature pyramid is built by extracting shallower subtree pyramids,
each one of them similar to the traditional architecture. Each subtree
pyramid S_i is extracted starting from each resolution level:

image resolution 0 -> S_0
image resolution 1 -> S_1
image resolution 2 -> S_2
...

If we denote the features at level j of subtree i as S_i_j, the cascaded pyramid
is constructed by concatenating features as follows (assuming subtree depth=3):

lvl
feat_0 = concat(                               S_0_0 )
feat_1 = concat(                         S_1_0 S_0_1 )
feat_2 = concat(                   S_2_0 S_1_1 S_0_2 )
feat_3 = concat(             S_3_0 S_2_1 S_1_2       )
feat_4 = concat(       S_4_0 S_3_1 S_2_2             )
feat_5 = concat( S_5_0 S_4_1 S_3_2                   )
   ....

In above, all levels except feat_0 and feat_1 have the same number of features
with similar semantic meaning. This enables training a single optical flow
predictor module shared by levels 2,3,4,5... . For more details and evaluation
see [1].

[1] Multi-view Image Fusion, Trinidad et al. 2019
"""

from typing import List

from . import options
import tensorflow as tf


def _relu(x: tf.Tensor) -> tf.Tensor:
  return tf.nn.leaky_relu(x, alpha=0.2)


def _conv(filters: int, name: str):
  return tf.keras.layers.Conv2D(
      name=name,
      filters=filters,
      kernel_size=3,
      padding='same',
      activation=_relu)


class SubTreeExtractor(tf.keras.layers.Layer):
  """Extracts a hierarchical set of features from an image.

  This is a conventional, hierarchical image feature extractor, that extracts
  [k, k*2, k*4... ] filters for the image pyramid where k=options.sub_levels.
  Each level is followed by average pooling.

  Attributes:
    name: Name for the layer
    config: Options for the fusion_net frame interpolator
  """

  def __init__(self, name: str, config: options.Options):
    super().__init__(name=name)
    k = config.filters
    n = config.sub_levels
    self.convs = []
    for i in range(n):
      self.convs.append(
          _conv(filters=(k << i), name='cfeat_conv_{}'.format(2 * i)))
      self.convs.append(
          _conv(filters=(k << i), name='cfeat_conv_{}'.format(2 * i + 1)))

  def call(self, image: tf.Tensor, n: int) -> List[tf.Tensor]:
    """Extracts a pyramid of features from the image.

    Args:
      image: tf.Tensor with shape BATCH_SIZE x HEIGHT x WIDTH x CHANNELS.
      n: number of pyramid levels to extract. This can be less or equal to
       options.sub_levels given in the __init__.
    Returns:
      The pyramid of features, starting from the finest level. Each element
      contains the output after the last convolution on the corresponding
      pyramid level.
    """
    head = image
    pool = tf.keras.layers.AveragePooling2D(
        pool_size=2, strides=2, padding='valid')
    pyramid = []
    for i in range(n):
      head = self.convs[2*i](head)
      head = self.convs[2*i+1](head)
      pyramid.append(head)
      if i < n-1:
        head = pool(head)
    return pyramid


class FeatureExtractor(tf.keras.layers.Layer):
  """Extracts features from an image pyramid using a cascaded architecture.

  Attributes:
    name: Name of the layer
    config: Options for the fusion_net frame interpolator
  """

  def __init__(self, name: str, config: options.Options):
    super().__init__(name=name)
    self.extract_sublevels = SubTreeExtractor('sub_extractor', config)
    self.options = config

  def call(self, image_pyramid: List[tf.Tensor]) -> List[tf.Tensor]:
    """Extracts a cascaded feature pyramid.

    Args:
      image_pyramid: Image pyramid as a list, starting from the finest level.
    Returns:
      A pyramid of cascaded features.
    """
    sub_pyramids = []
    for i in range(len(image_pyramid)):
      # At each level of the image pyramid, creates a sub_pyramid of features
      # with 'sub_levels' pyramid levels, re-using the same SubTreeExtractor.
      # We use the same instance since we want to share the weights.
      #
      # However, we cap the depth of the sub_pyramid so we don't create features
      # that are beyond the coarsest level of the cascaded feature pyramid we
      # want to generate.
      capped_sub_levels = min(len(image_pyramid) - i, self.options.sub_levels)
      sub_pyramids.append(
          self.extract_sublevels(image_pyramid[i], capped_sub_levels))
    # Below we generate the cascades of features on each level of the feature
    # pyramid. Assuming sub_levels=3, The layout of the features will be
    # as shown in the example on file documentation above.
    feature_pyramid = []
    for i in range(len(image_pyramid)):
      features = sub_pyramids[i][0]
      for j in range(1, self.options.sub_levels):
        if j <= i:
          features = tf.concat([features, sub_pyramids[i - j][j]], axis=-1)
      feature_pyramid.append(features)
    return feature_pyramid
