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
"""Dataset augmentation for frame interpolation."""
from typing import Callable, Dict, List

import gin.tf
import numpy as np
import tensorflow as tf
import tensorflow.math as tfm
import tensorflow_addons.image as tfa_image

_PI = 3.141592653589793


def _rotate_flow_vectors(flow: tf.Tensor, angle_rad: float) -> tf.Tensor:
  r"""Rotate the (u,v) vector of each pixel with angle in radians.

  Flow matrix system of coordinates.
  . . . . u (x)
  .
  .
  . v (-y)

  Rotation system of coordinates.
  . y
  .
  .
  . . . . x
  Args:
    flow: Flow map which has been image-rotated.
    angle_rad: The rotation angle in radians.

  Returns:
    A flow with the same map but each (u,v) vector rotated by angle_rad.
  """
  u, v = tf.split(flow, 2, axis=-1)
  # rotu = u * cos(angle) - (-v) * sin(angle)
  rot_u = tfm.cos(angle_rad) * u + tfm.sin(angle_rad) * v
  # rotv = -(u * sin(theta) + (-v) * cos(theta))
  rot_v = -tfm.sin(angle_rad) * u + tfm.cos(angle_rad) * v
  return tf.concat((rot_u, rot_v), axis=-1)


def flow_rot90(flow: tf.Tensor, k: int) -> tf.Tensor:
  """Rotates a flow by a multiple of 90 degrees.

  Args:
    flow: The flow image shaped (H, W, 2) to rotate by multiples of 90 degrees.
    k: The multiplier factor.

  Returns:
    A flow image of the same shape as the input rotated by multiples of 90
    degrees.
  """
  angle_rad = tf.cast(k, dtype=tf.float32) * 90. * (_PI/180.)
  flow = tf.image.rot90(flow, k)
  return _rotate_flow_vectors(flow, angle_rad)


def rotate_flow(flow: tf.Tensor, angle_rad: float) -> tf.Tensor:
  """Rotates a flow by a the provided angle in radians.

  Args:
    flow: The flow image shaped (H, W, 2) to rotate by multiples of 90 degrees.
    angle_rad: The angle to ratate the flow in radians.

  Returns:
    A flow image of the same shape as the input rotated by the provided angle in
    radians.
  """
  flow = tfa_image.rotate(
      flow,
      angles=angle_rad,
      interpolation='bilinear',
      fill_mode='reflect')
  return _rotate_flow_vectors(flow, angle_rad)


def flow_flip(flow: tf.Tensor) -> tf.Tensor:
  """Flips a flow left to right.

  Args:
    flow: The flow image shaped (H, W, 2) to flip left to right.

  Returns:
    A flow image of the same shape as the input flipped left to right.
  """
  flow = tf.image.flip_left_right(tf.identity(flow))
  flow_u, flow_v = tf.split(flow, 2, axis=-1)
  return tf.stack([-1 * flow_u, flow_v], axis=-1)


def random_image_rot90(images: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
  """Rotates a stack of images by a random multiples of 90 degrees.

  Args:
    images: A tf.Tensor shaped (H, W, num_channels) of images stacked along the
      channel's axis.
  Returns:
    A tf.Tensor of the same rank as the `images` after random rotation by
    multiples of 90 degrees applied counter-clock wise.
  """
  random_k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
  for key in images:
    images[key] = tf.image.rot90(images[key], k=random_k)
  return images


def random_flip(images: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
  """Flips a stack of images randomly.

  Args:
    images: A tf.Tensor shaped (H, W, num_channels) of images stacked along the
      channel's axis.

  Returns:
    A tf.Tensor of the images after random left to right flip.
  """
  prob = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
  prob = tf.cast(prob, tf.bool)

  def _identity(image):
    return image

  def _flip_left_right(image):
    return tf.image.flip_left_right(image)

  # pylint: disable=cell-var-from-loop
  for key in images:
    images[key] = tf.cond(prob, lambda: _flip_left_right(images[key]),
                          lambda: _identity(images[key]))
  return images


def random_reverse(images: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
  """Reverses a stack of images randomly.

  Args:
    images: A dictionary of tf.Tensors, each shaped (H, W, num_channels), with
      each tensor being a stack of iamges along the last channel axis.

  Returns:
    A dictionary of tf.Tensors, each shaped the same as the input images dict.
  """
  prob = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
  prob = tf.cast(prob, tf.bool)

  def _identity(images):
    return images

  def _reverse(images):
    images['x0'], images['x1'] = images['x1'], images['x0']
    return images

  return tf.cond(prob, lambda: _reverse(images), lambda: _identity(images))


def random_rotate(images: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
  """Rotates image randomly with [-45 to 45 degrees].

  Args:
    images: A tf.Tensor shaped (H, W, num_channels) of images stacked along the
      channel's axis.

  Returns:
    A tf.Tensor of the images after random rotation with a bound of -72 to 72
    degrees.
  """
  prob = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
  prob = tf.cast(prob, tf.float32)
  random_angle = tf.random.uniform((),
                                   minval=-0.25 * np.pi,
                                   maxval=0.25 * np.pi,
                                   dtype=tf.float32)

  for key in images:
    images[key] = tfa_image.rotate(
        images[key],
        angles=random_angle * prob,
        interpolation='bilinear',
        fill_mode='constant')
  return images


@gin.configurable('data_augmentation')
def data_augmentations(
    names: List[str]) -> Dict[str, Callable[..., tf.Tensor]]:
  """Creates the data augmentation functions.

  Args:
    names: The list of augmentation function names.
  Returns:
    A dictionary of Callables to the augmentation functions, keyed by their
    names.
  """
  augmentations = dict()
  for name in names:
    if name == 'random_image_rot90':
      augmentations[name] = random_image_rot90
    elif name == 'random_rotate':
      augmentations[name] = random_rotate
    elif name == 'random_flip':
      augmentations[name] = random_flip
    elif name == 'random_reverse':
      augmentations[name] = random_reverse
    else:
      raise AttributeError('Invalid augmentation function %s' % name)
  return augmentations
