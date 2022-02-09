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
"""Loss functions used to train the FILM interpolation model.

The losses for training and test loops are configurable via gin. Training can
use more than one loss function. Test loop can also evaluate one ore more loss
functions, each of which can be summarized separately.
"""
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from . import vgg19_loss as vgg19
import gin.tf
import numpy as np
import tensorflow as tf


@gin.configurable('vgg', denylist=['example', 'prediction'])
def vgg_loss(example: Mapping[str, tf.Tensor],
             prediction: Mapping[str, tf.Tensor],
             vgg_model_file: str,
             weights: Optional[List[float]] = None) -> tf.Tensor:
  """Perceptual loss for images in [0,1] color range.

  Args:
    example: A dictionary with the ground truth image as 'y'.
    prediction: The prediction dictionary with the image as 'image'.
    vgg_model_file: The path containing the vgg19 weights in MATLAB format.
    weights: An optional array of weights for different VGG layers. If None, the
      default weights are used (see vgg19.vgg_loss documentation).

  Returns:
    The perceptual loss.
  """
  return vgg19.vgg_loss(prediction['image'], example['y'], vgg_model_file,
                        weights)


@gin.configurable('style', denylist=['example', 'prediction'])
def style_loss(example: Mapping[str, tf.Tensor],
               prediction: Mapping[str, tf.Tensor],
               vgg_model_file: str,
               weights: Optional[List[float]] = None) -> tf.Tensor:
  """Computes style loss from images in [0..1] color range.

  Args:
    example: A dictionary with the ground truth image as 'y'.
    prediction: The prediction dictionary with the image as 'image'.
    vgg_model_file: The path containing the vgg19 weights in MATLAB format.
    weights: An optional array of weights for different VGG layers. If None, the
      default weights are used (see vgg19.vgg_loss documentation).

  Returns:
    A tf.Tensor of a scalar representing the style loss computed over multiple
    vgg layer features.
  """
  return vgg19.style_loss(prediction['image'], example['y'], vgg_model_file,
                          weights)


def l1_loss(example: Mapping[str, tf.Tensor],
            prediction: Mapping[str, tf.Tensor]) -> tf.Tensor:
  return tf.reduce_mean(tf.abs(prediction['image'] - example['y']))


def l1_warped_loss(example: Mapping[str, tf.Tensor],
                   prediction: Mapping[str, tf.Tensor]) -> tf.Tensor:
  """Computes an l1 loss using only warped images.

  Args:
    example: A dictionary with the ground truth image as 'y'.
    prediction: The prediction dictionary with the image(s) as 'x0_warped'
      and/or 'x1_warped'.

  Returns:
    A tf.Tensor of a scalar representing the linear combination of l1 losses
      between prediction images and y.
  """
  loss = tf.constant(0.0, dtype=tf.float32)
  if 'x0_warped' in prediction:
    loss += tf.reduce_mean(tf.abs(prediction['x0_warped'] - example['y']))
  if 'x1_warped' in prediction:
    loss += tf.reduce_mean(tf.abs(prediction['x1_warped'] - example['y']))
  return loss


def l2_loss(example: Mapping[str, tf.Tensor],
            prediction: Mapping[str, tf.Tensor]) -> tf.Tensor:
  return tf.reduce_mean(tf.square(prediction['image'] - example['y']))


def ssim_loss(example: Mapping[str, tf.Tensor],
              prediction: Mapping[str, tf.Tensor]) -> tf.Tensor:
  image = prediction['image']
  y = example['y']
  return tf.reduce_mean(tf.image.ssim(image, y, max_val=1.0))


def psnr_loss(example: Mapping[str, tf.Tensor],
              prediction: Mapping[str, tf.Tensor]) -> tf.Tensor:
  return tf.reduce_mean(
      tf.image.psnr(prediction['image'], example['y'], max_val=1.0))


def get_loss(loss_name: str) -> Callable[[Any, Any], tf.Tensor]:
  """Returns the loss function corresponding to the given name."""
  if loss_name == 'l1':
    return l1_loss
  elif loss_name == 'l2':
    return l2_loss
  elif loss_name == 'ssim':
    return ssim_loss
  elif loss_name == 'vgg':
    return vgg_loss
  elif loss_name == 'style':
    return style_loss
  elif loss_name == 'psnr':
    return psnr_loss
  elif loss_name == 'l1_warped':
    return l1_warped_loss
  else:
    raise ValueError('Invalid loss function %s' % loss_name)


# pylint: disable=unnecessary-lambda
def get_loss_op(loss_name):
  """Returns a function for creating a loss calculation op."""
  loss = get_loss(loss_name)
  return lambda example, prediction: loss(example, prediction)


def get_weight_op(weight_schedule):
  """Returns a function for creating an iteration dependent loss weight op."""
  return lambda iterations: weight_schedule(iterations)


def create_losses(
    loss_names: List[str], loss_weight_schedules: List[
        tf.keras.optimizers.schedules.LearningRateSchedule]
) -> Dict[str, Tuple[Callable[[Any, Any], tf.Tensor], Callable[[Any],
                                                               tf.Tensor]]]:
  """Returns a dictionary of functions for creating loss and loss_weight ops.

  As an example, create_losses(['l1', 'l2'], [PiecewiseConstantDecay(),
  PiecewiseConstantDecay()]) returns a dictionary with two keys, and each value
  being a tuple of ops for loss calculation and loss_weight sampling.

  Args:
      loss_names: Names of the losses.
      loss_weight_schedules: Instances of loss weight schedules.

  Returns:
    A dictionary that contains the loss and weight schedule ops keyed by the
    names.
  """
  losses = dict()
  for name, weight_schedule in zip(loss_names, loss_weight_schedules):
    unique_values = np.unique(weight_schedule.values)
    if len(unique_values) == 1 and unique_values[0] == 1.0:
      # Special case 'no weight' for prettier TensorBoard summaries.
      weighted_name = name
    else:
      # Weights are variable/scheduled, a constant "k" is used to
      # indicate weights are iteration dependent.
      weighted_name = 'k*' + name
    losses[weighted_name] = (get_loss_op(name), get_weight_op(weight_schedule))
  return losses


@gin.configurable
def training_losses(
    loss_names: List[str],
    loss_weights: Optional[List[float]] = None,
    loss_weight_schedules: Optional[List[
        tf.keras.optimizers.schedules.LearningRateSchedule]] = None,
    loss_weight_parameters: Optional[List[Mapping[str, List[Any]]]] = None
) -> Mapping[str, Tuple[Callable[[Any, Any], tf.Tensor], Callable[[Any],
                                                                  tf.Tensor]]]:
  """Creates the training loss functions and loss weight schedules."""
  weight_schedules = []
  if not loss_weights:
    for weight_schedule, weight_parameters in zip(loss_weight_schedules,
                                                  loss_weight_parameters):
      weight_schedules.append(weight_schedule(**weight_parameters))
  else:
    for loss_weight in loss_weights:
      weight_parameters = {
          'boundaries': [0],
          'values': 2 * [
              loss_weight,
          ]
      }
      weight_schedules.append(
          tf.keras.optimizers.schedules.PiecewiseConstantDecay(
              **weight_parameters))

  return create_losses(loss_names, weight_schedules)


@gin.configurable
def test_losses(
    loss_names: List[str],
    loss_weights: Optional[List[float]] = None,
    loss_weight_schedules: Optional[List[
        tf.keras.optimizers.schedules.LearningRateSchedule]] = None,
    loss_weight_parameters: Optional[List[Mapping[str, List[Any]]]] = None
) -> Mapping[str, Tuple[Callable[[Any, Any], tf.Tensor], Callable[[Any],
                                                                  tf.Tensor]]]:
  """Creates the test loss functions and loss weight schedules."""
  weight_schedules = []
  if not loss_weights:
    for weight_schedule, weight_parameters in zip(loss_weight_schedules,
                                                  loss_weight_parameters):
      weight_schedules.append(weight_schedule(**weight_parameters))
  else:
    for loss_weight in loss_weights:
      weight_parameters = {
          'boundaries': [0],
          'values': 2 * [
              loss_weight,
          ]
      }
      weight_schedules.append(
          tf.keras.optimizers.schedules.PiecewiseConstantDecay(
              **weight_parameters))

  return create_losses(loss_names, weight_schedules)


def aggregate_batch_losses(
    batch_losses: List[Mapping[str, float]]) -> Mapping[str, float]:
  """Averages per batch losses into single dictionary for the whole epoch.

  As an example, if the batch_losses contained per batch losses:
  batch_losses = { {'l1': 0.2, 'ssim': 0.9}, {'l1': 0.3, 'ssim': 0.8}}
  The returned dictionary would look like: { 'l1': 0.25, 'ssim': 0.95 }

  Args:
    batch_losses: A list of dictionary objects, with one entry for each loss.

  Returns:
    Single dictionary with the losses aggregated.
  """
  transp_losses = {}
  # Loop through all losses
  for batch_loss in batch_losses:
    # Loop through per batch losses of a single type:
    for loss_name, loss in batch_loss.items():
      if loss_name not in transp_losses:
        transp_losses[loss_name] = []
      transp_losses[loss_name].append(loss)
  aggregate_losses = {}
  for loss_name in transp_losses:
    aggregate_losses[loss_name] = np.mean(transp_losses[loss_name])
  return aggregate_losses
