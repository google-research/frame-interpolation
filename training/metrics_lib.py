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
"""A library for instantiating frame interpolation evaluation metrics."""

from typing import Callable, Dict, Text

from ..losses import losses
import tensorflow as tf


class TrainLossMetric(tf.keras.metrics.Metric):
  """Compute training loss for our example and prediction format.

  The purpose of this is to ensure that we always include a loss that is exactly
  like the training loss into the evaluation in order to detect possible
  overfitting.
  """

  def __init__(self, name='eval_loss', **kwargs):
    super(TrainLossMetric, self).__init__(name=name, **kwargs)
    self.acc = self.add_weight(name='train_metric_acc', initializer='zeros')
    self.count = self.add_weight(name='train_metric_count', initializer='zeros')

  def update_state(self,
                   batch,
                   predictions,
                   sample_weight=None,
                   checkpoint_step=0):
    loss_functions = losses.training_losses()
    loss_list = []
    for (loss_value, loss_weight) in loss_functions.values():
      loss_list.append(
          loss_value(batch, predictions) * loss_weight(checkpoint_step))
    loss = tf.add_n(loss_list)
    self.acc.assign_add(loss)
    self.count.assign_add(1)

  def result(self):
    return self.acc / self.count

  def reset_states(self):
    self.acc.assign(0)
    self.count.assign(0)


class L1Metric(tf.keras.metrics.Metric):
  """Compute L1 over our training example and prediction format.

  The purpose of this is to ensure that we have at least one metric that is
  compatible across all eval the session and allows us to quickly compare models
  against each other.
  """

  def __init__(self, name='eval_loss', **kwargs):
    super(L1Metric, self).__init__(name=name, **kwargs)
    self.acc = self.add_weight(name='l1_metric_acc', initializer='zeros')
    self.count = self.add_weight(name='l1_metric_count', initializer='zeros')

  def update_state(self, batch, prediction, sample_weight=None,
                   checkpoint_step=0):
    self.acc.assign_add(losses.l1_loss(batch, prediction))
    self.count.assign_add(1)

  def result(self):
    return self.acc / self.count

  def reset_states(self):
    self.acc.assign(0)
    self.count.assign(0)


class GenericLossMetric(tf.keras.metrics.Metric):
  """Metric based on any loss function."""

  def __init__(self, name: str, loss: Callable[..., tf.Tensor],
               weight: Callable[..., tf.Tensor], **kwargs):
    """Initializes a metric based on a loss function and a weight schedule.

    Args:
      name: The name of the metric.
      loss: The callable loss that calculates a loss value for a (prediction,
        target) pair.
      weight: The callable weight scheduling function that samples a weight
        based on iteration.
      **kwargs: Any additional keyword arguments to be passed.
    """
    super(GenericLossMetric, self).__init__(name=name, **kwargs)
    self.acc = self.add_weight(name='loss_metric_acc', initializer='zeros')
    self.count = self.add_weight(name='loss_metric_count', initializer='zeros')
    self.loss = loss
    self.weight = weight

  def update_state(self,
                   batch,
                   predictions,
                   sample_weight=None,
                   checkpoint_step=0):
    self.acc.assign_add(
        self.loss(batch, predictions) * self.weight(checkpoint_step))
    self.count.assign_add(1)

  def result(self):
    return self.acc / self.count

  def reset_states(self):
    self.acc.assign(0)
    self.count.assign(0)


def create_metrics_fn() -> Dict[Text, tf.keras.metrics.Metric]:
  """Create evaluation metrics.

  L1 and total training loss are added by default.
  The rest are the configured by the test_losses item via gin.

  Returns:
    A dictionary from metric name to Keras Metric object.
  """
  metrics = {}
  # L1 is explicitly added just so we always have some consistent numbers around
  # to compare across sessions.
  metrics['l1'] = L1Metric()
  # We also always include training loss for the eval set to detect overfitting:
  metrics['training_loss'] = TrainLossMetric()

  test_losses = losses.test_losses()
  for loss_name, (loss_value, loss_weight) in test_losses.items():
    metrics[loss_name] = GenericLossMetric(
        name=loss_name, loss=loss_value, weight=loss_weight)
  return metrics
