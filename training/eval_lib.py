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
"""Evaluation library for frame interpolation."""
from typing import Dict, Mapping, Text

from absl import logging
import tensorflow as tf


def _collect_tensors(tensors: tf.Tensor) -> tf.Tensor:
  """Collect tensors of the different replicas into a list."""
  return tf.nest.flatten(tensors, expand_composites=True)


@tf.function
def _distributed_eval_step(strategy: tf.distribute.Strategy,
                           batch: Dict[Text, tf.Tensor], model: tf.keras.Model,
                           metrics: Dict[Text, tf.keras.metrics.Metric],
                           checkpoint_step: int) -> Dict[Text, tf.Tensor]:
  """Distributed eval step.

  Args:
    strategy: A Tensorflow distribution strategy.
    batch: A batch of training examples.
    model: The Keras model to evaluate.
    metrics: The Keras metrics used for evaluation (a dictionary).
    checkpoint_step: The iteration number at which the checkpoint is restored.

  Returns:
    list of predictions from each replica.
  """

  def _eval_step(
      batch: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
    """Eval for one step."""
    predictions = model(batch, training=False)
    # Note: these metrics expect batch and prediction dictionaries rather than
    # tensors like standard TF metrics do. This allows our losses and metrics to
    # use a richer set of inputs than just the predicted final image.
    for metric in metrics.values():
      metric.update_state(batch, predictions, checkpoint_step=checkpoint_step)
    return predictions

  return strategy.run(_eval_step, args=(batch,))


def _summarize_image_tensors(combined, prefix, step):
  for name in combined:
    image = combined[name]
    if isinstance(image, tf.Tensor):
      if len(image.shape) == 4 and (image.shape[-1] == 1 or
                                    image.shape[-1] == 3):
        tf.summary.image(prefix + '/' + name, image, step=step)


def eval_loop(strategy: tf.distribute.Strategy,
              eval_base_folder: str,
              model: tf.keras.Model,
              metrics: Dict[str, tf.keras.metrics.Metric],
              datasets: Mapping[str, tf.data.Dataset],
              summary_writer: tf.summary.SummaryWriter,
              checkpoint_step: int):
  """Eval function that is strategy agnostic.

  Args:
    strategy: A Tensorflow distributed strategy.
    eval_base_folder: A path to where the summaries event files and
      checkpoints will be saved.
    model: A function that returns the model.
    metrics: A function that returns the metrics dictionary.
    datasets: A dict of tf.data.Dataset to evaluate on.
    summary_writer: Eval summary writer.
    checkpoint_step: The number of iterations completed.
  """
  logging.info('Saving eval summaries to: %s...', eval_base_folder)
  summary_writer.set_as_default()

  for dataset_name, dataset in datasets.items():
    for metric in metrics.values():
      metric.reset_states()

    logging.info('Loading %s testing data ...', dataset_name)
    dataset = strategy.experimental_distribute_dataset(dataset)

    logging.info('Evaluating %s ...', dataset_name)
    batch_idx = 0
    max_batches_to_summarize = 10
    for batch in dataset:
      predictions = _distributed_eval_step(strategy, batch, model, metrics,
                                           checkpoint_step)
      # Clip interpolator output to [0,1]. Clipping is done only
      # on the eval loop to get better metrics, but not on the training loop
      # so gradients are not killed.
      if strategy.num_replicas_in_sync > 1:
        predictions = {
            'image': tf.concat(predictions['image'].values, axis=0)
        }
      predictions['image'] = tf.clip_by_value(predictions['image'], 0., 1.)
      if batch_idx % 10 == 0:
        logging.info('Evaluating batch %s', batch_idx)
      batch_idx = batch_idx + 1
      if batch_idx < max_batches_to_summarize:
        # Loop through the global batch:
        prefix = f'{dataset_name}/eval_{batch_idx}'
        # Find all tensors that look like images, and summarize:
        combined = {**batch, **predictions}
        _summarize_image_tensors(combined, prefix, step=checkpoint_step)

      elif batch_idx == max_batches_to_summarize:
        tf.summary.flush()

    for name, metric in metrics.items():
      tf.summary.scalar(
          f'{dataset_name}/{name}', metric.result(), step=checkpoint_step)
      tf.summary.flush()
      logging.info('Step {:2}, {} {}'.format(checkpoint_step,
                                             f'{dataset_name}/{name}',
                                             metric.result().numpy()))
      metric.reset_states()
