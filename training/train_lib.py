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
r"""Training library for frame interpolation using distributed strategy."""
import functools
from typing import Any, Callable, Dict, Text, Tuple

from absl import logging
import tensorflow as tf


def _concat_tensors(tensors: tf.Tensor) -> tf.Tensor:
  """Concat tensors of the different replicas."""
  return tf.concat(tf.nest.flatten(tensors, expand_composites=True), axis=0)


@tf.function
def _distributed_train_step(strategy: tf.distribute.Strategy,
                            batch: Dict[Text, tf.Tensor], model: tf.keras.Model,
                            loss_functions: Dict[Text,
                                                 Tuple[Callable[..., tf.Tensor],
                                                       Callable[...,
                                                                tf.Tensor]]],
                            optimizer: tf.keras.optimizers.Optimizer,
                            iterations: int) -> Dict[Text, Any]:
  """Distributed training step.

  Args:
    strategy: A Tensorflow distribution strategy.
    batch: A batch of training examples.
    model: The Keras model to train.
    loss_functions: The list of Keras losses used to train the model.
    optimizer: The Keras optimizer used to train the model.
    iterations: Iteration number used to sample weights to each loss.

  Returns:
    A dictionary of train step outputs.
  """

  def _train_step(batch: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
    """Train for one step."""
    with tf.GradientTape() as tape:
      predictions = model(batch, training=True)
      losses = []
      for (loss_value, loss_weight) in loss_functions.values():
        losses.append(loss_value(batch, predictions) * loss_weight(iterations))
      loss = tf.add_n(losses)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # post process for visualization
    all_data = {'loss': loss}
    all_data.update(batch)
    all_data.update(predictions)
    return all_data

  step_outputs = strategy.run(_train_step, args=(batch,))

  loss = strategy.reduce(
      tf.distribute.ReduceOp.MEAN, step_outputs['loss'], axis=None)

  x0 = _concat_tensors(step_outputs['x0'])
  x1 = _concat_tensors(step_outputs['x1'])
  y = _concat_tensors(step_outputs['y'])
  pred_y = _concat_tensors(step_outputs['image'])

  scalar_summaries = {'training_loss': loss}

  image_summaries = {
      'x0': x0,
      'x1': x1,
      'y': y,
      'pred_y': pred_y
  }

  extra_images = {
      'importance0', 'importance1', 'x0_warped', 'x1_warped', 'fg_image',
      'bg_image', 'fg_alpha', 'x1_unfiltered_warped'
  }
  for image in extra_images:
    if image in step_outputs:
      image_summaries[image] = _concat_tensors(step_outputs[image])

  return {
      'loss': loss,
      'scalar_summaries': scalar_summaries,
      'image_summaries': {
          f'training/{name}': value for name, value in image_summaries.items()
      }
  }


def _summary_writer(summaries_dict: Dict[Text, Any]) -> None:
  """Adds scalar and image summaries."""
  # Adds scalar summaries.
  for key, scalars in summaries_dict['scalar_summaries'].items():
    tf.summary.scalar(key, scalars)
  # Adds image summaries.
  for key, images in summaries_dict['image_summaries'].items():
    tf.summary.image(key, tf.clip_by_value(images, 0.0, 1.0))
    tf.summary.histogram(key + '_h', images)


def train_loop(
    strategy: tf.distribute.Strategy,
    train_set: tf.data.Dataset,
    create_model_fn: Callable[..., tf.keras.Model],
    create_losses_fn: Callable[..., Dict[str, Tuple[Callable[..., tf.Tensor],
                                                    Callable[..., tf.Tensor]]]],
    create_optimizer_fn: Callable[..., tf.keras.optimizers.Optimizer],
    distributed_train_step_fn: Callable[[
        tf.distribute.Strategy, Dict[str, tf.Tensor], tf.keras.Model, Dict[
            str,
            Tuple[Callable[..., tf.Tensor],
                  Callable[..., tf.Tensor]]], tf.keras.optimizers.Optimizer, int
    ], Dict[str, Any]],
    eval_loop_fn: Callable[..., None],
    create_metrics_fn: Callable[..., Dict[str, tf.keras.metrics.Metric]],
    eval_folder: Dict[str, Any],
    eval_datasets: Dict[str, tf.data.Dataset],
    summary_writer_fn: Callable[[Dict[str, Any]], None],
    train_folder: str,
    saved_model_folder: str,
    num_iterations: int,
    save_summaries_frequency: int = 500,
    save_checkpoint_frequency: int = 500,
    checkpoint_max_to_keep: int = 10,
    checkpoint_save_every_n_hours: float = 2.,
    timing_frequency: int = 100,
    logging_frequency: int = 10):
  """A Tensorflow 2 eager mode training loop.

  Args:
    strategy: A Tensorflow distributed strategy.
    train_set: A tf.data.Dataset to loop through for training.
    create_model_fn: A callable that returns a tf.keras.Model.
    create_losses_fn: A callable that returns a tf.keras.losses.Loss.
    create_optimizer_fn: A callable that returns a
      tf.keras.optimizers.Optimizer.
    distributed_train_step_fn: A callable that takes a distribution strategy, a
      Dict[Text, tf.Tensor] holding the batch of training data, a
      tf.keras.Model, a tf.keras.losses.Loss, a tf.keras.optimizers.Optimizer,
      iteartion number to sample a weight value to loos functions,
      and returns a dictionary to be passed to the summary_writer_fn.
    eval_loop_fn: Eval loop function.
    create_metrics_fn: create_metric_fn.
    eval_folder: A path to where the summary event files and checkpoints will be
      saved.
    eval_datasets: A dictionary of evalution tf.data.Dataset to loop through for
      evaluation.
    summary_writer_fn: A callable that takes the output of
      distributed_train_step_fn and writes summaries to be visualized in
      TensorBoard.
    train_folder: A path to where the summaries event files and checkpoints
      will be saved.
    saved_model_folder: A path to where the saved models are stored.
    num_iterations: An integer, the number of iterations to train for.
    save_summaries_frequency: The iteration frequency with which summaries are
      saved.
    save_checkpoint_frequency: The iteration frequency with which model
      checkpoints are saved.
    checkpoint_max_to_keep: The maximum number of checkpoints to keep.
    checkpoint_save_every_n_hours: The frequency in hours to keep checkpoints.
    timing_frequency: The iteration frequency with which to log timing.
    logging_frequency: How often to output with logging.info().
  """
  logging.info('Creating training tensorboard summaries ...')
  summary_writer = tf.summary.create_file_writer(train_folder)

  if eval_datasets is not None:
    logging.info('Creating eval tensorboard summaries ...')
    eval_summary_writer = tf.summary.create_file_writer(eval_folder)

  train_set = strategy.experimental_distribute_dataset(train_set)
  with strategy.scope():
    logging.info('Building model ...')
    model = create_model_fn()
    loss_functions = create_losses_fn()
    optimizer = create_optimizer_fn()
    if eval_datasets is not None:
      metrics = create_metrics_fn()

  logging.info('Creating checkpoint ...')
  checkpoint = tf.train.Checkpoint(
      model=model,
      optimizer=optimizer,
      step=optimizer.iterations,
      epoch=tf.Variable(0, dtype=tf.int64, trainable=False),
      training_finished=tf.Variable(False, dtype=tf.bool, trainable=False))

  logging.info('Restoring old model (if exists) ...')
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=train_folder,
      max_to_keep=checkpoint_max_to_keep,
      keep_checkpoint_every_n_hours=checkpoint_save_every_n_hours)

  with strategy.scope():
    if checkpoint_manager.latest_checkpoint:
      checkpoint.restore(checkpoint_manager.latest_checkpoint)

  logging.info('Creating Timer ...')
  timer = tf.estimator.SecondOrStepTimer(every_steps=timing_frequency)
  timer.update_last_triggered_step(optimizer.iterations.numpy())

  logging.info('Training on devices: %s.', [
      el.name.split('/physical_device:')[-1]
      for el in tf.config.get_visible_devices()
  ])

  # Re-assign training_finished=False, in case we restored a checkpoint.
  checkpoint.training_finished.assign(False)
  while optimizer.iterations.numpy() < num_iterations:
    for i_batch, batch in enumerate(train_set):
      summary_writer.set_as_default()
      iterations = optimizer.iterations.numpy()

      if iterations % logging_frequency == 0:
        # Log epoch, total iterations and batch index.
        logging.info('epoch %d; iterations %d; i_batch %d',
                     checkpoint.epoch.numpy(), iterations,
                     i_batch)

      # Break if the number of iterations exceeds the max.
      if iterations >= num_iterations:
        break

      # Compute distributed step outputs.
      distributed_step_outputs = distributed_train_step_fn(
          strategy, batch, model, loss_functions, optimizer, iterations)

      # Save checkpoint, and optionally run the eval loops.
      if iterations % save_checkpoint_frequency == 0:
        checkpoint_manager.save(checkpoint_number=iterations)
        if eval_datasets is not None:
          eval_loop_fn(
              strategy=strategy,
              eval_base_folder=eval_folder,
              model=model,
              metrics=metrics,
              datasets=eval_datasets,
              summary_writer=eval_summary_writer,
              checkpoint_step=iterations)

      # Write summaries.
      if iterations % save_summaries_frequency == 0:
        tf.summary.experimental.set_step(step=iterations)
        summary_writer_fn(distributed_step_outputs)
        tf.summary.scalar('learning_rate',
                          optimizer.learning_rate(iterations).numpy())

      # Log steps/sec.
      if timer.should_trigger_for_step(iterations):
        elapsed_time, elapsed_steps = timer.update_last_triggered_step(
            iterations)
        if elapsed_time is not None:
          steps_per_second = elapsed_steps / elapsed_time
          tf.summary.scalar(
              'steps/sec', steps_per_second, step=optimizer.iterations)

    # Increment epoch.
    checkpoint.epoch.assign_add(1)

  # Assign training_finished variable to True after training is finished and
  # save the last checkpoint.
  checkpoint.training_finished.assign(True)
  checkpoint_manager.save(checkpoint_number=optimizer.iterations.numpy())

  # Generate a saved model.
  model.save(saved_model_folder)


def train(strategy: tf.distribute.Strategy, train_folder: str,
          saved_model_folder: str, n_iterations: int,
          create_model_fn: Callable[..., tf.keras.Model],
          create_losses_fn: Callable[..., Dict[str,
                                               Tuple[Callable[..., tf.Tensor],
                                                     Callable[...,
                                                              tf.Tensor]]]],
          create_metrics_fn: Callable[..., Dict[str, tf.keras.metrics.Metric]],
          dataset: tf.data.Dataset,
          learning_rate: tf.keras.optimizers.schedules.LearningRateSchedule,
          eval_loop_fn: Callable[..., None],
          eval_folder: str,
          eval_datasets: Dict[str, tf.data.Dataset]):
  """Training function that is strategy agnostic.

  Args:
    strategy: A Tensorflow distributed strategy.
    train_folder: A path to where the summaries event files and checkpoints
      will be saved.
    saved_model_folder: A path to where the saved models are stored.
    n_iterations: An integer, the number of iterations to train for.
    create_model_fn: A callable that returns tf.keras.Model.
    create_losses_fn: A callable that returns the losses.
    create_metrics_fn: A function that returns the metrics dictionary.
    dataset: The tensorflow dataset object.
    learning_rate: Keras learning rate schedule object.
    eval_loop_fn: eval loop function.
    eval_folder: A path to where eval summaries event files and checkpoints
      will be saved.
    eval_datasets: The tensorflow evaluation dataset objects.
  """
  train_loop(
      strategy=strategy,
      train_set=dataset,
      create_model_fn=create_model_fn,
      create_losses_fn=create_losses_fn,
      create_optimizer_fn=functools.partial(
          tf.keras.optimizers.Adam, learning_rate=learning_rate),
      distributed_train_step_fn=_distributed_train_step,
      eval_loop_fn=eval_loop_fn,
      create_metrics_fn=create_metrics_fn,
      eval_folder=eval_folder,
      eval_datasets=eval_datasets,
      summary_writer_fn=_summary_writer,
      train_folder=train_folder,
      saved_model_folder=saved_model_folder,
      num_iterations=n_iterations,
      save_summaries_frequency=3000,
      save_checkpoint_frequency=3000)


def get_strategy(mode) -> tf.distribute.Strategy:
  """Creates a distributed strategy."""
  strategy = None
  if mode == 'cpu':
    strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
  elif mode == 'gpu':
    strategy = tf.distribute.MirroredStrategy()
  else:
    raise ValueError('Unsupported distributed mode.')
  return strategy
