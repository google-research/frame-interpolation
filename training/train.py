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
r"""The training loop for frame interpolation.

gin_config: The gin configuration file containing model, losses and datasets.

To run on GPUs:
  python3 -m frame_interpolation.training.train \
      --gin_config <path to  network.gin> \
      --base_folder <base folder for all training runs> \
      --label <descriptive label for the run>

To debug the training loop on CPU:
  python3 -m frame_interpolation.training.train \
      --gin_config <path to config.gin> \
      --base_folder /tmp
      --label test_run \
      --mode cpu

The training output directory will be created at <base_folder>/<label>.
"""
import os

from . import augmentation_lib
from . import data_lib
from . import eval_lib
from . import metrics_lib
from . import model_lib
from . import train_lib
from absl import app
from absl import flags
from absl import logging
import gin.tf
from ..losses import losses

# Reduce tensorflow logs to ERRORs only.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf  # pylint: disable=g-import-not-at-top
tf.get_logger().setLevel('ERROR')


_GIN_CONFIG = flags.DEFINE_string('gin_config', None, 'Gin config file.')
_LABEL = flags.DEFINE_string('label', 'run0',
                             'Descriptive label for this run.')
_BASE_FOLDER = flags.DEFINE_string('base_folder', None,
                                   'Path to checkpoints/summaries.')
_MODE = flags.DEFINE_enum('mode', 'gpu', ['cpu', 'gpu'],
                          'Distributed strategy approach.')


@gin.configurable('training')
class TrainingOptions(object):
  """Training-related options."""

  def __init__(self, learning_rate: float, learning_rate_decay_steps: int,
               learning_rate_decay_rate: int, learning_rate_staircase: int,
               num_steps: int):
    self.learning_rate = learning_rate
    self.learning_rate_decay_steps = learning_rate_decay_steps
    self.learning_rate_decay_rate = learning_rate_decay_rate
    self.learning_rate_staircase = learning_rate_staircase
    self.num_steps = num_steps


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  output_dir = os.path.join(_BASE_FOLDER.value, _LABEL.value)
  logging.info('Creating output_dir @ %s ...', output_dir)

  # Copy config file to <base_folder>/<label>/config.gin.
  tf.io.gfile.makedirs(output_dir)
  tf.io.gfile.copy(
      _GIN_CONFIG.value, os.path.join(output_dir, 'config.gin'), overwrite=True)

  gin.external_configurable(
      tf.keras.optimizers.schedules.PiecewiseConstantDecay,
      module='tf.keras.optimizers.schedules')

  gin_configs = [_GIN_CONFIG.value]
  gin.parse_config_files_and_bindings(
      config_files=gin_configs, bindings=None, skip_unknown=True)

  training_options = TrainingOptions()  # pylint: disable=no-value-for-parameter

  learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
      training_options.learning_rate,
      training_options.learning_rate_decay_steps,
      training_options.learning_rate_decay_rate,
      training_options.learning_rate_staircase,
      name='learning_rate')

  # Initialize data augmentation functions
  augmentation_fns = augmentation_lib.data_augmentations()

  saved_model_folder = os.path.join(_BASE_FOLDER.value, _LABEL.value,
                                    'saved_model')
  train_folder = os.path.join(_BASE_FOLDER.value, _LABEL.value, 'train')
  eval_folder = os.path.join(_BASE_FOLDER.value, _LABEL.value, 'eval')

  train_lib.train(
      strategy=train_lib.get_strategy(_MODE.value),
      train_folder=train_folder,
      saved_model_folder=saved_model_folder,
      n_iterations=training_options.num_steps,
      create_model_fn=model_lib.create_model,
      create_losses_fn=losses.training_losses,
      create_metrics_fn=metrics_lib.create_metrics_fn,
      dataset=data_lib.create_training_dataset(
          augmentation_fns=augmentation_fns),
      learning_rate=learning_rate,
      eval_loop_fn=eval_lib.eval_loop,
      eval_folder=eval_folder,
      eval_datasets=data_lib.create_eval_datasets() or None)


if __name__ == '__main__':
  app.run(main)
