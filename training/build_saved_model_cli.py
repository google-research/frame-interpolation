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
r"""Converts TF2 training checkpoint to a saved model.

The model must match the checkpoint, so the gin config must be given.

Usage example:
  python3 -m frame_interpolation.training.build_saved_model_cli \
    --gin_config <filepath of the gin config the training session was based> \
    --base_folder <base folder of training sessions> \
    --label <the name of the run>

This will produce a saved model into: <base_folder>/<label>/saved_model
"""
import os
from typing import Sequence

from . import model_lib
from absl import app
from absl import flags
from absl import logging
import gin.tf
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

_GIN_CONFIG = flags.DEFINE_string(
    name='gin_config',
    default='config.gin',
    help='Gin config file, saved in the training session <root folder>.')
_LABEL = flags.DEFINE_string(
    name='label',
    default=None,
    required=True,
    help='Descriptive label for the training session.')
_BASE_FOLDER = flags.DEFINE_string(
    name='base_folder',
    default=None,
    help='Path to all training sessions.')
_MODE = flags.DEFINE_enum(
    name='mode',
    default=None,
    enum_values=['cpu', 'gpu', 'tpu'],
    help='Distributed strategy approach.')


def _build_saved_model(checkpoint_path: str, config_files: Sequence[str],
                       output_model_path: str):
  """Builds a saved model based on the checkpoint directory."""
  gin.parse_config_files_and_bindings(
      config_files=config_files,
      bindings=None,
      skip_unknown=True)
  model = model_lib.create_model()
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
  try:
    logging.info('Restoring from %s', checkpoint_file)
    status = checkpoint.restore(checkpoint_file)
    status.assert_existing_objects_matched()
    status.expect_partial()
    model.save(output_model_path)
  except (tf.errors.NotFoundError, AssertionError) as err:
    logging.info('Failed to restore checkpoint from %s. Error:\n%s',
                 checkpoint_file, err)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  checkpoint_path = os.path.join(_BASE_FOLDER.value, _LABEL.value, 'train')
  if not tf.io.gfile.exists(_GIN_CONFIG.value):
    config_file = os.path.join(_BASE_FOLDER.value, _LABEL.value,
                               _GIN_CONFIG.value)
  else:
    config_file = _GIN_CONFIG.value
  output_model_path = os.path.join(_BASE_FOLDER.value, _LABEL.value,
                                   'saved_model')
  _build_saved_model(
      checkpoint_path=checkpoint_path,
      config_files=[config_file],
      output_model_path=output_model_path)
  logging.info('The saved model stored into %s/.', output_model_path)

if __name__ == '__main__':
  app.run(main)
