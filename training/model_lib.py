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
"""A library for instantiating the model for training frame interpolation.

All models are expected to use three inputs: input image batches 'x0' and 'x1'
and 'time', the fractional time where the output should be generated.

The models are expected to output the prediction as a dictionary that contains
at least the predicted image batch as 'image' plus optional data for debug,
analysis or custom losses.
"""

import gin.tf
from ..models.film_net import interpolator as film_net_interpolator
from ..models.film_net import options as film_net_options

import tensorflow as tf


@gin.configurable('model')
def create_model(name: str) -> tf.keras.Model:
  """Creates the frame interpolation model based on given model name."""
  if name == 'film_net':
    return _create_film_net_model()  # pylint: disable=no-value-for-parameter
  else:
    raise ValueError(f'Model {name} not implemented.')


def _create_film_net_model() -> tf.keras.Model:
  """Creates the film_net interpolator."""
  # Options are gin-configured in the Options class directly.
  options = film_net_options.Options()

  x0 = tf.keras.Input(
      shape=(None, None, 3), batch_size=None, dtype=tf.float32, name='x0')
  x1 = tf.keras.Input(
      shape=(None, None, 3), batch_size=None, dtype=tf.float32, name='x1')
  time = tf.keras.Input(
      shape=(1,), batch_size=None, dtype=tf.float32, name='time')

  return film_net_interpolator.create_model(x0, x1, time, options)
