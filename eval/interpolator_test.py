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
r"""A test script for mid frame interpolation from two input frames.

Usage example:
 python3 -m frame_interpolation.eval.interpolator_test \
   --frame1 <filepath of the first frame> \
   --frame2 <filepath of the second frame> \
   --model_path <The filepath of the TF2 saved model to use>
   --tiles <The number of tiles to split the input frames into>
   --padding <The number of pixels to pad the input frames with>

The output is saved to <the directory of the input frames>/output_frame.png. If
`--output_frame` filepath is provided, it will be used instead.
"""
import os
from typing import Sequence

from . import interpolator
from . import util
from . import splitter
from absl import app
from absl import flags
import numpy as np

_FRAME1 = flags.DEFINE_string(
    name='frame1',
    default=None,
    help='The filepath of the first input frame.',
    required=True)
_FRAME2 = flags.DEFINE_string(
    name='frame2',
    default=None,
    help='The filepath of the second input frame.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.')
_OUTPUT_FRAME = flags.DEFINE_string(
    name='output_frame',
    default=None,
    help='The output filepath of the interpolated mid-frame.')
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=None,
    help='If >1, pad the input size so it is evenly divisible by this value.')
_SLICES = flags.DEFINE_integer(
    name='slices',
    default=None,
    help='Number of slices to process larger images.')
_PADDING = flags.DEFINE_integer(
    name='padding',
    default=20,
    help='Padding to add to the sliced frames.')


def _run_interpolator() -> None:
  """Writes interpolated mid frame from a given two input frame filepaths."""

  model_wrapper = interpolator.Interpolator(_MODEL_PATH.value, _ALIGN.value)

  if _SLICES.value is not None:
    # First image.
    image_1 = util.read_image(_FRAME1.value)
    split_images_1 = splitter.splitFrame(image_1, _SLICES.value, _PADDING.value)

    # Second image.
    image_2 = util.read_image(_FRAME2.value)
    split_images_2 = splitter.splitFrame(image_2, _SLICES.value, _PADDING.value)

    mid_frame = np.zeros((_SLICES.value**2, image_1.shape[0], image_1.shape[1], 3))

    for i in range (_SLICES.value**2):

      tile1 = split_images_1[i]
      tile2 = split_images_2[i]

      image_batch_1 = np.expand_dims(tile1, axis=0)
      image_batch_2 = np.expand_dims(tile2, axis=0)

      # Batched time.
      batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

      # Invoke the model once.
      mid_frame[i,:,:,:] = model_wrapper.interpolate(image_batch_1, image_batch_2,
                                            batch_dt)[0]
    mid_frame = splitter.stitchFrames(mid_frame, _SLICES.value, _PADDING.value)

  else:
    # First batched image.
    image_1 = util.read_image(_FRAME1.value)
    image_batch_1 = np.expand_dims(image_1, axis=0)

    # Second batched image.
    image_2 = util.read_image(_FRAME2.value)
    image_batch_2 = np.expand_dims(image_2, axis=0)

    # Batched time.
    batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

    # Invoke the model once.
    mid_frame = model_wrapper.interpolate(image_batch_1, image_batch_2,
                                          batch_dt)[0]

  # Write interpolated mid-frame.
  mid_frame_filepath = _OUTPUT_FRAME.value
  if not mid_frame_filepath:
    mid_frame_filepath = os.path.join(
        os.path.dirname(_FRAME1.value), 'output_frame.png')
  util.write_image(mid_frame_filepath, mid_frame)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  _run_interpolator()


if __name__ == '__main__':
  app.run(main)
