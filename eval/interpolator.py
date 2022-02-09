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
"""A wrapper class for running a frame interpolation TF2 saved model.

Usage:
  model_path='/tmp/saved_model/'
  it = Interpolator(model_path)
  result_batch = it.interpolate(image_batch_0, image_batch_1, batch_dt)

  Where image_batch_1 and image_batch_2 are numpy tensors with TF standard
  (B,H,W,C) layout, batch_dt is the sub-frame time in range [0,1], (B,) layout.
"""
from typing import Optional
import numpy as np
import tensorflow as tf


def _pad_to_align(x, align):
  """Pad image batch x so width and height divide by align.

  Args:
    x: Image batch to align.
    align: Number to align to.

  Returns:
    1) An image padded so width % align == 0 and height % align == 0.
    2) A bounding box that can be fed readily to tf.image.crop_to_bounding_box
      to undo the padding.
  """
  # Input checking.
  assert np.ndim(x) == 4
  assert align > 0, 'align must be a positive number.'

  height, width = x.shape[-3:-1]
  height_to_pad = (align - height % align) if height % align != 0 else 0
  width_to_pad = (align - width % align) if width % align != 0 else 0

  bbox_to_pad = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_width': height + height_to_pad,
      'target_height': width + width_to_pad
  }
  padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
  bbox_to_crop = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_width': height,
      'target_height': width
  }
  return padded_x, bbox_to_crop


class Interpolator:
  """A class for generating interpolated frames between two input frames.

  Uses TF2 saved model format.
  """

  def __init__(self, model_path: str,
               align: Optional[int] = None) -> None:
    """Loads a saved model.

    Args:
      model_path: Path to the saved model. If none are provided, uses the
        default model.
      align: 'If >1, pad the input size so it divides with this before
        inference.'
    """
    self._model = tf.compat.v2.saved_model.load(model_path)
    self._align = align

  def interpolate(self, x0: np.ndarray, x1: np.ndarray,
                  dt: np.ndarray) -> np.ndarray:
    """Generates an interpolated frame between given two batches of frames.

    All input tensors should be np.float32 datatype.

    Args:
      x0: First image batch. Dimensions: (batch_size, height, width, channels)
      x1: Second image batch. Dimensions: (batch_size, height, width, channels)
      dt: Sub-frame time. Range [0,1]. Dimensions: (batch_size,)

    Returns:
      The result with dimensions (batch_size, height, width, channels).
    """
    if self._align is not None:
      x0, bbox_to_crop = _pad_to_align(x0, self._align)
      x1, _ = _pad_to_align(x1, self._align)

    inputs = {'x0': x0, 'x1': x1, 'time': dt[..., np.newaxis]}
    result = self._model(inputs, training=False)
    image = result['image'].numpy()

    if self._align is not None:
      return tf.image.crop_to_bounding_box(image, **bbox_to_crop)
    return image
