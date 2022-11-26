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
from typing import List, Optional
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
      'target_height': height + height_to_pad,
      'target_width': width + width_to_pad
  }
  padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
  bbox_to_crop = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height,
      'target_width': width
  }
  return padded_x, bbox_to_crop


def image_to_patches(image: np.ndarray, block_shape: List[int]) -> np.ndarray:
  """Folds an image into patches and stacks along the batch dimension.

  Args:
    image: The input image of shape [B, H, W, C].
    block_shape: The number of patches along the height and width to extract.
      Each patch is shaped (H/block_shape[0], W/block_shape[1])

  Returns:
    The extracted patches shaped [num_blocks, patch_height, patch_width,...],
      with num_blocks = block_shape[0] * block_shape[1].
  """
  block_height, block_width = block_shape
  num_blocks = block_height * block_width

  height, width, channel = image.shape[-3:]
  patch_height, patch_width = height//block_height, width//block_width
 
  assert height == (
      patch_height * block_height
  ), 'block_height=%d should evenly divide height=%d.'%(block_height, height)
  assert width == (
      patch_width * block_width
  ), 'block_width=%d should evenly divide width=%d.'%(block_width, width)

  patch_size = patch_height * patch_width
  paddings = 2*[[0, 0]]

  patches = tf.space_to_batch(image, [patch_height, patch_width], paddings)
  patches = tf.split(patches, patch_size, 0)
  patches = tf.stack(patches, axis=3)
  patches = tf.reshape(patches,
                       [num_blocks, patch_height, patch_width, channel])
  return patches.numpy()


def patches_to_image(patches: np.ndarray, block_shape: List[int]) -> np.ndarray:
  """Unfolds patches (stacked along batch) into an image.

  Args:
    patches: The input patches, shaped [num_patches, patch_H, patch_W, C].
    block_shape: The number of patches along the height and width to unfold.
      Each patch assumed to be shaped (H/block_shape[0], W/block_shape[1]).

  Returns:
    The unfolded image shaped [B, H, W, C].
  """
  block_height, block_width = block_shape
  paddings = 2 * [[0, 0]]

  patch_height, patch_width, channel = patches.shape[-3:]
  patch_size = patch_height * patch_width

  patches = tf.reshape(patches,
                       [1, block_height, block_width, patch_size, channel])
  patches = tf.split(patches, patch_size, axis=3)
  patches = tf.stack(patches, axis=0)
  patches = tf.reshape(patches,
                       [patch_size, block_height, block_width, channel])
  image = tf.batch_to_space(patches, [patch_height, patch_width], paddings)
  return image.numpy()


class Interpolator:
  """A class for generating interpolated frames between two input frames.

  Uses TF2 saved model format.
  """

  def __init__(self, model_path: str,
               align: Optional[int] = None,
               block_shape: Optional[List[int]] = None) -> None:
    """Loads a saved model.

    Args:
      model_path: Path to the saved model. If none are provided, uses the
        default model.
      align: 'If >1, pad the input size so it divides with this before
        inference.'
      block_shape: Number of patches along the (height, width) to sid-divide
        input images. 
    """
    self._model = tf.compat.v2.saved_model.load(model_path)
    self._align = align or None
    self._block_shape = block_shape or None

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
    image = result['image']

    if self._align is not None:
      image = tf.image.crop_to_bounding_box(image, **bbox_to_crop)
    return image.numpy()

  def __call__(self, x0: np.ndarray, x1: np.ndarray,
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
    if self._block_shape is not None and np.prod(self._block_shape) > 1:
      # Subdivide high-res images into managable non-overlapping patches.
      x0_patches = image_to_patches(x0, self._block_shape)
      x1_patches = image_to_patches(x1, self._block_shape)

      # Run the interpolator on each patch pair.
      output_patches = []
      for image_0, image_1 in zip(x0_patches, x1_patches):
        mid_patch = self.interpolate(image_0[np.newaxis, ...],
                                     image_1[np.newaxis, ...], dt)
        output_patches.append(mid_patch)

      # Reconstruct interpolated image by stitching interpolated patches.
      output_patches = np.concatenate(output_patches, axis=0)
      return patches_to_image(output_patches, self._block_shape)
    else:
      # Invoke the interpolator once.
      return self.interpolate(x0, x1, dt)
