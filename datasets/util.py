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
"""Utility functions for creating a tf.train.Example proto of image triplets."""

import io
import os
from typing import Any, List, Mapping, Optional

from absl import logging
import apache_beam as beam
import numpy as np
import PIL.Image
import six
from skimage import transform
import tensorflow as tf

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
_GAMMA = 2.2


def _resample_image(image: np.ndarray, resample_image_width: int,
                    resample_image_height: int) -> np.ndarray:
  """Re-samples and returns an `image` to be `resample_image_size`."""
  # Convert image from uint8 gamma [0..255] to float linear [0..1].
  image = image.astype(np.float32) / _UINT8_MAX_F
  image = np.power(np.clip(image, 0, 1), _GAMMA)

  # Re-size the image
  resample_image_size = (resample_image_height, resample_image_width)
  image = transform.resize_local_mean(image, resample_image_size)

  # Convert back from float linear [0..1] to uint8 gamma [0..255].
  image = np.power(np.clip(image, 0, 1), 1.0 / _GAMMA)
  image = np.clip(image * _UINT8_MAX_F + 0.5, 0.0,
                  _UINT8_MAX_F).astype(np.uint8)
  return image


def generate_image_triplet_example(
    triplet_dict: Mapping[str, str],
    scale_factor: int = 1,
    center_crop_factor: int = 1) -> Optional[tf.train.Example]:
  """Generates and serializes a tf.train.Example proto from an image triplet.

  Default setting creates a triplet Example with the input images unchanged.
  Images are processed in the order of center-crop then downscale.

  Args:
    triplet_dict: A dict of image key to filepath of the triplet images.
    scale_factor: An integer scale factor to isotropically downsample images.
    center_crop_factor: An integer cropping factor to center crop images with
      the original resolution but isotropically downsized by the factor.

  Returns:
    tf.train.Example proto, or None upon error.

  Raises:
    ValueError if triplet_dict length is different from three or the scale input
    arguments are non-positive.
  """
  if len(triplet_dict) != 3:
    raise ValueError(
        f'Length of triplet_dict must be exactly 3, not {len(triplet_dict)}.')

  if scale_factor <= 0 or center_crop_factor <= 0:
    raise ValueError(f'(scale_factor, center_crop_factor) must be positive, '
                     f'Not ({scale_factor}, {center_crop_factor}).')

  feature = {}

  # Keep track of the path where the images came from for debugging purposes.
  mid_frame_path = os.path.dirname(triplet_dict['frame_1'])
  feature['path'] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[six.ensure_binary(mid_frame_path)]))

  for image_key, image_path in triplet_dict.items():
    if not tf.io.gfile.exists(image_path):
      logging.error('File not found: %s', image_path)
      return None

    # Note: we need both the raw bytes and the image size.
    # PIL.Image does not expose a method to grab the original bytes.
    # (Also it is not aware of non-local file systems.)
    # So we read with tf.io.gfile.GFile to get the bytes, and then wrap the
    # bytes in BytesIO to let PIL.Image open the image.
    try:
      byte_array = tf.io.gfile.GFile(image_path, 'rb').read()
    except tf.errors.InvalidArgumentError:
      logging.exception('Cannot read image file: %s', image_path)
      return None
    try:
      pil_image = PIL.Image.open(io.BytesIO(byte_array))
    except PIL.UnidentifiedImageError:
      logging.exception('Cannot decode image file: %s', image_path)
      return None
    width, height = pil_image.size
    pil_image_format = pil_image.format

    # Optionally center-crop images and downsize images
    # by `center_crop_factor`.
    if center_crop_factor > 1:
      image = np.array(pil_image)
      quarter_height = image.shape[0] // (2 * center_crop_factor)
      quarter_width = image.shape[1] // (2 * center_crop_factor)
      image = image[quarter_height:-quarter_height,
                    quarter_width:-quarter_width, :]
      pil_image = PIL.Image.fromarray(image)

      # Update image properties.
      height, width, _ = image.shape
      buffer = io.BytesIO()
      try:
        pil_image.save(buffer, format='PNG')
      except OSError:
        logging.exception('Cannot encode image file: %s', image_path)
        return None
      byte_array = buffer.getvalue()

    # Optionally downsample images by `scale_factor`.
    if scale_factor > 1:
      image = np.array(pil_image)
      image = _resample_image(image, image.shape[1] // scale_factor,
                              image.shape[0] // scale_factor)
      pil_image = PIL.Image.fromarray(image)

      # Update image properties.
      height, width, _ = image.shape
      buffer = io.BytesIO()
      try:
        pil_image.save(buffer, format='PNG')
      except OSError:
        logging.exception('Cannot encode image file: %s', image_path)
        return None
      byte_array = buffer.getvalue()

    # Create tf Features.
    image_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[byte_array]))
    height_feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[height]))
    width_feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[width]))
    encoding = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[six.ensure_binary(pil_image_format.lower())]))

    # Update feature map.
    feature[f'{image_key}/encoded'] = image_feature
    feature[f'{image_key}/format'] = encoding
    feature[f'{image_key}/height'] = height_feature
    feature[f'{image_key}/width'] = width_feature

  # Create tf Example.
  features = tf.train.Features(feature=feature)
  example = tf.train.Example(features=features)
  return example


class ExampleGenerator(beam.DoFn):
  """Generate a tf.train.Example per input image triplet filepaths."""

  def __init__(self,
               images_map: Mapping[str, Any],
               scale_factor: int = 1,
               center_crop_factor: int = 1):
    """Initializes the map of 3 images to add to each tf.train.Example.

    Args:
      images_map: Map from image key to image filepath.
      scale_factor: A scale factor to downsample frames.
      center_crop_factor: A factor to centercrop and downsize frames.
    """
    super().__init__()
    self._images_map = images_map
    self._scale_factor = scale_factor
    self._center_crop_factor = center_crop_factor

  def process(self, triplet_dict: Mapping[str, str]) -> List[bytes]:
    """Generates a serialized tf.train.Example for a triplet of images.

    Args:
      triplet_dict: A dict of image key to filepath of the triplet images.

    Returns:
      A serialized tf.train.Example proto. No shuffling is applied.
    """
    example = generate_image_triplet_example(triplet_dict, self._scale_factor,
                                             self._center_crop_factor)
    if example:
      return [example.SerializeToString()]
    else:
      return []
