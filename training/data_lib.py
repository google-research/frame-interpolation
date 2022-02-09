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
"""Dataset creation for frame interpolation."""
from typing import Callable, Dict, List, Optional

from absl import logging
import gin.tf
import tensorflow as tf


def _create_feature_map() -> Dict[str, tf.io.FixedLenFeature]:
  """Creates the feature map for extracting the frame triplet."""
  feature_map = {
      'frame_0/encoded':
          tf.io.FixedLenFeature((), tf.string, default_value=''),
      'frame_0/format':
          tf.io.FixedLenFeature((), tf.string, default_value='jpg'),
      'frame_0/height':
          tf.io.FixedLenFeature((), tf.int64, default_value=0),
      'frame_0/width':
          tf.io.FixedLenFeature((), tf.int64, default_value=0),
      'frame_1/encoded':
          tf.io.FixedLenFeature((), tf.string, default_value=''),
      'frame_1/format':
          tf.io.FixedLenFeature((), tf.string, default_value='jpg'),
      'frame_1/height':
          tf.io.FixedLenFeature((), tf.int64, default_value=0),
      'frame_1/width':
          tf.io.FixedLenFeature((), tf.int64, default_value=0),
      'frame_2/encoded':
          tf.io.FixedLenFeature((), tf.string, default_value=''),
      'frame_2/format':
          tf.io.FixedLenFeature((), tf.string, default_value='jpg'),
      'frame_2/height':
          tf.io.FixedLenFeature((), tf.int64, default_value=0),
      'frame_2/width':
          tf.io.FixedLenFeature((), tf.int64, default_value=0),
      'path':
          tf.io.FixedLenFeature((), tf.string, default_value=''),
  }
  return feature_map


def _parse_example(sample):
  """Parses a serialized sample.

  Args:
    sample: A serialized tf.Example to be parsed.

  Returns:
    dictionary containing the following:
      encoded_image
      image_height
      image_width
  """
  feature_map = _create_feature_map()
  features = tf.io.parse_single_example(sample, feature_map)
  output_dict = {
      'x0': tf.io.decode_image(features['frame_0/encoded'], dtype=tf.float32),
      'x1': tf.io.decode_image(features['frame_2/encoded'], dtype=tf.float32),
      'y': tf.io.decode_image(features['frame_1/encoded'], dtype=tf.float32),
      # The fractional time value of frame_1 is not included in our tfrecords,
      # but is always at 0.5. The model will expect this to be specificed, so
      # we insert it here.
      'time': 0.5,
      # Store the original mid frame filepath for identifying examples.
      'path': features['path'],
  }

  return output_dict


def _random_crop_images(crop_size: int, images: tf.Tensor,
                        total_channel_size: int) -> tf.Tensor:
  """Crops the tensor with random offset to the given size."""
  if crop_size > 0:
    crop_shape = tf.constant([crop_size, crop_size, total_channel_size])
    images = tf.image.random_crop(images, crop_shape)
  return images


def crop_example(example: tf.Tensor, crop_size: int,
                 crop_keys: Optional[List[str]] = None):
  """Random crops selected images in the example to given size and keys.

  Args:
    example: Input tensor representing images to be cropped.
    crop_size: The size to crop images to. This value is used for both
      height and width.
    crop_keys: The images in the input example to crop.

  Returns:
    Example with cropping applied to selected images.
  """
  if crop_keys is None:
    crop_keys = ['x0', 'x1', 'y']
    channels = [3, 3, 3]

  # Stack images along channel axis, and perform a random crop once.
  image_to_crop = [example[key] for key in crop_keys]
  stacked_images = tf.concat(image_to_crop, axis=-1)
  cropped_images = _random_crop_images(crop_size, stacked_images, sum(channels))
  cropped_images = tf.split(
      cropped_images, num_or_size_splits=channels, axis=-1)
  for key, cropped_image in zip(crop_keys, cropped_images):
    example[key] = cropped_image
  return example


def apply_data_augmentation(
    augmentation_fns: Dict[str, Callable[..., tf.Tensor]],
    example: tf.Tensor,
    augmentation_keys: Optional[List[str]] = None) -> tf.Tensor:
  """Applies random augmentation in succession to selected image keys.

  Args:
    augmentation_fns: A Dict of Callables to data augmentation functions.
    example: Input tensor representing images to be augmented.
    augmentation_keys: The images in the input example to augment.

  Returns:
    Example with augmentation applied to selected images.
  """
  if augmentation_keys is None:
    augmentation_keys = ['x0', 'x1', 'y']

  # Apply each augmentation in sequence
  augmented_images = {key: example[key] for key in augmentation_keys}
  for augmentation_function in augmentation_fns.values():
    augmented_images = augmentation_function(augmented_images)

  for key in augmentation_keys:
    example[key] = augmented_images[key]
  return example


def _create_from_tfrecord(batch_size, file, augmentation_fns,
                          crop_size) -> tf.data.Dataset:
  """Creates a dataset from TFRecord."""
  dataset = tf.data.TFRecordDataset(file)
  dataset = dataset.map(
      _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # Perform data_augmentation before cropping and batching
  if augmentation_fns is not None:
    dataset = dataset.map(
        lambda x: apply_data_augmentation(augmentation_fns, x),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if crop_size > 0:
    dataset = dataset.map(
        lambda x: crop_example(x, crop_size=crop_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  return dataset


def _generate_sharded_filenames(filename: str) -> List[str]:
  """Generates filenames of the each file in the sharded filepath.

  Based on github.com/google/revisiting-self-supervised/blob/master/datasets.py.

  Args:
    filename: The sharded filepath.

  Returns:
    A list of filepaths for each file in the shard.
  """
  base, count = filename.split('@')
  count = int(count)
  return ['{}-{:05d}-of-{:05d}'.format(base, i, count) for i in range(count)]


def _create_from_sharded_tfrecord(batch_size,
                                  train_mode,
                                  file,
                                  augmentation_fns,
                                  crop_size,
                                  max_examples=-1) -> tf.data.Dataset:
  """Creates a dataset from a sharded tfrecord."""
  dataset = tf.data.Dataset.from_tensor_slices(
      _generate_sharded_filenames(file))

  # pylint: disable=g-long-lambda
  dataset = dataset.interleave(
      lambda x: _create_from_tfrecord(
          batch_size,
          file=x,
          augmentation_fns=augmentation_fns,
          crop_size=crop_size),
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=not train_mode)
  # pylint: enable=g-long-lambda
  dataset = dataset.prefetch(buffer_size=2)
  if max_examples > 0:
    return dataset.take(max_examples)
  return dataset


@gin.configurable('training_dataset')
def create_training_dataset(
    batch_size: int,
    file: Optional[str] = None,
    files: Optional[List[str]] = None,
    crop_size: int = -1,
    crop_sizes: Optional[List[int]] = None,
    augmentation_fns: Optional[Dict[str, Callable[..., tf.Tensor]]] = None
) -> tf.data.Dataset:
  """Creates the training dataset.

  The given tfrecord should contain data in a format produced by
  frame_interpolation/datasets/create_*_tfrecord.py

  Args:
    batch_size: The number of images to batch per example.
    file: (deprecated) A path to a sharded tfrecord in <tfrecord>@N format.
      Deprecated. Use 'files' instead.
    files: A list of paths to sharded tfrecords in <tfrecord>@N format.
    crop_size: (deprecated) If > 0, images are cropped to crop_size x crop_size
      using tensorflow's random cropping. Deprecated: use 'files' and
      'crop_sizes' instead.
    crop_sizes: List of crop sizes. If > 0, images are cropped to
      crop_size x crop_size using tensorflow's random cropping.
    augmentation_fns: A Dict of Callables to data augmentation functions.
  Returns:
    A tensorflow dataset for accessing examples that contain the input images
    'x0', 'x1', ground truth 'y' and time of the ground truth 'time'=[0,1] in a
    dictionary of tensors.
  """
  if file:
    logging.warning('gin-configurable training_dataset.file is deprecated. '
                    'Use training_dataset.files instead.')
    return _create_from_sharded_tfrecord(batch_size, True, file,
                                         augmentation_fns, crop_size)
  else:
    if not crop_sizes or len(crop_sizes) != len(files):
      raise ValueError('Please pass crop_sizes[] with training_dataset.files.')
    if crop_size > 0:
      raise ValueError(
          'crop_size should not be used with files[], use crop_sizes[] instead.'
      )
    tables = []
    for file, crop_size in zip(files, crop_sizes):
      tables.append(
          _create_from_sharded_tfrecord(batch_size, True, file,
                                        augmentation_fns, crop_size))
    return tf.data.experimental.sample_from_datasets(tables)


@gin.configurable('eval_datasets')
def create_eval_datasets(batch_size: int,
                         files: List[str],
                         names: List[str],
                         crop_size: int = -1,
                         max_examples: int = -1) -> Dict[str, tf.data.Dataset]:
  """Creates the evaluation datasets.

  As opposed to create_training_dataset this function makes sure that the
  examples for each dataset are always read in a deterministic (same) order.

  Each given tfrecord should contain data in a format produced by
  frame_interpolation/datasets/create_*_tfrecord.py

  The (batch_size, crop_size, max_examples) are specified for all eval datasets.

  Args:
    batch_size: The number of images to batch per example.
    files: List of paths to a sharded tfrecord in <tfrecord>@N format.
    names: List of names of eval datasets.
    crop_size: If > 0, images are cropped to crop_size x crop_size using
      tensorflow's random cropping.
    max_examples: If > 0, truncate the dataset to 'max_examples' in length. This
      can be useful for speeding up evaluation loop in case the tfrecord for the
      evaluation set is very large.
  Returns:
    A dict of name to tensorflow dataset for accessing examples that contain the
    input images 'x0', 'x1', ground truth 'y' and time of the ground truth
    'time'=[0,1] in a dictionary of tensors.
  """
  return {
      name: _create_from_sharded_tfrecord(batch_size, False, file, None,
                                          crop_size, max_examples)
      for name, file in zip(names, files)
  }
