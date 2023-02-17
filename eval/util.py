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
"""Utility functions for frame interpolation on a set of video frames."""
import os
import shutil
from typing import Generator, Iterable, List, Optional

from . import interpolator as interpolator_lib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
_CONFIG_FFMPEG_NAME_OR_PATH = 'ffmpeg'


def read_image(filename: str) -> np.ndarray:
  """Reads an sRgb 16-bit image.

  Args:
    filename: The input filename to read.

  Returns:
    A float32 3-channel (RGB) ndarray with colors in the [0..1] range.
  """
  image_data = tf.io.read_file(filename)
  image = tf.io.decode_png(image_data, channels=3, dtype=tf.uint16)
  # Rescale to [0..1] range from [0..2^16-1]
  image_numpy = tf.cast(image, dtype=tf.float32) / (2**16 - 1)
  # Apply gamma correction, could be 2.2
  image_numpy = tf.pow(image_numpy, 1)

  return image_numpy


def write_image(filename: str, image: np.ndarray) -> None:
  """Writes a float32 3-channel RGB ndarray image, with colors in range [0..1].

  Args:
    filename: The output filename to save.
    image: A float32 3-channel (RGB) ndarray with colors in the [0..1] range.
  """
  image_in_uint16_range = np.clip(image * (2**16 - 1), 0, 2**16 - 1)
  # Add some noise to the image to reduce color banding
  noise = np.random.normal(scale=2, size=image_in_uint16_range.shape)
  noisy_image = image_in_uint16_range + noise
  # Convert to uint16
  noisy_image = np.clip(noisy_image, 0, 2**16 - 1).astype(np.uint16)

  # Use PNG with 48-bit compression
  image_data = tf.image.encode_png(noisy_image, compression=-1)
  tf.io.write_file(filename, image_data)


def _recursive_generator(
    frame1: np.ndarray, frame2: np.ndarray, num_recursions: int,
    interpolator: interpolator_lib.Interpolator,
    bar: Optional[tqdm] = None
) -> Generator[np.ndarray, None, None]:
  """Splits halfway to repeatedly generate more frames.

  Args:
    frame1: Input image 1.
    frame2: Input image 2.
    num_recursions: How many times to interpolate the consecutive image pairs.
    interpolator: The frame interpolator instance.

  Yields:
    The interpolated frames, including the first frame (frame1), but excluding
    the final frame2.
  """
  if num_recursions == 0:
    yield frame1
  else:
    # Adds the batch dimension to all inputs before calling the interpolator,
    # and remove it afterwards.
    time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    mid_frame = interpolator(frame1[np.newaxis, ...], frame2[np.newaxis, ...],
                             time)[0]
    bar.update(1) if bar is not None else bar
    yield from _recursive_generator(frame1, mid_frame, num_recursions - 1,
                                    interpolator, bar)
    yield from _recursive_generator(mid_frame, frame2, num_recursions - 1,
                                    interpolator, bar)


def interpolate_recursively_from_files(
    frames: List[str], times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  Loads the files on demand and uses the yield paradigm to return the frames
  to allow streamed processing of longer videos.

  Recursive interpolation is useful if the interpolator is trained to predict
  frames at midpoint only and is thus expected to perform poorly elsewhere.

  Args:
    frames: List of input frames. Expected shape (H, W, 3). The colors should be
      in the range[0, 1] and in gamma space.
    times_to_interpolate: Number of times to do recursive midpoint
      interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frames)
  num_frames = (n - 1) * (2**(times_to_interpolate) - 1)
  bar = tqdm(total=num_frames, ncols=100, colour='green')
  for i in range(1, n):
    yield from _recursive_generator(
        read_image(frames[i - 1]), read_image(frames[i]), times_to_interpolate,
        interpolator, bar)
  # Separately yield the final frame.
  yield read_image(frames[-1])

def interpolate_recursively_from_memory(
    frames: List[np.ndarray], times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  This is functionally equivalent to interpolate_recursively_from_files(), but
  expects the inputs frames in memory, instead of loading them on demand.

  Recursive interpolation is useful if the interpolator is trained to predict
  frames at midpoint only and is thus expected to perform poorly elsewhere.

  Args:
    frames: List of input frames. Expected shape (H, W, 3). The colors should be
      in the range[0, 1] and in gamma space.
    times_to_interpolate: Number of times to do recursive midpoint
      interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frames)
  num_frames = (n - 1) * (2**(times_to_interpolate) - 1)
  bar = tqdm(total=num_frames, ncols=100, colour='green')
  for i in range(1, n):
    yield from _recursive_generator(frames[i - 1], frames[i],
                                    times_to_interpolate, interpolator, bar)
  # Separately yield the final frame.
  yield frames[-1]


def get_ffmpeg_path() -> str:
  path = shutil.which(_CONFIG_FFMPEG_NAME_OR_PATH)
  if not path:
    raise RuntimeError(
        f"Program '{_CONFIG_FFMPEG_NAME_OR_PATH}' is not found;"
        " perhaps install ffmpeg using 'apt-get install ffmpeg'.")
  return path