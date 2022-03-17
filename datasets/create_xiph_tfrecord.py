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
r"""Beam pipeline that generates Xiph triplet TFRecords.

Xiph is a frame sequence dataset commonly used to assess video compression. See
here: https://media.xiph.org/video/derf/

The SoftSplat paper selected eight 4K clips with the most amount of motion and
extracted the first 100 frames from each clip. Each frame is then either resized
from 4K to 2K, or a 2K center crop from them is performed before interpolating
the even frames from the odd frames. These datasets are denoted as `Xiph-2K`
and `Xiph-4K` respectively. For more information see the project page:
https://github.com/sniklaus/softmax-splatting

Input is the root folder that contains the 800 frames of the eight clips. Set
center_crop_factor=2 and scale_factor=1 to generate `Xiph-4K`,and scale_factor=2
, center_crop_factor=1 to generate `Xiph-2K`. The scripts defaults to `Xiph-2K`.

Output TFRecord is a tf.train.Example proto of each image triplet.
The feature_map takes the form:
  feature_map {
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

Usage example:
  python3 -m frame_interpolation.datasets.create_xiph_tfrecord \
    --input_dir=<root folder of xiph dataset> \
    --scale_factor=<scale factor for image resizing, default=2> \
    --center_crop_factor=<center cropping factor, default=1> \
    --output_tfrecord_filepath=<output tfrecord filepath>
"""
import os

from . import util
from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import tensorflow as tf

_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    default='/root/path/to/selected/xiph/clips',
    help='Path to the root directory of the `Xiph` interpolation evaluation '
    'data. We expect the data to have been downloaded and unzipped.')
_CENTER_CROP_FACTOR = flags.DEFINE_integer(
    'center_crop_factor',
    default=1,
    help='Factor to center crop image. If set to 2, an image of the same '
    'resolution as the inputs but half the size is created.')
_SCALE_FACTOR = flags.DEFINE_integer(
    'scale_factor',
    default=2,
    help='Factor to downsample frames.')
_NUM_CLIPS = flags.DEFINE_integer(
    'num_clips', default=8, help='Number of clips.')
_NUM_FRAMES = flags.DEFINE_integer(
    'num_frames', default=100, help='Number of frames per clip.')
_OUTPUT_TFRECORD_FILEPATH = flags.DEFINE_string(
    'output_tfrecord_filepath',
    default=None,
    required=True,
    help='Filepath to the output TFRecord file.')
_NUM_SHARDS = flags.DEFINE_integer('num_shards',
    default=2,
    help='Number of shards used for the output.')

# Image key -> offset for frame interpolator: start / middle / end frame offset.
_INTERPOLATOR_IMAGES_MAP = {
    'frame_0': -1,
    'frame_1': 0,
    'frame_2': 1,
}


def main(unused_argv):
  """Creates and runs a Beam pipeline to write frame triplets as a TFRecord."""
  # Collect the list of frame filenames.
  frames_list = sorted(tf.io.gfile.listdir(_INPUT_DIR.value))

  # Collect the triplets, even frames serving as golden to interpolate odds.
  triplets_dict = []
  for clip_index in range(_NUM_CLIPS.value):
    for frame_index in range(1, _NUM_FRAMES.value - 1, 2):
      index = clip_index * _NUM_FRAMES.value + frame_index
      triplet_dict = {
          image_key: os.path.join(_INPUT_DIR.value,
                                  frames_list[index + image_offset])
          for image_key, image_offset in _INTERPOLATOR_IMAGES_MAP.items()
      }
      triplets_dict.append(triplet_dict)

  p = beam.Pipeline('DirectRunner')
  (p | 'ReadInputTripletDicts' >> beam.Create(triplets_dict)  # pylint: disable=expression-not-assigned
   | 'GenerateSingleExample' >> beam.ParDo(
       util.ExampleGenerator(_INTERPOLATOR_IMAGES_MAP, _SCALE_FACTOR.value,
                             _CENTER_CROP_FACTOR.value))
   | 'WriteToTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
       file_path_prefix=_OUTPUT_TFRECORD_FILEPATH.value,
       num_shards=_NUM_SHARDS.value,
       coder=beam.coders.BytesCoder()))
  result = p.run()
  result.wait_until_finish()

  logging.info('Succeeded in creating the output TFRecord file: \'%s@%s\'.',
    _OUTPUT_TFRECORD_FILEPATH.value, str(_NUM_SHARDS.value))

if __name__ == '__main__':
  app.run(main)
