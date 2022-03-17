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
r"""Beam pipeline that generates UCF101 `interp_test` triplet TFRecords.

UCF101 interpolation evaluation dataset consists of 379 triplets, with the
middle frame being the golden intermediate. The dataset is available here:
https://people.cs.umass.edu/~hzjiang/projects/superslomo/UCF101_results.zip.

Input to the script is the root folder that contains the unzipped
`UCF101_results` folder.

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
  python3 -m frame_interpolation.datasets.create_ucf101_tfrecord \
    --input_dir=<root folder of UCF101_results> \
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
    default='/root/path/to/UCF101_results/ucf101_interp_ours',
    help='Path to the root directory of the `UCF101_results` of the UCF101 '
    'interpolation evaluation data. '
    'We expect the data to have been downloaded and unzipped. \n'
    'Folder structures:\n'
    '| raw_UCF101_results/\n'
    '|  ucf101_interp_ours/\n'
    '|  |  1/\n'
    '|  |  |  frame_00.png\n'
    '|  |  |  frame_01_gt.png\n'
    '|  |  |  frame_01_ours.png\n'
    '|  |  |  frame_02.png\n'
    '|  |  2/\n'
    '|  |  |  frame_00.png\n'
    '|  |  |  frame_01_gt.png\n'
    '|  |  |  frame_01_ours.png\n'
    '|  |  |  frame_02.png\n'
    '|  |  ...\n'
    '|  ucf101_sepconv/\n'
    '|  ...\n')

_OUTPUT_TFRECORD_FILEPATH = flags.DEFINE_string(
    'output_tfrecord_filepath',
    default=None,
    required=True,
    help='Filepath to the output TFRecord file.')

_NUM_SHARDS = flags.DEFINE_integer('num_shards',
    default=2,
    help='Number of shards used for the output.')

# Image key -> basename for frame interpolator: start / middle / end frames.
_INTERPOLATOR_IMAGES_MAP = {
    'frame_0': 'frame_00.png',
    'frame_1': 'frame_01_gt.png',
    'frame_2': 'frame_02.png',
}


def main(unused_argv):
  """Creates and runs a Beam pipeline to write frame triplets as a TFRecord."""
  # Collect the list of folder paths containing the input and golden frames.
  triplets_list = tf.io.gfile.listdir(_INPUT_DIR.value)

  triplet_dicts = []
  for triplet in triplets_list:
    triplet_dicts.append({
        image_key: os.path.join(_INPUT_DIR.value, triplet, image_basename)
        for image_key, image_basename in _INTERPOLATOR_IMAGES_MAP.items()
    })

  p = beam.Pipeline('DirectRunner')
  (p | 'ReadInputTripletDicts' >> beam.Create(triplet_dicts)  # pylint: disable=expression-not-assigned
   | 'GenerateSingleExample' >> beam.ParDo(
       util.ExampleGenerator(_INTERPOLATOR_IMAGES_MAP))
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
