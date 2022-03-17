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
r"""Beam pipeline that generates Vimeo-90K (train or test) triplet TFRecords.

Vimeo-90K dataset is built upon 5,846 videos downloaded from vimeo.com. The list
of the original video links are available here:
https://github.com/anchen1011/toflow/blob/master/data/original_vimeo_links.txt.
Each video is further cropped into a fixed spatial size of (448 x 256) to create
89,000 video clips.

The Vimeo-90K dataset is designed for four video processing tasks. This script
creates the TFRecords of frame triplets for frame interpolation task.

Temporal frame interpolation triplet dataset:
  - 73,171 triplets of size (448x256) extracted from 15K subsets of Vimeo-90K.
  - The triplets are pre-split into (train,test) = (51313,3782)
  - Download links:
    Test-set: http://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip
    Train+test-set: http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip

For more information, see the arXiv paper, project page or the GitHub link.
@article{xue17toflow,
  author = {Xue, Tianfan and
            Chen, Baian and
            Wu, Jiajun and
            Wei, Donglai and
            Freeman, William T},
  title = {Video Enhancement with Task-Oriented Flow},
  journal = {arXiv},
  year = {2017}
}
Project: http://toflow.csail.mit.edu/
GitHub: https://github.com/anchen1011/toflow

Inputs to the script are (1) the directory to the downloaded and unzipped folder
(2) the filepath of the text-file that lists the subfolders of the triplets.

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
          tf.io.FixedLenFeature((), tf.int64, default_value=0)
      'path':
          tf.io.FixedLenFeature((), tf.string, default_value='')
  }

Usage example:
  python3 -m frame_interpolation.datasets.create_vimeo90K_tfrecord \
    --input_dir=<root folder of vimeo90K dataset> \
    --input_triplet_list_filepath=<filepath of tri_{test|train}list.txt> \
    --output_tfrecord_filepath=<output tfrecord filepath>
"""
import os

from . import util
from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import numpy as np
import tensorflow as tf


_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    default='/path/to/raw_vimeo_interp/sequences',
    help='Path to the root directory of the vimeo frame interpolation dataset. '
    'We expect the data to have been downloaded and unzipped.\n'
    'Folder structures:\n'
    '| raw_vimeo_dataset/\n'
    '|  sequences/\n'
    '|  |  00001\n'
    '|  |  |  0389/\n'
    '|  |  |  |  im1.png\n'
    '|  |  |  |  im2.png\n'
    '|  |  |  |  im3.png\n'
    '|  |  |  ...\n'
    '|  |  00002/\n'
    '|  |  ...\n'
    '|  readme.txt\n'
    '|  tri_trainlist.txt\n'
    '|  tri_testlist.txt \n')

_INTPUT_TRIPLET_LIST_FILEPATH = flags.DEFINE_string(
    'input_triplet_list_filepath',
    default='/path/to/raw_vimeo_dataset/tri_{test|train}list.txt',
    help='Text file containing a list of sub-directories of input triplets.')

_OUTPUT_TFRECORD_FILEPATH = flags.DEFINE_string(
    'output_tfrecord_filepath',
    default=None,
    help='Filepath to the output TFRecord file.')

_NUM_SHARDS = flags.DEFINE_integer('num_shards',
    default=200, # set to 3 for vimeo_test, and 200 for vimeo_train.
    help='Number of shards used for the output.')

# Image key -> basename for frame interpolator: start / middle / end frames.
_INTERPOLATOR_IMAGES_MAP = {
    'frame_0': 'im1.png',
    'frame_1': 'im2.png',
    'frame_2': 'im3.png',
}


def main(unused_argv):
  """Creates and runs a Beam pipeline to write frame triplets as a TFRecord."""
  with tf.io.gfile.GFile(_INTPUT_TRIPLET_LIST_FILEPATH.value, 'r') as fid:
    triplets_list = np.loadtxt(fid, dtype=str)

  triplet_dicts = []
  for triplet in triplets_list:
    triplet_dict = {
        image_key: os.path.join(_INPUT_DIR.value, triplet, image_basename)
        for image_key, image_basename in _INTERPOLATOR_IMAGES_MAP.items()
    }
    triplet_dicts.append(triplet_dict)
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
