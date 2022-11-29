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
r"""Runs the FILM frame interpolator on a pair of frames on beam.

This script is used evaluate the output quality of the FILM Tensorflow frame
interpolator. Optionally, it outputs a video of the interpolated frames.

A beam pipeline for invoking the frame interpolator on a set of directories
identified by a glob (--pattern). Each directory is expected to contain two
input frames that are the inputs to the frame interpolator. If a directory has
more than two frames, then each contiguous frame pair is treated as input to
generate in-between frames.

The output video is stored to interpolator.mp4 in each directory. The number of
frames is determined by --times_to_interpolate, which controls the number of
times the frame interpolator is invoked. When the number of input frames is 2,
the number of output frames is 2^times_to_interpolate+1.

This expects a directory structure such as:
  <root directory of the eval>/01/frame1.png
                                  frame2.png
  <root directory of the eval>/02/frame1.png
                                  frame2.png
  <root directory of the eval>/03/frame1.png
                                  frame2.png
  ...

And will produce:
  <root directory of the eval>/01/interpolated_frames/frame0.png
                                                      frame1.png
                                                      frame2.png
  <root directory of the eval>/02/interpolated_frames/frame0.png
                                                      frame1.png
                                                      frame2.png
  <root directory of the eval>/03/interpolated_frames/frame0.png
                                                      frame1.png
                                                      frame2.png
  ...

And optionally will produce:
  <root directory of the eval>/01/interpolated.mp4
  <root directory of the eval>/02/interpolated.mp4
  <root directory of the eval>/03/interpolated.mp4
  ...

Usage example:
  python3 -m frame_interpolation.eval.interpolator_cli \
    --model_path <path to TF2 saved model> \
    --pattern "<root directory of the eval>/*" \
    --times_to_interpolate <Number of times to interpolate>
"""

import functools
import os
from typing import List, Sequence

from . import interpolator as interpolator_lib
from . import util
from absl import app
from absl import flags
from absl import logging
from tqdm.autonotebook import tqdm
import apache_beam as beam
import mediapy as media
import natsort
import numpy as np
import tensorflow as tf


_PATTERN = flags.DEFINE_string(
    name='pattern',
    default=None,
    help='The pattern to determine the directories with the input frames.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.')
_TIMES_TO_INTERPOLATE = flags.DEFINE_integer(
    name='times_to_interpolate',
    default=5,
    help='The number of times to run recursive midpoint interpolation. '
    'The number of output frames will be 2^times_to_interpolate+1.')
_FPS = flags.DEFINE_integer(
    name='fps',
    default=60,
    help='Frames per second to play interpolated videos in slow motion.')
_OUTPUT_VIDEO = flags.DEFINE_boolean(
    name='output_video',
    default=False,
    help='If true, creates a video of the frames in the interpolated_frames/ '
    'subdirectory')

# Add other extensions, if not either.
_INPUT_EXT = ['png', 'jpg']


def _output_frames(frames: List[np.ndarray], frames_dir: str):
  """Writes PNG-images to a directory.

  If frames_dir doesn't exist, it is created. If frames_dir contains existing
  PNG-files, they are removed before saving the new ones.

  Args:
    frames: List of images to save.
    frames_dir: The output directory to save the images.

  """
  if tf.io.gfile.isdir(frames_dir):
    old_frames = tf.io.gfile.glob(os.path.join(frames_dir, 'frame_*.png'))
    if old_frames:
      logging.info('Removing existing frames from %s.', frames_dir)
      for old_frame in old_frames:
        tf.io.gfile.remove(old_frame)
  else:
    tf.io.gfile.makedirs(frames_dir)
  for idx, frame in tqdm(enumerate(frames),total=len(frames),ncols=100,colour='green'):
    util.write_image(
        os.path.join(frames_dir, f'frame_{idx:03d}.png'), frame)
  logging.info('Output frames saved in %s.', frames_dir)


class ProcessDirectory(beam.DoFn):
  """DoFn for running the interpolator on a single directory at the time."""

  def setup(self):
    self.interpolator = interpolator_lib.Interpolator(
        _MODEL_PATH.value)

    if _OUTPUT_VIDEO.value:
      ffmpeg_path = util.get_ffmpeg_path()
      media.set_ffmpeg(ffmpeg_path)

  def process(self, directory: str):
    input_frames_list = [
        natsort.natsorted(tf.io.gfile.glob(f'{directory}/*.{ext}'))
        for ext in _INPUT_EXT
    ]
    input_frames = functools.reduce(lambda x, y: x + y, sorted(input_frames_list))
    logging.info('Generating in-between frames for %s.', directory)
    frames = list(
        util.interpolate_recursively_from_files(
            input_frames, _TIMES_TO_INTERPOLATE.value, self.interpolator))
    _output_frames(frames, os.path.join(directory, 'interpolated_frames'))
    if _OUTPUT_VIDEO.value:
      media.write_video(f'{directory}/interpolated.mp4', frames, fps=_FPS.value)
      logging.info('Output video saved at %s/interpolated.mp4.', directory)


def _run_pipeline() -> None:
  directories = tf.io.gfile.glob(_PATTERN.value)
  pipeline = beam.Pipeline('DirectRunner')
  (pipeline | 'Create directory names' >> beam.Create(directories)  # pylint: disable=expression-not-assigned
   | 'Process directories' >> beam.ParDo(ProcessDirectory()))

  result = pipeline.run()
  result.wait_until_finish()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  _run_pipeline()


if __name__ == '__main__':
  app.run(main)
