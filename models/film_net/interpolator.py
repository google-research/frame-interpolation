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
"""The film_net frame interpolator main model code.

Basics
======
The film_net is an end-to-end learned neural frame interpolator implemented as
a TF2 model. It has the following inputs and outputs:

Inputs:
  x0: image A.
  x1: image B.
  time: desired sub-frame time.

Outputs:
  image: the predicted in-between image at the chosen time in range [0, 1].

Additional outputs include forward and backward warped image pyramids, flow
pyramids, etc., that can be visualized for debugging and analysis.

Note that many training sets only contain triplets with ground truth at
time=0.5. If a model has been trained with such training set, it will only work
well for synthesizing frames at time=0.5. Such models can only generate more
in-between frames using recursion.

Architecture
============
The inference consists of three main stages: 1) feature extraction 2) warping
3) fusion. On high-level, the architecture has similarities to Context-aware
Synthesis for Video Frame Interpolation [1], but the exact architecture is
closer to Multi-view Image Fusion [2] with some modifications for the frame
interpolation use-case.

Feature extraction stage employs the cascaded multi-scale architecture described
in [2]. The advantage of this architecture is that coarse level flow prediction
can be learned from finer resolution image samples. This is especially useful
to avoid overfitting with moderately sized datasets.

The warping stage uses a residual flow prediction idea that is similar to
PWC-Net [3], Multi-view Image Fusion [2] and many others.

The fusion stage is similar to U-Net's decoder where the skip connections are
connected to warped image and feature pyramids. This is described in [2].

Implementation Conventions
====================
Pyramids
--------
Throughtout the model, all image and feature pyramids are stored as python lists
with finest level first followed by downscaled versions obtained by successively
halving the resolution. The depths of all pyramids are determined by
options.pyramid_levels. The only exception to this is internal to the feature
extractor, where smaller feature pyramids are temporarily constructed with depth
options.sub_levels.

Color ranges & gamma
--------------------
The model code makes no assumptions on whether the images are in gamma or
linearized space or what is the range of RGB color values. So a model can be
trained with different choices. This does not mean that all the choices lead to
similar results. In practice the model has been proven to work well with RGB
scale = [0,1] with gamma-space images (i.e. not linearized).

[1] Context-aware Synthesis for Video Frame Interpolation, Niklaus and Liu, 2018
[2] Multi-view Image Fusion, Trinidad et al, 2019
[3] PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume
"""

from . import feature_extractor
from . import fusion
from . import options
from . import pyramid_flow_estimator
from . import util
import tensorflow as tf


def create_model(x0: tf.Tensor, x1: tf.Tensor, time: tf.Tensor,
                 config: options.Options) -> tf.keras.Model:
  """Creates a frame interpolator model.

  The frame interpolator is used to warp the two images to the in-between frame
  at given time. Note that training data is often restricted such that
  supervision only exists at 'time'=0.5. If trained with such data, the model
  will overfit to predicting images that are halfway between the two inputs and
  will not be as accurate elsewhere.

  Args:
    x0: first input image as BxHxWxC tensor.
    x1: second input image as BxHxWxC tensor.
    time: ignored by film_net. We always infer a frame at t = 0.5.
    config: FilmNetOptions object.

  Returns:
    A tf.Model that takes 'x0', 'x1', and 'time' as input and returns a
          dictionary with the interpolated result in 'image'. For additional
          diagnostics or supervision, the following intermediate results are
          also stored in the dictionary:
          'x0_warped': an intermediate result obtained by warping from x0
          'x1_warped': an intermediate result obtained by warping from x1
          'forward_residual_flow_pyramid': pyramid with forward residual flows
          'backward_residual_flow_pyramid': pyramid with backward residual flows
          'forward_flow_pyramid': pyramid with forward flows
          'backward_flow_pyramid': pyramid with backward flows

  Raises:
    ValueError, if config.pyramid_levels < config.fusion_pyramid_levels.
  """
  if config.pyramid_levels < config.fusion_pyramid_levels:
    raise ValueError('config.pyramid_levels must be greater than or equal to '
                     'config.fusion_pyramid_levels.')

  x0_decoded = x0
  x1_decoded = x1

  # shuffle images
  image_pyramids = [
      util.build_image_pyramid(x0_decoded, config),
      util.build_image_pyramid(x1_decoded, config)
  ]

  # Siamese feature pyramids:
  extract = feature_extractor.FeatureExtractor('feat_net', config)
  feature_pyramids = [extract(image_pyramids[0]), extract(image_pyramids[1])]

  predict_flow = pyramid_flow_estimator.PyramidFlowEstimator(
      'predict_flow', config)

  # Predict forward flow.
  forward_residual_flow_pyramid = predict_flow(feature_pyramids[0],
                                               feature_pyramids[1])
  # Predict backward flow.
  backward_residual_flow_pyramid = predict_flow(feature_pyramids[1],
                                                feature_pyramids[0])

  # Concatenate features and images:

  # Note that we keep up to 'fusion_pyramid_levels' levels as only those
  # are used by the fusion module.
  fusion_pyramid_levels = config.fusion_pyramid_levels

  forward_flow_pyramid = util.flow_pyramid_synthesis(
      forward_residual_flow_pyramid)[:fusion_pyramid_levels]
  backward_flow_pyramid = util.flow_pyramid_synthesis(
      backward_residual_flow_pyramid)[:fusion_pyramid_levels]

  # We multiply the flows with t and 1-t to warp to the desired fractional time.
  #
  # Note: In film_net we fix time to be 0.5, and recursively invoke the interpo-
  # lator for multi-frame interpolation. Below, we create a constant tensor of
  # shape [B]. We use the `time` tensor to infer the batch size.
  mid_time = tf.keras.layers.Lambda(lambda x: tf.ones_like(x) * 0.5)(time)
  backward_flow = util.multiply_pyramid(backward_flow_pyramid, mid_time[:, 0])
  forward_flow = util.multiply_pyramid(forward_flow_pyramid, 1 - mid_time[:, 0])

  pyramids_to_warp = [
      util.concatenate_pyramids(image_pyramids[0][:fusion_pyramid_levels],
                                feature_pyramids[0][:fusion_pyramid_levels]),
      util.concatenate_pyramids(image_pyramids[1][:fusion_pyramid_levels],
                                feature_pyramids[1][:fusion_pyramid_levels])
  ]

  # Warp features and images using the flow. Note that we use backward warping
  # and backward flow is used to read from image 0 and forward flow from
  # image 1.
  forward_warped_pyramid = util.pyramid_warp(pyramids_to_warp[0], backward_flow)
  backward_warped_pyramid = util.pyramid_warp(pyramids_to_warp[1], forward_flow)

  aligned_pyramid = util.concatenate_pyramids(forward_warped_pyramid,
                                              backward_warped_pyramid)
  aligned_pyramid = util.concatenate_pyramids(aligned_pyramid, backward_flow)
  aligned_pyramid = util.concatenate_pyramids(aligned_pyramid, forward_flow)

  fuse = fusion.Fusion('fusion', config)
  prediction = fuse(aligned_pyramid)

  output_color = prediction[..., :3]
  outputs = {'image': output_color}

  if config.use_aux_outputs:
    outputs.update({
        'x0_warped': forward_warped_pyramid[0][..., 0:3],
        'x1_warped': backward_warped_pyramid[0][..., 0:3],
        'forward_residual_flow_pyramid': forward_residual_flow_pyramid,
        'backward_residual_flow_pyramid': backward_residual_flow_pyramid,
        'forward_flow_pyramid': forward_flow_pyramid,
        'backward_flow_pyramid': backward_flow_pyramid,
    })

  model = tf.keras.Model(
      inputs={
          'x0': x0,
          'x1': x1,
          'time': time
      }, outputs=outputs)
  return model
