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
"""Options for the film_net video frame interpolator."""

import gin.tf


@gin.configurable('film_net')
class Options(object):
  """Options for the film_net video frame interpolator.

  To further understand these options, see the paper here:
  https://augmentedperception.github.io/pixelfusion/.

  The default values are suitable for up to 64 pixel motions. For larger motions
  the number of flow convolutions and/or pyramid levels can be increased, but
  usually with the cost of accuracy on solving the smaller motions.

  The maximum motion in pixels that the system can resolve is equivalent to
  2^(pyramid_levels-1) * flow_convs[-1]. I.e. the downsampling factor times
  the receptive field radius on the coarsest pyramid level. This, of course,
  assumes that the training data contains such motions.

  Note that to avoid a run-time error, the input image width and height have to
  be divisible by 2^(pyramid_levels-1).

  Attributes:
    pyramid_levels: How many pyramid levels to use for the feature pyramid and
      the flow prediction.
    fusion_pyramid_levels: How many pyramid levels to use for the fusion module
      this must be less or equal to 'pyramid_levels'.
    specialized_levels: How many fine levels of the pyramid shouldn't share the
      weights. If specialized_levels = 3, it means that two finest levels are
      independently learned, whereas the third will be learned together with the
      rest of the pyramid. Valid range [1, pyramid_levels].
    flow_convs: Convolutions per residual flow predictor. This array should have
      specialized_levels+1 items on it, the last item representing the number of
      convs used by any pyramid level that uses shared weights.
    flow_filters: Base number of filters in residual flow predictors. This array
      should have specialized_levels+1 items on it, the last item representing
      the number of filters used by any pyramid level that uses shared weights.
    sub_levels: The depth of the cascaded feature tree each pyramid level
      concatenates together to compute the flow. This must be within range [1,
      specialized_level+1]. It is recommended to set this to specialized_levels
      + 1
    filters: Base number of features to extract. On each pyramid level the
      number doubles. This is used by both feature extraction and fusion stages.
    use_aux_outputs: Set to True to include auxiliary outputs along with the
      predicted image.
  """

  def __init__(self,
               pyramid_levels=5,
               fusion_pyramid_levels=5,
               specialized_levels=3,
               flow_convs=None,
               flow_filters=None,
               sub_levels=4,
               filters=16,
               use_aux_outputs=True):
    self.pyramid_levels = pyramid_levels
    self.fusion_pyramid_levels = fusion_pyramid_levels
    self.specialized_levels = specialized_levels
    self.flow_convs = flow_convs or [4, 4, 4, 4]
    self.flow_filters = flow_filters or [64, 128, 256, 256]
    self.sub_levels = sub_levels
    self.filters = filters
    self.use_aux_outputs = use_aux_outputs

