/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.pvanet.model

import com.intel.analytics.bigdl.optim.SGD.LearningRateSchedule
import com.intel.analytics.bigdl.optim.{OptimMethod, Trigger}
import com.intel.analytics.bigdl.pvanet.model.Model.ModelType
import com.intel.analytics.bigdl.pvanet.model.Phase.PhaseType


object Phase extends Enumeration {
  type PhaseType = Value
  val TRAIN, TEST, FINETUNE = Value

}

object Model extends Enumeration {
  type ModelType = Value
  val VGG16, PVANET = Value
}

abstract class FasterRcnnParam(phase: PhaseType = Phase.TEST) {
  val anchorScales: Array[Float]
  val anchorRatios: Array[Float]
  val anchorNum: Int
  val featStride = 16
  var numClasses: Int = 21

  // Pixel mean values (BGR order) as a (1, 1, 3) array
  // We use the same pixel mean for all networks even though it"s not exactly what
  // they were trained with
  var PIXEL_MEANS = List(List(List(102.9801, 115.9465, 122.7717)))

  // Scales to use during training (can list multiple scales)
  // Each scale is the pixel size of an image"s shortest side
  var SCALES = Array(600)

  // Resize test images so that its width and height are multiples of ...
  val SCALE_MULTIPLE_OF = 1

  // Max pixel size of the longest side of a scaled input image
  val MAX_SIZE = 1000

  // Images to use per minibatch
  var IMS_PER_BATCH = 1

  // Minibatch size (number of regions of interest [ROIs])
  val BATCH_SIZE = 3

  // Fraction of minibatch that is labeled foreground (i.e. class > 0)
  val FG_FRACTION = 0.25

  // Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
  val FG_THRESH = 0.5

  // Overlap threshold for a ROI to be considered background (class = 0 if
  // overlap in [LO, HI))
  val BG_THRESH_HI = 0.5
  val BG_THRESH_LO = 0.1

  // Use horizontally-flipped images during training?
  // todo: change tmp
  val USE_FLIPPED = false

  // Overlap required between a ROI and ground-truth box in order for that ROI to
  // be used as a bounding-box regression training example
  val BBOX_THRESH = 0.5

  // Iterations between snapshots
  val SNAPSHOT_ITERS = 10000

  // solver.prototxt specifies the snapshot path prefix, this adds an optional
  // infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
  val SNAPSHOT_INFIX = ""

  // Use a prefetch thread in roi_data_layer.layer
  // So far I haven"t found this useful; likely more engineering work is required
  val USE_PREFETCH = false

  // Normalize the targets (subtract empirical mean, divide by empirical stddev)
  val BBOX_NORMALIZE_TARGETS = true
  // Deprecated (inside weights)
  val BBOX_INSIDE_WEIGHTS = Array(1.0f, 1.0f, 1.0f, 1.0f)
  // Normalize the targets using "precomputed" (or made up) means and stdevs
  // (BBOX_NORMALIZE_TARGETS must also be true)
  var BBOX_NORMALIZE_TARGETS_PRECOMPUTED = false
  val BBOX_NORMALIZE_MEANS = Array(0.0f, 0.0f, 0.0f, 0.0f)
  val BBOX_NORMALIZE_STDS = Array(0.1f, 0.1f, 0.2f, 0.2f)

  // Make minibatches from images that have similar aspect ratios (i.e. both
  // tall and thin or both short and wide) in order to avoid wasting computation
  // on zero-padding.
  val ASPECT_GROUPING = true

  // IOU >= thresh: positive example
  val RPN_POSITIVE_OVERLAP = 0.7
  // IOU < thresh: negative example
  val RPN_NEGATIVE_OVERLAP = 0.3
  // If an anchor statisfied by positive and negative conditions set to negative
  val RPN_CLOBBER_POSITIVES = false
  // Max number of foreground examples
  val RPN_FG_FRACTION = 0.5
  // Total number of examples
  val RPN_BATCHSIZE = 256
  // NMS threshold used on RPN proposals
  val RPN_NMS_THRESH = 0.7
  // Number of top scoring boxes to keep before apply NMS to RPN proposals
  var RPN_PRE_NMS_TOP_N = 12000
  // Number of top scoring boxes to keep after applying NMS to RPN proposals
  var RPN_POST_NMS_TOP_N = 2000
  // Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
  val RPN_MIN_SIZE = 16
  // Deprecated (outside weights)
  val RPN_BBOX_INSIDE_WEIGHTS = Array(1.0, 1.0, 1.0, 1.0)
  // Give the positive RPN examples weight of p * 1 / {num positives}
  // and give negatives a weight of (1 - p)
  // Set to -1.0 to use uniform example weighting
  val RPN_POSITIVE_WEIGHT = -1.0

  // Overlap threshold used for non-maximum suppression (suppress boxes with
  // IoU >= this threshold)
  val NMS = 0.3

  // Apply bounding box voting
  val BBOX_VOTE = false

  var BBOX_REG = true

  val optimizeConfig: OptimizeConfig
}

case class OptimizeConfig(
  optimMethod: OptimMethod[Float],
  momentum: Double,
  weightDecay: Double,
  testTrigger: Trigger,
  cacheTrigger: Trigger,
  endWhen: Trigger,
  learningRate: Double,
  learningRateSchedule: LearningRateSchedule
)

object FasterRcnnParam {
  def getNetParam(net: ModelType, phase: PhaseType): FasterRcnnParam = {
    net match {
      case Model.VGG16 => new VggParam(phase)
      case Model.PVANET => new PvanetParam(phase)
      case _ => throw new UnsupportedOperationException
    }
  }
}

