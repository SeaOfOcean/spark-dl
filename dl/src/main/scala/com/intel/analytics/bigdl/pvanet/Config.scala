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

package com.intel.analytics.bigdl.pvanet

import java.io.File

import com.intel.analytics.bigdl.pvanet.datasets.Imdb

object Config {

  val DEBUG: Boolean = true

  var _feat_stride = 16


  object TRAIN {
    // Scales to use during training (can list multiple scales)
    // Each scale is the pixel size of an image"s shortest side
    var SCALES = List(600)

    // Resize test images so that its width and height are multiples of ...
    var SCALE_MULTIPLE_OF = 1

    // Max pixel size of the longest side of a scaled input image
    var MAX_SIZE = 1000

    // Images to use per minibatch
    var IMS_PER_BATCH = 1

    // Minibatch size (number of regions of interest [ROIs])
    var BATCH_SIZE = 3

    // Fraction of minibatch that is labeled foreground (i.e. class > 0)
    var FG_FRACTION = 0.25

    // Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    var FG_THRESH = 0.5

    // Overlap threshold for a ROI to be considered background (class = 0 if
    // overlap in [LO, HI))
    var BG_THRESH_HI = 0.5
    var BG_THRESH_LO = 0.1

    // Use horizontally-flipped images during training?
    // todo: change tmp
    var USE_FLIPPED = false

    // Train bounding-box regressors
    var BBOX_REG = true

    // Overlap required between a ROI and ground-truth box in order for that ROI to
    // be used as a bounding-box regression training example
    var BBOX_THRESH = 0.5

    // Iterations between snapshots
    var SNAPSHOT_ITERS = 10000

    // solver.prototxt specifies the snapshot path prefix, this adds an optional
    // infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
    var SNAPSHOT_INFIX = ""

    // Use a prefetch thread in roi_data_layer.layer
    // So far I haven"t found this useful; likely more engineering work is required
    var USE_PREFETCH = false

    // Normalize the targets (subtract empirical mean, divide by empirical stddev)
    var BBOX_NORMALIZE_TARGETS = true
    // Deprecated (inside weights)
    var BBOX_INSIDE_WEIGHTS = Array(1.0f, 1.0f, 1.0f, 1.0f)
    // Normalize the targets using "precomputed" (or made up) means and stdevs
    // (BBOX_NORMALIZE_TARGETS must also be true)
    var BBOX_NORMALIZE_TARGETS_PRECOMPUTED = false
    var BBOX_NORMALIZE_MEANS = Array(0.0f, 0.0f, 0.0f, 0.0f)
    var BBOX_NORMALIZE_STDS = Array(0.1f, 0.1f, 0.2f, 0.2f)

    // Train using these proposals
    var PROPOSAL_METHOD = "gt"

    // Make minibatches from images that have similar aspect ratios (i.e. both
    // tall and thin or both short and wide) in order to avoid wasting computation
    // on zero-padding.
    var ASPECT_GROUPING = true

    // Use RPN to detect objects
    var HAS_RPN = true
    // IOU >= thresh: positive example
    var RPN_POSITIVE_OVERLAP = 0.7
    // IOU < thresh: negative example
    var RPN_NEGATIVE_OVERLAP = 0.3
    // If an anchor statisfied by positive and negative conditions set to negative
    var RPN_CLOBBER_POSITIVES = false
    // Max number of foreground examples
    var RPN_FG_FRACTION = 0.5
    // Total number of examples
    var RPN_BATCHSIZE = 256
    // NMS threshold used on RPN proposals
    var RPN_NMS_THRESH = 0.7
    // Number of top scoring boxes to keep before apply NMS to RPN proposals
    var RPN_PRE_NMS_TOP_N = 12000
    // Number of top scoring boxes to keep after applying NMS to RPN proposals
    var RPN_POST_NMS_TOP_N = 2000
    // Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
    var RPN_MIN_SIZE = 16
    // Deprecated (outside weights)
    var RPN_BBOX_INSIDE_WEIGHTS = Array(1.0, 1.0, 1.0, 1.0)
    // Give the positive RPN examples weight of p * 1 / {num positives}
    // and give negatives a weight of (1 - p)
    // Set to -1.0 to use uniform example weighting
    var RPN_POSITIVE_WEIGHT = -1.0
  }

  // Testing options
  object TEST {


    // Scales to use during testing (can list multiple scales)
    // Each scale is the pixel size of an image"s shortest side
    var SCALES = List(600)

    // Resize test images so that its width and height are multiples of ...
    var SCALE_MULTIPLE_OF = 1

    // Max pixel size of the longest side of a scaled input image
    var MAX_SIZE = 1000

    // Overlap threshold used for non-maximum suppression (suppress boxes with
    // IoU >= this threshold)
    var NMS = 0.3

    // Experimental: treat the (K+1) units in the cls_score layer as linear
    // predictors (trained, eg, with one-vs-rest SVMs).
    var SVM = false

    // Test using bounding-box regressors
    var BBOX_REG = true

    // Propose boxes
    var HAS_RPN = false

    // Test using these proposals
    var PROPOSAL_METHOD = "gt"

    // NMS threshold used on RPN proposals
    var RPN_NMS_THRESH = 0.7
    // Number of top scoring boxes to keep before apply NMS to RPN proposals
    var RPN_PRE_NMS_TOP_N = 6000
    // Number of top scoring boxes to keep after applying NMS to RPN proposals
    var RPN_POST_NMS_TOP_N = 300
    // Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
    var RPN_MIN_SIZE = 16

    // Apply bounding box voting
    var BBOX_VOTE = false

  }

  // MISC

  // The mapping from image coordinates to feature map coordinates might cause
  // some boxes that are distinct in image space to become identical in feature
  // coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
  // for identifying duplicate boxes.
  // 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
  var DEDUP_BOXES = 1.0 / 16.0

  // Pixel mean values (BGR order) as a (1, 1, 3) array
  // We use the same pixel mean for all networks even though it"s not exactly what
  // they were trained with
  var PIXEL_MEANS = List(List(List(102.9801, 115.9465, 122.7717)))

  // For reproducibility
  var RNG_SEED = 3

  // A small number that"s used many times
  var EPS = 1e-14

  // Root directory of project
  var ROOT_DIR = System.getProperty("user.dir")

  // Data directory
  var DATA_DIR = ROOT_DIR + "/data"

  // Model directory
  var MODELS_DIR = ROOT_DIR + "/models/pascal_voc"

  // Name (or path to) the matlab executable
  var MATLAB = "matlab"

  // Place outputs under an experiments directory
  var EXP_DIR = "default"

  // Use GPU implementation of non-maximum suppression
  var USE_GPU_NMS = true

  // Default GPU device id
  var GPU_ID = 0


  def getOutputDir(imdb: Imdb, netName: String): String = {
    // Return the directory where experimental artifacts are placed.
    // If the directory does not exist, it is created.

    // A canonical path is built using the name from an imdb and a network
    // (if not None).
    var outdir = ROOT_DIR + "/" + EXP_DIR + "/" + imdb.name
    if (netName != None && !netName.isEmpty) {
      outdir = outdir + "/" + netName
    }
    if (!new File(outdir).exists()) {
      new File(outdir).mkdirs()
    }
    outdir
  }

  def getOutputDir(imdb: Imdb): String = {
    getOutputDir(imdb, "")
  }

  def cachePath: String = {
    val path = DATA_DIR + "/cache"
    if (!existFile(path)) new File(path.toString).mkdirs()
    path
  }

  def modelPath: String = {
    val path = DATA_DIR + "/model"
    if (!existFile(path)) new File(path).mkdirs()
    path
  }

  def existFile(f: String): Boolean = new java.io.File(f).exists()
}
