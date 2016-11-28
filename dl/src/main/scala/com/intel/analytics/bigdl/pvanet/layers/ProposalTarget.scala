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

package com.intel.analytics.bigdl.pvanet.layers

import breeze.linalg.{DenseMatrix, DenseVector, min, sum}
import breeze.numerics.round
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.pvanet.{Bbox, Config, MatrixUtil}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag
import scala.util.Random

/**
 * Assign object detection proposals to ground-truth targets. Produces proposal
 * classification labels and bounding-box regression targets.
 *
 * @param ev$1
 * @param ev
 * @tparam T
 */
class ProposalTarget[@specialized(Float, Double) T: ClassTag](numClasses: Int)
  (implicit ev: TensorNumeric[T]) extends Module[Table, Table, T] {

  /**
   * Compute bounding-box regression targets for an image.
   *
   * @param ex_rois
   * @param gt_rois
   * @param labels
   */
  def computeTargets(ex_rois: DenseMatrix[Float],
    gt_rois: DenseMatrix[Float],
    labels: Array[Float]): DenseMatrix[Float] = {
    assert(ex_rois.rows == gt_rois.rows)
    assert(ex_rois.cols == 4)
    assert(gt_rois.cols == 4)

    val targets = Bbox.bboxTransform(ex_rois, gt_rois)

    if (Config.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED) {
      // Optionally normalize targets by a precomputed mean and stdev
      for (r <- 0 until targets.rows) {
        targets(r, ::) :-= DenseVector(Config.TRAIN.BBOX_NORMALIZE_MEANS).t
        targets(r, ::) :/= DenseVector(Config.TRAIN.BBOX_NORMALIZE_STDS).t
      }
    }
    DenseMatrix.horzcat(DenseVector(labels).toDenseMatrix.t, targets)
  }

  /**
   * Bounding-box regression targets (bbox_target_data) are stored in a
   * compact form N x (class, tx, ty, tw, th)
   * *
   * This function expands those targets into the 4-of-4*K representation used
   * by the network (i.e. only one class has non-zero targets).
   * *
   * Returns:
   * bbox_target (ndarray): N x 4K blob of regression targets
   * bbox_inside_weights (ndarray): N x 4K blob of loss weights
   *
   * @param bbox_target_data
   * @param numClasses
   */
  def getBboxRegressionLabels(bbox_target_data: DenseMatrix[Float],
    numClasses: Int): (DenseMatrix[Float], DenseMatrix[Float]) = {
    val clss = bbox_target_data(::, 0).toArray
    val bbox_targets = DenseMatrix.zeros[Float](clss.size, 4 * numClasses)
    val bbox_inside_weights = DenseMatrix.zeros[Float](bbox_targets.rows, bbox_targets.cols)
    val inds = clss.zipWithIndex.filter(x => x._1 > 0).map(x => x._2)
    inds.foreach(ind => {
      val cls = clss(ind)
      val start = 4 * cls
      val end = start + 4
      (1 until bbox_target_data.cols).foreach(x => {
        bbox_targets(ind, (x + start.toInt - 1)) = bbox_target_data(ind, x)
        bbox_inside_weights(ind, (x + start.toInt - 1)) = Config.TRAIN.BBOX_INSIDE_WEIGHTS(x - 1)
      })
    })
    (bbox_targets, bbox_inside_weights)
  }

  // Generate a random sample of RoIs comprising foreground and background examples.
  def sampleRois(all_rois: DenseMatrix[Float],
    gt_boxes: DenseMatrix[Float],
    fg_rois_per_image: Int,
    rois_per_image: Int,
    numClasses: Int)
  : (Array[Float], DenseMatrix[Float], DenseMatrix[Float], DenseMatrix[Float]) = {
    // overlaps: (rois x gt_boxes)
    val overlaps = Bbox.bboxOverlap(all_rois(::, 1 until 5), gt_boxes(::, 0 until 4))
    val gt_assignment = MatrixUtil.argmax2(overlaps, 1)
    val max_overlaps = MatrixUtil.max2(overlaps, 1)
    var labels = MatrixUtil.selectMatrix(gt_boxes, gt_assignment, 0)(::, 4).toArray

    // Select foreground RoIs as those with >= FG_THRESH overlap
    var fg_inds = max_overlaps.zipWithIndex
      .filter(x => x._1 >= Config.TRAIN.FG_THRESH).map(x => x._2)
    // Guard against the case when an image has fewer than fg_rois_per_image
    // foreground RoIs
    val fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    // Sample foreground regions without replacement
    if (fg_inds.size > 0) {
      fg_inds = Random.shuffle(fg_inds.toList).slice(0, fg_rois_per_this_image).toArray
    }

    // Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    var bg_inds = max_overlaps.zipWithIndex
      .filter(x => (x._1 < Config.TRAIN.BG_THRESH_HI) && (x._1 >= Config.TRAIN.BG_THRESH_LO))
      .map(x => x._2)
    // Compute number of background RoIs to take from this image (guarding
    // against there being fewer than desired)
    var bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    // Sample background regions without replacement
    if (bg_inds.length > 0) {
      bg_inds = Random.shuffle(bg_inds.toList).slice(0, bg_rois_per_this_image).toArray
    }

    // The indices that we're selecting (both fg and bg)
    var keep_inds = fg_inds ++ bg_inds
    // Select sampled values from various arrays:
    labels = keep_inds.map(x => labels(x))
    // Clamp labels for the background RoIs to 0
    Range(fg_rois_per_this_image, labels.length).map(i => labels(i) = 0)

    val rois = MatrixUtil.selectMatrix(all_rois, keep_inds, 0)

    val keepInds2 = keep_inds.map(x => gt_assignment(x))
    val bbox_target_data = computeTargets(
      rois(::, 1 until 5), MatrixUtil.selectMatrix(gt_boxes, keepInds2, 0)(::, 0 until 4), labels)

    val (bbox_targets, bbox_inside_weights) =
      getBboxRegressionLabels(bbox_target_data, numClasses)

    (labels, rois, bbox_targets, bbox_inside_weights)
  }

  override def updateOutput(input: Table): Table = {

    // Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    // (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    val all_roisTen = input(1).asInstanceOf[Tensor[Float]]
    // GT boxes (x1, y1, x2, y2, label)
    // and other times after box coordinates -- normalize to one format
    val gt_boxes = input(2).asInstanceOf[Tensor[Float]].toBreezeMatrix()

    // Include ground-truth boxes in the set of candidate rois
    val zeros = DenseMatrix.zeros[Float](gt_boxes.rows, 1)
    val all_rois = DenseMatrix.vertcat(all_roisTen.toBreezeMatrix(),
      DenseMatrix.horzcat(zeros, gt_boxes(::, 0 until gt_boxes.cols - 1)))
    // Sanity check: single batch only
    assert(all_rois(::, 0).forall(x => x == 0), "Only single item batches are supported")

    val numImages = 1
    val rois_per_image = Config.TRAIN.BATCH_SIZE / numImages

    val fg_rois_per_image = round(Config.TRAIN.FG_FRACTION * rois_per_image).toInt

    // Sample rois with classification labels and bounding box regression
    // targets
    val (labels, rois, bbox_targets, bbox_inside_weights) = sampleRois(
      all_rois, gt_boxes, fg_rois_per_image,
      rois_per_image, numClasses)

    if (Config.DEBUG) {
      println("num fg: %d ".format(sum(labels.map(x => if (x > 0) 1 else 0))))
      println("num bg: %d ".format(sum(labels.map(x => if (x == 0) 1 else 0))))
    }


    // sampled rois (0, x1, y1, x2, y2) (1,5)
    output.insert(matrix2tensor(rois))
    // labels (1,1)
    output.insert(Tensor(Storage(labels)))
    // bbox_targets (1, numClasses * 4)
    output.insert(matrix2tensor(bbox_targets))
    // bbox_inside_weights (1, numClasses * 4)
    output.insert(matrix2tensor(bbox_inside_weights))
    // bbox_outside_weights (1, numClasses * 4)
    output.insert(matrix2tensor(bbox_inside_weights.map(x => if (x > 0) 1f else 0f)))
    output
  }

  def matrix2tensor(mat: DenseMatrix[Float]): Tensor[Float] = {
    val out = Tensor[Float]().resize(mat.rows, mat.cols)
    for (i <- 0 until mat.rows) {
      for (j <- 0 until mat.cols) {
        out.setValue(i + 1, j + 1, mat(i, j))
      }
    }
    out
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    null
  }
}
