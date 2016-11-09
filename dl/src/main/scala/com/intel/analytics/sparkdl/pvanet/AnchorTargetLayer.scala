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

package com.intel.analytics.sparkdl.pvanet

import java.util.logging.Logger

import breeze.linalg.{*, DenseMatrix, DenseVector, convert, max}
import com.intel.analytics.sparkdl.pvanet.Roidb.ImageWithRoi
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


case class AnchorTarget(val labels: DenseVector[Int],
                        val bboxTargets: DenseMatrix[Float],
                        val bboxInsideWeights: DenseMatrix[Float],
                        val bboxOutsideWeights: DenseMatrix[Float]) {

}

//object AnchorGenerator extends Transformer[ImageWithRoi, ImageWithRoi] {
//  override def transform(prev: Iterator[ImageWithRoi]): Iterator[ImageWithRoi] = {
//    prev.map(data => {
//      data.anchorTarget = Some(AnchorTargetLayer.generateAnchors(data))
//      data
//    })
//  }
//}


class AnchorTargetLayer(anchorScales: Tensor[Float] = Tensor(Storage(Array[Float](3, 6, 9, 16, 32))),
                        anchorRatios: Tensor[Float] = Tensor(Storage(Array(0.5f, 0.667f, 1.0f, 1.5f, 2.0f)))) {
  val logger = Logger.getLogger(this.getClass.getName)
  //todo: now hard code

  val featStride = 16
  val anchors = Anchor.generateAnchors(ratios = anchorRatios, scales = anchorScales)
  val num_anchors = anchors.rows
  //n_scales * n_ratios
  val allowedBorder = 0
  assert(num_anchors == anchorRatios.nElement() * anchorScales.nElement())

  //debug info
  var _counts = Config.EPS
  var _fg_sum = 0
  var _bg_sum = 0
  var _count = 0

  def generateShifts(width: Int, height: Int, featStride: Float): Option[DenseMatrix[Float]] = {
    val shift_x = DenseVector.range(0, width).map(x => x * featStride)
    val shift_y = DenseVector.range(0, height).map(x => x * featStride)
    var shifts = MatrixUtil.meshgrid(shift_x, shift_y) match {
      case Some((x1Mesh, x2Mesh)) => {
        return Some(DenseMatrix.vertcat(x1Mesh.t.toDenseVector.toDenseMatrix,
          x2Mesh.t.toDenseVector.toDenseMatrix,
          x1Mesh.t.toDenseVector.toDenseMatrix,
          x2Mesh.t.toDenseVector.toDenseMatrix).t)
      }
    }
    None
  }

  /**
    * Compute bounding-box regression targets for an image.
    *
    * @param ex_rois
    * @param gt_rois
    * @return
    */
  def _compute_targets(ex_rois: DenseMatrix[Float], gt_rois: DenseMatrix[Float]): DenseMatrix[Float] = {
    require(ex_rois.rows == gt_rois.rows)
    require(ex_rois.cols == 4)
    require(gt_rois.cols == 5)
    Bbox.bboxTransform(ex_rois, MatrixUtil.select(gt_rois, Array.range(0, 4), 1).get)
  }


  def getInsideAnchors(indsInside: ArrayBuffer[Int], allAnchors: DenseMatrix[Float]): DenseMatrix[Float] = {
    var insideAnchors = new DenseMatrix[Float](indsInside.length, 4)
    indsInside.zipWithIndex.foreach(i => {
      insideAnchors(i._2, 0) = allAnchors(i._1, 0)
      insideAnchors(i._2, 1) = allAnchors(i._1, 1)
      insideAnchors(i._2, 2) = allAnchors(i._1, 2)
      insideAnchors(i._2, 3) = allAnchors(i._1, 3)
    })
    insideAnchors
  }

  /**
    * Algorithm:
    *
    * for each (H, W) location i
    * --- generate scale_size * ratio_size anchor boxes centered on cell i
    * --- apply predicted bbox deltas at cell i to each of the 9 anchors
    * filter out-of-image anchors
    * measure GT overlap
    */

  def generateAnchors(data: ImageWithRoi, height: Int, width: Int): AnchorTarget = {
    logger.info("start generating anchors ----------------------")
    //1. Generate proposals from bbox deltas and shifted anchors
    val shifts = generateShifts(width, height, featStride).get
    val totalAnchors = shifts.rows * num_anchors
    var allAnchors: DenseMatrix[Float] = getAllAnchors(shifts, anchors)
    var indsInside: ArrayBuffer[Int] = getIndsInside(width, height, allAnchors, allowedBorder)


    //keep only inside anchors
    val insideAnchors: DenseMatrix[Float] = getInsideAnchors(indsInside, allAnchors)

    // label: 1 is positive, 0 is negative, -1 is dont care
    var labels = DenseVector.fill(indsInside.length, -1)

    // overlaps between the anchors and the gt boxes
    // overlaps (ex, gt)
    val overlaps = Bbox.bboxOverlap(insideAnchors, data.gt_boxes.get)

    val argmax_overlaps = MatrixUtil.argmax2(overlaps, 1).get

    var max_overlaps = argmax_overlaps.zipWithIndex.map(x => overlaps(x._2, x._1))
    var gt_argmax_overlaps = MatrixUtil.argmax2(overlaps, 0).get

    val gt_max_overlaps = gt_argmax_overlaps.zipWithIndex.map(x => {
      overlaps(x._1, x._2)
    })

    gt_argmax_overlaps = Array.range(0, overlaps.rows).filter(r => {
      def isFilter(): Boolean = {
        for (i <- 0 until overlaps.cols) {
          if (overlaps(r, i) == gt_max_overlaps(i)) {
            return true
          }
        }
        return false
      }
      isFilter
    })


    if (!Config.TRAIN.RPN_CLOBBER_POSITIVES) {
      // assign bg labels first so that positive labels can clobber them
      max_overlaps.zipWithIndex.foreach(x => {
        if (x._1 < Config.TRAIN.RPN_NEGATIVE_OVERLAP) labels(x._2) = 0
      })
    }

    // fg label: for each gt, anchor with highest overlap
    gt_argmax_overlaps.foreach(x => labels(x) = 1)

    // fg label: above threshold IOU
    max_overlaps.zipWithIndex.foreach(x => {
      if (x._1 >= Config.TRAIN.RPN_POSITIVE_OVERLAP) max_overlaps(x._2) = 1
    })

    if (Config.TRAIN.RPN_CLOBBER_POSITIVES) {
      //assign bg labels last so that negative labels can clobber positives
      max_overlaps.zipWithIndex.foreach(x => {
        if (x._1 < Config.TRAIN.RPN_NEGATIVE_OVERLAP) max_overlaps(x._2) = 0
      })
    }

    // subsample positive labels if we have too many
    val num_fg = Config.TRAIN.RPN_FG_FRACTION * Config.TRAIN.RPN_BATCHSIZE
    val fg_inds = labels.findAll(_ == 1)
    if (fg_inds.length > num_fg) {
      val disable_inds = Random.shuffle(fg_inds).take(fg_inds.length - num_fg.toInt)
      disable_inds.foreach(x => labels(x) = -1)
    }

    var bbox_targets = _compute_targets(insideAnchors, MatrixUtil.select(data.gt_boxes.get, argmax_overlaps, 0).get)

    var bbox_inside_weights = DenseMatrix.zeros[Float](indsInside.length, 4)
    labels.foreachPair((k, v) => {
      if (v == 1) {
        bbox_inside_weights(k, ::) := convert(DenseVector(Config.TRAIN.RPN_BBOX_INSIDE_WEIGHTS), Float).t
      }
    })

    var bbox_outside_weights = DenseMatrix.zeros[Float](indsInside.length, 4)

    val labelGe0 = labels.findAll(x => x >= 0).toArray
    val labelE1 = labels.findAll(x => x == 1).toArray
    val labelE0 = labels.findAll(x => x == 0).toArray
    var positive_weights = None: Option[DenseMatrix[Float]]
    var negative_weights = None: Option[DenseMatrix[Float]]
    if (Config.TRAIN.RPN_POSITIVE_WEIGHT < 0) {
      // uniform weighting of examples (given non -uniform sampling)
      val num_examples = labelGe0.length
      positive_weights = Some(DenseMatrix.ones[Float](1, 4) * (1.0f / num_examples))
      negative_weights = Some(DenseMatrix.ones[Float](1, 4) * (1.0f / num_examples))
    }
    else {
      require((Config.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
        (Config.TRAIN.RPN_POSITIVE_WEIGHT < 1))
      positive_weights = Some(DenseMatrix(Config.TRAIN.RPN_POSITIVE_WEIGHT.toFloat / labelE1.length))
      negative_weights = Some(DenseMatrix((1.0f - Config.TRAIN.RPN_POSITIVE_WEIGHT.toFloat) / labelE0.length))
    }

    labelE1.foreach(x => bbox_outside_weights(x, ::) := positive_weights.get.toDenseVector.t)
    labelE0.foreach(x => bbox_outside_weights(x, ::) := negative_weights.get.toDenseVector.t)

    if (Config.DEBUG) {
      _counts += labelE1.length
    }


    // map up to original set of anchors
    labels = convert(_unmap(convert(DenseMatrix(labels.data.array).t, Float), totalAnchors, indsInside, -1).toDenseVector, Int)
    bbox_targets = _unmap(bbox_targets, totalAnchors, indsInside, 0)
    bbox_inside_weights = _unmap(bbox_inside_weights, totalAnchors, indsInside, 0)
    bbox_outside_weights = _unmap(bbox_outside_weights, totalAnchors, indsInside, 0)

    if (Config.DEBUG) {
      println("generate anchors done")
      println("rpn: max max_overlap %s".format(if (max_overlaps.length != 0) max(max_overlaps) else ""))
      println("rpn: num_positive %d".format(labelE1.length))
      println("rpn: num_negative %d".format(labelE0.length))
      _fg_sum += labelE1.length
      _bg_sum += labelE0.length
      _count += 1
      println("rpn: num_positive avg " + _fg_sum / _count)
      println("rpn: num_negative avg " + _bg_sum / _count)

      println("total anchors: " + num_anchors)
      println("num shifts" + shifts.rows)
      println("bbox target shape: " + bbox_targets.rows + ", " + bbox_targets.cols)
      println("bbox_inside_weights shape: " + bbox_inside_weights.rows + ", " + bbox_inside_weights.cols)
      println("bbox_outside_weights shape: " + bbox_outside_weights.rows + ", " + bbox_outside_weights.cols)
    }
    
    new AnchorTarget(labels, bbox_targets, bbox_inside_weights, bbox_outside_weights)

  }

  def getIndsInside(width: Int, height: Int, allAnchors: DenseMatrix[Float], allowed_border: Float): ArrayBuffer[Int] = {
    var indsInside = ArrayBuffer[Int]()
    for (i <- 0 until allAnchors.rows) {
      if ((allAnchors(i, 0) >= -allowed_border) &&
        (allAnchors(i, 1) >= -allowed_border) &&
        (allAnchors(i, 2) < width.toFloat + allowed_border) &&
        (allAnchors(i, 3) < height.toFloat + allowed_border)) {
        indsInside += i
      }
    }
    indsInside
  }

  def getAllAnchors(shifts: DenseMatrix[Float], anchors: DenseMatrix[Float]): DenseMatrix[Float] = {
    var allAnchors = new DenseMatrix[Float](shifts.rows * anchors.rows, 4)
    var k = 0
    for (s <- 0 until shifts.rows) {
      allAnchors(s * anchors.rows to (s + 1) * anchors.rows - 1, 0 to 3) := (anchors.t(::, *) + shifts.t(::, s)).t
    }
    allAnchors
  }

  /**
    * Unmap a subset of item (data) back to the original set of items (of size count) 
    *
    * @param data
    * @param count
    * @param inds
    * @param fillValue
    * @return
    */
  def _unmap(data: DenseMatrix[Float], count: Int, inds: ArrayBuffer[Int], fillValue: Int): DenseMatrix[Float] = {
    var ret = DenseMatrix.fill[Float](count, data.cols) {
      fillValue
    }
    inds.zipWithIndex.foreach(ind => {
      ret(ind._1, ::) := data(ind._2, ::)
    })
    ret
  }
}
