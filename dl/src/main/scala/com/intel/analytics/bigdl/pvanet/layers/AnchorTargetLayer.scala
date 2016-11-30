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

import java.util.logging.Logger

import breeze.linalg.{*, DenseMatrix, DenseVector, convert, max}
import com.intel.analytics.bigdl.pvanet.datasets.Roidb.ImageWithRoi
import com.intel.analytics.bigdl.pvanet.util.{Anchor, Bbox, Config, MatrixUtil}
import com.intel.analytics.bigdl.pvanet.{Config, MatrixUtil}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


case class AnchorTarget(labels: DenseVector[Int],
  bboxTargets: DenseMatrix[Float],
  bboxInsideWeights: DenseMatrix[Float],
  bboxOutsideWeights: DenseMatrix[Float]) {

}

class AnchorTargetLayer(val scales: Array[Float] = Array[Float](3, 6, 9, 16, 32),
  val ratios: Array[Float] = Array(0.5f, 0.667f, 1.0f, 1.5f, 2.0f)) {
  val logger = Logger.getLogger(this.getClass.getName)
  // todo: now hard code

  val featStride = 16
  val anchors = Anchor.generateAnchors(ratios = ratios, scales = scales)
  val numAnchors = anchors.rows
  // n_scales * n_ratios
  val allowedBorder = 0
  assert(numAnchors == ratios.length * scales.length)

  // debug info
  var counts = Config.EPS
  var fgSum = 0
  var bgSum = 0
  var count = 0

  def generateShifts(width: Int, height: Int, featStride: Float): DenseMatrix[Float] = {
    val shiftX = DenseVector.range(0, width).map(x => x * featStride)
    val shiftY = DenseVector.range(0, height).map(x => x * featStride)
    MatrixUtil.meshgrid(shiftX, shiftY) match {
      case (x1Mesh, x2Mesh) =>
        return DenseMatrix.vertcat(x1Mesh.t.toDenseVector.toDenseMatrix,
          x2Mesh.t.toDenseVector.toDenseMatrix,
          x1Mesh.t.toDenseVector.toDenseMatrix,
          x2Mesh.t.toDenseVector.toDenseMatrix).t
    }
    new DenseMatrix(0, 0)
  }

  /**
   * Compute bounding-box regression targets for an image.
   *
   * @return
   */
  def computeTargets(exRois: DenseMatrix[Float], gtRois: DenseMatrix[Float]): DenseMatrix[Float] = {
    require(exRois.rows == gtRois.rows)
    require(exRois.cols == 4)
    require(gtRois.cols == 5)
    Bbox.bboxTransform(exRois, MatrixUtil.selectMatrix(gtRois, Array.range(0, 4), 1))
  }


  def getInsideAnchors(indsInside: ArrayBuffer[Int],
    allAnchors: DenseMatrix[Float]): DenseMatrix[Float] = {
    val insideAnchors = new DenseMatrix[Float](indsInside.length, 4)
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
    // 1. Generate proposals from bbox deltas and shifted anchors
    val shifts = generateShifts(width, height, featStride)
    val totalAnchors = shifts.rows * numAnchors
    val allAnchors: DenseMatrix[Float] = getAllAnchors(shifts, anchors)
    val indsInside: ArrayBuffer[Int] = getIndsInside(data.scaledImage.width(),
      data.scaledImage.height(), allAnchors, allowedBorder)


    // keep only inside anchors
    val insideAnchors: DenseMatrix[Float] = getInsideAnchors(indsInside, allAnchors)

    // label: 1 is positive, 0 is negative, -1 is dont care
    var labels = DenseVector.fill(indsInside.length, -1)

    // overlaps between the anchors and the gt boxes
    // overlaps (ex, gt)
    val overlaps = Bbox.bboxOverlap(insideAnchors, data.gtBoxes.get)

    val argmaxOverlaps = MatrixUtil.argmax2(overlaps, 1)

    val maxOverlaps = argmaxOverlaps.zipWithIndex.map(x => overlaps(x._2, x._1))
    var gtArgmaxOverlaps = MatrixUtil.argmax2(overlaps, 0)

    val gtMaxOverlaps = gtArgmaxOverlaps.zipWithIndex.map(x => {
      overlaps(x._1, x._2)
    })

    gtArgmaxOverlaps = Array.range(0, overlaps.rows).filter(r => {
      def isFilter: Boolean = {
        for (i <- 0 until overlaps.cols) {
          if (overlaps(r, i) == gtMaxOverlaps(i)) {
            return true
          }
        }
        false
      }
      isFilter
    })


    if (!Config.TRAIN.RPN_CLOBBER_POSITIVES) {
      // assign bg labels first so that positive labels can clobber them
      maxOverlaps.zipWithIndex.foreach(x => {
        if (x._1 < Config.TRAIN.RPN_NEGATIVE_OVERLAP) labels(x._2) = 0
      })
    }

    // fg label: for each gt, anchor with highest overlap
    gtArgmaxOverlaps.foreach(x => labels(x) = 1)

    // fg label: above threshold IOU
    maxOverlaps.zipWithIndex.foreach(x => {
      if (x._1 >= Config.TRAIN.RPN_POSITIVE_OVERLAP) maxOverlaps(x._2) = 1
    })

    if (Config.TRAIN.RPN_CLOBBER_POSITIVES) {
      // assign bg labels last so that negative labels can clobber positives
      maxOverlaps.zipWithIndex.foreach(x => {
        if (x._1 < Config.TRAIN.RPN_NEGATIVE_OVERLAP) maxOverlaps(x._2) = 0
      })
    }

    // subsample positive labels if we have too many
    val numFg = Config.TRAIN.RPN_FG_FRACTION * Config.TRAIN.RPN_BATCHSIZE
    val fgInds = labels.findAll(_ == 1)
    if (fgInds.length > numFg) {
      val disableInds = Random.shuffle(fgInds).take(fgInds.length - numFg.toInt)
      disableInds.foreach(x => labels(x) = -1)
    }

    var bboxTargets = computeTargets(insideAnchors,
      MatrixUtil.selectMatrix(data.gtBoxes.get, argmaxOverlaps, 0))

    var bboxInsideWeights = DenseMatrix.zeros[Float](indsInside.length, 4)
    labels.foreachPair((k, v) => {
      if (v == 1) {
        bboxInsideWeights(k, ::) :=
          convert(DenseVector(Config.TRAIN.RPN_BBOX_INSIDE_WEIGHTS), Float).t
      }
    })

    var bboxOutSideWeights = DenseMatrix.zeros[Float](indsInside.length, 4)

    val labelGe0 = labels.findAll(x => x >= 0).toArray
    val labelE1 = labels.findAll(x => x == 1).toArray
    val labelE0 = labels.findAll(x => x == 0).toArray
    var positiveWeights = None: Option[DenseMatrix[Float]]
    var negative_weights = None: Option[DenseMatrix[Float]]
    if (Config.TRAIN.RPN_POSITIVE_WEIGHT < 0) {
      // uniform weighting of examples (given non -uniform sampling)
      val numExamples = labelGe0.length
      positiveWeights = Some(DenseMatrix.ones[Float](1, 4) * (1.0f / numExamples))
      negative_weights = Some(DenseMatrix.ones[Float](1, 4) * (1.0f / numExamples))
    }
    else {
      require((Config.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
        (Config.TRAIN.RPN_POSITIVE_WEIGHT < 1))
      positiveWeights = Some(
        DenseMatrix(Config.TRAIN.RPN_POSITIVE_WEIGHT.toFloat / labelE1.length))
      negative_weights = Some(
        DenseMatrix((1.0f - Config.TRAIN.RPN_POSITIVE_WEIGHT.toFloat) / labelE0.length))
    }

    labelE1.foreach(x => bboxOutSideWeights(x, ::) := positiveWeights.get.toDenseVector.t)
    labelE0.foreach(x => bboxOutSideWeights(x, ::) := negative_weights.get.toDenseVector.t)

    if (Config.DEBUG) {
      counts += labelE1.length
    }


    // map up to original set of anchors
    labels = convert(unmap(convert(
      DenseMatrix(labels.data.array).t, Float), totalAnchors, indsInside, -1).toDenseVector, Int)
    bboxTargets = unmap(bboxTargets, totalAnchors, indsInside, 0)
    bboxInsideWeights = unmap(bboxInsideWeights, totalAnchors, indsInside, 0)
    bboxOutSideWeights = unmap(bboxOutSideWeights, totalAnchors, indsInside, 0)

    if (Config.DEBUG) {
      println("generate anchors done")
      println("rpn: max max_overlap %s".format(
        if (maxOverlaps.length != 0) max(maxOverlaps) else ""))
      println("rpn: num_positive %d".format(labelE1.length))
      println("rpn: num_negative %d".format(labelE0.length))
      fgSum += labelE1.length
      bgSum += labelE0.length
      count += 1
      println("rpn: num_positive avg " + fgSum / count)
      println("rpn: num_negative avg " + bgSum / count)

      println("total anchors: " + numAnchors)
      println("num shifts" + shifts.rows)
      println("bbox target shape: " + bboxTargets.rows + ", " + bboxTargets.cols)
      println("bbox_inside_weights shape: " +
        bboxInsideWeights.rows + ", " + bboxInsideWeights.cols)
      println("bbox_outside_weights shape: " +
        bboxOutSideWeights.rows + ", " + bboxOutSideWeights.cols)
    }

    AnchorTarget(labels, bboxTargets, bboxInsideWeights, bboxOutSideWeights)

  }

  def getIndsInside(width: Int, height: Int,
    allAnchors: DenseMatrix[Float], allowedBorder: Float): ArrayBuffer[Int] = {
    var indsInside = ArrayBuffer[Int]()
    for (i <- 0 until allAnchors.rows) {
      if ((allAnchors(i, 0) >= -allowedBorder) &&
        (allAnchors(i, 1) >= -allowedBorder) &&
        (allAnchors(i, 2) < width.toFloat + allowedBorder) &&
        (allAnchors(i, 3) < height.toFloat + allowedBorder)) {
        indsInside += i
      }
    }
    indsInside
  }

  def getAllAnchors(shifts: DenseMatrix[Float],
    anchors: DenseMatrix[Float] = anchors): DenseMatrix[Float] = {
    val allAnchors = new DenseMatrix[Float](shifts.rows * anchors.rows, 4)
    for (s <- 0 until shifts.rows) {
      allAnchors(s * anchors.rows until (s + 1) * anchors.rows, 0 until 4) :=
        (anchors.t(::, *) + shifts.t(::, s)).t
    }
    allAnchors
  }


  /**
   * Unmap a subset of item (data) back to the original set of items (of size count)
   */
  def unmap(data: DenseMatrix[Float],
    count: Int,
    inds: ArrayBuffer[Int],
    fillValue: Int): DenseMatrix[Float] = {
    val ret = DenseMatrix.fill[Float](count, data.cols) {
      fillValue
    }
    inds.zipWithIndex.foreach(ind => {
      ret(ind._1, ::) := data(ind._2, ::)
    })
    ret
  }
}
