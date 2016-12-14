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

import breeze.linalg.{DenseMatrix, DenseVector, convert}
import com.intel.analytics.bigdl.pvanet.datasets.ImageWithRoi
import com.intel.analytics.bigdl.pvanet.model.FasterRcnnParam
import com.intel.analytics.bigdl.pvanet.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


case class BboxTarget(labels: Tensor[Float],
  private val bboxTargets: Tensor[Float],
  private val bboxInsideWeights: Tensor[Float],
  private val bboxOutsideWeights: Tensor[Float]) {

  val targetsTable: Table = new Table()
  targetsTable.insert(bboxTargets)
  targetsTable.insert(bboxInsideWeights)
  targetsTable.insert(bboxOutsideWeights)
}

object AnchorTarget {
  var basicAnchors: DenseMatrix[Float] = _

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

  def apply(data: ImageWithRoi, height: Int, width: Int, param: FasterRcnnParam): BboxTarget = {
    if (basicAnchors == null) {
      basicAnchors = Anchor.generateAnchors(param.anchorRatios, param.anchorScales)
    }
    // 1. Generate proposals from bbox deltas and shifted anchors
    val shifts = Anchor.generateShifts(width, height, param.featStride)
    val totalAnchors = shifts.rows * param.anchorNum
    val allAnchors: DenseMatrix[Float] = Anchor.getAllAnchors(shifts, basicAnchors)
    val indsInside: ArrayBuffer[Int] = getIndsInside(data.scaledImage.width(),
      data.scaledImage.height(), allAnchors, 0)


    // keep only inside anchors
    val insideAnchors: DenseMatrix[Float] = getInsideAnchors(indsInside, allAnchors)

    // label: 1 is positive, 0 is negative, -1 is dont care
    val labels = DenseVector.fill[Float](indsInside.length, -1)

    // overlaps between the anchors and the gt boxes
    // overlaps (ex, gt)
    val overlaps = Bbox.bboxOverlap(insideAnchors, data.gtBoxes.get.toBreezeMatrix())

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


    if (!param.RPN_CLOBBER_POSITIVES) {
      // assign bg labels first so that positive labels can clobber them
      maxOverlaps.zipWithIndex.foreach(x => {
        if (x._1 < param.RPN_NEGATIVE_OVERLAP) labels(x._2) = 0
      })
    }

    // fg label: for each gt, anchor with highest overlap
    gtArgmaxOverlaps.foreach(x => labels(x) = 1)

    // fg label: above threshold IOU
    maxOverlaps.zipWithIndex.foreach(x => {
      if (x._1 >= param.RPN_POSITIVE_OVERLAP) maxOverlaps(x._2) = 1
    })

    if (param.RPN_CLOBBER_POSITIVES) {
      // assign bg labels last so that negative labels can clobber positives
      maxOverlaps.zipWithIndex.foreach(x => {
        if (x._1 < param.RPN_NEGATIVE_OVERLAP) maxOverlaps(x._2) = 0
      })
    }

    // subsample positive labels if we have too many
    val numFg = param.RPN_FG_FRACTION * param.RPN_BATCHSIZE
    val fgInds = labels.findAll(_ == 1)
    if (fgInds.length > numFg) {
      val disableInds = Random.shuffle(fgInds).take(fgInds.length - numFg.toInt)
      disableInds.foreach(x => labels(x) = -1)
    }

    var bboxTargets = computeTargets(insideAnchors,
      MatrixUtil.selectMatrix(data.gtBoxes.get.toBreezeMatrix(), argmaxOverlaps, 0))

    var bboxInsideWeights = DenseMatrix.zeros[Float](indsInside.length, 4)
    labels.foreachPair((k, v) => {
      if (v == 1) {
        bboxInsideWeights(k, ::) :=
          convert(DenseVector(param.RPN_BBOX_INSIDE_WEIGHTS), Float).t
      }
    })

    var bboxOutSideWeights = DenseMatrix.zeros[Float](indsInside.length, 4)

    val labelGe0 = labels.findAll(x => x >= 0).toArray
    val labelE1 = labels.findAll(x => x == 1).toArray
    val labelE0 = labels.findAll(x => x == 0).toArray
    var positiveWeights = None: Option[DenseMatrix[Float]]
    var negative_weights = None: Option[DenseMatrix[Float]]
    if (param.RPN_POSITIVE_WEIGHT < 0) {
      // uniform weighting of examples (given non -uniform sampling)
      val numExamples = labelGe0.length
      positiveWeights = Some(DenseMatrix.ones[Float](1, 4) * (1.0f / numExamples))
      negative_weights = Some(DenseMatrix.ones[Float](1, 4) * (1.0f / numExamples))
    }
    else {
      require((param.RPN_POSITIVE_WEIGHT > 0) &
        (param.RPN_POSITIVE_WEIGHT < 1))
      positiveWeights = Some(
        DenseMatrix(param.RPN_POSITIVE_WEIGHT.toFloat / labelE1.length))
      negative_weights = Some(
        DenseMatrix((1.0f - param.RPN_POSITIVE_WEIGHT.toFloat) / labelE0.length))
    }

    labelE1.foreach(x => bboxOutSideWeights(x, ::) := positiveWeights.get.toDenseVector.t)
    labelE0.foreach(x => bboxOutSideWeights(x, ::) := negative_weights.get.toDenseVector.t)

    // map up to original set of anchors
    val labels2 = unmap(Tensor(DenseMatrix(labels.data.array).t), totalAnchors, indsInside, -1)
    val bboxTargets2 = unmap(Tensor(bboxTargets), totalAnchors, indsInside, 0)
    val bboxInsideWeights2 = unmap(Tensor(bboxInsideWeights), totalAnchors, indsInside, 0)
    val bboxOutSideWeights2 = unmap(Tensor(bboxOutSideWeights), totalAnchors, indsInside, 0)

    BboxTarget(labels2, bboxTargets2, bboxInsideWeights2,bboxOutSideWeights2)

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


  /**
   * Unmap a subset of item (data) back to the original set of items (of size count)
   */
  def unmap(data: Tensor[Float],
    count: Int,
    inds: ArrayBuffer[Int],
    fillValue: Float): Tensor[Float] = {
    val ret = Tensor[Float](count, data.size(2)).fill(fillValue)
    inds.zip(Stream.from(1)).foreach(ind => {
      ret.update(ind._1, data(ind._2))
    })
    ret
  }
}
