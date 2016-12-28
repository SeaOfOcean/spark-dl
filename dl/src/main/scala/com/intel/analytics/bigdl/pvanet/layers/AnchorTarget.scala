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

import com.intel.analytics.bigdl.pvanet.model.FasterRcnnParam
import com.intel.analytics.bigdl.pvanet.utils._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


case class BboxTarget(labels: Tensor[Float],
  bboxTargets: Tensor[Float],
  bboxInsideWeights: Tensor[Float],
  bboxOutsideWeights: Tensor[Float]) {

  val targetsTable: Table = new Table()
  targetsTable.insert(bboxTargets)
  targetsTable.insert(bboxInsideWeights)
  targetsTable.insert(bboxOutsideWeights)
}

class AnchorTarget(param: FasterRcnnParam) {
  val logger = Logger.getLogger(getClass)
  val basicAnchors = Anchor.generateBasicAnchors2(param.anchorRatios, param.anchorScales)


  /**
   * Compute bounding-box regression targets for an image.
   *
   * @return
   */
  def computeTargets(exRois: Tensor[Float], gtBoxes: Tensor[Float],
    insideAnchorsGtOverlaps: Tensor[Float]): Tensor[Float] = {
    val gtRois = TensorUtil.selectMatrix(gtBoxes,
      TensorUtil.argmax2(insideAnchorsGtOverlaps, 2), 1)
    require(exRois.size(1) == gtRois.size(1))
    require(exRois.size(2) == 4)
    require(gtRois.size(2) == 5)
    Bbox.bboxTransform(exRois, TensorUtil.selectMatrix(gtRois, Array.range(1, 5), 2))
  }


  def getAnchors(featureW: Int, featureH: Int, imgW: Int, imgH: Int)
  : (Array[Int], Tensor[Float], Int) = {
    // 1. Generate proposals from bbox deltas and shifted anchors
    val shifts = Anchor.generateShifts2(featureW, featureH, param.featStride)
    totalAnchors = shifts.size(1) * param.anchorNum
    logger.info(s"totalAnchors: $totalAnchors")
    val allAnchors = Anchor.getAllAnchors(shifts, basicAnchors)
    // keep only inside anchors
    val indsInside = getIndsInside(imgW, imgH, allAnchors, 0)
    logger.info(s"indsInside: ${indsInside.length}")
    val insideAnchors = Tensor[Float](indsInside.length, 4)
    indsInside.zip(Stream.from(1)).foreach(i => {
      insideAnchors.setValue(i._2, 1, allAnchors.valueAt(i._1, 1))
      insideAnchors.setValue(i._2, 2, allAnchors.valueAt(i._1, 2))
      insideAnchors.setValue(i._2, 3, allAnchors.valueAt(i._1, 3))
      insideAnchors.setValue(i._2, 4, allAnchors.valueAt(i._1, 4))
    })
    (indsInside, insideAnchors, totalAnchors)
  }

  var totalAnchors: Int = 0


  /**
   * Algorithm:
   *
   * for each (H, W) location i
   * --- generate scale_size * ratio_size anchor boxes centered on cell i
   * --- apply predicted bbox deltas at cell i to each of the 9 anchors
   * filter out-of-image anchors
   * measure GT overlap
   */
  def getAnchorTarget(featureH: Int, featureW: Int,
    imgH: Int, imgW: Int, gtBoxes: Tensor[Float]): BboxTarget = {

    val (indsInside, insideAnchors, totalAnchors) = getAnchors(featureW, featureH, imgW, imgH)

    // overlaps between the anchors and the gt boxes
    val insideAnchorsGtOverlaps = Bbox.bboxOverlap(insideAnchors, gtBoxes)

    // label: 1 is positive, 0 is negative, -1 is don't care
    var labels = getAllLabels(indsInside, insideAnchorsGtOverlaps)
    labels = sampleLabels(labels)

    var bboxTargets = computeTargets(insideAnchors, gtBoxes, insideAnchorsGtOverlaps)

    var bboxInsideWeights = getBboxInsideWeights(indsInside, labels)
    var bboxOutSideWeights = getBboxOutsideWeights(indsInside, labels)

    // map up to original set of anchors
    // mapUpToOriginal(labels, bboxTargets, bboxInsideWeights, bboxOutSideWeights, indsInside)
    labels = unmap(labels, totalAnchors, indsInside, -1)
    bboxTargets = unmap(bboxTargets, totalAnchors, indsInside, 0)
    bboxInsideWeights = unmap(bboxInsideWeights, totalAnchors, indsInside, 0)
    bboxOutSideWeights = unmap(bboxOutSideWeights, totalAnchors, indsInside, 0)

    labels = labels.reshape(Array(1, featureH, featureW, param.anchorNum))
      .transpose(2, 3).transpose(2, 4).reshape(Array(1, 1, param.anchorNum * featureH, featureW))
    bboxTargets = bboxTargets.reshape(Array(1, featureH, featureW, param.anchorNum * 4))
      .transpose(2, 3).transpose(2, 4)
    bboxInsideWeights = bboxInsideWeights.reshape(Array(1, featureH, featureW, param.anchorNum * 4))
      .transpose(2, 3).transpose(2, 4)
    bboxOutSideWeights = bboxOutSideWeights.reshape(Array(1, featureH,
      featureW, param.anchorNum * 4)).transpose(2, 3).transpose(2, 4)
    BboxTarget(labels.contiguous(),
      bboxTargets.contiguous(),
      bboxInsideWeights.contiguous(),
      bboxOutSideWeights.contiguous())
  }

  // label: 1 is positive, 0 is negative, -1 is don't care
  def getAllLabels(indsInside: Array[Int],
    insideAnchorsGtOverlaps: Tensor[Float]): Tensor[Float] = {
    val labels = Tensor[Float](indsInside.length).fill(-1f)
    // todo: argmaxOverlaps may not be needed here
    val argmaxOverlaps = TensorUtil.argmax2(insideAnchorsGtOverlaps, 2)
    val maxOverlaps = argmaxOverlaps.zip(Stream.from(1)).map(x =>
      insideAnchorsGtOverlaps.valueAt(x._2, x._1))
    val gtArgmaxOverlaps = TensorUtil.argmax2(insideAnchorsGtOverlaps, 1)

    val gtMaxOverlaps = gtArgmaxOverlaps.zip(Stream.from(1)).map(x => {
      insideAnchorsGtOverlaps.valueAt(x._1, x._2)
    })

    val gtArgmaxOverlaps2 = (1 to insideAnchorsGtOverlaps.size(1)).filter(r => {
      def isFilter: Boolean = {
        for (i <- 1 to insideAnchorsGtOverlaps.size(2)) {
          if (insideAnchorsGtOverlaps.valueAt(r, i) == gtMaxOverlaps(i - 1)) {
            return true
          }
        }
        false
      }
      isFilter
    })


    if (!param.RPN_CLOBBER_POSITIVES) {
      // assign bg labels first so that positive labels can clobber them
      maxOverlaps.zip(Stream from 1).foreach(x => {
        if (x._1 < param.RPN_NEGATIVE_OVERLAP) labels.setValue(x._2, 0)
      })
    }

    // fg label: for each gt, anchor with highest overlap
    gtArgmaxOverlaps2.foreach(x => labels.setValue(x, 1))

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
    labels
  }

  def sampleLabels(labels: Tensor[Float]): Tensor[Float] = {
    // subsample positive labels if we have too many
    val numFg = param.RPN_FG_FRACTION * param.RPN_BATCHSIZE
    val fgInds = (1 to labels.size(1)).filter(x => x == 1)
    if (fgInds.length > numFg) {
      val disableInds = Random.shuffle(fgInds).take(fgInds.length - numFg.toInt)
      disableInds.foreach(x => labels.update(x, -1))
      logger.info(s"${disableInds.length} fg inds are disabled")
    }
    logger.info(s"fg: ${fgInds.length}")

    // subsample negative labels if we have too many
    val numBg = param.RPN_BATCHSIZE - fgInds.length
    val bgInds = (1 to labels.size(1)).filter(x => labels.valueAt(x) == 0)
    if (bgInds.length > numBg) {
      val disableInds = Random.shuffle(bgInds).take(bgInds.length - numBg.toInt)
      disableInds.foreach(x => labels.setValue(x, -1))
      logger.info(s"${disableInds.length} bg inds are disabled, " +
        s"now ${(1 to labels.size(1)).count(x => labels.valueAt(x) == 0)} inds")
    }
    labels
  }

  def getBboxInsideWeights(indsInside: Array[Int],
    labels: Tensor[Float]): Tensor[Float] = {
    val bboxInsideWeights = Tensor[Float](indsInside.length, 4)

    labels.map(Tensor[Float].range(1, labels.size(1)), (v, k) => {
      if (v == 1) {
        bboxInsideWeights.update(k.toInt, Tensor(Storage(param.RPN_BBOX_INSIDE_WEIGHTS)))
      }
      v
    })
    bboxInsideWeights
  }

  def getBboxOutsideWeights(indsInside: Array[Int], labels: Tensor[Float])
  : Tensor[Float] = {
    val bboxOutSideWeights = Tensor[Float](indsInside.length, 4)

    val labelGe0 = (1 to labels.nElement()).filter(x => labels.valueAt(x) >= 0).toArray
    val labelE1 = (1 to labels.nElement()).filter(x => labels.valueAt(x) == 1).toArray
    val labelE0 = (1 to labels.nElement()).filter(x => labels.valueAt(x) == 0).toArray
    var positiveWeights: Tensor[Float] = null
    var negative_weights: Tensor[Float] = null
    if (param.RPN_POSITIVE_WEIGHT < 0) {
      // uniform weighting of examples (given non -uniform sampling)
      val numExamples = labelGe0.length
      positiveWeights = Tensor[Float](1, 4).fill(1f).mul(1.0f / numExamples)
      negative_weights = Tensor[Float](1, 4).fill(1f).mul(1.0f / numExamples)
    }
    else {
      require((param.RPN_POSITIVE_WEIGHT > 0) &
        (param.RPN_POSITIVE_WEIGHT < 1))
      positiveWeights = Tensor(Storage(Array(param.RPN_POSITIVE_WEIGHT / labelE1.length)))
      negative_weights = Tensor(Storage(Array(1.0f - param.RPN_POSITIVE_WEIGHT / labelE0.length)))
    }
    labelE1.foreach(x => bboxOutSideWeights.update(x, positiveWeights))
    labelE0.foreach(x => bboxOutSideWeights.update(x, negative_weights))
    bboxOutSideWeights
  }


  /**
   * map up to original set of anchors
   */
  def mapUpToOriginal(labels: Tensor[Float],
    bboxTargets: Tensor[Float],
    bboxInsideWeights: Tensor[Float],
    bboxOutSideWeights: Tensor[Float],
    indsInside: Array[Int]): BboxTarget = {
    val labels2 = unmap(labels, totalAnchors, indsInside, -1)
    val bboxTargets2 = unmap(bboxTargets, totalAnchors, indsInside, 0)
    val bboxInsideWeights2 = unmap(bboxInsideWeights, totalAnchors, indsInside, 0)
    val bboxOutSideWeights2 = unmap(bboxOutSideWeights, totalAnchors, indsInside, 0)

    BboxTarget(labels2, bboxTargets2, bboxInsideWeights2, bboxOutSideWeights2)
  }

  def getIndsInside(width: Int, height: Int,
    allAnchors: Tensor[Float], allowedBorder: Float): Array[Int] = {
    var indsInside = ArrayBuffer[Int]()
    for (i <- 1 to allAnchors.size(1)) {
      if ((allAnchors.valueAt(i, 1) >= -allowedBorder) &&
        (allAnchors.valueAt(i, 2) >= -allowedBorder) &&
        (allAnchors.valueAt(i, 3) < width.toFloat + allowedBorder) &&
        (allAnchors.valueAt(i, 4) < height.toFloat + allowedBorder)) {
        indsInside += i
      }
    }
    indsInside.toArray
  }


  /**
   * Unmap a subset of item (data) back to the original set of items (of size count)
   */
  def unmap(data: Tensor[Float], count: Int, inds: Array[Int],
    fillValue: Float): Tensor[Float] = {
    if (data.nDimension() == 1) {
      val ret = Tensor[Float](count).fill(fillValue)
      inds.zip(Stream.from(1)).foreach(ind => ret.setValue(ind._1, data.valueAt(ind._2)))
      ret
    } else {
      val ret = Tensor[Float](count, data.size(2)).fill(fillValue)
      inds.zip(Stream.from(1)).foreach(ind => {
        ret.update(ind._1, data(ind._2))
      })
      ret
    }
  }
}
