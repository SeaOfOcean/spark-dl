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
import com.intel.analytics.bigdl.pvanet.model.FasterRcnnParam
import com.intel.analytics.bigdl.pvanet.utils._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


case class BboxTarget(labels: Tensor[Float],
  val bboxTargets: Tensor[Float],
  val bboxInsideWeights: Tensor[Float],
  val bboxOutsideWeights: Tensor[Float]) {

  val targetsTable: Table = new Table()
  targetsTable.insert(bboxTargets)
  targetsTable.insert(bboxInsideWeights)
  targetsTable.insert(bboxOutsideWeights)
}

class AnchorTarget(param: FasterRcnnParam) {
  val logger = Logger.getLogger(getClass)
  val basicAnchors = Anchor.generateBasicAnchors(param.anchorRatios, param.anchorScales)
  val basicAnchors2 = Anchor.generateBasicAnchors2(param.anchorRatios, param.anchorScales)

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

  /**
   * Compute bounding-box regression targets for an image.
   *
   * @return
   */
  def computeTargets(exRois: Tensor[Float], gtRois: Tensor[Float]): Tensor[Float] = {
    require(exRois.size(1) == gtRois.size(1))
    require(exRois.size(2) == 4)
    require(gtRois.size(2) == 5)
    Bbox.bboxTransform(exRois, TensorUtil.selectMatrix(gtRois, Array.range(1, 5), 2))
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

  def getInsideAnchors(indsInside: ArrayBuffer[Int],
    allAnchors: Tensor[Float]): Tensor[Float] = {
    val insideAnchors = Tensor[Float](indsInside.length, 4)
    indsInside.zip(Stream.from(1)).foreach(i => {
      insideAnchors.setValue(i._2, 1, allAnchors.valueAt(i._1, 1))
      insideAnchors.setValue(i._2, 2, allAnchors.valueAt(i._1, 2))
      insideAnchors.setValue(i._2, 3, allAnchors.valueAt(i._1, 3))
      insideAnchors.setValue(i._2, 4, allAnchors.valueAt(i._1, 4))
    })
    insideAnchors
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
    logger.info("img size", imgH, imgW)
    // 1. Generate proposals from bbox deltas and shifted anchors
    val shifts = Anchor.generateShifts(featureW, featureH, param.featStride)
    totalAnchors = shifts.rows * param.anchorNum
    logger.info(s"totalAnchors: $totalAnchors")
    val allAnchors = Anchor.getAllAnchors(shifts, basicAnchors)


    // keep only inside anchors
    val indsInside = getIndsInside(imgW, imgH, allAnchors, 0)
    logger.info(s"indsInside: ${indsInside.length}")
    val insideAnchors = getInsideAnchors(indsInside, allAnchors)

    // overlaps between the anchors and the gt boxes
    // val insideAnchorsGtOverlaps = Bbox.bboxOverlap(insideAnchors, gtBoxes.toBreezeMatrix())
    val insideAnchorsGtOverlaps =
    FileUtil.loadFeatures("insideAnchorsGtOverlaps", "data/middle/vgg16/step1").toBreezeMatrix()
    val exp = FileUtil.loadFeatures("insideAnchorsGtOverlaps", "data/middle/vgg16/step1")
    FileUtil.assertEqualIgnoreSize(exp,
      Tensor(insideAnchorsGtOverlaps),
      "compare insideAnchorsGtOverlaps", 1e-2)

    // label: 1 is positive, 0 is negative, -1 is don't care
    val labels = getLabels(indsInside, insideAnchorsGtOverlaps)
    compare("labelsBefore3354", labels)

    val bboxTargets = computeTargets(insideAnchors,
      MatrixUtil.selectMatrix(gtBoxes.toBreezeMatrix(),
        MatrixUtil.argmax2(insideAnchorsGtOverlaps, 1), 0))

    // todo: precision may not be enough
    compare("targetBefore3354", bboxTargets, 0.01)
    val bboxInsideWeights = getBboxInsideWeights(indsInside, labels)
    compare("inwBefore3354", bboxInsideWeights, 1e-6)
    val bboxOutSideWeights = getBboxOutsideWeights(indsInside, labels)
    compare("outWBefore3354", bboxOutSideWeights, 1e-6)
    compare("labelsBefore3354", labels)

    // map up to original set of anchors
    mapUpToOriginal(labels, bboxTargets, bboxInsideWeights, bboxOutSideWeights, indsInside)

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
  def getAnchorTarget2(featureH: Int, featureW: Int,
    imgH: Int, imgW: Int, gtBoxes: Tensor[Float]): BboxTarget = {
    logger.info("img size", imgH, imgW)
    // 1. Generate proposals from bbox deltas and shifted anchors
    val shifts = Anchor.generateShifts2(featureW, featureH, param.featStride)
    totalAnchors = shifts.size(1) * param.anchorNum
    logger.info(s"totalAnchors: $totalAnchors")
    val allAnchors = Anchor.getAllAnchors(shifts, basicAnchors2)


    // keep only inside anchors
    val indsInside = getIndsInside(imgW, imgH, allAnchors, 0)
    logger.info(s"indsInside: ${indsInside.length}")
    val insideAnchors = getInsideAnchors(indsInside, allAnchors)

    // overlaps between the anchors and the gt boxes
    // val insideAnchorsGtOverlaps = Bbox.bboxOverlap(insideAnchors, gtBoxes.toBreezeMatrix())
    val insideAnchorsGtOverlaps =
    FileUtil.loadFeatures("insideAnchorsGtOverlaps", "data/middle/vgg16/step1")
    val exp = FileUtil.loadFeatures("insideAnchorsGtOverlaps", "data/middle/vgg16/step1")
    FileUtil.assertEqualIgnoreSize(exp,
      Tensor(insideAnchorsGtOverlaps),
      "compare insideAnchorsGtOverlaps", 1e-2)

    // label: 1 is positive, 0 is negative, -1 is don't care
    var labels = getLabels(indsInside, insideAnchorsGtOverlaps)
    compare("labelsBefore3354", labels, 1e-6)

    var bboxTargets = computeTargets(insideAnchors,
      TensorUtil.selectMatrix(gtBoxes,
        TensorUtil.argmax2(insideAnchorsGtOverlaps, 2), 1))

    // todo: precision may not be enough
    compare("targetBefore3354", bboxTargets, 0.01)
    var bboxInsideWeights = getBboxInsideWeights(indsInside, labels)
    compare("inwBefore3354", bboxInsideWeights, 1e-6)
    var bboxOutSideWeights = getBboxOutsideWeights(indsInside, labels)
    compare("outWBefore3354", bboxOutSideWeights, 1e-6)
    compare("labelsBefore3354", labels, 1e-6)

    // map up to original set of anchors
    // mapUpToOriginal(labels, bboxTargets, bboxInsideWeights, bboxOutSideWeights, indsInside)
    labels = unmap(labels, totalAnchors, indsInside, -1)
    bboxTargets = unmap(bboxTargets, totalAnchors, indsInside, 0)
    bboxInsideWeights = unmap(bboxInsideWeights, totalAnchors, indsInside, 0)
    bboxOutSideWeights = unmap(bboxOutSideWeights, totalAnchors, indsInside, 0)

    compare("labelUnmap", labels, 1e-6)
    compare("targetUnmap", bboxTargets, 1e-2)
    compare("inwUnmap", bboxInsideWeights, 1e-6)
    compare("outWUnmap", bboxOutSideWeights, 1e-6)

    labels = labels.reshape(Array(1, featureH, featureW, param.anchorNum))
      .transpose(2, 3).transpose(2, 4).reshape(Array(1, 1, param.anchorNum * featureH, featureW))
    bboxTargets = bboxTargets.reshape(Array(1, featureH, featureW, param.anchorNum * 4))
      .transpose(2, 3).transpose(2, 4)
    bboxInsideWeights = bboxInsideWeights.reshape(Array(1, featureH, featureW, param.anchorNum * 4))
      .transpose(2, 3).transpose(2, 4)
    bboxOutSideWeights = bboxOutSideWeights.reshape(Array(1, featureH,
      featureW, param.anchorNum * 4)).transpose(2, 3).transpose(2, 4)
    BboxTarget(labels, bboxTargets, bboxInsideWeights, bboxOutSideWeights)
  }

  // label: 1 is positive, 0 is negative, -1 is don't care
  def getLabels(indsInside: ArrayBuffer[Int],
    insideAnchorsGtOverlaps: DenseMatrix[Float]): DenseVector[Float] = {
    val labels = DenseVector.fill[Float](indsInside.length, -1)
    // todo: argmaxOverlaps may not be needed here
    val argmaxOverlaps = MatrixUtil.argmax2(insideAnchorsGtOverlaps, 1)
    compare("argmax_overlaps", argmaxOverlaps)
    val maxOverlaps = argmaxOverlaps.zipWithIndex.map(x => insideAnchorsGtOverlaps(x._2, x._1))
    val gtArgmaxOverlaps = MatrixUtil.argmax2(insideAnchorsGtOverlaps, 0)

    val gtMaxOverlaps = gtArgmaxOverlaps.zipWithIndex.map(x => {
      insideAnchorsGtOverlaps(x._1, x._2)
    })

    val gtArgmaxOverlaps2 = Array.range(0, insideAnchorsGtOverlaps.rows).filter(r => {
      def isFilter: Boolean = {
        for (i <- 0 until insideAnchorsGtOverlaps.cols) {
          if (insideAnchorsGtOverlaps(r, i) == gtMaxOverlaps(i)) {
            return true
          }
        }
        false
      }
      isFilter
    })
    compare("gt_argmax_overlaps", gtArgmaxOverlaps2)


    if (!param.RPN_CLOBBER_POSITIVES) {
      // assign bg labels first so that positive labels can clobber them
      maxOverlaps.zipWithIndex.foreach(x => {
        if (x._1 < param.RPN_NEGATIVE_OVERLAP) labels(x._2) = 0
      })
    }

    // fg label: for each gt, anchor with highest overlap
    gtArgmaxOverlaps2.foreach(x => labels(x) = 1)
    compare("label111", labels)

    // fg label: above threshold IOU
    maxOverlaps.zipWithIndex.foreach(x => {
      if (x._1 >= param.RPN_POSITIVE_OVERLAP) maxOverlaps(x._2) = 1
    })
    compare("label222", labels)

    if (param.RPN_CLOBBER_POSITIVES) {
      // assign bg labels last so that negative labels can clobber positives
      maxOverlaps.zipWithIndex.foreach(x => {
        if (x._1 < param.RPN_NEGATIVE_OVERLAP) maxOverlaps(x._2) = 0
      })
    }

    compare("labelbeforesample", labels)
    sampleLabels(labels)
    labels
  }

  // label: 1 is positive, 0 is negative, -1 is don't care
  def getLabels(indsInside: ArrayBuffer[Int],
    insideAnchorsGtOverlaps: Tensor[Float]): Tensor[Float] = {
    val labels = Tensor[Float](indsInside.length).fill(-1f)
    // todo: argmaxOverlaps may not be needed here
    val argmaxOverlaps = TensorUtil.argmax2(insideAnchorsGtOverlaps, 2)
    //    compare("argmax_overlaps", argmaxOverlaps)
    val maxOverlaps = argmaxOverlaps.zip(Stream.from(1)).map(x =>
      insideAnchorsGtOverlaps.valueAt(x._2, x._1))
    val gtArgmaxOverlaps = TensorUtil.argmax2(insideAnchorsGtOverlaps, 1)

    val gtMaxOverlaps = gtArgmaxOverlaps.zip(Stream.from(1)).map(x => {
      insideAnchorsGtOverlaps.valueAt(x._1, x._2)
    })
    compare("gt_max_overlaps", gtMaxOverlaps, 1e-6)

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
//    compare("gt_argmax_overlaps", gtArgmaxOverlaps2.toArray)


    if (!param.RPN_CLOBBER_POSITIVES) {
      // assign bg labels first so that positive labels can clobber them
      maxOverlaps.zip(Stream from (1)).foreach(x => {
        if (x._1 < param.RPN_NEGATIVE_OVERLAP) labels.setValue(x._2, 0)
      })
    }

    // fg label: for each gt, anchor with highest overlap
    gtArgmaxOverlaps2.foreach(x => labels.setValue(x, 1))
    compare("label111", labels, 1e-6)

    // fg label: above threshold IOU
    maxOverlaps.zipWithIndex.foreach(x => {
      if (x._1 >= param.RPN_POSITIVE_OVERLAP) maxOverlaps(x._2) = 1
    })
    compare("label222", labels, 1e-6)

    if (param.RPN_CLOBBER_POSITIVES) {
      // assign bg labels last so that negative labels can clobber positives
      maxOverlaps.zipWithIndex.foreach(x => {
        if (x._1 < param.RPN_NEGATIVE_OVERLAP) maxOverlaps(x._2) = 0
      })
    }

    compare("labelbeforesample", labels, 1e-6)
    sampleLabels(labels)
    labels
  }

  def compare(name: String, vec: DenseVector[Float]): Unit = {
    val exp = FileUtil.loadFeatures(name, "data/middle/vgg16/step1")
    FileUtil.assertEqualIgnoreSize(exp, Tensor(vec), name, 1e-6)
  }

  def compare(name: String, vec: Array[Int]): Unit = {
    val exp = FileUtil.loadFeatures(name, "data/middle/vgg16/step1")
    FileUtil.assertEqualIgnoreSize(exp,
      Tensor(Storage(vec.map(x => x.toFloat))), name, 1e-6)
  }

  def compare(name: String, vec: DenseMatrix[Float], prec: Double): Unit = {
    val exp = FileUtil.loadFeatures(name, "data/middle/vgg16/step1")
    FileUtil.assertEqualIgnoreSize(exp, Tensor(vec), name, prec)
  }

  def compare(name: String, vec: Tensor[Float], prec: Double): Unit = {
    val exp = FileUtil.loadFeatures(name, "data/middle/vgg16/step1")
    FileUtil.assertEqualIgnoreSize(exp, vec, name, prec)
  }

  def compare(name: String, vec: Array[Float], prec: Double): Unit = {
    val exp = FileUtil.loadFeatures(name, "data/middle/vgg16/step1")
    FileUtil.assertEqualIgnoreSize(exp, Tensor(Storage(vec)), name, prec)
  }

  def sampleLabels(labels: DenseVector[Float]): DenseVector[Float] = {
    // subsample positive labels if we have too many
    val numFg = param.RPN_FG_FRACTION * param.RPN_BATCHSIZE
    val fgInds = labels.findAll(_ == 1)
    if (fgInds.length > numFg) {
      val disableInds = Random.shuffle(fgInds).take(fgInds.length - numFg.toInt)
      disableInds.foreach(x => labels(x) = -1)
      logger.info(s"${disableInds.length} fg inds are disabled")
    }
    logger.info(s"fg: ${fgInds.length}")

    // subsample negative labels if we have too many
    val numBg = param.RPN_BATCHSIZE - fgInds.length
    val bgInds = labels.findAll(_ == 0)
    if (bgInds.length > numBg) {
      //      val disableInds = Random.shuffle(bgInds).take(bgInds.length - numBg.toInt)
      val disableInds = FileUtil.loadFeaturesFullName("disablebg3354", false)
        .storage().array()
      disableInds.foreach(x => labels(x.toInt) = -1)
      logger.info(s"${disableInds.length} bg inds are disabled, " +
        s"now ${labels.findAll(_ == 0).length} inds")
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
      //      val disableInds = Random.shuffle(bgInds).take(bgInds.length - numBg.toInt)
      val disableInds = FileUtil.loadFeaturesFullName("disablebg3354", false)
        .storage().array()
      // temp +1 for they start from 0
      disableInds.foreach(x => labels.setValue(x.toInt + 1, -1))
      logger.info(s"${disableInds.length} bg inds are disabled, " +
        s"now ${(1 to labels.size(1)).filter(x => x == 0).length} inds")
    }
    labels
  }

  def getBboxInsideWeights(indsInside: ArrayBuffer[Int],
    labels: DenseVector[Float]): DenseMatrix[Float] = {
    val bboxInsideWeights = DenseMatrix.zeros[Float](indsInside.length, 4)
    labels.foreachPair((k, v) => {
      if (v == 1) {
        bboxInsideWeights(k, ::) :=
          convert(DenseVector(param.RPN_BBOX_INSIDE_WEIGHTS), Float).t
      }
    })
    bboxInsideWeights
  }

  def getBboxInsideWeights(indsInside: ArrayBuffer[Int],
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

  def getBboxOutsideWeights(indsInside: ArrayBuffer[Int], labels: DenseVector[Float])
  : DenseMatrix[Float] = {
    val bboxOutSideWeights = DenseMatrix.zeros[Float](indsInside.length, 4)

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
    bboxOutSideWeights
  }

  def getBboxOutsideWeights(indsInside: ArrayBuffer[Int], labels: Tensor[Float])
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
  def mapUpToOriginal(labels: DenseVector[Float],
    bboxTargets: DenseMatrix[Float],
    bboxInsideWeights: DenseMatrix[Float],
    bboxOutSideWeights: DenseMatrix[Float],
    indsInside: ArrayBuffer[Int]): BboxTarget = {
    val labels2 = unmap(Tensor(DenseMatrix(labels.data.array).t), totalAnchors, indsInside, -1)
    val bboxTargets2 = unmap(Tensor(bboxTargets), totalAnchors, indsInside, 0)
    val bboxInsideWeights2 = unmap(Tensor(bboxInsideWeights), totalAnchors, indsInside, 0)
    val bboxOutSideWeights2 = unmap(Tensor(bboxOutSideWeights), totalAnchors, indsInside, 0)

    compare("labelUnmap", labels2.toBreezeMatrix(), 1e-6)
    compare("targetUnmap", bboxTargets2.toBreezeMatrix(), 1e-6)
    compare("inwUnmap", bboxInsideWeights2.toBreezeMatrix(), 1e-6)
    compare("outWUnmap", bboxOutSideWeights2.toBreezeMatrix(), 1e-6)
    BboxTarget(labels2, bboxTargets2, bboxInsideWeights2, bboxOutSideWeights2)
  }

  /**
   * map up to original set of anchors
   */
  def mapUpToOriginal(labels: Tensor[Float],
    bboxTargets: Tensor[Float],
    bboxInsideWeights: Tensor[Float],
    bboxOutSideWeights: Tensor[Float],
    indsInside: ArrayBuffer[Int]): BboxTarget = {
    val labels2 = unmap(labels, totalAnchors, indsInside, -1)
    val bboxTargets2 = unmap(bboxTargets, totalAnchors, indsInside, 0)
    val bboxInsideWeights2 = unmap(bboxInsideWeights, totalAnchors, indsInside, 0)
    val bboxOutSideWeights2 = unmap(bboxOutSideWeights, totalAnchors, indsInside, 0)

    compare("labelUnmap", labels2, 1e-6)
    compare("targetUnmap", bboxTargets2, 1e-2)
    compare("inwUnmap", bboxInsideWeights2, 1e-6)
    compare("outWUnmap", bboxOutSideWeights2, 1e-6)

    BboxTarget(labels2, bboxTargets2, bboxInsideWeights2, bboxOutSideWeights2)
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

  def getIndsInside(width: Int, height: Int,
    allAnchors: Tensor[Float], allowedBorder: Float): ArrayBuffer[Int] = {
    var indsInside = ArrayBuffer[Int]()
    for (i <- 1 to allAnchors.size(1)) {
      if ((allAnchors.valueAt(i, 1) >= -allowedBorder) &&
        (allAnchors.valueAt(i, 2) >= -allowedBorder) &&
        (allAnchors.valueAt(i, 3) < width.toFloat + allowedBorder) &&
        (allAnchors.valueAt(i, 4) < height.toFloat + allowedBorder)) {
        indsInside += i
      }
    }
    indsInside
  }


  /**
   * Unmap a subset of item (data) back to the original set of items (of size count)
   */
  def unmap(data: Tensor[Float], count: Int, inds: ArrayBuffer[Int],
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
