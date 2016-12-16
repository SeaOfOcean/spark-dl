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

import breeze.linalg.{DenseMatrix, DenseVector, argsort}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.pvanet.model.FasterRcnnParam
import com.intel.analytics.bigdl.pvanet.utils._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Proposal[@specialized(Float, Double) T: ClassTag](param: FasterRcnnParam)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Table, T] {
  /**
   * Outputs object detection proposals by applying estimated bounding-box
   * transformations to a set of regular boxes (called "anchors").
   */

  // rois blob: holds R regions of interest, each is a 5-tuple
  // (n, x1, y1, x2, y2) specifying an image batch index n and a
  // rectangle (x1, y1, x2, y2)
  //  top[ 0].reshape(1, 5)

  // scores blob: holds scores for R regions of interest
  //  if len(top) > 1:
  //    top[ 1].reshape(1, 1, 1, 1)

  val basicAnchors = Anchor.generateBasicAnchors(param.anchorRatios, param.anchorScales)

  /**
   *
   * @param input
   * input(1): cls scores
   * input(2): bbox pred
   * input(3): im_info
   * @return output
   *         output(1): rpn_rois
   *         output(2): rpn_scores
   */
  override def updateOutput(input: Table): Table = {
    // Algorithm:
    //
    // for each (H, W) location i
    //   generate A anchor boxes centered on cell i
    //   apply predicted bbox deltas at cell i to each of the A anchors
    // clip predicted boxes to image
    // remove predicted boxes with either height or width < threshold
    // sort all (proposal, score) pairs by score from highest to lowest
    // take top pre_nms_topN proposals before NMS
    // apply NMS with threshold 0.7 to remaining proposals
    // take after_nms_topN proposals after NMS
    // return the top proposals (-> RoIs top, scores top)
    val data = input(1).asInstanceOf[Tensor[Float]].clone()
    assert(data.size(1) == 1, "Only single item batches are supported")

    val pre_nms_topN = param.RPN_PRE_NMS_TOP_N
    val post_nms_topN = param.RPN_POST_NMS_TOP_N
    val nms_thresh = param.RPN_NMS_THRESH
    val min_size = param.RPN_MIN_SIZE

    // the first set of _num_anchors channels are bg probs
    // the second set are the fg probs, which we want
    val dataSize = data.size()
    data.resize(dataSize(1), data.nElement() / dataSize(1))
    var scoresTensor = data.narrow(1, param.anchorNum + 1, param.anchorNum)
    scoresTensor.resize(1, param.anchorNum, dataSize(2), dataSize(3))
    // bbox_deltas: (1, 4A, H, W)
    var bboxDeltas = input(2).asInstanceOf[Tensor[Float]].clone()

    // Transpose and reshape predicted bbox transformations to get them
    // into the same order as the anchors:
    //
    // bbox deltas will be (1, 4 * A, H, W) format
    // transpose to (1, H, W, 4 * A)
    // reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    // in slowest to fastest order
    def transposeAndReshape(mat: Tensor[Float], cols: Int): Tensor[Float] = {
      val out = Tensor[Float](mat.nElement() / cols, cols)
      var ind = 1
      for (r <- 1 to mat.size(3)) {
        for (c <- 1 to mat.size(4)) {
          for (i <- 1 to mat.size(2) by cols) {
            for (j <- 1 to cols) {
              out.setValue(ind, j, mat.valueAt(1, i + j - 1, r, c))
            }
            ind += 1
          }
        }
      }
      out
    }

    bboxDeltas = transposeAndReshape(bboxDeltas, 4)

    scoresTensor = transposeAndReshape(scoresTensor, 1)

    val imInfo = input(3).asInstanceOf[Tensor[Float]].clone()

    // 1. Generate proposals from bbox deltas and shifted anchors
    val height = dataSize(2)
    val width = dataSize(3)

    // Enumerate all shifts
    val shifts = Anchor.generateShifts(width, height, param.featStride)

    val anchors: DenseMatrix[Float] = Anchor.getAllAnchors(shifts, basicAnchors)

    // Convert anchors into proposals via bbox transformations
    var proposals = Bbox.bboxTransformInv(anchors, bboxDeltas.toBreezeMatrix())

    // 2. clip predicted boxes to image
    proposals = Bbox.clipBoxes(proposals, imInfo.valueAt(1), imInfo.valueAt(2))

    // 3. remove predicted boxes with either height or width < threshold
    // (NOTE: convert min_size to input image scale stored in im_info[2])
    var keep = filterBoxes(proposals, min_size * imInfo.valueAt(3))

    proposals = MatrixUtil.selectMatrix(proposals, keep, 0)
    var scores = MatrixUtil.selectMatrix(scoresTensor.toBreezeMatrix(), keep, 0)

    // 4. sort all (proposal, score) pairs by score from highest to lowest
    // 5. take top pre_nms_topN (e.g. 6000)
    var order = argsort(scores.toDenseVector).reverse.toArray
    if (pre_nms_topN > 0) {
      order = order.slice(0, pre_nms_topN)
    }
    proposals = MatrixUtil.selectMatrix(proposals, order, 0)
    scores = MatrixUtil.selectMatrix(scores, order, 0)

    // 6. apply nms (e.g. threshold = 0.7)
    // 7. take after_nms_topN (e.g. 300)
    // 8. return the top proposals (-> RoIs top)
    keep = Nms.nms(DenseMatrix.horzcat(proposals, scores), nms_thresh.toFloat)
    if (post_nms_topN > 0) {
      keep = keep.slice(0, post_nms_topN)
    }
    proposals = MatrixUtil.selectMatrix(proposals, keep, 0)
    scores = MatrixUtil.selectMatrix(scores, keep, 0)

    // Output rois blob
    // Our RPN implementation only supports a single input image, so all
    // batch inds are 0
    val mat = DenseMatrix.horzcat(DenseMatrix.zeros[Float](proposals.rows, 1), proposals)
    val rpn_rois = Tensor[Float]()
    rpn_rois.resize(mat.rows, mat.cols)
    for (i <- 1 to mat.rows) {
      for (j <- 1 to mat.cols) {
        rpn_rois.setValue(i, j, mat(i - 1, j - 1))
      }
    }
    if (output.length == 0) {
      output.insert(rpn_rois)
      output.insert(Tensor(Storage(scores.toArray)))
    } else {
      output.update(1, rpn_rois)
      output.update(2, Tensor(Storage(scores.toArray)))
    }
//    FileUtil.assertEqual("rpn_rois", rpn_rois)
    output
  }

  /**
   * Remove all boxes with any side smaller than min_size
   *
   */
  private def filterBoxes(boxes: DenseMatrix[Float], minSize: Float): Array[Int] = {
    val ws: DenseVector[Float] = boxes(::, 2) - boxes(::, 0) + 1f
    val hs: DenseVector[Float] = boxes(::, 3) - boxes(::, 1) + 1f

    var keep = Array[Int]()
    for (i <- 0 until boxes.rows) {
      if (ws(i) >= minSize && hs(i) >= minSize) {
        keep :+= i
      }
    }
    keep
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = gradOutput
    gradInput
  }
}


