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

package com.intel.analytics.bigdl.pvanet.utils

import breeze.linalg.{*, DenseMatrix, min}
import breeze.numerics._

object Bbox {
  def bboxVote(detsNMSed: DenseMatrix[Float], clsDets: DenseMatrix[Float]): DenseMatrix[Float] = {
    throw new UnsupportedOperationException
  }


  /**
   *
   * @param boxes      (N, 4) ndarray of float
   * @param queryBoxes (K, >=4) ndarray of float
   * @return overlaps: (N, K) ndarray of overlap between boxes and query_boxes
   */
  def bboxOverlap(boxes: DenseMatrix[Float], queryBoxes: DenseMatrix[Float]): DenseMatrix[Float] = {
    require(boxes.cols >= 4)
    require(queryBoxes.cols >= 4)
    val N = boxes.rows
    val K = queryBoxes.rows
    val overlaps = new DenseMatrix[Float](N, K)

    for (k <- 0 until K) {
      val boxArea: Float = (queryBoxes(k, 2) - queryBoxes(k, 0) + 1) *
        (queryBoxes(k, 3) - queryBoxes(k, 1) + 1)
      for (n <- 0 until N) {
        val iw: Float = Math.min(boxes(n, 2), queryBoxes(k, 2)) -
          Math.max(boxes(n, 0), queryBoxes(k, 0)) + 1
        if (iw > 0) {
          val ih: Float =
            Math.min(boxes(n, 3), queryBoxes(k, 3)) - Math.max(boxes(n, 1), queryBoxes(k, 1)) + 1

          if (ih > 0) {
            val ua: Float = (boxes(n, 2) - boxes(n, 0) + 1) *
              (boxes(n, 3) - boxes(n, 1) + 1) + boxArea - iw * ih
            overlaps(n, k) = iw * ih / ua
          }
        }
      }
    }
    overlaps
  }

  /**
   * copy value to corresponding cols of mat, the start col ind is cid, with step
   *
   */
  private def setCols(mat: DenseMatrix[Float], cid: Int, step: Int, value: DenseMatrix[Float]) = {
    var ind = 0
    for (i <- cid until mat.cols by step) {
      mat(::, i) := value(::, ind)
      ind += 1
    }
  }

  def bboxTransform(exRois: DenseMatrix[Float], gtRois: DenseMatrix[Float]): DenseMatrix[Float] = {
    val exWidths = MatrixUtil.selectCol(exRois, 2) - MatrixUtil.selectCol(exRois, 0) + 1.0f
    val exHeights = MatrixUtil.selectCol(exRois, 3) - MatrixUtil.selectCol(exRois, 1) + 1.0f
    val exCtrX = MatrixUtil.selectCol(exRois, 0) + exWidths * 0.5f
    val exCtrY = MatrixUtil.selectCol(exRois, 1) + exHeights * 0.5f

    val gtWidths = MatrixUtil.selectCol(gtRois, 2) - MatrixUtil.selectCol(gtRois, 0) + 1.0f
    val gtHeights = MatrixUtil.selectCol(gtRois, 3) - MatrixUtil.selectCol(gtRois, 1) + 1.0f
    val gtCtrX = MatrixUtil.selectCol(gtRois, 0) + gtWidths * 0.5f
    val gtCtrY = MatrixUtil.selectCol(gtRois, 1) + gtHeights * 0.5f

    val targetsDx = (gtCtrX - exCtrX) / exWidths
    val targetsDy = (gtCtrY - exCtrY) / exHeights
    val targetsDw = log(gtWidths / exWidths)
    val targetsDh = log(gtHeights / exHeights)

    val res = DenseMatrix.vertcat(targetsDx, targetsDy, targetsDw, targetsDh)
    res.reshape(res.size / 4, 4)
  }


  /**
   *
   * @param boxes  (N, 4)
   * @param deltas (N, 4)
   * @return
   */
  def bboxTransformInv(boxes: DenseMatrix[Float],
    deltas: DenseMatrix[Float]): DenseMatrix[Float] = {
    if (boxes.rows == 0) {
      return DenseMatrix.fill(0, deltas.cols) {
        0f
      }
    }
    val widths = MatrixUtil.selectCol(boxes, 2) - MatrixUtil.selectCol(boxes, 0) + 1.0f
    val heights = MatrixUtil.selectCol(boxes, 3) - MatrixUtil.selectCol(boxes, 1) + 1.0f
    val ctrX = MatrixUtil.selectCol(boxes, 0) + widths * 0.5f
    val ctrY = MatrixUtil.selectCol(boxes, 1) + heights * 0.5f

    val dx = MatrixUtil.selectCols(deltas, 0, 4)
    val dy = MatrixUtil.selectCols(deltas, 1, 4)
    var dw = MatrixUtil.selectCols(deltas, 2, 4)
    var dh = MatrixUtil.selectCols(deltas, 3, 4)

    dx(::, *) :*= widths.toDenseVector
    dx(::, *) :+= ctrX.toDenseVector
    dy(::, *) :*= heights.toDenseVector
    dy(::, *) :+= ctrY.toDenseVector
    dw = exp(dw)
    dw(::, *) :*= widths.toDenseVector
    dh = exp(dh)
    dh(::, *) :*= heights.toDenseVector

    val predBoxes = DenseMatrix.zeros[Float](deltas.rows, deltas.cols)

    setCols(predBoxes, 0, 4, dx - dw * 0.5f)
    setCols(predBoxes, 1, 4, dy - dh * 0.5f)
    setCols(predBoxes, 2, 4, dx + dw * 0.5f)
    setCols(predBoxes, 3, 4, dy + dh * 0.5f)

    predBoxes
  }

  /**
   * Clip boxes to image boundaries.
   *
   * @return
   */
  def clipBoxes(boxes: DenseMatrix[Float], height: Float, width: Float): DenseMatrix[Float] = {
    for (r <- 0 until boxes.rows) {
      for (c <- 0 until boxes.cols by 4) {
        boxes(r, c) = Math.max(Math.min(boxes(r, c), width - 1f), 0)
        boxes(r, c + 1) = Math.max(Math.min(boxes(r, c + 1), height - 1f), 0)
        boxes(r, c + 2) = Math.max(min(boxes(r, c + 2), width - 1f), 0)
        boxes(r, c + 3) = Math.max(min(boxes(r, c + 3), height - 1f), 0)
      }
    }
    boxes
  }
}
