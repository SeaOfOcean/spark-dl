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

import breeze.linalg.{*, DenseMatrix, DenseVector, min}
import breeze.numerics._
import com.intel.analytics.bigdl.tensor.Tensor

object Bbox {
  def bboxVote(detsNMS: DenseMatrix[Float], detsAll: DenseMatrix[Float]): DenseMatrix[Float] = {
    val detsVoted = new DenseMatrix[Float](detsNMS.rows, detsNMS.cols)
    var accBox: DenseVector[Float] = null
    var accScore = 0f
    var det: Array[Float] = null
    for (i <- 0 until detsNMS.rows) {
      det = MatrixUtil.selectRow(detsNMS, i)
      if (accBox == null) {
        accBox = DenseVector.zeros[Float](4)
      } else {
        accBox := 0f
      }
      accScore = 0f
      for (m <- 0 until detsAll.rows) {
        val det2 = MatrixUtil.selectRow(detsAll, m)

        val bis0 = Math.max(det(0), det2(0))
        val bis1 = Math.max(det(1), det2(1))
        val bis2 = Math.min(det(2), det2(2))
        val bis3 = Math.min(det(3), det2(3))

        val iw = bis2 - bis0 + 1
        val ih = bis3 - bis1 + 1

        if (iw > 0 && ih > 0) {
          val ua = (det(2) - det(0) + 1) * (det(3) - det(1) + 1) +
            (det2(2) - det2(0) + 1) * (det2(3) - det2(1) + 1) - iw * ih
          val ov = iw * ih / ua
          if (ov >= 0.5) {
            accBox :+= (DenseVector(det2.slice(0, 4)) :* det2(4))
            accScore += det2(4)
          }
        }
      }
      (0 until 4).foreach(x => detsVoted(i, x) = accBox(x) / accScore)
      detsVoted(i, 4) = det(4)
    }
    detsVoted
  }

  def bboxVote(detsNMS: Tensor[Float], detsAll: Tensor[Float]): Tensor[Float] = {
    val detsVoted = Tensor[Float](detsNMS.size(1), detsNMS.size(2))
    var accBox: Tensor[Float] = null
    var accScore = 0f
    var det: Tensor[Float] = null
    for (i <- 1 to detsNMS.size(1)) {
      det = MatrixUtil.selectRow(detsNMS, i)
      if (accBox == null) {
        accBox = Tensor[Float](4)
      } else {
        accBox.fill(0f)
      }
      accScore = 0f
      for (m <- 1 to detsAll.size(1)) {
        val det2 = MatrixUtil.selectRow(detsAll, m, newSpace = true)

        val bis0 = Math.max(det.valueAt(1), det2.valueAt(1))
        val bis1 = Math.max(det.valueAt(2), det2.valueAt(2))
        val bis2 = Math.min(det.valueAt(3), det2.valueAt(3))
        val bis3 = Math.min(det.valueAt(4), det2.valueAt(4))

        val iw = bis2 - bis0 + 1
        val ih = bis3 - bis1 + 1

        if (iw > 0 && ih > 0) {
          val ua = (det.valueAt(3) - det.valueAt(1) + 1) * (det.valueAt(4) - det.valueAt(2) + 1) +
            (det2.valueAt(3) - det2.valueAt(1) + 1) *
              (det2.valueAt(4) - det2.valueAt(2) + 1) - iw * ih
          val ov = iw * ih / ua
          if (ov >= 0.5) {
            accBox.add(det2.narrow(1, 1, 4).mul(det2.valueAt(5)))
            accScore += det2.valueAt(5)
          }
        }
      }
      (1 to 4).foreach(x => detsVoted.setValue(i, x, accBox.valueAt(x) / accScore))
      detsVoted.setValue(i, 5, det.valueAt(5))
    }
    detsVoted
  }


  /**
   *
   * @param boxes      (N, 4) ndarray of float
   * @param queryBoxes (K, >=4) ndarray of float
   * @return overlaps: (N, K) ndarray of overlap between boxes and query_boxes
   */
  def bboxOverlap(boxes: DenseMatrix[Float], queryBoxes: DenseMatrix[Float])
  : DenseMatrix[Float] = {
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


  def bboxOverlap(boxes: Tensor[Float], queryBoxes: Tensor[Float]): Tensor[Float] = {
    require(boxes.size(2) >= 4)
    require(queryBoxes.size(2) >= 4)
    val N = boxes.size(1)
    val K = queryBoxes.size(1)
    val overlaps = Tensor[Float](N, K)

    for (k <- 1 to K) {
      val boxArea = (queryBoxes.valueAt(k, 3) - queryBoxes.valueAt(k, 1) + 1) *
        (queryBoxes.valueAt(k, 4) - queryBoxes.valueAt(k, 2) + 1)
      for (n <- 1 to N) {
        val iw = Math.min(boxes.valueAt(n, 3), queryBoxes.valueAt(k, 3)) -
          Math.max(boxes.valueAt(n, 1), queryBoxes.valueAt(k, 1)) + 1
        if (iw > 0) {
          val ih = Math.min(boxes.valueAt(n, 4), queryBoxes.valueAt(k, 4)) -
            Math.max(boxes.valueAt(n, 2), queryBoxes.valueAt(k, 2)) + 1

          if (ih > 0) {
            val ua = (boxes.valueAt(n, 3) - boxes.valueAt(n, 1) + 1) *
              (boxes.valueAt(n, 4) - boxes.valueAt(n, 2) + 1) + boxArea - iw * ih
            overlaps.setValue(n, k, iw * ih / ua)
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

  /**
   * copy value to corresponding cols of mat, the start col ind is cid, with step
   *
   */
  private def setCols(mat: Tensor[Float], cid: Int, step: Int, value: Tensor[Float]) = {
    var ind = 1
    for (i <- cid to mat.size(2) by step) {
      (1 to mat.size(1)).foreach(rid => mat.setValue(rid, i, value.valueAt(rid, ind)))
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

  def bboxTransform(exRois: Tensor[Float], gtRois: Tensor[Float]): Tensor[Float] = {
    val exWidths = MatrixUtil.selectCol(exRois, 3) - MatrixUtil.selectCol(exRois, 1) + 1.0f
    val exHeights = MatrixUtil.selectCol(exRois, 4) - MatrixUtil.selectCol(exRois, 2) + 1.0f
    val exCtrX = MatrixUtil.selectCol(exRois, 1) + exWidths * 0.5f
    val exCtrY = MatrixUtil.selectCol(exRois, 2) + exHeights * 0.5f

    val gtWidths = MatrixUtil.selectCol(gtRois, 3) - MatrixUtil.selectCol(gtRois, 1) + 1.0f
    val gtHeights = MatrixUtil.selectCol(gtRois, 4) - MatrixUtil.selectCol(gtRois, 2) + 1.0f
    val gtCtrX = MatrixUtil.selectCol(gtRois, 1) + gtWidths * 0.5f
    val gtCtrY = MatrixUtil.selectCol(gtRois, 2) + gtHeights * 0.5f

    val targetsDx = (gtCtrX - exCtrX) / exWidths
    val targetsDy = (gtCtrY - exCtrY) / exHeights
    val targetsDw = gtWidths.cdiv(exWidths).log()
    val targetsDh = gtHeights.cdiv(exHeights).log()

    val res = TensorUtil.vertConcat(targetsDx, targetsDy, targetsDw, targetsDh)
    res.t().contiguous()
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

  def bboxTransformInv(boxes: Tensor[Float], deltas: Tensor[Float]): Tensor[Float] = {
    if (boxes.size(1) == 0) {
      return Tensor[Float](0, deltas.size(2))
    }
    val widths = MatrixUtil.selectCol(boxes, 3) - MatrixUtil.selectCol(boxes, 1) + 1.0f
    val heights = MatrixUtil.selectCol(boxes, 4) - MatrixUtil.selectCol(boxes, 2) + 1.0f
    val ctrX = MatrixUtil.selectCol(boxes, 1) + widths * 0.5f
    val ctrY = MatrixUtil.selectCol(boxes, 2) + heights * 0.5f

    val dx = MatrixUtil.selectCols(deltas, 1, 4)
    val dy = MatrixUtil.selectCols(deltas, 2, 4)
    var dw = MatrixUtil.selectCols(deltas, 3, 4)
    var dh = MatrixUtil.selectCols(deltas, 4, 4)

    MatrixUtil.mulVecToMatCols(dx, widths)
    MatrixUtil.addVecToMatCols(dx, ctrX)
    MatrixUtil.mulVecToMatCols(dy, heights)
    MatrixUtil.addVecToMatCols(dy, ctrY)
    dw = dw.exp()
    MatrixUtil.mulVecToMatCols(dw, widths)
    dh = dh.exp()
    MatrixUtil.mulVecToMatCols(dh, heights)

    val predBoxes = Tensor[Float](deltas.size(1), deltas.size(2))

    setCols(predBoxes, 1, 4, dx - dw * 0.5f)
    setCols(predBoxes, 2, 4, dy - dh * 0.5f)
    setCols(predBoxes, 3, 4, dx + dw * 0.5f)
    setCols(predBoxes, 4, 4, dy + dh * 0.5f)

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

  /**
   * Clip boxes to image boundaries.
   *
   * @return
   */
  def clipBoxes(boxes: Tensor[Float], height: Float, width: Float): Tensor[Float] = {
    for (r <- 1 to boxes.size(1)) {
      for (c <- 1 to boxes.size(2) by 4) {
        boxes.setValue(r, c, Math.max(Math.min(boxes.valueAt(r, c), width - 1f), 0))
        boxes.setValue(r, c + 1, Math.max(Math.min(boxes.valueAt(r, c + 1), height - 1f), 0))
        boxes.setValue(r, c + 2, Math.max(min(boxes.valueAt(r, c + 2), width - 1f), 0))
        boxes.setValue(r, c + 3, Math.max(min(boxes.valueAt(r, c + 3), height - 1f), 0))
      }
    }
    boxes
  }
}
