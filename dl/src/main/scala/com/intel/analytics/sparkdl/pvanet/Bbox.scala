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

import breeze.linalg.DenseMatrix
import breeze.numerics._

object Bbox {
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
    var overlaps = new DenseMatrix[Float](N, K)

    for (k <- 0 until K) {
      val boxArea: Float = (queryBoxes(k, 2) - queryBoxes(k, 0) + 1) * (queryBoxes(k, 3) - queryBoxes(k, 1) + 1)
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

  def selectCol(mat: DenseMatrix[Float], cid: Int) = MatrixUtil.select(mat, Array(cid), 1).get

  def bboxTransform(exRois: DenseMatrix[Float], gtRois: DenseMatrix[Float]): DenseMatrix[Float] = {
    val exWidths = selectCol(exRois, 2) - selectCol(exRois, 0) + 1.0f
    val exHeights = selectCol(exRois, 3) - selectCol(exRois, 1) + 1.0f
    val exCtrX = selectCol(exRois, 0) + exWidths * 0.5f
    val exCtrY = selectCol(exRois, 1) + exHeights * 0.5f

    val gtWidths = selectCol(gtRois, 2) - selectCol(gtRois, 0) + 1.0f
    val gtHeights = selectCol(gtRois, 3) - selectCol(gtRois, 1) + 1.0f
    val gtCtrX = selectCol(gtRois, 0) + gtWidths * 0.5f
    val gtCtrY = selectCol(gtRois, 1) + gtHeights * 0.5f

    val targetsDx = (gtCtrX - exCtrX) / exWidths
    val targetsDy = (gtCtrY - exCtrY) / exHeights
    val targetsDw = log(gtWidths / exWidths)
    val targetsDh = log(gtHeights / exHeights)

    val res = DenseMatrix.vertcat(targetsDx, targetsDy, targetsDw, targetsDh)
    res.reshape(res.size / 4, 4)
  }
}
