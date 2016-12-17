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

import breeze.linalg.{DenseMatrix, DenseVector, argsort, max, min}
import com.intel.analytics.bigdl.tensor.Tensor

object Nms {
  def nms(dets: DenseMatrix[Float], thresh: Float): Array[Int] = {
    var keep = Array[Int]()
    if (dets.size == 0) return keep
    val x1 = dets(::, 0)
    val y1 = dets(::, 1)
    val x2 = dets(::, 2)
    val y2 = dets(::, 3)
    val scores = dets(::, 4)

    val areas = (x2 - x1 + 1f) :* (y2 - y1 + 1f)
    var order = argsort(scores).reverse.toArray

    while (order.size > 0) {
      val i = order(0)
      keep :+= i
      def getMax(vec: DenseVector[Float]) =
        MatrixUtil.selectMatrix(vec.toDenseMatrix.t, order.slice(1, order.length), 0)
          .map(x => max(x, vec(i)))
      val xx1 = getMax(x1)
      val yy1 = getMax(y1)
      def getMin(vec: DenseVector[Float]) = {
        MatrixUtil.selectMatrix(vec.toDenseMatrix.t, order.slice(1, order.length), 0)
          .map(x => min(x, vec(i)))
      }


      val xx2 = getMin(x2)
      val yy2 = getMin(y2)

      val w = (xx2 - xx1 + 1f).map(x => max(x, 0f))
      val h = (yy2 - yy1 + 1f).map(x => max(x, 0f))

      val inter = w :* h


      val selectedArea = MatrixUtil.selectMatrix(areas.toDenseMatrix.t,
        order.slice(1, order.length).array, 0)
      val ovr = inter :/ (selectedArea + areas(i) :- inter)
      val inds = ovr.findAll(x => x <= thresh).map(x => x._1).toArray
      order = inds.map(x => order(x + 1))
    }
    keep
  }

  def nms(dets: Tensor[Float], thresh: Float): Array[Int] = {
    var keep = Array[Int]()
    if (dets.size == 0) return keep
    val x1 = dets.select(2, 1)
    val y1 = dets.select(2, 2)
    val x2 = dets.select(2, 3)
    val y2 = dets.select(2, 4)
    val scores = dets.select(2, 5)

    val areas = x2.clone().add(-1, x1).add(1f).cmul(y2.clone().add(-1, y1).add(1f))
    var order = scores.topk(scores.nElement(), increase = false)._2.storage().array().map(x => x.toInt)

    while (order.size > 0) {
      val i = order(0)
      keep :+= i

      def getMax(vec: Tensor[Float]) =
        MatrixUtil.selectMatrix(vec, order.slice(1, order.length), 1)
          .apply1(x => Math.max(x, vec.valueAt(i)))
      val xx1 = getMax(x1)
      val yy1 = getMax(y1)

      def getMin(vec: Tensor[Float]) =
        MatrixUtil.selectMatrix(vec, order.slice(1, order.length), 1)
          .apply1(x => Math.min(x, vec.valueAt(i)))
      val xx2 = getMin(x2)
      val yy2 = getMin(y2)

      xx2.add(-1, xx1).add(1f).update(x => x < 0f, 0f) // w
      yy2.add(-1, yy1).add(1f).update(x => x < 0f, 0f) // h

      val inter = xx2.cmul(yy2)

      val selectedArea = MatrixUtil.selectMatrix(areas,
        order.slice(1, order.length).array, 0)
      val ovr = inter.cdiv(selectedArea.add(areas.valueAt(i)).add(-1, inter))
      val inds = ovr.contiguous().storage().array().zipWithIndex
        .filter(x => x._1 <= thresh).map(x => x._2)
      order = inds.map(ind => order(ind + 1))
    }
    keep
  }
}
