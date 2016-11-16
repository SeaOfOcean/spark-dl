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

import breeze.linalg.{DenseMatrix, DenseVector, argsort, max, min}

object Nms {
  def nms(dets: DenseMatrix[Float], thresh: Float): Array[Int] = {
    val x1 = dets(::, 0)
    val y1 = dets(::, 1)
    val x2 = dets(::, 2)
    val y2 = dets(::, 3)
    val scores = dets(::, 4)

    val areas = (x2 - x1 + 1f) :* (y2 - y1 + 1f)
    var order = argsort(scores).reverse.toArray

    var keep = Array[Int]()
    while (order.size > 0) {
      val i = order(0)
      keep :+= i
      def getMax(vec: DenseVector[Float]) =
        MatrixUtil.selectMatrix(vec.toDenseMatrix.t, order.slice(1, order.length), 0).map(x => max(x, vec(i)))
      val xx1 = getMax(x1)
      val yy1 = getMax(y1)
      def getMin(vec: DenseVector[Float]) =
        MatrixUtil.selectMatrix(vec.toDenseMatrix.t, order.slice(1, order.length), 0).map(x => min(x, vec(i)))
      val xx2 = getMin(x2)
      val yy2 = getMin(y2)

      val w = (xx2 - xx1 + 1f).map(x => max(x, 0f))
      val h = (yy2 - yy1 + 1f).map(x => max(x, 0f))

      val inter = w :* h

      val selectedArea = MatrixUtil.selectMatrix(areas.toDenseMatrix.t, order.slice(1, order.length).array, 0)
      val ovr = inter :/ (selectedArea + areas(i) :- inter)
      val inds = ovr.findAll(x => x <= thresh).map(x => x._1).toArray
      order = inds.zipWithIndex.map(x => order(x._1 + 1))
    }
    keep
  }
}
