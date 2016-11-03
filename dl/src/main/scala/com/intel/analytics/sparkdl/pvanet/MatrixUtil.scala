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

import breeze.linalg.{DenseMatrix, DenseVector, argmax}

object MatrixUtil {

  def select(matrix: DenseMatrix[Float], selectInds: Array[Int], dim: Int): Option[DenseMatrix[Float]] = {
    //select rows
    if (dim == 0) {
      val res = new DenseMatrix[Float](selectInds.length, matrix.cols)
      selectInds.zipWithIndex.map(x => res(x._2, ::) := matrix(x._1, ::))
      Some(res)
    } else if (dim == 1) {
      val res = new DenseMatrix[Float](matrix.rows, selectInds.length)
      selectInds.zipWithIndex.map(x => res(::, x._2) := matrix(::, x._1))
      Some(res)
    } else {
      None
    }
  }


  def meshgrid(x1: DenseVector[Float], x2: DenseVector[Float]): Option[(DenseMatrix[Float], DenseMatrix[Float])] = {
    val x1Mesh = DenseMatrix.zeros[Float](x2.length, x1.length)
    for (i <- 0 until x2.length) {
      x1Mesh(i, ::) := x1.t
    }
    val x2Mesh = DenseMatrix.zeros[Float](x2.length, x1.length)
    for (i <- 0 until x1.length) {
      x2Mesh(::, i) := x2
    }
    Some(x1Mesh, x2Mesh)
  }

  def argmax2(arr: DenseMatrix[Float], d: Int): Option[Array[Int]] = {
    if(arr.size == 0) return Some(Array[Int]())
    if (d == 0) {
      Some(Array.range(0, arr.cols).map(i => {
        argmax(arr(::, i))
      }))
    } else if (d == 1) {
      Some(Array.range(0, arr.rows).map(i => {
        argmax(arr(i, ::))
      }))
    } else {
      None
    }
  }
}
