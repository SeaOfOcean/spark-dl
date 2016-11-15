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

  def selectMatrix(matrix: DenseMatrix[Float], selectInds: Array[Int], dim: Int): DenseMatrix[Float] = {
    assert(dim == 0 || dim == 1)
    //select rows
    if (dim == 0) {
      val res = new DenseMatrix[Float](selectInds.length, matrix.cols)
      selectInds.zipWithIndex.map(x => res(x._2, ::) := matrix(x._1, ::))
      res
    } else {
      val res = new DenseMatrix[Float](matrix.rows, selectInds.length)
      selectInds.zipWithIndex.map(x => res(::, x._2) := matrix(::, x._1))
      res
    }
  }


  def meshgrid(x1: DenseVector[Float], x2: DenseVector[Float]): (DenseMatrix[Float], DenseMatrix[Float]) = {
    val x1Mesh = DenseMatrix.zeros[Float](x2.length, x1.length)
    for (i <- 0 until x2.length) {
      x1Mesh(i, ::) := x1.t
    }
    val x2Mesh = DenseMatrix.zeros[Float](x2.length, x1.length)
    for (i <- 0 until x1.length) {
      x2Mesh(::, i) := x2
    }
    (x1Mesh, x2Mesh)
  }

  def argmax2(arr: DenseMatrix[Float], d: Int): Array[Int] = {
    assert(d == 0 || d == 1)
    if (arr.size == 0) return Array[Int]()
    if (d == 0) {
      Array.range(0, arr.cols).map(i => {
        argmax(arr(::, i))
      })
    } else {
      Array.range(0, arr.rows).map(i => {
        argmax(arr(i, ::))
      })
    }
  }

  def selectCol(mat: DenseMatrix[Float], cid: Int) = MatrixUtil.selectMatrix(mat, Array(cid), 1)

  def selectCols(mat: DenseMatrix[Float], cid: Int, step: Int): DenseMatrix[Float] = {
    val out = new DenseMatrix[Float](mat.rows, mat.cols / step)
    var ind = 0
    for (i <- cid until mat.cols by step) {
      out(::, ind) := selectCol(mat, i).toDenseVector
      ind += 1
    }
    out
  }

  def printSelect(info: String, data: DenseMatrix[Float]) = {
    println(s"=========================$info======================")
    if (data.rows <= 10) println(data)
    else {
      for (i <- 0 until 5) {
        for (j <- 0 until data.cols) {
          print(data.valueAt(i, j) + " ")
        }
        println()
      }
      println("-------------")
      for (i <- 1 to 5) {
        for (j <- 0 until data.cols) {
          print(data.valueAt(data.rows - 6 + i, j) + " ")
        }
        println()
      }
    }
    println("... shape: ", data.rows, data.cols)
  }

}
