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

import breeze.linalg.{DenseMatrix, DenseVector, argmax, max}
import com.intel.analytics.bigdl.tensor.Tensor

object MatrixUtil {
  def mulVecToMatCols(mat: Tensor[Float], vec: Tensor[Float]): Unit = {
    assert(mat.size(1) == vec.nElement())
    (1 to mat.size(1)).foreach(rid => mat(rid).mul(vec.valueAt(rid)))
  }

  def addVecToMatCols(mat: Tensor[Float], vec: Tensor[Float]): Unit = {
    (1 to mat.size(2)).foreach(cid => {
      (1 to mat.size(1)).foreach(rid => {
        mat.setValue(rid, cid, mat.valueAt(rid, cid) + vec.valueAt(rid))
      })
    })
  }


  def selectMatrix2(mat: DenseMatrix[Float],
    rows: Array[Int], cols: Array[Int]): DenseMatrix[Float] = {
    val out = new DenseMatrix[Float](rows.length, cols.length)
    rows.zipWithIndex.map(r => {
      cols.zipWithIndex.map(c => {
        out(r._2, c._2) = mat(r._1, c._1)
      })
    })
    out
  }

  def selectMatrix2(mat: Tensor[Float],
    rows: Array[Int], cols: Array[Int]): Tensor[Float] = {
    val out = Tensor[Float](rows.length, cols.length)
    rows.zip(Stream from (1)).map(r => {
      cols.zip(Stream from (1)).map(c => {
        out.setValue(r._2, c._2, mat.valueAt(r._1, c._1))
      })
    })
    out
  }


  def selectMatrix(matrix: DenseMatrix[Float],
    selectInds: Array[Int], dim: Int): DenseMatrix[Float] = {
    assert(dim == 0 || dim == 1)
    // select rows
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

  def selectMatrix(matrix: Tensor[Float],
    selectInds: Array[Int], dim: Int): Tensor[Float] = {
    assert(dim == 1 || dim == 2)
    if (matrix.nDimension() == 1) {
      val res = Tensor[Float](selectInds.length)
      selectInds.zip(Stream.from(1)).map { x =>
        res.update(x._2, matrix.valueAt(x._1))
      }
      return res
    }
    // select rows
    if (dim == 1) {
      val res = Tensor[Float](selectInds.length, matrix.size(2))
      selectInds.zip(Stream.from(1)).map(x => res.update(x._2, matrix(x._1)))
      res
    } else {
      val res = Tensor[Float](matrix.size(1), selectInds.length)
      selectInds.zip(Stream.from(1)).map(x => updateCol(res, x._2, selectCol(matrix, x._1)))
      res
    }
  }


  def meshgrid(x1: DenseVector[Float],
    x2: DenseVector[Float]): (DenseMatrix[Float], DenseMatrix[Float]) = {
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

  /**
   * If x1 = [1, 2, 3]
   * x2 = [4, 5, 6]
   * return [1 2 3
   * 1 2 3
   * 1 2 3
   * 4 4 4
   * 5 5 5
   * 6 6 6]
   *
   * @param x1
   * @param x2
   * @return
   */
  def meshgrid(x1: Tensor[Float],
    x2: Tensor[Float]): (Tensor[Float], Tensor[Float]) = {
    val x1Mesh = Tensor[Float](x2.nElement(), x1.nElement())
    (1 to x2.nElement()).foreach(i => x1Mesh.update(i, x1))
    val x2Mesh = Tensor[Float](x2.nElement(), x1.nElement())
    (1 to x1.nElement()).foreach { i =>
      (1 to x2Mesh.size(1)).foreach(x => x2Mesh.setValue(x, i, x2.valueAt(x)))
    }
    (x1Mesh, x2Mesh)
  }

  /**
   * return the max value in rows(d=0) or in cols(d=1)
   * arr = [4 9
   * 5 7
   * 8 5]
   *
   * argmax2(arr, 0) will return 2, 0
   * argmax2(arr, 1) will return 1, 1, 0
   *
   * @param arr
   * @param d
   * @return
   */
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

  /**
   * return the max value in rows(d=0) or in cols(d=1)
   * arr = [4 9
   * 5 7
   * 8 5]
   *
   * argmax2(arr, 1) will return 3, 1
   * argmax2(arr, 2) will return 2, 2, 1
   *
   * @param arr
   * @param d
   * @return
   * todo: this maybe removed
   */
  def argmax2(arr: Tensor[Float], d: Int): Array[Int] = {
    arr.max(d)._2.storage().array().map(x => x.toInt)
//    if (arr.size == 0) return Array[Int]()
//    if (d == 0) {
//      Array.range(0, arr.size(2)).map(i => {
//        argmax(arr(::, i))
//      })
//    } else {
//      Array.range(0, arr.rows).map(i => {
//        argmax(arr(i, ::))
//      })
//    }
  }

  def max2(arr: DenseMatrix[Float], d: Int): Array[Float] = {
    assert(d == 0 || d == 1)
    if (arr.size == 0) return Array[Float]()
    if (d == 0) {
      Array.range(0, arr.cols).map(i => {
        max(arr(::, i))
      })
    } else {
      Array.range(0, arr.rows).map(i => {
        max(arr(i, ::))
      })
    }
  }

  def max2(arr: Tensor[Float], d: Int): Array[Float] = {
    arr.max(d)._1.storage().array()
//    if (arr.size == 0) return Array[Float]()
//    if (d == 1) {
//      (1 to arr.size(2)).map(i => {
//        arr.
//        max(arr(::, i))
//      })
//    } else {
//      (1 to arr.size(1)).map(i => {
//        max(arr(i, ::))
//      })
//    }
  }

  def selectCol(mat: DenseMatrix[Float], cid: Int): DenseMatrix[Float] =
    selectMatrix(mat, Array(cid), 1)

  def selectCol(mat: Tensor[Float], cid: Int): Tensor[Float] = {
    if (mat.nElement() == 0) return Tensor[Float](0)
    mat.select(2, cid)
//    val col = Tensor[Float](mat.size(1))
//    (1 to mat.size(1)).foreach(rid => col.setValue(rid, mat.valueAt(rid, cid)))
//    col
  }

  def selectColAsArray(mat: Tensor[Float], cid: Int): Array[Float] = {
    val res = selectCol(mat, cid)
    if (res.nElement() == 0) Array[Float]()
    else res.clone().storage().array()
  }

  def selectCols(mat: DenseMatrix[Float], cid: Int, step: Int): DenseMatrix[Float] = {
    val out = new DenseMatrix[Float](mat.rows, mat.cols / step)
    var ind = 0
    for (i <- cid until mat.cols by step) {
      out(::, ind) := selectCol(mat, i).toDenseVector
      ind += 1
    }
    out
  }

  def updateAllCols(mat: Tensor[Float], vec: Tensor[Float]) = {
    (1 to mat.size(2)).foreach(cid => MatrixUtil.updateCol(mat, cid, vec))
  }

  def updateCol(tensor: Tensor[Float], cid: Int, value: Tensor[Float]): Unit = {
    require(tensor.size(1) == value.nElement())
    (1 to tensor.size(1)).foreach(rid => tensor.setValue(rid, cid, value.valueAt(rid)))
  }

  def selectCols(mat: Tensor[Float], cid: Int, step: Int): Tensor[Float] = {
    val out = Tensor[Float](mat.size(1), mat.size(2) / step)
    var ind = 1
    for (i <- cid to mat.size(2) by step) {
      updateCol(out, ind, selectCol(mat, i))
      ind += 1
    }
    out
  }

  def selectRow(mat: DenseMatrix[Float], rid: Int): Array[Float] = {
    selectMatrix(mat, Array(rid), 0).toArray
  }

  def selectRow(mat: Tensor[Float], rid: Int, newSpace: Boolean = false): Tensor[Float] = {
    if (newSpace) mat.apply(rid).clone()
    else mat.apply(rid)
  }

  def printMatrix(info: String, data: DenseMatrix[Float]): Unit = {
    println(s"=========================$info======================")
    println(data)
  }

  def printSelectMatrix(info: String, data: DenseMatrix[Float]): Unit = {
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

  def printSelectedVector(info: String, data: DenseVector[Float]): Unit = {
    println(s"=========================$info======================")
    if (data.length <= 10) println(data)
    else {
      for (i <- 0 until 5) {
        print(data.valueAt(i) + " ")
      }
    }
    println("......")
    for (i <- 1 to 5) {
      print(data.valueAt(data.length - 6 + i) + " ")
    }
    println("... length: ", data.length)
  }

  def printArrayFloat(info: String, data: Array[Float]): Unit = {
    println(s"=========================$info======================")
    if (data.length <= 100) println(data.mkString(" "))
    else {
      for (i <- 0 until 5) {
        print(data(i) + " ")
      }
      print("...")
      for (i <- 1 to 5) {
        print(data(data.length - 6 + i) + " ")
      }
      println("... length: ", data.length)
    }
  }

  def printArrayInt(info: String, data: Array[Int]): Unit = {
    println(s"=========================$info======================")
    if (data.length <= 100) println(data.mkString(" "))
    else {
      for (i <- 0 until 5) {
        print(data(i) + " ")
      }
      print("...")
      for (i <- 1 to 5) {
        print(data(data.length - 6 + i) + " ")
      }
      println("... length: ", data.length)
    }
  }
}
