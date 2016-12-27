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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object TensorUtil {

  /**
   * element add vec with each mat row
   *
   * @param mat
   * @param vec
   * @return
   */
  def addMatrixWithVector(mat: Tensor[Float], vec: Tensor[Float]): Tensor[Float] = {
    assert(mat.size(2) == vec.nElement(), "maxtrix cols should be same size as vec")
    val res = mat.clone()
    (1 to res.size(1)).foreach(r => res.update(r, vec))
    res
  }

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

  def selectMatrix2(mat: Tensor[Float],
    rows: Array[Int], cols: Array[Int]): Tensor[Float] = {
    val out = Tensor[Float](rows.length, cols.length)
    rows.zip(Stream from 1).map(r => {
      cols.zip(Stream from 1).map(c => {
        out.setValue(r._2, c._2, mat.valueAt(r._1, c._1))
      })
    })
    out
  }

  def selectMatrix2(mat: Tensor[Float],
    rows: Tensor[Float], cols: Array[Int]): Tensor[Float] = {
    selectMatrix2(mat, rows.storage().array().map(x => x.toInt), cols)
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

  def selectMatrix(matrix: Tensor[Float],
    selectInds: Tensor[Float], dim: Int): Tensor[Float] = {
    selectMatrix(matrix, selectInds.contiguous().storage().array().map(x => x.toInt), dim)
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
   * argmax2(arr, 1) will return 3, 1
   * argmax2(arr, 2) will return 2, 2, 1
   *
   * @param arr
   * @param d
   * @return
   * todo: this maybe removed
   */
  def argmax2(arr: Tensor[Float], d: Int): Array[Int] = {
    require(d >= 1)
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


  def updateAllCols(mat: Tensor[Float], vec: Tensor[Float]): Unit = {
    (1 to mat.size(2)).foreach(cid => updateCol(mat, cid, vec))
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


  def selectRow(mat: Tensor[Float], rid: Int, newSpace: Boolean = false): Tensor[Float] = {
    if (newSpace) mat.apply(rid).clone()
    else mat.apply(rid)
  }


  /**
   * update with 2d tensor, the range must be equal to the src tensor size
   *
   * @param startR
   * @param endR
   * @param startC
   * @param endC
   * @param dest
   * @param src
   */
  def updateRange(dest: Tensor[Float], startR: Int, endR: Int, startC: Int, endC: Int,
    src: Tensor[Float]): Unit = {
    assert(src.size(1) == endR - startR + 1)
    assert(src.size(2) == endC - startC + 1)
    (startR to endR).zip(Stream.from(1)).foreach(r => {
      (startC to endC).zip(Stream.from(1)).foreach(c => {
        dest.setValue(r._1, c._1, src.valueAt(r._2, c._2))
      })
    })
  }

  def concat[T: ClassTag](tensors: Tensor[T]*)
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(tensors(0).dim() == 1, "currently only support 1D")
    var length = tensors(0).size(1)
    for (i <- 1 until tensors.length) {
      require(tensors(i).dim() == 1, "currently only support 1D")
      length += tensors(i).nElement()
    }
    val resData = Tensor[T](length)
    var id = 1
    tensors.foreach { tensor =>
      (1 to tensor.nElement()).foreach { i =>
        resData.setValue(id, tensor.valueAt(i))
        id = id + 1
      }
    }
    resData
  }

  def horzcat(tensors: Tensor[Float]*): Tensor[Float] = {
    require(tensors(0).dim() == 2, "currently only support 2D")
    val nRows = tensors(0).size(1)
    var nCols = tensors(0).size(2)
    for (i <- 1 until tensors.length) {
      require(tensors(i).size(1) == nRows, "the rows length must be equal")
      nCols += tensors(i).size(2)
    }
    val resData = Tensor[Float](nRows, nCols)
    var id = 1
    tensors.foreach { tensor =>
      TensorUtil.updateRange(resData, 1, nRows, id, id + tensor.size(2) - 1, tensor)
      id = id + tensor.size(2)
    }
    resData
  }


  def vertcat[T: ClassTag](tensors: Tensor[T]*)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(tensors(0).dim() <= 2, "currently only support 1D or 2D")

    def getRowCol(tensor: Tensor[T]): (Int, Int) = {
      if (tensors(0).nDimension() == 2) {
        (tensor.size(1), tensor.size(2))
      } else {
        (1, tensor.size(1))
      }
    }

    var nRows = getRowCol(tensors(0))._1
    val nCols = getRowCol(tensors(0))._2
    for (i <- 1 until tensors.length) {
      require(getRowCol(tensors(i))._2 == nCols, "the cols length must be equal")
      nRows += getRowCol(tensors(i))._1
    }
    val resData = Tensor[T](nRows, nCols)
    var id = 0
    tensors.foreach { tensor =>
      if (tensor.nDimension() == 1) {
        id = id + 1
        resData.update(id, tensor)
      } else {
        (1 to getRowCol(tensor)._1).foreach(rid => {
          id = id + 1
          resData.update(id, tensor(rid))
        })
      }
    }
    resData
  }
}
