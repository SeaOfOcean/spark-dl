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


  def vertConcat[T: ClassTag](tensors: Tensor[T]*)(implicit ev: TensorNumeric[T]): Tensor[T] = {
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
    tensors.foreach(tensor =>
      (1 to getRowCol(tensor)._1).foreach(rid => {
        id = id + 1
        resData.update(id, tensor)
      }))
    resData
  }
}
