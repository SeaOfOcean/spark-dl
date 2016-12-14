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

  def horzConcat[T: ClassTag](tensors: Tensor[T]*)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(tensors(0).dim() == 2, "currently only support 2D")
    val nRows = tensors(0).size(1)
    var nCols = tensors(0).size(2)
    for (i <- 1 until tensors.length) {
      require(tensors(i).size(1) == nRows, "the rows length must be equal")
      nCols += tensors(i).size(2)
    }
    val resData = Tensor[T](nRows, nCols)
    var id = 1
    tensors.foreach { tensor =>
      (1 to tensor.size(2)).foreach { cid =>
        (1 to tensor.size(1)).foreach(rid =>
          resData.setValue(rid, id, tensor.valueAt(rid, cid)))
      }
      id = id + 1
    }
    resData
  }

  def vertConcat[T: ClassTag](tensors: Tensor[T]*)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(tensors(0).dim() == 2, "currently only support 2D")
    var nRows = tensors(0).size(1)
    val nCols = tensors(0).size(2)
    for (i <- 1 until tensors.length) {
      require(tensors(i).size(2) == nCols, "the cols length must be equal")
      nRows += tensors(i).size(1)
    }
    val resData = Tensor[T](nRows, nCols)
    var id = 1
    tensors.foreach { tensor =>
      (1 to tensor.size(1)).foreach { rid =>
        (1 to tensor.size(2)).foreach(cid =>
          resData.setValue(rid, id, tensor.valueAt(rid, cid)))
      }
      id = id + 1
    }
    resData
  }
}
