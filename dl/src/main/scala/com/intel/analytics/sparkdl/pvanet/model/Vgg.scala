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

package com.intel.analytics.sparkdl.pvanet.model

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.Table

import scala.reflect.ClassTag
object Vgg_16 {
  def apply[T: ClassTag](classNum: Int)
                        (implicit ev: TensorNumeric[T]): Module[Tensor[T], Table, T] = {
    val vggNet = new Sequential[Tensor[T], Tensor[T], T]()
    vggNet.add(new SpatialConvolution[T](3, 64, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](64, 64, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    vggNet.add(new SpatialConvolution[T](64, 128, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](128, 128, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    vggNet.add(new SpatialConvolution[T](128, 256, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    vggNet.add(new SpatialConvolution[T](256, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    vggNet.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))

    //todo lr
    vggNet.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))

    val clsAndReg = new ConcatTable

    val cls = new SpatialConvolution[T](512, 18, 1, 1, 1, 1)
    val reg = new SpatialConvolution[T](512, 36, 1, 1, 1, 1)

    clsAndReg
      .add(cls)
      .add(reg)

    val rpnModel = new Sequential[Tensor[T], Table, T]()
    rpnModel.add(vggNet)
    rpnModel.add(clsAndReg)

    rpnModel
  }
}
object Vgg_16_RPN {
  def apply[T: ClassTag](classNum: Int)
                        (implicit ev: TensorNumeric[T]): Module[Tensor[T], Table, T] = {
    val vggNet = new Sequential[Tensor[T], Tensor[T], T]()
    vggNet.add(new SpatialConvolution[T](3, 64, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](64, 64, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    vggNet.add(new SpatialConvolution[T](64, 128, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](128, 128, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    vggNet.add(new SpatialConvolution[T](128, 256, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    vggNet.add(new SpatialConvolution[T](256, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    vggNet.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))

    //todo lr
    vggNet.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[T](true))

    val clsAndReg = new ConcatTable

    val cls = new SpatialConvolution[T](512, 18, 1, 1, 1, 1)
    val reg = new SpatialConvolution[T](512, 36, 1, 1, 1, 1)

    clsAndReg
      .add(cls)
      .add(reg)

    val rpnModel = new Sequential[Tensor[T], Table, T]()
    rpnModel.add(vggNet)
    rpnModel.add(clsAndReg)

    rpnModel
  }
}

