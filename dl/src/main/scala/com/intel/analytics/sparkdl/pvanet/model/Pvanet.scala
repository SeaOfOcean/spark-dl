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

import scala.reflect.ClassTag

object Pvanet {
  def getModel[T: ClassTag](implicit ev: TensorNumeric[T]): Module[Tensor[T], Tensor[T], T] = {
    val pvanet = new Sequential[Tensor[T], Tensor[T], T]()

    def convBNReLU(nInputPlane: Int,
      nOutPutPlane: Int,
      kernelW: Int,
      kernelH: Int,
      strideW: Int,
      strideH: Int,
      padW: Int,
      padH: Int):
    Sequential[Tensor[T], Tensor[T], T] = {
      pvanet.add(new SpatialConvolution[T](nInputPlane, nOutPutPlane,
        kernelW, kernelH, strideW, strideH, padW, padH))
      pvanet.add(new SpatialBatchNormalization[T](nOutPutPlane))
      pvanet.add(new ReLU[T](true))
      pvanet
    }

    pvanet.add(new SpatialConvolution[T](16, 16, 7, 7, 2, 2, 3, 3, initMethod = Xavier))
//    pvanet.add(new BatchNormalization[T]())
  }

}
