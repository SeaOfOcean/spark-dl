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

package com.intel.analytics.bigdl.pvanet.layers

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class LogSoftMax[@specialized(Float, Double) T: ClassTag]
(implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  //  @transient
  //  private var results: Array[Future[Unit]] = null

  @transient
  private var startIndex: Int = 0

  @transient
  private var sum = ev.fromType(0)

  @transient
  private var maxInput = ev.fromType(0)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() <= 4, "1D, 2D, 3D or 4D tensor expected")
    require(input.isContiguous())
    require(output.isContiguous())

    val (nframe, dim, stride) =
      if (input.nDimension() == 1) {
        (1, input.size(1), 1)
      } else if (input.nDimension == 2) {
        (input.size(1), input.size(2), 1)
      } else if (input.nDimension() == 3) {
        (1, input.size(1), input.size(2) * input.size(3))
      } else {
        // (input.nDimension() == 4) {
        (input.size(1), input.size(2), input.size(3) * input.size(4))
      }
    output.resizeAs(input)
    var t = 1
    val inputData = input.storage().array()
    var outputData = output.storage().array()
    for (t <- 0 until stride * nframe) {
      sum = ev.fromType[Int](0)
      maxInput = input.max()
      startIndex = (t / stride) * dim * stride + t % stride
      for (d <- 0 until dim) {
        maxInput = ev.max(maxInput, inputData(startIndex + d * stride))
      }
      for (d <- 0 until dim) {
        sum = ev.plus(sum, ev.exp(ev.minus(inputData(startIndex + d * stride), maxInput)))
      }
      sum = ev.plus(maxInput, ev.log(sum))
      for (d <- 0 until dim) {
        outputData(startIndex + d * stride) = ev.minus(inputData(startIndex + d * stride), sum)
      }
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() <= 4, "1D, 2D, 3D or 4D tensor expected")
    require(input.nElement() == gradOutput.nElement())

    val (nframe, dim, stride) =
      if (output.nDimension() == 1) {
        (1, output.size(1), 1)
      }
      else if (output.nDimension() == 2) {
        (output.size(1), output.size(2), 1)
      }
      else if (output.nDimension() == 3) {
        (1, output.size(1), output.size(2) * output.size(3))
      }
      else {
        // if (output -> nDimension == 4)
        (output.size(1), output.size(2), output.size(3) * output.size(4))
      }

    gradInput.resizeAs(output)
    var gradInputData = gradInput.storage().array()
    val outputData = output.storage().array()
    val gradOutputData = gradOutput.storage().array()

    for (t <- 0 until stride * nframe) {
      sum = ev.fromType(0)
      startIndex = (t / stride) * dim * stride + t % stride

      for (d <- 0 until dim) {
        sum = ev.plus(sum, gradOutputData(startIndex + d * stride))
      }
      for (d <- 0 until dim) {
        gradInputData(startIndex + d * stride) =
          ev.minus(
            gradOutputData(startIndex + d * stride),
            ev.times(
              ev.exp(outputData(startIndex + d * stride)),
              sum))
      }
    }

    gradInput
  }

  override def toString(): String = {
    s"nn.LogSoftMax"
  }
}

