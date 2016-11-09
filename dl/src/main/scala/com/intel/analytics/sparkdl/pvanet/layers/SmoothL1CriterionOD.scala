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

package com.intel.analytics.sparkdl.pvanet.layers

import com.intel.analytics.sparkdl.nn.TensorCriterion
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class SmoothL1CriterionOD[T: ClassTag](@transient val sigma: T, @transient val num: Int)
                                      (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
  @transient var gradInput: Tensor[T] = null
  @transient var buffer: Tensor[T] = null
  // diff holds (input - gt) * w_in
  @transient var diff: Tensor[T] = null
  @transient val sigma2 = ev.times(sigma, sigma)
  @transient var hasWeights = true

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    // the target are composed of gt, inside_weight, outside_weight
    if (input.nElement() * 3 != target.nElement()) {
      hasWeights = false
      require(input.nElement() == target.nElement())
      target.resize(1, target.nElement())
    } else {
      hasWeights = true
      target.resize(3, target.nElement() / 3)
    }

    if (diff == null) {
      diff = Tensor[T]()
    }
    diff.resizeAs(input).copy(input)
    // input - gt
    diff.add(ev.fromType(-1), target.apply(1))
    if (hasWeights) {
      // apply "inside" weights, (input - gt) * w_in
      diff.cmul(target.apply(2))
    }


    if (buffer == null) {
      buffer = Tensor[T]
    }
    // |input - gt| * w_in
    buffer.resizeAs(diff).copy(diff).abs()
    var data = buffer.storage().array()
    for (i <- 0 until data.length) {
      // f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
      //        |x| - 0.5 / sigma / sigma    otherwise
      if (ev.isGreater(ev.divide(ev.fromType(1.0), sigma2), data(i))) {
        data(i) = ev.times(sigma2, ev.times(ev.fromType(0.5), ev.times(data(i), data(i))))
      }
      else {
        data(i) = ev.minus(data(i), ev.divide(ev.fromType(0.5), sigma2))
      }
    }
    if (hasWeights) {
      // apply "outside" weights,  w_out * SmoothL1(|input - gt| * w_in)
      buffer.cmul(target.apply(3))
    }
    target.resize(target.nElement())
    ev.divide(buffer.sum(), ev.fromType(num))
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    if (input.nElement() * 3 != target.nElement()) {
      hasWeights = false
      require(input.nElement() == target.nElement())
    } else {
      hasWeights = true
      target.resize(3, input.nElement())
    }
    if (gradInput == null) {
      gradInput = Tensor[T]()
    }
    var data = diff.storage().array()
    for (i <- 0 until data.length) {
      // f'(x) = sigma * sigma * x         if |x| < 1 / sigma / sigma
      //       = sign(x)
      val x = data(i)
      if (ev.isGreater(ev.divide(ev.fromType(1.0), sigma2), ev.abs(x))) {
        data(i) = ev.times(sigma2, x)
      } else {
        // sign(x) == (0<x) - (x<0)
        if (ev.isGreater(data(i), ev.fromType(0))) {
          data(i) = ev.fromType(1)
        } else if (ev.isGreater(ev.fromType(0), data(i))) {
          data(i) = ev.fromType(-1)
        } else {
          data(i) = ev.fromType(0)
        }
      }
    }
    var sign = ev.fromType(1)
    var alpha = ev.divide(sign, ev.fromType(num))
    gradInput.resizeAs(diff).copy(diff).mul(alpha)
    if (hasWeights) {
      //scale by inside weight
      gradInput.cmul(target.apply(2))
      //scale by outside weight
      gradInput.cmul(target.apply(3))
    }
    target.resize(target.nElement())
    gradInput
  }
}