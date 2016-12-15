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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.reflect.ClassTag

class SpatialFullConvolutionMap[T: ClassTag](
  val connTable: Tensor[T],
  val kW: Int,
  val kH: Int,
  val dW: Int = 1,
  val dH: Int = 1,
  val padW: Int = 0,
  val padH: Int = 0,
  val noBias: Boolean = false,
  private var initMethod: InitializationMethod = Default
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  val nInputPlane = ev.toType[Int](connTable.select(2, 1).max())
  val nOutputPlane = ev.toType[Int](connTable.select(2, 2).max())

  val weight: Tensor[T] = Tensor[T](nInputPlane, kH, kW)
  val bias: Tensor[T] = if (noBias) null else Tensor[T](nOutputPlane)

  val gradWeight: Tensor[T] = Tensor[T](nInputPlane, kH, kW)
  val gradBias: Tensor[T] = if (noBias) null else Tensor[T](nOutputPlane)

  reset()

  def setInitMethod(initMethod: InitializationMethod): this.type = {
    this.initMethod = initMethod
    this
  }

  override def reset(): Unit = {
    initMethod match {
      case Default =>
        val stdv = 1.0 / math.sqrt(kW * kH * nInputPlane)
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
        if (!noBias) {
          bias.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
        }
      case Xavier =>
        val fanIn = nInputPlane * kH * kW
        val fanOut = nOutputPlane * kH * kW
        val stdv = math.sqrt(6.0 / (fanIn + fanOut))
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
        if (null != bias) {
          bias.fill(ev.fromType(0))
        }
      case BilinearFiller =>
        require(weight.nDimension() == 4, "weight must be 4 dim")
        require(kH == kW, "Kernel must be square")
        val f = Math.ceil(kW / 2.0).toInt
        val c = (2 * f - 1 - f % 2) / (2.0f * f)
        val weightArray = weight.storage().array()
        val weightOffset = weight.storageOffset() - 1
        var i = 0
        while (i < weight.nElement()) {
          val x: Float = i % kW
          val y: Float = (i / kW) % kH
          weightArray(i + weightOffset) = ev.fromType[Float](
            (1f - math.abs(x / f - c)) * (1f - math.abs(y / f - c)))
          i += 1
        }
    }
    zeroGradParameters()
  }


  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val batch = if (input.nDimension() == 4 && input.size(1) == 1) {
      input.resize(input.size().slice(1, 4))
      true
    } else {
      false
    }
    require(input.nDimension() == 3, "3D input tensor expected")
    require(input.size(1) >= nInputPlane, "invalid number of input planes")

    output.resize(nOutputPlane, (input.size(2) - 1) * dH + kH - 2 * padH,
      (input.size(3) - 1) * dW + kW - 2 * padW)

    require(input.isContiguous())

    val inputH = input.size(2)
    val inputW = input.size(3)
    val outputH = output.size(2)
    val outputW = output.size(3)

    val connTableIndex = new Array[Int](2)
    val biasIndex = new Array[Int](1)

    var p = 1
    while (p <= nOutputPlane) {
      if (noBias) {
        output.zero()
      } else {
        biasIndex(0) = p
        output(p).fill(bias(biasIndex))
      }

      val nWeight = connTable.size(1)
      var k = 1
      while (k <= nWeight) {
        connTableIndex(0) = k
        connTableIndex(1) = 2
        val o = ev.toType[Int](connTable(connTableIndex))
        connTableIndex(1) = 1
        val i = ev.toType[Int](connTable(connTableIndex))

        if (o == p) {
          DenseTensorConv.fullConv2Dptr[T](
            output.storage(),
            output.storageOffset() - 1 + (o - 1) * outputH * outputW,
            ev.fromType[Int](1), input.storage(),
            input.storageOffset() - 1 + (i - 1) * inputW * inputH, inputH,
            inputW,
            weight.storage(), weight.storageOffset() - 1 + (k - 1) * kW * kH, kH, kW, dH, dW, padH, padW)
        }
        k += 1
      }

      p += 1
    }

    if (batch) {
      input.resize(Array(1) ++ input.size())
      output.resize(Array(1) ++ output.size())
    }

    println(getName() + "================" + output.size().mkString(","))
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {

    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
    scale: Double = 1.0): Unit = {

  }

  override def updateParameters(learningRate: T): Unit = {
    weight.map(gradWeight, (a, b) => ev.minus(a, ev.times(learningRate, b)))
    bias.map(gradBias, (a, b) => ev.minus(a, ev.times(learningRate, b)))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    if (!noBias) gradBias.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialFullConvolutionMap[T]]) {
      return false
    }
    val other = obj.asInstanceOf[SpatialFullConvolutionMap[T]]
    if (this.eq(other)) {
      return true
    }

    nInputPlane == other.nInputPlane &&
      nOutputPlane == other.nOutputPlane &&
      kW == other.kW &&
      kH == other.kH &&
      dW == other.dW &&
      dH == other.dH &&
      padW == other.padW &&
      padH == other.padH &&
      weight == other.weight &&
      bias == other.bias &&
      gradWeight == other.gradWeight &&
      gradBias == other.gradBias
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + nInputPlane.hashCode()
    hash = hash * seed + nOutputPlane.hashCode()
    hash = hash * seed + kW.hashCode()
    hash = hash * seed + kH.hashCode()
    hash = hash * seed + dW.hashCode()
    hash = hash * seed + dH.hashCode()
    hash = hash * seed + padW.hashCode()
    hash = hash * seed + padH.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()

    hash
  }

  override def toString(): String = {
    s"nn.SpatialFullConvolution($nInputPlane -> $nOutputPlane, " +
      s"$kW x $kH, $dW, $dH, $padW, $padH)"
  }
}
