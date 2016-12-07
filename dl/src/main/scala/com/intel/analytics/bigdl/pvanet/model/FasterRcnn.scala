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

package com.intel.analytics.bigdl.pvanet.model

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.pvanet.caffe.CaffeReader
import com.intel.analytics.bigdl.pvanet.model.Model._
import com.intel.analytics.bigdl.pvanet.model.Phase._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag


abstract class FasterRcnn[T: ClassTag](var phase: Phase)
  (implicit ev: TensorNumeric[T]) {

  val modelType: Model
  val param: FasterRcnnParam
  var caffeReader: CaffeReader[T] = null

  def modelName: String = modelType.toString

  def setCaffeReader(caffeReader: CaffeReader[T]): Unit = {
    this.caffeReader = caffeReader
  }

  def setPhase(phase: Phase): Unit = this.phase = phase

  /**
   *
   * @param p    parameter: (nIn: Int, nOut: Int, ker: Int, stride: Int, pad: Int)
   * @param name name of layer
   * @return
   */
  def conv(p: (Int, Int, Int, Int, Int), name: String): SpatialConvolution[T] = {
    if (caffeReader != null) {
      val out = caffeReader.mapConvolution(name)
      assert(out.nInputPlane == p._1)
      assert(out.nOutputPlane == p._2)
      assert(out.kernelH == p._3)
      assert(out.strideH == p._4)
      assert(out.padH == p._5)
      out
    } else {
      new SpatialConvolution[T](p._1, p._2, p._3, p._3, p._4, p._4,
        p._5, p._5, initMethod = Xavier).setName(name)
    }
  }

  def scale(size: Array[Int], name: String): (CMul[T], CAdd[T]) = {
    if (caffeReader != null) {
      val out = caffeReader.mapScale(name)
      (size zip out._1.size).foreach(x => assert(x._1 == x._2))
      out
    } else {
      (new CMul[T](size), new CAdd[T](size))
    }
  }

  /**
   *
   * @param p    (nIn, nOut)
   * @param name name of layer
   * @return
   */
  def linear(p: (Int, Int), name: String): Linear[T] = {
    if (caffeReader != null) {
      val out = caffeReader.mapInnerProduct(name)
      assert(out.weight.size(1) == p._2 && out.weight.size(2) == p._1)
      out
    } else {
      new Linear[T](p._1, p._2).setName(name)
    }
  }

  def spatialFullConv(p: (Int, Int, Int, Int, Boolean), name: String)
  : SpatialFullConvolutionMap[T] = {
    if (caffeReader != null) {
      val out = caffeReader.mapDeconvolution(name).setName(name)
      assert(out.connTable.size(1) == p._1)
      assert(out.kW == p._2)
      assert(out.dW == p._3)
      assert(out.padH == p._4)
      assert(out.noBias == !p._5)
      out
    } else {
      new SpatialFullConvolutionMap[T](SpatialConvolutionMap.oneToOne[T](p._1),
        p._2, p._2, p._3, p._3, p._4, p._4, p._5).setName(name)
    }
  }

  def criterion4: ParallelCriterion[T]

  def featureAndRpnNet(): Module[Tensor[T], Table, T]

  def fastRcnn(): Module[Table, Table, T]

  def rpn(): Module[Tensor[T], Table, T]

  def fullModel(): Module[Table, Table, T]
}
