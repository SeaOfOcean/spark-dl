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
import com.intel.analytics.bigdl.pvanet.layers.{ParallelCriterion, ReshapeInfer, RoiPooling}
import com.intel.analytics.bigdl.pvanet.model.Model._
import com.intel.analytics.bigdl.pvanet.model.Phase._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag


abstract class FasterRcnn[T: ClassTag](var phase: PhaseType)
  (implicit ev: TensorNumeric[T]) {
  val modelType: ModelType
  val param: FasterRcnnParam
  var caffeReader: CaffeReader[T] = _

  def modelName: String = modelType.toString

  def setCaffeReader(caffeReader: CaffeReader[T]): Unit = {
    this.caffeReader = caffeReader
  }

  def train: Unit = setPhase(TRAIN)

  def evaluate: Unit = setPhase(TEST)

  def isTrain: Boolean = phase == TRAIN

  def isTest: Boolean = phase == TEST

  private def setPhase(phase: PhaseType): Unit = this.phase = phase

  def copyParamToTest(model: Module[_, _, T]) = {
    val name2model = Utils.getParamModules[T](model)
    evaluate
    val testModel = getTestModel()
    Utils.copyParamModules[T](testModel, name2model)
  }

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

  type Scale[T] = (CMul[T], CAdd[T])

  def scale(size: Array[Int], name: String): Scale[T] = {
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

  var testModel: Option[STT] = None
  var trainModel: Option[STT] = None

  def getTestModel(): MTT = {
    testModel match {
      case None => testModel = Some(createTestModel())
      case _ =>
    }
    testModel.get
  }

  def getTrainModel(): MTT = {
    trainModel match {
      case None => trainModel = Some(createTrainModel())
      case _ =>
    }
    trainModel.get
  }

  def criterion4: ParallelCriterion[T]

  def featureAndRpnNet(): StT

  // pool is the parameter of RoiPooling
  val pool: Int

  def fastRcnn(): STT = {
    val model = new STT()
      .add(new RoiPooling[T](pool, pool, ev.fromType(0.0625f)))
      .add(new ReshapeInfer[T](Array(-1, 512 * pool * pool)))
      .add(linear((512 * pool * pool, 4096), "fc6"))
      .add(new ReLU[T]())
      .add(linear((4096, 4096), "fc7"))
      .add(new ReLU[T]())

    val cls = new Stt().add(linear((4096, 21), "cls_score"))
    if (isTest) cls.add(new SoftMax[T]())
    val clsReg = new CT()
      .add(cls)
      .add(linear((4096, 84), "bbox_pred"))

    model.add(clsReg)
    model
  }

  def rpn(): StT

  protected def createTestModel(): STT

  protected def createTrainModel(): STT

  type STT = Sequential[Table, Table, T]
  type STt = Sequential[Table, Tensor[T], T]
  type StT = Sequential[Tensor[T], Table, T]
  type Stt = Sequential[Tensor[T], Tensor[T], T]
  type CT = ConcatTable[Table, T]
  type Ct = ConcatTable[Tensor[T], T]
  type PC = ParallelCriterion[T]
  type MTT = Module[Table, Table, T]

  /**
   * select tensor from nested tables
   *
   * @param depths a serious of depth to use when fetching certain tensor
   * @return a wanted tensor
   */
  def selectTensor(depths: Int*): STt = {
    val module = new STt()
    depths.slice(0, depths.length - 1).foreach(depth =>
      module.add(new SelectTable[Table, T](depth)))
    module.add(new SelectTable[Tensor[T], T](depths(depths.length - 1)))
  }

  def selectTensor1(depth: Int): SelectTable[Tensor[T], T] = {
    new SelectTable[Tensor[T], T](depth)
  }

  def selectTable(depths: Int*): STT = {
    val module = new STT()
    depths.slice(0, depths.length).foreach(depth =>
      module.add(new SelectTable[Table, T](depth)))
    module
  }

  def selectTable1(depth: Int): SelectTable[Table, T] = {
    new SelectTable[Table, T](depth)
  }
}
