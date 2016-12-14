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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.pvanet.caffe.CaffeReader
import com.intel.analytics.bigdl.pvanet.model.Model._
import com.intel.analytics.bigdl.pvanet.model.Phase._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{File, Table}

import scala.reflect.ClassTag
import scala.util.Random


abstract class FasterRcnn[T: ClassTag](var phase: PhaseType)
  (implicit ev: TensorNumeric[T]) {
  val modelType: ModelType
  val param: FasterRcnnParam
  var caffeReader: CaffeReader[T] = _

  def modelName: String = modelType.toString

  def setCaffeReader(caffeReader: CaffeReader[T]): Unit = {
    this.caffeReader = caffeReader
  }

  def train(): Unit = setPhase(TRAIN)

  def evaluate(): Unit = setPhase(TEST)

  def isTrain: Boolean = phase == TRAIN

  def isTest: Boolean = phase == TEST

  private def setPhase(phase: PhaseType): Unit = this.phase = phase

  def copyParamToTest(model: Module[T]) = {
    val name2model = Utils.getParamModules[T](model)
    evaluate
    Utils.copyParamModules[T](getTestModel, name2model)
  }

  /**
   *
   * @param p    parameter: (nIn: Int, nOut: Int, ker: Int, stride: Int, pad: Int)
   * @param name name of layer
   * @return
   */
  def conv(p: (Int, Int, Int, Int, Int),
    name: String, isBack: Boolean = true,
    initMethod: InitializationMethod = Xavier,
    init: (Double, Double) = null): SpatialConvolution[T] = {
    if (caffeReader != null) {
      val out = caffeReader.mapConvolution(name, isBack)
      if (out != null) {
        assert(out.nInputPlane == p._1)
        assert(out.nOutputPlane == p._2)
        assert(out.kernelH == p._3)
        assert(out.strideH == p._4)
        assert(out.padH == p._5)
        return out
      }
    }
    val module = new SpatialConvolution[T](p._1, p._2, p._3, p._3, p._4, p._4,
      p._5, p._5, propagateBack = isBack, initMethod = initMethod).setName(name)
    if (init != null) initParameters(module, init)
    module
  }

  type Scale = (CMul[T], CAdd[T])

  def scale(size: Array[Int], name: String): Scale = {
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
  def linear(p: (Int, Int), name: String, init: (Double, Double) = null): Linear[T] = {
    if (caffeReader != null) {
      val out = caffeReader.mapInnerProduct(name)
      if (out != null) {
        assert(out.weight.size(1) == p._2 && out.weight.size(2) == p._1)
        return out
      }
    }
    val module = new Linear[T](p._1, p._2).setName(name)
    if (init != null) initParameters(module, init)
    module
  }

  def spatialFullConv(p: (Int, Int, Int, Int, Boolean), name: String)
  : SpatialFullConvolutionMap[T] = {
    if (caffeReader != null) {
      val out = caffeReader.mapDeconvolution(name).setName(name)
      if (out != null) {
        assert(out.connTable.size(1) == p._1)
        assert(out.kW == p._2)
        assert(out.dW == p._3)
        assert(out.padH == p._4)
        assert(out.noBias == !p._5)
        return out
      }
    }
    new SpatialFullConvolutionMap[T](SpatialConvolutionMap.oneToOne[T](p._1),
      p._2, p._2, p._3, p._3, p._4, p._4, p._5).setName(name)
  }

  private var testModel: Option[Sequential[T]] = None
  private var trainModel: Option[Sequential[T]] = None

  def getTestModel: Module[T] = {
    testModel match {
      case None => testModel = Some(createTestModel())
      case _ =>
    }
    testModel.get
  }

  def getTrainModel: Module[T] = {
    trainModel match {
      case None => trainModel = Some(createTrainModel())
      case _ =>
    }
    trainModel.get
  }

  def criterion4: ParallelCriterion[T]

  protected def createFeatureAndRpnNet(): Sequential[T]

  // pool is the parameter of RoiPooling
  val pool: Int
  private[this] var _featureAndRpnNet: Sequential[T] = _

  def featureAndRpnNet: Sequential[T] = {
    if (_featureAndRpnNet == null) {
      _featureAndRpnNet = createFeatureAndRpnNet()
    }
    _featureAndRpnNet
  }

  def setFeatureAndRpnNet(value: Sequential[T]): Unit = {
    _featureAndRpnNet = value
  }

  private[this] var _fastRcnn: Sequential[T] = _

  def fastRcnn: Sequential[T] = {
    if (_fastRcnn == null) {
      _fastRcnn = createFastRcnn()
    }
    _fastRcnn
  }

  def setFastRcnn(value: Sequential[T]): Unit = {
    _fastRcnn = value
  }

  protected def createFastRcnn(): Sequential[T]


  def copyFromCaffe(caffeReader: CaffeReader[T]): FasterRcnn[T] = {
    val mod = caffeReader.loadModuleFromFile[(Sequential[T], Sequential[T])](modelName)
    mod match {
      case Some((featureAndRpn, fastRcnn)) =>
        setFeatureAndRpnNet(featureAndRpn)
        setFastRcnn(fastRcnn)
      case _ =>
        setCaffeReader(caffeReader)
        getModel
        caffeReader.writeToFile((featureAndRpnNet, fastRcnn), modelName)
    }
    this
  }

  def createRpn(): Sequential[T]

  protected def createTestModel(): Sequential[T]

  protected def createTrainModel(): Sequential[T]

  def getModel: Module[T] = {
    if (isTest) getTestModel
    else getTrainModel
  }

  def selectTensorNoBack(depths: Int*): Sequential[T] = {
    val module = new Sequential[T]().setPropagateBack(false)
    depths.slice(0, depths.length - 1).foreach(depth =>
      module.add(new SelectTable[Table, T](depth).setPropagateBack(false)))
    module.add(new SelectTable[Tensor[T], T](depths(depths.length - 1)).setPropagateBack(false))
  }

  /**
   * select tensor from nested tables
   *
   * @param depths a serious of depth to use when fetching certain tensor
   * @return a wanted tensor
   */
  def selectTensor(depths: Int*): Sequential[T] = {
    val module = new Sequential[T]()
    depths.slice(0, depths.length - 1).foreach(depth =>
      module.add(new SelectTable[Table, T](depth)))
    module.add(new SelectTable[Tensor[T], T](depths(depths.length - 1)))
  }

  def selectTensor1(depth: Int): SelectTable[Tensor[T], T] = {
    new SelectTable[Tensor[T], T](depth)
  }

  def selectTensor1NoBack(depth: Int): SelectTable[Tensor[T], T] = {
    new SelectTable[Tensor[T], T](depth).setPropagateBack(false)
  }

  def selectTable(depths: Int*): Sequential[T] = {
    val module = new Sequential[T]()
    depths.slice(0, depths.length).foreach(depth =>
      module.add(new SelectTable[Table, T](depth)))
    module
  }

  def selectTableNoBack(depths: Int*): Sequential[T] = {
    val module = new Sequential[T]().setPropagateBack(false)
    depths.slice(0, depths.length).foreach(depth =>
      module.add(new SelectTable[Table, T](depth).setPropagateBack(false)))
    module
  }

  def selectTable1(depth: Int): SelectTable[Table, T] = {
    new SelectTable[Table, T](depth)
  }

  def selectTable1NoBack(depth: Int): SelectTable[Table, T] = {
    new SelectTable[Table, T](depth).setPropagateBack(false)
  }

  def initParameters(module: Module[T], init: (Double, Double)): Unit = {
    val params = module.getParameters()
    val rand = new Random()
    params._1.apply1(_ => ev.fromType[Double](RNG.normal(0, init._1)))
    params._2.apply1(_ => ev.fromType[Double](init._2))
  }
}

object FasterRcnn {
  def apply[@specialized(Float, Double) T: ClassTag]
  (modelType: ModelType, phase: PhaseType = TEST, pretrained: Any = None)
    (implicit ev: TensorNumeric[T]): FasterRcnn[T] = {

    def getFasterRcnn(modelType: ModelType): FasterRcnn[T] = {
      modelType match {
        case VGG16 =>
          new VggFRcnn[T](phase)
        case PVANET =>
          new PvanetFRcnn[T](phase)
        case _ =>
          throw new Exception("unsupport network")
      }
    }

    pretrained match {
      case mp: String =>
        // big dl faster rcnn models
        File.load[FasterRcnn[T]](mp)
      case (dp: String, mp: String) =>
        // caffe pretrained model
        val fm = getFasterRcnn(modelType).copyFromCaffe(new CaffeReader[T](dp, mp))
        fm
      case _ => getFasterRcnn(modelType)
    }
  }
}
