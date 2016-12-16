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
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.pvanet.caffe.CaffeReader
import com.intel.analytics.bigdl.pvanet.model.Model._
import com.intel.analytics.bigdl.pvanet.model.Phase._
import com.intel.analytics.bigdl.pvanet.utils.FileUtil
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Table, File => DlFile}

import scala.util.Random


abstract class FasterRcnn(var phase: PhaseType) {

  val modelType: ModelType
  val param: FasterRcnnParam
  var caffeReader: CaffeReader[Float] = _

  def modelName: String = modelType.toString

  def setCaffeReader(caffeReader: CaffeReader[Float]): Unit = {
    this.caffeReader = caffeReader
  }

  def train(): Unit = setPhase(TRAIN)

  def evaluate(): Unit = setPhase(TEST)

  def isTrain: Boolean = phase == TRAIN

  def isTest: Boolean = phase == TEST

  private def setPhase(phase: PhaseType): Unit = this.phase = phase

  /**
   *
   * @param p    parameter: (nIn: Int, nOut: Int, ker: Int, stride: Int, pad: Int)
   * @param name name of layer
   * @return
   */
  def conv(p: (Int, Int, Int, Int, Int),
    name: String, isBack: Boolean = true,
    initMethod: InitializationMethod = Xavier,
    init: (Double, Double) = null): SpatialConvolution[Float] = {
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
    val module: SpatialConvolution[Float] = new SpatialConvolution(p._1, p._2, p._3, p._3, p._4, p._4,
      p._5, p._5, propagateBack = isBack, initMethod = initMethod).setName(name)
    if (init != null) initParameters(module, init)
    module
  }

  type Scale = (CMul[Float], CAdd[Float])

  def scale(size: Array[Int], name: String): Scale = {
    if (caffeReader != null) {
      val out = caffeReader.mapScale(name)
      (size zip out._1.size).foreach(x => assert(x._1 == x._2))
      out
    } else {
      (new CMul(size), new CAdd(size))
    }
  }

  /**
   *
   * @param p    (nIn, nOut)
   * @param name name of layer
   * @return
   */
  def linear(p: (Int, Int), name: String, init: (Double, Double) = null): Linear[Float] = {
    if (caffeReader != null) {
      val out = caffeReader.mapInnerProduct(name)
      if (out != null) {
        assert(out.weight.size(1) == p._2 && out.weight.size(2) == p._1)
        return out
      }
    }
    val module = new Linear(p._1, p._2).setName(name)
    if (init != null) initParameters(module, init)
    module
  }

  def spatialFullConv(p: (Int, Int, Int, Int, Boolean), name: String)
  : SpatialFullConvolutionMap[Float] = {
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
    new SpatialFullConvolutionMap(SpatialConvolutionMap.oneToOne[Float](p._1),
      p._2, p._2, p._3, p._3, p._4, p._4, p._5).setName(name)
  }

  private var testModel: Option[Sequential[Float]] = None
  private var trainModel: Option[Sequential[Float]] = None

  def getTestModel: Module[Float] = {
    testModel match {
      case None => testModel = Some(createTestModel())
      case _ =>
    }
    testModel.get
  }

  def getTrainModel: Module[Float] = {
    trainModel match {
      case None => trainModel = Some(createTrainModel())
      case _ =>
    }
    trainModel.get
  }

  def criterion4: ParallelCriterion[Float]

  protected def createFeatureAndRpnNet(): Sequential[Float]

  // pool is the parameter of RoiPooling
  val pool: Int
  private[this] var _featureAndRpnNet: Sequential[Float] = _

  def featureAndRpnNet: Sequential[Float] = {
    if (_featureAndRpnNet == null) {
      _featureAndRpnNet = createFeatureAndRpnNet()
    }
    _featureAndRpnNet
  }

  def setFeatureAndRpnNet(value: Sequential[Float]): Unit = {
    _featureAndRpnNet = value
  }

  private[this] var _fastRcnn: Sequential[Float] = _

  def fastRcnn: Sequential[Float] = {
    if (_fastRcnn == null) {
      _fastRcnn = createFastRcnn()
    }
    _fastRcnn
  }

  def setFastRcnn(value: Sequential[Float]): Unit = {
    _fastRcnn = value
  }

  protected def createFastRcnn(): Sequential[Float]

  def createRpn(): Sequential[Float]

  protected def createTestModel(): Sequential[Float]

  protected def createTrainModel(): Sequential[Float]

  def getModel: Module[Float] = {
    if (isTest) getTestModel
    else getTrainModel
  }

  def selectTensorNoBack(depths: Int*): Sequential[Float] = {
    val module = new Sequential[Float]().setPropagateBack(false)
    depths.slice(0, depths.length - 1).foreach(depth =>
      module.add(new SelectTable[Table, Float](depth).setPropagateBack(false)))
    module.add(new SelectTable[Tensor[Float], Float](depths(depths.length - 1)).setPropagateBack(false))
  }

  /**
   * select tensor from nested tables
   *
   * @param depths a serious of depth to use when fetching certain tensor
   * @return a wanted tensor
   */
  def selectTensor(depths: Int*): Sequential[Float] = {
    val module = new Sequential[Float]()
    depths.slice(0, depths.length - 1).foreach(depth =>
      module.add(new SelectTable[Table, Float](depth)))
    module.add(new SelectTable[Tensor[Float], Float](depths(depths.length - 1)))
  }

  def selectTensor1(depth: Int): SelectTable[Tensor[Float], Float] = {
    new SelectTable[Tensor[Float], Float](depth)
  }

  def selectTensor1NoBack(depth: Int): SelectTable[Tensor[Float], Float] = {
    new SelectTable[Tensor[Float], Float](depth).setPropagateBack(false)
  }

  def selectTable(depths: Int*): Sequential[Float] = {
    val module = new Sequential[Float]()
    depths.slice(0, depths.length).foreach(depth =>
      module.add(new SelectTable[Table, Float](depth)))
    module
  }

  def selectTableNoBack(depths: Int*): Sequential[Float] = {
    val module = new Sequential[Float]().setPropagateBack(false)
    depths.slice(0, depths.length).foreach(depth =>
      module.add(new SelectTable[Table, Float](depth).setPropagateBack(false)))
    module
  }

  def selectTable1(depth: Int): SelectTable[Table, Float] = {
    new SelectTable[Table, Float](depth)
  }

  def selectTable1NoBack(depth: Int): SelectTable[Table, Float] = {
    new SelectTable[Table, Float](depth).setPropagateBack(false)
  }

  def initParameters(module: Module[Float], init: (Double, Double)): Unit = {
    val params = module.getParameters()
    params._1.apply1(_ => RNG.normal(0, init._1).toFloat)
    params._2.apply1(_ => init._2.toFloat)
  }

  def loadFromCaffeOrCache(dp: String, mp: String): this.type = {
    val cachedPath = mp.substring(0, mp.lastIndexOf(".")) + ".bigdl"
    val mod = FileUtil.load[(Sequential[Float], Sequential[Float])](cachedPath)
    mod match {
      case Some((featureAndRpn, fastRcnn)) =>
        println(s"load model with caffe weight from cache $cachedPath")
        setFeatureAndRpnNet(featureAndRpn)
        setFastRcnn(fastRcnn)
      case _ =>
        Module.loadCaffeParameters[Float](getModel, dp, mp, phase == TEST)
        DlFile.save((featureAndRpnNet, fastRcnn), cachedPath, true)
    }
    this
  }

  def copyFromCaffe(caffeReader: CaffeReader[Float]): this.type = {
    val mod = caffeReader.loadModuleFromFile[(Sequential[Float], Sequential[Float])](modelName)
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

}

object FasterRcnn {
  def apply(modelType: ModelType, phase: PhaseType = TEST, pretrained: Any = None): FasterRcnn = {

    def getFasterRcnn(modelType: ModelType): FasterRcnn = {
      modelType match {
        case VGG16 =>
          new VggFRcnn(phase)
        case PVANET =>
          new PvanetFRcnn(phase)
        case _ =>
          throw new Exception("unsupport network")
      }
    }

    val fasterRcnnModel = pretrained match {
      case mp: String =>
        // big dl faster rcnn models
        DlFile.load[FasterRcnn](mp)
      case (dp: String, mp: String) =>
        // caffe pretrained model
        getFasterRcnn(modelType)
//          .copyFromCaffe(new CaffeReader[Float](dp, mp))
          .loadFromCaffeOrCache(dp, mp)
      case _ => getFasterRcnn(modelType)
    }
    Random.setSeed(fasterRcnnModel.param.RANDOM_SEED)
    fasterRcnnModel
  }
}
