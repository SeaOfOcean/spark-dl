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
import com.intel.analytics.bigdl.pvanet.layers.{Proposal, ProposalTarget, ReshapeInfer}
import com.intel.analytics.bigdl.pvanet.model.Model._
import com.intel.analytics.bigdl.pvanet.model.Phase._
import com.intel.analytics.bigdl.pvanet.utils.FileUtil
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{File => DlFile}
import org.apache.log4j.Logger

import scala.util.Random


abstract class FasterRcnn(var phase: PhaseType) {

  val logger = Logger.getLogger(getClass)
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
    val module: SpatialConvolution[Float] =
      new SpatialConvolution(p._1, p._2, p._3, p._3, p._4, p._4,
        p._5, p._5, propagateBack = isBack, initMethod = initMethod).setName(name)
    if (phase == TRAIN && init != null) initParameters(module, init)
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
    if (phase == TRAIN && init != null) initParameters(module, init)
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
    testModel.get.evaluate()
  }

  def getTrainModel: Module[Float] = {
    trainModel match {
      case None => trainModel = Some(createTrainModel())
      case _ =>
    }
    trainModel.get.training()
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

  def createTestModel(): Sequential[Float] = {
    val model = new Sequential()
    val model1 = new ParallelTable()
    model1.add(featureAndRpnNet)
    model1.add(new Identity())
    model.add(model1)
    // connect rpn and fast-rcnn
    val middle = new ConcatTable()
    val left = new Sequential()
    val left1 = new ConcatTable()
    left1.add(selectTensor(1, 1, 1))
    left1.add(selectTensor(1, 1, 2))
    left1.add(selectTensor1(2))
    left.add(left1)
    left.add(new Proposal(param))
    left.add(selectTensor1(1))
    // first add feature from feature net
    middle.add(selectTensor(1, 2))
    // then add rois from proposal
    middle.add(left)
    model.add(middle)
    // get the fast rcnn results and rois
    model.add(new ConcatTable().add(fastRcnn).add(selectTensor(2)))
    model
  }


  def createTrainModel(): Sequential[Float] = {
    val model = new Sequential()

    val rpnFeatureWithInfoGt = new ParallelTable()
    rpnFeatureWithInfoGt.add(featureAndRpnNet)
    // im_info
    rpnFeatureWithInfoGt.add(new Identity())
    // gt_boxes
    rpnFeatureWithInfoGt.add(new Identity())
    model.add(rpnFeatureWithInfoGt)

    val lossModels = new ConcatTable()
    model.add(lossModels)

    lossModels.add(selectTensor(1, 1, 1).setName("rpn_cls"))
    lossModels.add(selectTensor(1, 1, 2).setName("rpn_reg"))
    val fastRcnnLossModel = new Sequential().setName("loss from fast rcnn")
    // get ((rois, otherProposalTargets), features
    val fastRcnnInputModel = new ConcatTable().setName("fast-rcnn")
    fastRcnnInputModel.add(selectTensor(1, 2).setName("features"))
    val sampleRoisModel = new Sequential().setPropagateBack(false)
    fastRcnnInputModel.add(sampleRoisModel)
    // add sample rois
    lossModels.add(fastRcnnLossModel)

    val proposalTargetInput = new ConcatTable()
    // get rois from proposal layer
    val proposalModel = new Sequential()
      .add(new ConcatTable()
        .add(new Sequential()
          .add(selectTensor(1, 1, 1).setName("rpn_cls"))
          .add(new SoftMax())
          .add(new ReshapeInfer(Array(1, 2 * param.anchorNum, -1, 0)))
          .setName("rpn_cls_softmax_reshape"))
        .add(selectTensor(1, 1, 2).setName("rpn_reg"))
        .add(selectTensor1NoBack(2).setName("im_info")))
      .add(new Proposal(param))
      .add(selectTensor1NoBack(1).setName("rpn_rois"))

    proposalTargetInput.add(proposalModel)
    proposalTargetInput.add(selectTensor1NoBack(3).setName("gtBoxes"))
    sampleRoisModel.add(proposalTargetInput)
    sampleRoisModel.add(new ProposalTarget(param))

    // ( features, (rois, otherProposalTargets))
    fastRcnnLossModel.add(fastRcnnInputModel)
    fastRcnnLossModel.add(
      new ConcatTable()
        .add(new ConcatTable()
          .add(selectTensor1(1).setName("features"))
          .add(selectTensorNoBack(2, 1).setName("rois")))
        .add(selectTableNoBack(2, 2).setName("other targets info")))
    fastRcnnLossModel.add(new ParallelTable()
      .add(fastRcnn.setName("fast rcnn"))
      .add(new Identity()).setName("other targets info"))
    // make each res a tensor
    model.add(new ConcatTable()
      .add(selectTensor1(1).setName("rpn cls"))
      .add(selectTensor1(2).setName("rpn reg"))
      .add(selectTensor(3, 1, 1).setName("cls"))
      .add(selectTensor(3, 1, 2).setName("reg"))
      .add(selectTensorNoBack(3, 2).setName("other target info")))
    model
  }

  def getModel: Module[Float] = if (isTest) getTestModel else getTrainModel

  def selectTensorNoBack(depths: Int*): Sequential[Float] = {
    val module = new Sequential[Float]().setPropagateBack(false)
    depths.slice(0, depths.length - 1).foreach(depth =>
      module.add(new SelectTable(depth).setPropagateBack(false)))
    module.add(new SelectTable(depths(depths.length - 1)).setPropagateBack(false))
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
      module.add(new SelectTable(depth)))
    module.add(new SelectTable(depths(depths.length - 1)))
  }

  def selectTensor1(depth: Int): SelectTable[Float] = {
    new SelectTable(depth)
  }

  def selectTensor1NoBack(depth: Int): SelectTable[Float] = {
    new SelectTable(depth).setPropagateBack(false)
  }

  def selectTable(depths: Int*): Sequential[Float] = {
    val module = new Sequential[Float]()
    depths.slice(0, depths.length).foreach(depth =>
      module.add(new SelectTable(depth)))
    module
  }

  def selectTableNoBack(depths: Int*): Sequential[Float] = {
    val module = new Sequential[Float]().setPropagateBack(false)
    depths.slice(0, depths.length).foreach(depth =>
      module.add(new SelectTable(depth).setPropagateBack(false)))
    module
  }

  def selectTable1(depth: Int): SelectTable[Float] = {
    new SelectTable(depth)
  }

  def selectTable1NoBack(depth: Int): SelectTable[Float] = {
    new SelectTable(depth).setPropagateBack(false)
  }

  def initParameters(module: Module[Float], init: (Double, Double)): Unit = {
    val params = module.parameters()
    params._1(0).apply1(_ => RNG.normal(0, init._1).toFloat)
    params._1(1).apply1(_ => init._2.toFloat)
  }

  def loadFromCaffeOrCache(dp: String, mp: String): this.type = {
    val cachedPath = mp.substring(0, mp.lastIndexOf(".")) + ".bigdl"
    val mod = FileUtil.load[(Sequential[Float], Sequential[Float])](cachedPath)
    mod match {
      case Some((featureAndRpn, fastRcnn)) =>
        logger.info(s"load model with caffe weight from cache $cachedPath")
        setFeatureAndRpnNet(featureAndRpn)
        setFastRcnn(fastRcnn)
      case _ =>
        Module.loadCaffe[Float](getModel, dp, mp, phase == TEST)
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
  def apply(modelType: ModelType, phase: PhaseType = TEST,
    caffeModel: Option[(String, String)] = None): FasterRcnn = {

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

    Random.setSeed(FasterRcnnParam.RANDOM_SEED)
    val fasterRcnnModel = caffeModel match {
      case Some((dp: String, mp: String)) =>
        // caffe pretrained model
        getFasterRcnn(modelType)
          .loadFromCaffeOrCache(dp, mp)
      case _ => getFasterRcnn(modelType)
    }
    fasterRcnnModel
  }
}
