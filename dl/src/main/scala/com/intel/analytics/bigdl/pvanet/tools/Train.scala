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

package com.intel.analytics.bigdl.pvanet.tools

import com.intel.analytics.bigdl.dataset.ImageNetLocal.Config
import com.intel.analytics.bigdl.pvanet.model.Model._
import com.intel.analytics.bigdl.pvanet.model.Phase._
import com.intel.analytics.bigdl.models.imagenet.{AlexNet, GoogleNet_v1}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, ParallelCriterion}
import com.intel.analytics.bigdl.optim.SGD.EpochStep
import com.intel.analytics.bigdl.optim.{SGD, Trigger}
import com.intel.analytics.bigdl.pvanet.datasets.Roidb.ImageWithRoi
import com.intel.analytics.bigdl.pvanet.datasets.{AnchorToTensor, ImageToTensor, PascolVocDataSource}
import com.intel.analytics.bigdl.pvanet.layers.{AnchorTargetLayer, SmoothL1Criterion2, SoftmaxWithCriterion}
import com.intel.analytics.bigdl.pvanet.model._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import scopt.OptionParser

object Train {
  private val configs = Map(
    "vgg16" -> Config(
      AlexNet[Float](classNum = 1000),
      new ClassNLLCriterion[Float](),
      new SGD[Float](),
      imageSize = 227,
      batchSize = 256,
      momentum = 0.9,
      weightDecay = 0.0005,
      testTrigger = Trigger.severalIteration(10000),
      cacheTrigger = Trigger.severalIteration(10000),
      endWhen = Trigger.maxIteration(450000),
      learningRate = 0.01,
      learningRateSchedule = SGD.Step(100000, 0.1)),
    "pvanet" -> Config(
      GoogleNet_v1[Float](classNum = 1000),
      new ClassNLLCriterion[Float](),
      new SGD[Float](),
      imageSize = 224,
      batchSize = 32,
      momentum = 0.9,
      weightDecay = 0.0002,
      testTrigger = Trigger.severalIteration(4000),
      cacheTrigger = Trigger.severalIteration(40000),
      endWhen = Trigger.maxIteration(2400000),
      learningRate = 0.01,
      learningRateSchedule = SGD.Poly(0.5, 2400000))
  )

  def trainNet(net: FasterRcnn[Float], dataSource: PascolVocDataSource,
    valSource: PascolVocDataSource, param: PascolVocLocalParam): Unit = {
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 1 RPN, init from ImageNet model")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val rpnStage1Model = trainRpn(net, dataSource, valSource, "pretrained_path", 10)

    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 1 RPN, generate proposals")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val rpnStage1Proposals = rpnGenerate(dataSource, rpnStage1Model)
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    val fastRcnnStage1Model = trainFastRcnn(
      dataSource, "pretrained_path", 10, rpnStage1Proposals)

    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 2 RPN, init from stage 1 Fast R-CNN model")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    val rpnStage2Model = trainRpn(net, dataSource, valSource, fastRcnnStage1Model, 10)

    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 2 RPN, generate proposals")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val rpnStage2Proposals = rpnGenerate(dataSource, rpnStage2Model)



    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 2 Fast R-CNN, init from stage 2 RPN R-CNN model")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val fastRcnnStage2Model = trainFastRcnn(
      dataSource, rpnStage2Model, 10, rpnStage2Proposals)
    val finalModelPath = fastRcnnStage2Model
  }

//  def rpnTest(imageToTensor: ImageToTensor,
//    net: FasterRcnn[Float], d: ImageWithRoi): Unit = {
//    val res = net.featureAndRpnNet.forward(imageToTensor.apply(d))
//    require(res.length() == 2)
//    val (height: Int, width: Int, nAnchors: Int, target: Table) = getRpnTarget(d, res, net.param)
//    res(1).asInstanceOf[Tensor[Float]].resize(2, nAnchors * height * width)
//    res(2).asInstanceOf[Tensor[Float]].resize(nAnchors * 4 * height * width)
//    val output: Float = rpnLoss(res, target)
//    println("output from parallel criterion: " + output)
//  }

  def getRpnTarget(d: ImageWithRoi, res: Table, param: FasterRcnnParam): (Int, Int, Int, Table) = {
    val clsOut = res(1).asInstanceOf[Tensor[Float]]
    val sizes = clsOut.size()
    val height = sizes(sizes.length - 2)
    val width = sizes(sizes.length - 1)
    val anchorTargetLayer = new AnchorTargetLayer(param)
    val anchors = anchorTargetLayer.generateAnchors(d, height, width)
    val anchorToTensor = new AnchorToTensor(1, height, width)
    val anchorTensors = anchorToTensor.apply(anchors)
    val target = new Table
    target.insert(anchorTensors._1)
    target.insert(anchorTensors._2)
    (height, width, param.anchorNum, target)
  }

  def rpnLoss(res: Table, target: Table): Float = {
    val slc = new SmoothL1Criterion2[Float](3f, 1)
    val sfm = new SoftmaxWithCriterion[Float](ignoreLabel = Some(-1))
    val pc = new ParallelCriterion[Float]()
    pc.add(sfm, 1.0f)
    pc.add(slc, 1.0f)
    val output = pc.forward(res, target)
    output
  }

  /**
   *
   * Train a Region Proposal Network in a separate training process.
   */
  def trainRpn(net: FasterRcnn[Float], dataSource: PascolVocDataSource,
    valDataSource: PascolVocDataSource, initModel: String, maxIters: Int): String = {
    // Not using any proposals, just ground-truth boxes
    net.param.BBOX_REG = false
    net.param.IMS_PER_BATCH = 1
    val optimizer = new RpnOptimizer(
      data = dataSource,
      net = net,
      optimMethod = new SGD[Float](),
      state = T(
        "learningRate" -> 0.01,
        "weightDecay" -> 0.0005,
        "momentum" -> 0.9,
        "dampening" -> 0.0,
        "learningRateSchedule" -> EpochStep(25, 0.5)
      ),
      endWhen = Trigger.maxEpoch(90))

    val config = configs(net.modelName)
    optimizer.setCache(param.cache + "/" + param.net, config.cacheTrigger)
    optimizer.setValidationTrigger(config.testTrigger)
    optimizer.addValidation(new Top1Accuracy[Float])
    optimizer.addValidation(new Top5Accuracy[Float])
    optimizer.overWriteCache()
    optimizer.optimizeRpn()
    param.cache + "/" + param.net
  }

  def rpnGenerate(dataSource: PascolVocDataSource, rpnModelPath: String): String = {
    ""
  }

  def trainFastRcnn(
    dataSource: PascolVocDataSource,
    initModel: String,
    maxIters: Int,
    rpnProposals: String): String = {
    ""
  }

  case class PascolVocLocalParam(
    folder: String = "/home/xianyan/objectRelated/VOCdevkit",
    net: String = "pvanet",
    nThread: Int = 4,
    cache: String = ".")

  private val parser = new OptionParser[PascolVocLocalParam]("Spark-DL PascolVoc Local Example") {
    head("Spark-DL PascolVoc Local Example")
    opt[String]('f', "folder")
      .text("where you put the PascolVoc data")
      .action((x, c) => c.copy(folder = x))
    opt[String]('n', "net")
      .text("net type : vgg16 | pvanet")
      .action((x, c) => c.copy(net = x.toLowerCase))
    opt[String]('t', "mkl thread number")
      .action((x, c) => c.copy(nThread = x.toInt))
  }

  var param: PascolVocLocalParam = null

  def main(args: Array[String]) {
    import com.intel.analytics.bigdl.mkl.MKL
    param = parser.parse(args, PascolVocLocalParam()).get

    var model: FasterRcnn[Float] = null
    param.net match {
      case "vgg16" =>
        model = VggFRcnn.model()
      case "pvanet" =>
        model = PvanetFRcnn.model()
    }
    MKL.setNumThreads(param.nThread)
    val dataSource = new PascolVocDataSource("2007", "train", param.folder,
      false, model.param)
    val valSource = new PascolVocDataSource("2007", "val", param.folder,
      false, model.param)
    trainNet(model, dataSource, valSource, param)
  }
}
