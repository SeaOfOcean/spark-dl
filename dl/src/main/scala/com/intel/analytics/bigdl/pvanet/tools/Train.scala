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

import com.intel.analytics.bigdl.optim.SGD.EpochStep
import com.intel.analytics.bigdl.optim.{SGD, Trigger}
import com.intel.analytics.bigdl.pvanet.datasets.{ImageScalerAndMeanSubstractor, ObjectDataSource}
import com.intel.analytics.bigdl.pvanet.model.Model._
import com.intel.analytics.bigdl.pvanet.model._
import com.intel.analytics.bigdl.utils.T
import scopt.OptionParser

object Train {

  case class PascolVocLocalParam(
    folder: String = "/home/xianyan/objectRelated/VOCdevkit",
    net: ModelType = Model.VGG16,
    nThread: Int = 8,
    cache: String = ".")

  private val parser = new OptionParser[PascolVocLocalParam]("Spark-DL PascolVoc Local Example") {
    head("Spark-DL PascolVoc Local Example")
    opt[String]('f', "folder")
      .text("where you put the PascolVoc data")
      .action((x, c) => c.copy(folder = x))
    opt[String]('n', "net")
      .text("net type : vgg16 | pvanet")
      .action((x, c) => c.copy(net = Model.withName(x)))
    opt[String]('t', "mkl thread number")
      .action((x, c) => c.copy(nThread = x.toInt))
  }

  val model2caffePath = Map(
    VGG16 -> ("/home/xianyan/objectRelated/faster_rcnn_models/VGG16/" +
      "faster_rcnn_alt_opt/rpn_test.pt",
      "dl/data/imagenet_models/VGG16.v2.caffemodel"),
    PVANET -> ("/home/xianyan/objectRelated/pvanet/full/test.pt",
      "/home/xianyan/objectRelated/pvanet/full/test.model"))

  var param: PascolVocLocalParam = null

  def main(args: Array[String]) {
    import com.intel.analytics.bigdl.mkl.MKL
    param = parser.parse(args, PascolVocLocalParam()).get

    val fasterRcnnModel = FasterRcnn(param.net, Phase.TRAIN, model2caffePath(param.net))
    MKL.setNumThreads(param.nThread)
    val dataSource = new ObjectDataSource("voc_2007_testcode1", param.folder,
      true, fasterRcnnModel.param)
    val valSource = new ObjectDataSource("voc_2007_testcode1", param.folder,
      false, fasterRcnnModel.param)
    val config = fasterRcnnModel.param.optimizeConfig
    val imgScaler = new ImageScalerAndMeanSubstractor(fasterRcnnModel.param)
    fasterRcnnModel.train
    val optimizer = new FasterRcnnOptimizer(
      data = dataSource -> imgScaler,
      validationData = valSource,
      net = fasterRcnnModel,
      model = fasterRcnnModel.getTrainModel,
      criterion = fasterRcnnModel.criterion4,
      optimMethod = new SGD[Float](),
      state = T(
        "learningRate" -> config.learningRate,
        "weightDecay" -> config.weightDecay,
        "momentum" -> config.momentum,
        "dampening" -> 0.0,
        "learningRateSchedule" -> EpochStep(25, 0.5)
      ),
      endWhen = Trigger.maxIteration(5))

    optimizer.setCache(param.cache + "/" + param.net, config.cacheTrigger)
    optimizer.setValidationTrigger(config.testTrigger)
    optimizer.overWriteCache()
    optimizer.optimize()
  }
}
