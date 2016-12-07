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

import com.intel.analytics.bigdl.optim.SGD.{EpochStep, LearningRateSchedule}
import com.intel.analytics.bigdl.optim.{OptimMethod, SGD, Trigger}
import com.intel.analytics.bigdl.pvanet.datasets.{ImageScalerAndMeanSubstractor, ObjectDataSource}
import com.intel.analytics.bigdl.pvanet.model.{FasterRcnn, Model, PvanetFRcnn, VggFRcnn}
import com.intel.analytics.bigdl.pvanet.model.Model._
import com.intel.analytics.bigdl.utils.T
import scopt.OptionParser

object Train {

  case class PascolVocLocalParam(
    folder: String = "/home/xianyan/objectRelated/VOCdevkit",
    net: Model = VGG16,
    nThread: Int = 4,
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

  var param: PascolVocLocalParam = null

  case class Config(
    optimMethod: OptimMethod[Float],
    momentum: Double,
    weightDecay: Double,
    testTrigger: Trigger,
    cacheTrigger: Trigger,
    endWhen: Trigger,
    learningRate: Double,
    learningRateSchedule: LearningRateSchedule
  )

  private val configs = Map(
    VGG16 -> Config(
      new SGD[Float](),
      momentum = 0.9,
      weightDecay = 0.0005,
      testTrigger = Trigger.severalIteration(10000),
      cacheTrigger = Trigger.severalIteration(10000),
      endWhen = Trigger.maxIteration(450000),
      learningRate = 0.001,
      learningRateSchedule = SGD.Step(50000, 0.1)),
    PVANET -> Config(
      new SGD[Float](),
      momentum = 0.9,
      weightDecay = 0.0002,
      testTrigger = Trigger.severalIteration(4000),
      cacheTrigger = Trigger.severalIteration(40000),
      endWhen = Trigger.maxIteration(2400000),
      learningRate = 0.001,
      learningRateSchedule = SGD.Step(50000, 0.1)))

  def main(args: Array[String]) {
    import com.intel.analytics.bigdl.mkl.MKL
    param = parser.parse(args, PascolVocLocalParam()).get

    var model: FasterRcnn[Float] = null
    param.net match {
      case VGG16 =>
        model = VggFRcnn.model()
      case PVANET =>
        model = PvanetFRcnn.model()
    }
    MKL.setNumThreads(param.nThread)
    val dataSource = new ObjectDataSource("voc_2007_train", param.folder,
      true, model.param)
    val valSource = new ObjectDataSource("voc_2007_val", param.folder,
      false, model.param)
    val config = configs(model.modelType)
    val imgScaler = new ImageScalerAndMeanSubstractor(model.param)
    val optimizer = new FasterRcnnOptimizer(
      data = dataSource -> imgScaler,
      validationData = valSource,
      net = model,
      optimMethod = new SGD[Float](),
      state = T(
        "learningRate" -> 0.001,
        "weightDecay" -> 0.0005,
        "momentum" -> config.momentum,
        "dampening" -> 0.0,
        "learningRateSchedule" -> EpochStep(25, 0.5)
      ),
      endWhen = Trigger.maxEpoch(90))

    optimizer.setCache(param.cache + "/" + param.net, config.cacheTrigger)
    optimizer.setValidationTrigger(config.testTrigger)
    optimizer.overWriteCache()
    optimizer.optimize(model.fullModel())
  }
}
