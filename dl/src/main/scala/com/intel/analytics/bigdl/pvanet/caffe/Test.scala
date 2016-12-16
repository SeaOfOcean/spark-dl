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

package com.intel.analytics.bigdl.pvanet.caffe

import java.nio.file.Paths

import com.intel.analytics.bigdl.models.googlenet.{DataSet, Options}
import com.intel.analytics.bigdl.models.imagenet.AlexNet
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{LocalValidator, Top1Accuracy}
import scopt.OptionParser


object Test {

  case class TestLocalParams(
    folder: String = "./",
    caffeDefPath: Option[String] = None,
    caffeModelPath: Option[String] = None,
    coreNumber: Int = (Runtime.getRuntime().availableProcessors() / 2)
  )

  val testLocalParser = new OptionParser[TestLocalParams]("BigDL AlexNet Example") {
    head("Train AlexNet model on single node")
    opt[String]('f', "folder")
      .text("where you put your local hadoop sequence files")
      .action((x, c) => c.copy(folder = x))
    opt[String]("caffeDefPath")
      .text("caffe define path")
      .action((x, c) => c.copy(caffeDefPath = Some(x)))
    opt[String]("caffeModelPath")
      .text("caffe model path")
      .action((x, c) => c.copy(caffeModelPath = Some(x)))
    opt[Int]('c', "core")
      .text("cores number to train the model")
      .action((x, c) => c.copy(coreNumber = x))
  }

  def main(args: Array[String]): Unit = {
    val batchSize = 32
    val imageSize = 224

        val defPath: String = "data/model/alexnet/deploy.prototxt"
    val modelPath: String = "data/model/alexnet/bvlc_alexnet.caffemodel"
    val model = Module.loadCaffeParameters[Float](AlexNet.apply(1000),
      defPath, modelPath)

    testLocalParser.parse(args, new TestLocalParams()).map(param => {

      val validationData = Paths.get(param.folder, "val")
      val validateDataSet = DataSet.localDataSet(validationData, imageSize, batchSize,
        param.coreNumber, false)

      //      val model = Module.loadCaffeParameters[Float](GoogleNet_v1_NoAuxClassifier.apply(1000),
//        param.caffeDefPath.get, param.caffeModelPath.get)
val model = Module.loadCaffeParameters[Float](AlexNet.apply(1000),
  param.caffeDefPath.get, param.caffeModelPath.get)
      val validator = new LocalValidator[Float](model, param.coreNumber)
      println("dataset size", validateDataSet.size())
      val result = validator.test(validateDataSet, Array(new Top1Accuracy[Float]))
      result.foreach(r => {
        println(s"${r._2} is ${r._1}")
      })
    })
  }
}
