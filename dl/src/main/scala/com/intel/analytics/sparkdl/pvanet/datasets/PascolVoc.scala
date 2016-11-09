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

package com.intel.analytics.sparkdl.pvanet.datasets

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.pvanet.AnchorTargetLayer
import com.intel.analytics.sparkdl.pvanet.layers.{SmoothL1CriterionOD, SoftmaxWithCriterion}
import com.intel.analytics.sparkdl.pvanet.model.Vgg_16_RPN
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import com.intel.analytics.sparkdl.utils.Table
import scopt.OptionParser

object PascolVoc {

  case class PascolVocLocalParam(folder: String = "/home/xianyan/objectRelated/VOCdevkit",
                                 net: String = "vgg")

  private val parser = new OptionParser[PascolVocLocalParam]("Spark-DL PascolVoc Local Example") {
    head("Spark-DL PascolVoc Local Example")
    opt[String]('f', "folder")
      .text("where you put the PascolVoc data")
      .action((x, c) => c.copy(folder = x))
    opt[String]('n', "net")
      .text("net type : vgg | pvanet")
      .action((x, c) => c.copy(net = x.toLowerCase))
  }

  def main(args: Array[String]) {
    parser.parse(args, new PascolVocLocalParam()).map(param => {
      val year = "2007"

      val validationDataSource = new PascolVocDataSource(year, "val", param.folder, false)
      val trainDataSource = new PascolVocDataSource(year, imageSet = "train", param.folder, false)

      val imageScaler = ImageScalerAndMeanSubstractor(trainDataSource)
      val imageToTensor = new ImageToTensor(batchSize = 1)

      val data = trainDataSource -> imageScaler

      val model = param.net match {
        case "vgg" => Vgg_16_RPN[Float](classNum = 21)
        case _ => throw new IllegalArgumentException
      }

      val d = data.next()
      val res = model.forward(imageToTensor.apply(d))
      require(res.length() == 2)
      val clsOut = res(1).asInstanceOf[Tensor[Float]]
      val sizes = clsOut.size()
      println("clsOut: (" + sizes.mkString(", ") + ")")
      val regOut = res(2).asInstanceOf[Tensor[Float]]
      println("regOut: (" + regOut.size().mkString(", ") + ")")

      val height = sizes(sizes.length - 2)
      val width = sizes(sizes.length - 1)

      val scales = Tensor(Storage(Array[Float](8, 16, 32)))
      val ratios = Tensor(Storage(Array(0.5f, 1.0f, 2.0f)))
      val nAnchors = scales.nElement() * ratios.nElement()
      val anchorTargetLayer = new AnchorTargetLayer(scales, ratios)
      val anchors = anchorTargetLayer.generateAnchors(d, height, width)
      val anchorToTensor = new AnchorToTensor(1, height, width)
      val anchorTensors = anchorToTensor.apply(anchors)

      res(1).asInstanceOf[Tensor[Float]].resize(2, nAnchors * height * width)
      res(2).asInstanceOf[Tensor[Float]].resize(nAnchors * 4 * height * width)

      val target = new Table
      target.insert(anchorTensors._1)
      target.insert(anchorTensors._2)
      val slc = new SmoothL1CriterionOD[Float](0.3f, 1)
      val sfm = new SoftmaxWithCriterion[Float](ignoreLabel = Some(-1))
      val pc = new ParallelCriterion[Float]()
      pc.add(sfm, 1.0f)
      pc.add(slc, 1.0f)

      val output = pc.forward(res, target)
      println("output from parallel criterion: " + output)
    })
  }
}
