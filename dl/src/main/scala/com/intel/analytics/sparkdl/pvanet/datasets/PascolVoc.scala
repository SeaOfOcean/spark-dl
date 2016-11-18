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
import com.intel.analytics.sparkdl.pvanet.Roidb.ImageWithRoi
import com.intel.analytics.sparkdl.pvanet.caffe.VggCaffeModel
import com.intel.analytics.sparkdl.pvanet.layers.{Proposal, Reshape2, SmoothL1Criterion2, SoftmaxWithCriterion}
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import com.intel.analytics.sparkdl.utils.Table
import scopt.OptionParser

object PascolVoc {
  val scales = VggCaffeModel.scales
  val ratios = VggCaffeModel.ratios
  val A = scales.length * ratios.length

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

      //      val model = param.net match {
      //        case "vgg" => VggCaffeModel.Vgg_16_RPN
      //        case _ => throw new IllegalArgumentException
      //      }

      var start = 0L
      var end = 0L
      for (i <- 0 until 10) {
        val d = data.next()
        println(s"process ${d.imagePath} ...............")
        val imgTensor = imageToTensor(d)

        println("start generating features...")
        start = System.nanoTime()
        val featureModel = VggCaffeModel.vgg_16
        val featureOut = featureModel.forward(imgTensor)
        end = System.nanoTime()
        println(s"generate features done, ${(end - start) / 1e9}s")

        println("start go to rpn...")
        start = System.nanoTime()
        val rpnModel = VggCaffeModel.RPN
        val clsRegOut = rpnModel.forward(featureOut)

        val rpnClsScore = clsRegOut(1).asInstanceOf[Tensor[Float]]
        val clsProc = new Sequential[Tensor[Float], Tensor[Float], Float]()
        clsProc.add(new Reshape2[Float](Array(0, 2, -1, 0)))
        clsProc.add(new SoftMax[Float]())
        clsProc.add(new Reshape2[Float](Array(0, 2 * A, -1, 0)))
        val rpn_bbox_pred = clsRegOut(2).asInstanceOf[Tensor[Float]]
        end = System.nanoTime()
        println(s"rpn done, ${(end - start) / 1e9}s")

        println("start fast rcnn...")
        start = System.nanoTime()
        val proposalInput = new Table
        proposalInput.insert(rpnClsScore)
        proposalInput.insert(rpn_bbox_pred)
        proposalInput.insert(d.imInfo.get)

        println("rpnClsScore: ", rpnClsScore.size().mkString(","))
        println("bbox pred: ", rpn_bbox_pred.size().mkString(","))
        println(d.imInfo.get.mkString(", "))
        val propoal = new Proposal[Float]
        val proposalOut = propoal.forward(proposalInput)

        //todo: the resize here is used for linear which support only 2d
        //todo: but not sure whether the resize is correct
        val rois = proposalOut(1).asInstanceOf[Tensor[Float]]
        val propDecInput = new Table()
//        propDecInput.insert(featureOut.resize(featureOut.size(2), featureOut.nElement() / featureOut.size(2)))
//        propDecInput.insert(rois.resize(rois.size(2), rois.nElement() / rois.size(2)))
        propDecInput.insert(featureOut)
        propDecInput.insert(rois)

        println("featureOut:", featureOut.size().mkString(", "))
        println("rois: ", proposalOut(1).asInstanceOf[Tensor[Float]].size().mkString(","))

        val propDecModel = VggCaffeModel.FastRCNN

        val result = propDecModel.forward(propDecInput)
        end = System.nanoTime()
        println(s"fast rcnn done, ${(end - start) / 1e9}s")
        println(result)
      }
    })
  }

  def fullModelTest(imageToTensor: ImageToTensor, model: Module[Tensor[Float], Table, Float], d: ImageWithRoi): Unit = {
    // get rpn_cls_score and rpn_bbox_pred
    val res = model.forward(imageToTensor(d))
    val rpnClsScore = res(1).asInstanceOf[Tensor[Float]]
    val clsProc = new Sequential[Tensor[Float], Tensor[Float], Float]()
    clsProc.add(new Reshape2[Float](Array(0, 2, -1, 0)))
    clsProc.add(new SoftMax[Float]())
    clsProc.add(new Reshape2[Float](Array(0, 2 * A, -1, 0)))
    val rpn_bbox_pred = res(2).asInstanceOf[Tensor[Float]]

    val proposalInput = new Table
    proposalInput.insert(rpnClsScore)
    proposalInput.insert(rpn_bbox_pred)
    proposalInput.insert(Tensor(Storage(d.imInfo.get)))

    val roiPoolInput = new ConcatTable[Table, Float]()
    roiPoolInput

  }


  def rpnTest(imageToTensor: ImageToTensor, model: Module[Tensor[Float], Table, Float], d: ImageWithRoi): Unit = {
    val res = model.forward(imageToTensor.apply(d))
    require(res.length() == 2)
    val (height: Int, width: Int, nAnchors: Int, target: Table) = getRpnTarget(d, res)
    res(1).asInstanceOf[Tensor[Float]].resize(2, nAnchors * height * width)
    res(2).asInstanceOf[Tensor[Float]].resize(nAnchors * 4 * height * width)
    val output: Float = rpnLoss(res, target)
    println("output from parallel criterion: " + output)
  }

  def getRpnTarget(d: ImageWithRoi, res: Table): (Int, Int, Int, Table) = {
    val clsOut = res(1).asInstanceOf[Tensor[Float]]
    val sizes = clsOut.size()
    val regOut = res(2).asInstanceOf[Tensor[Float]]
    val height = sizes(sizes.length - 2)
    val width = sizes(sizes.length - 1)
    val nAnchors = scales.length * ratios.length
    val anchorTargetLayer = new AnchorTargetLayer(scales, ratios)
    val anchors = anchorTargetLayer.generateAnchors(d, height, width)
    val anchorToTensor = new AnchorToTensor(1, height, width)
    val anchorTensors = anchorToTensor.apply(anchors)
    val target = new Table
    target.insert(anchorTensors._1)
    target.insert(anchorTensors._2)
    (height, width, nAnchors, target)
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
}
