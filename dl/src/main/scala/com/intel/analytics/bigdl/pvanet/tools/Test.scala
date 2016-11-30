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

import breeze.linalg.DenseMatrix
import com.intel.analytics.bigdl.nn.{Module, Sequential, SoftMax}
import com.intel.analytics.bigdl.pvanet.datasets.Roidb.ImageWithRoi
import com.intel.analytics.bigdl.pvanet.datasets.{ImageScalerAndMeanSubstractor, ImageToTensor, PascolVocDataSource}
import com.intel.analytics.bigdl.pvanet.layers.{Proposal, Reshape2}
import com.intel.analytics.bigdl.pvanet.model.{FasterPvanet, FasterRCNN, FasterVgg}
import com.intel.analytics.bigdl.pvanet.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Table, Timer}
import scopt.OptionParser

object Test {

  case class PascolVocLocalParam(folder: String = "/home/xianyan/objectRelated/VOCdevkit",
    net: String = "vgg16", nThread: Int = 4)

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

  def testNet(net: FasterRCNN[Float], dataSource: PascolVocDataSource, maxPerImage: Int = 100,
    thresh: Double = 0.05, vis: Boolean = false): Unit = {
    val imdb = dataSource.imdb
    val allBoxes: Array[Array[DenseMatrix[Float]]] = {
      val out = new Array[Array[DenseMatrix[Float]]](imdb.numClasses)
      Range(0, imdb.numClasses).foreach(x => {
        out(x) = new Array[DenseMatrix[Float]](imdb.numImages)
      })
      out
    }
    val imageScaler = new ImageScalerAndMeanSubstractor(dataSource, isShuffle = false)

    // timer
    val imDetectTimer = new Timer
    val miscTimer = new Timer

    for (i <- 0 until imdb.numImages) {
      val d = imageScaler.apply(dataSource.next())
      println(s"process ${d.imagePath} ...............")

      imDetectTimer.tic()
      val (scores: DenseMatrix[Float], boxes: DenseMatrix[Float]) = imDetect(net, d)
      imDetectTimer.toc()

      miscTimer.tic()
      // skip j = 0, because it's the background class
      for (j <- 1 until imdb.numClasses) {
        def getClsDet: DenseMatrix[Float] = {
          val inds = Range(0, scores.rows).filter(ind => scores(ind, j) > thresh).toArray
          if (inds.length == 0) return new DenseMatrix[Float](0, 5)
          val clsScores = MatrixUtil.selectMatrix2(scores, inds, Array(j))
          val clsBoxes = MatrixUtil.selectMatrix2(boxes,
            inds, Range(j * 4, (j + 1) * 4).toArray)

          var clsDets = DenseMatrix.horzcat(clsBoxes, clsScores)
          val keep = Nms.nms(clsDets, Config.TEST.NMS.toFloat)

          val detsNMSed = MatrixUtil.selectMatrix(clsDets, keep, 0)

          if (Config.TEST.BBOX_VOTE) {
            clsDets = Bbox.bboxVote(detsNMSed, clsDets)
          } else {
            clsDets = detsNMSed
          }
          clsDets
        }
        val clsDets = getClsDet

        if (vis) {
          visDetection(d, imdb.classes(j), clsDets)
        }
        allBoxes(j)(i) = clsDets
      }

      // Limit to max_per_image detections *over all classes*
      // todo: has not been tested
      if (maxPerImage > 0) {
        var imageScores = Array[Float]()
        for (j <- Range(1, imdb.numClasses)) {
          imageScores = imageScores ++ allBoxes(j)(i)(::, -1).toArray
        }
        if (imageScores.length > maxPerImage) {
          val imageThresh = imageScores.sortWith(_ < _)(imageScores.length - maxPerImage)
          for (j <- Range(1, imdb.numClasses)) {
            val box = allBoxes(j)(i)
            val keep = (0 until box.rows).filter(x => box(x, box.cols - 1) >= imageThresh).toArray
            allBoxes(j)(i) = MatrixUtil.selectMatrix(box, keep, 0)
          }
        }
      }
      miscTimer.toc()
      println(s"im detect: $i/${imdb.numImages} " +
        s"${imDetectTimer.averageTime / 1e9}s ${miscTimer.averageTime / 1e9}s")
    }

    println("Evaluating detections")

    val outputDir = Config.getOutputDir(imdb, "VGG16")
    imdb.evaluateDetections(allBoxes, outputDir)
  }

  def imDetect(d: ImageWithRoi,
    rpnWithFeatureModel: Module[Tensor[Float], Table, Float],
    fastRcnn: Module[Table, Table, Float], A: Int): (DenseMatrix[Float], DenseMatrix[Float]) = {
    val imgTensor = ImageToTensor(d)
    val rpnWithFeature = rpnWithFeatureModel.forward(imgTensor)

    val rpnBboxPred = rpnWithFeature(1).asInstanceOf[Table](2).asInstanceOf[Tensor[Float]]
    val rpnClsScore = rpnWithFeature(1).asInstanceOf[Table](1).asInstanceOf[Tensor[Float]]
    val featureOut = rpnWithFeature(2).asInstanceOf[Tensor[Float]]

    val clsProc = new Sequential[Tensor[Float], Tensor[Float], Float]()
    clsProc.add(new Reshape2[Float](Array(2, -1), Some(false)))
    clsProc.add(new SoftMax[Float]())
    clsProc.add(new Reshape2[Float](Array(1, 2 * A, -1, rpnBboxPred.size(4)), Some(false)))
    val rpnClsScoreReshape = clsProc.forward(rpnClsScore)

    val proposalInput = new Table
    proposalInput.insert(rpnClsScoreReshape)
    proposalInput.insert(rpnBboxPred)
    proposalInput.insert(d.imInfo.get)

    val propoal = new Proposal[Float](phase = 1)
    val proposalOut = propoal.forward(proposalInput)

    val rois = proposalOut(1).asInstanceOf[Tensor[Float]]

    val propDecInput = new Table()
    propDecInput.insert(featureOut)
    propDecInput.insert(rois)

    val result = fastRcnn.forward(propDecInput)

    val scores = result(1).asInstanceOf[Tensor[Float]]
    val boxDeltas = result(2).asInstanceOf[Tensor[Float]]

    // post process
    // unscale back to raw image space
    val boxes = rois.narrow(2, 2, 4).div(d.imInfo.get(2))
    // Apply bounding-box regression deltas
    var predBoxes = Bbox.bboxTransformInv(boxes.toBreezeMatrix(), boxDeltas.toBreezeMatrix())
    predBoxes = Bbox.clipBoxes(predBoxes, d.oriHeight, d.oriWidth)

    (scores.toBreezeMatrix(), predBoxes)
  }

  def visDetection(d: ImageWithRoi, clsname: String, clsDets: DenseMatrix[Float],
    thresh: Float = 0.3f): Unit = {
    Draw.vis(d.imagePath, clsname, clsDets,
      Config.demoPath + s"/${clsname}_" + d.imagePath.substring(d.imagePath.lastIndexOf("/") + 1))
  }

  def imDetect(net: FasterRCNN[Float], d: ImageWithRoi)
  : (DenseMatrix[Float], DenseMatrix[Float]) = {
    imDetect(d, net.featureAndRpnNetWithCache, net.fastRcnnWithCache,
      net.param.A)
  }


  def main(args: Array[String]) {
    import com.intel.analytics.bigdl.mkl.MKL
    val param = parser.parse(args, PascolVocLocalParam()).get
    MKL.setNumThreads(param.nThread)
    var model: FasterRCNN[Float] = null
    val testDataSource = new PascolVocDataSource("2007", "testcode", param.folder, false)
    param.net match {
      case "vgg16" =>
        model = FasterVgg.model
      case "pvanet" =>
        model = FasterPvanet.model
    }
    testNet(model, testDataSource)
  }

}
