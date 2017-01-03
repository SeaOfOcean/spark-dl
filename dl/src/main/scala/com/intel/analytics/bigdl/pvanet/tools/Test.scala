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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.pvanet.dataset._
import com.intel.analytics.bigdl.pvanet.model.Model.ModelType
import com.intel.analytics.bigdl.pvanet.model.{FasterRcnn, Model, Phase}
import com.intel.analytics.bigdl.pvanet.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Table, Timer}
import org.apache.log4j.Logger
import scopt.OptionParser

object Test {
  val logger = Logger.getLogger(getClass)

  def testNet(net: FasterRcnn, dataSource: ObjectDataSource,
    maxPerImage: Int = 100, thresh: Double = 0.05, vis: Boolean = false): Unit = {
    val imdb = dataSource.imdb
    val model = net.getTestModel
    val allBoxes: Array[Array[Tensor[Float]]] = {
      val out = new Array[Array[Tensor[Float]]](imdb.numClasses)
      Range(0, imdb.numClasses).foreach(x => {
        out(x) = new Array[Tensor[Float]](dataSource.size().toInt)
      })
      out
    }
    val imageScaler = new ImageScalerWithNormalizer(net.param)

    // timer
    val imDetectTimer = new Timer
    val miscTimer = new Timer

    val dataIter = dataSource.data(false)
    for (i <- 0 until imdb.numImages) {
      val d = dataIter.next()
      val imgWithRoi = imageScaler.apply(d)
      logger.info(s"process ${d.imagePath} ...............")

      imDetectTimer.tic()
      val (scores: Tensor[Float], boxes: Tensor[Float]) = imDetect(model, imgWithRoi)
      imDetectTimer.toc()

      miscTimer.tic()
      // skip j = 0, because it's the background class
      for (cls <- 1 until imdb.numClasses) {
        def getClsDet: Tensor[Float] = {
          val inds = (1 to scores.size(1)).filter(ind =>
            scores.valueAt(ind, cls + 1) > thresh).toArray
          if (inds.length == 0) return Tensor[Float](0, 5)
          val clsScores = TensorUtil.selectMatrix2(scores, inds, Array(cls + 1))
          val clsBoxes = TensorUtil.selectMatrix2(boxes,
            inds, Range(cls * 4 + 1, (cls + 1) * 4 + 1).toArray)

          var clsDets = TensorUtil.horzcat(clsBoxes, clsScores)
          val keep = Nms.nms(clsDets, net.param.NMS.toFloat)

          val detsNMSed = TensorUtil.selectMatrix(clsDets, keep, 1)

          if (net.param.BBOX_VOTE) {
            clsDets = Bbox.bboxVote(detsNMSed, clsDets)
          } else {
            clsDets = detsNMSed
          }
          clsDets
        }
        val clsDets = getClsDet

        if (vis) {
          visDetection(d, imdb.classes(cls), clsDets)
        }
        allBoxes(cls)(i) = clsDets
      }

      // Limit to max_per_image detections *over all classes*
      // todo: has not been tested
      if (maxPerImage > 0) {
        var imageScores = Array[Float]()
        for (j <- Range(1, imdb.numClasses)) {
          imageScores = imageScores ++
            TensorUtil.selectColAsArray(allBoxes(j)(i), allBoxes(j)(i).size(2))
        }
        if (imageScores.length > maxPerImage) {
          val imageThresh = imageScores.sortWith(_ < _)(imageScores.length - maxPerImage)
          for (j <- Range(1, imdb.numClasses)) {
            val box = allBoxes(j)(i)
            val keep = (1 to box.size(1)).filter(x =>
              box.valueAt(x, box.size(2)) >= imageThresh).toArray
            allBoxes(j)(i) = TensorUtil.selectMatrix(box, keep, 1)
          }
        }
      }
      miscTimer.toc()
      logger.info(s"im detect: $i/${imdb.numImages} " +
        s"${imDetectTimer.averageTime / 1e9}s ${miscTimer.averageTime / 1e9}s")
    }

    logger.info("Evaluating detections")

//    val outputDir = FileUtil.getOutputDir(imdb, "VGG16")
    imdb.evaluateDetections(allBoxes)
  }

  def imDetect(model: Module[Float], d: ImageWithRoi):
  (Tensor[Float], Tensor[Float]) = {
    val input = new Table
    input.insert(ImageToTensor(d))
    input.insert(d.imInfo.get)
    val result = model.forward(input).asInstanceOf[Table]

    val scores = result(1).asInstanceOf[Table](1).asInstanceOf[Tensor[Float]]
    val boxDeltas = result(1).asInstanceOf[Table](2).asInstanceOf[Tensor[Float]]
    val rois = result(2).asInstanceOf[Tensor[Float]]

    // post process
    // unscale back to raw image space
    val boxes = rois.narrow(2, 2, 4).div(d.imInfo.get.valueAt(3))
    // Apply bounding-box regression deltas
    var predBoxes = Bbox.bboxTransformInv(boxes, boxDeltas)
    predBoxes = Bbox.clipBoxes(predBoxes, d.oriHeight, d.oriWidth)
    (scores, predBoxes)
  }

  def visDetection(d: Roidb, clsname: String, clsDets: Tensor[Float],
    thresh: Float = 0.3f): Unit = {
    Draw.vis(d.imagePath, clsname, clsDets,
      FileUtil.demoPath + s"/${clsname}_"
        + d.imagePath.substring(d.imagePath.lastIndexOf("/") + 1))
  }

//  val model2caffePath = Map(
//    VGG16 -> ("/home/xianyan/objectRelated/faster_rcnn_models/VGG16/" +
//      "faster_rcnn_alt_opt/rpn_test.pt",
//      "/home/xianyan/objectRelated/faster_rcnn_models/" +
//        "VGG16_faster_rcnn_final.caffemodel"),
//    PVANET -> ("/home/xianyan/objectRelated/pvanet/full/test.pt",
//      "/home/xianyan/objectRelated/pvanet/full/test.model"))

  case class PascolVocLocalParam(
    folder: String = "data/VOCdevkit",
    net: ModelType = Model.VGG16,
    imageSet: String = "voc_2007_testcode",
    caffeDefPath: String = "data/faster_rcnn_models/VGG16/" +
      "faster_rcnn_alt_opt/rpn_test.pt",
    caffeModelPath: String = "data/faster_rcnn_models/VGG16/" +
      "VGG16_faster_rcnn_final.caffemodel",
    nThread: Int = 8)

  val parser = new OptionParser[PascolVocLocalParam]("Spark-DL PascolVoc Local Example") {
    head("Spark-DL PascolVoc Local Example")
    opt[String]('f', "folder")
      .text("where you put the PascolVoc data")
      .action((x, c) => c.copy(folder = x))
    opt[String]('n', "net")
      .text("net type : VGG16 | PVANET")
      .action((x, c) => c.copy(net = Model.withName(x)))
    opt[String]('i', "imageset")
      .text("imageset: voc_2007_test")
      .action((x, c) => c.copy(imageSet = x))
    opt[String]("caffeDefPath")
      .text("caffe prototxt")
      .action((x, c) => c.copy(caffeDefPath = x))
    opt[String]("caffeModelPath")
      .text("caffe model path")
      .action((x, c) => c.copy(caffeModelPath = x))
    opt[String]('t', "mkl thread number")
      .action((x, c) => c.copy(nThread = x.toInt))
  }

  def main(args: Array[String]) {
    import com.intel.analytics.bigdl.mkl.MKL
    val param = parser.parse(args, PascolVocLocalParam()).get

    val model = FasterRcnn(param.net, Phase.TEST, Some((param.caffeDefPath, param.caffeModelPath)))
    MKL.setNumThreads(param.nThread)
    val testDataSource = ObjectDataSource(param.imageSet, param.folder, useFlipped = false)
    testNet(model, testDataSource)
  }
}
