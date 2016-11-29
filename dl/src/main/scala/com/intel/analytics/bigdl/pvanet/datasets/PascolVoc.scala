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

package com.intel.analytics.bigdl.pvanet.datasets

import breeze.linalg.DenseMatrix
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.pvanet.Roidb.ImageWithRoi
import com.intel.analytics.bigdl.pvanet._
import com.intel.analytics.bigdl.pvanet.caffe.VggCaffeModel
import com.intel.analytics.bigdl.pvanet.layers._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Table, Timer}
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

  def select(tensor: Tensor[Float], rows: Array[Int], cols: Array[Int]): Tensor[Float] = {
    assert(tensor.nDimension() == 2)
    val out = Tensor[Float](rows.length, cols.length)
    rows.zip(Stream.from(1)).map(r => {
      cols.zip(Stream.from(1)).map(c => {
        out.setValue(r._2, c._2, tensor.valueAt(r._1, c._1))
      })
    })
    out
  }

  def hstack(clsBoxes: Array[Tensor[Float]], clsScores: Array[Float]): DenseMatrix[Float] = {
    if (clsBoxes == null || clsScores.length == 0) return new DenseMatrix[Float](0, 0)
    val cols = clsBoxes(0).nElement()
    val out = new DenseMatrix[Float](clsBoxes.length, cols + 1)
    clsBoxes.indices.foreach { i =>
      Range(0, cols).foreach(j => out(i, j) = clsBoxes(i).valueAt(j + 1))
      out(i, cols) = clsScores(i)
    }
    out
  }

  def visDetection(d: ImageWithRoi, clsname: String, clsDets: DenseMatrix[Float], thresh: Float = 0.3f): Unit = {
    Draw.vis(d.imagePath, clsname, clsDets,
      Config.demoPath + s"/${clsname}_" + d.imagePath.substring(d.imagePath.lastIndexOf("/") + 1))
  }

  def imDetect(d: ImageWithRoi): (DenseMatrix[Float], DenseMatrix[Float]) = {
    imDetect(d, VggCaffeModel.vgg16,
      VggCaffeModel.rpn,
      VggCaffeModel.fastRcnn
    )
  }

  def imDetect(d: ImageWithRoi,
    featureModel: Module[Tensor[Float], Tensor[Float], Float],
    rpnModel: Module[Tensor[Float], Table, Float],
    fastRcnn: Module[Table, Table, Float]): (DenseMatrix[Float], DenseMatrix[Float]) = {
    val imgTensor = ImageToTensor(d)

    val timer = new Timer
    if (Config.DEBUG) {
      println("===================================== start generating features")
      println(s"------- input size: ${imgTensor.size().mkString(",")}")
      timer.tic()
    }

    val featureOut = featureModel.forward(imgTensor)
    if (Config.DEBUG) {
      timer.toc()
      println(s"------- output size: ${featureOut.size().mkString(",")}")
      println(s"------- time: ${timer.diff / 1e9}s")
      println("===================================== start rpn ")
      timer.tic()
    }

    val clsRegOut = rpnModel.forward(featureOut)

    val rpnBboxPred = clsRegOut(2).asInstanceOf[Tensor[Float]]
    val rpnClsScore = clsRegOut(1).asInstanceOf[Tensor[Float]]

    val clsProc = new Sequential[Tensor[Float], Tensor[Float], Float]()
    clsProc.add(new Reshape2[Float](Array(2, -1), Some(false)))
    clsProc.add(new SoftMax[Float]())
    clsProc.add(new Reshape2[Float](Array(1, 2 * A, -1, rpnBboxPred.size(4)), Some(false)))
    val rpnClsScoreReshape = clsProc.forward(rpnClsScore)

    if (Config.DEBUG) {
      timer.toc()
      println(s"------- output size: cls (${rpnClsScoreReshape.size().mkString(",")}), " +
        s"bbox (${rpnBboxPred.size().mkString(",")})")
      println(s"------- time:  ${timer.diff / 1e9}s")

      println("=====================================  proposal")
      timer.tic()
    }
    val proposalInput = new Table
    proposalInput.insert(rpnClsScoreReshape)
    proposalInput.insert(rpnBboxPred)
    proposalInput.insert(d.imInfo.get)

    val propoal = new Proposal[Float](phase = 1)
    val proposalOut = propoal.forward(proposalInput)

    val rois = proposalOut(1).asInstanceOf[Tensor[Float]]

    if (Config.DEBUG) {
      timer.toc()
      println(s"------- output size: rois (${rois.size().mkString(",")})")
      println(s"------- time:  ${timer.diff / 1e9}s")

      println("=====================================  fast rcnn")
      timer.tic()
    }
    val propDecInput = new Table()
    propDecInput.insert(featureOut)
    propDecInput.insert(rois)

    val result = fastRcnn.forward(propDecInput)

    if (Config.DEBUG) {
      timer.toc()
      println(s"------- time: ${timer.diff / 1e9}s")
    }

    val scores = result(1).asInstanceOf[Tensor[Float]]
    val boxDeltas = result(2).asInstanceOf[Tensor[Float]]

    // post process
    // unscale back to raw image space
    if (Config.DEBUG) {
      println("=====================================  post process")
      timer.tic()
    }
    val boxes = rois.narrow(2, 2, 4).div(d.imInfo.get(2))
    // Apply bounding-box regression deltas
    var predBoxes = Bbox.bboxTransformInv(boxes.toBreezeMatrix(), boxDeltas.toBreezeMatrix())
    predBoxes = Bbox.clipBoxes(predBoxes, d.oriHeight, d.oriWidth)

    if (Config.DEBUG) {
      timer.toc()
      println(s"------- time: ${timer.diff / 1e9}s")
    }
    (scores.toBreezeMatrix(), predBoxes)
  }


  def testNet(dataSource: PascolVocDataSource, maxPerImage: Int = 100,
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
      val (scores: DenseMatrix[Float], boxes: DenseMatrix[Float]) = imDetect(d)
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

  def rpnTest(imageToTensor: ImageToTensor,
    model: Module[Tensor[Float], Table, Float], d: ImageWithRoi): Unit = {
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
    //    val regOut = res(2).asInstanceOf[Tensor[Float]]
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

  def main(args: Array[String]) {
    val param = parser.parse(args, PascolVocLocalParam()).get
    val testDataSource = new PascolVocDataSource("2007", "testcode1", param.folder, false)
    testNet(testDataSource)
  }

}
