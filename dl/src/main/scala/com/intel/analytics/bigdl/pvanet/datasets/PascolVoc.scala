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
import com.intel.analytics.bigdl.pvanet.layers.{Proposal, Reshape2, SmoothL1Criterion2, SoftmaxWithCriterion}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table
import scopt.OptionParser

import scala.io.Source

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
    for (i <- 0 until clsBoxes.length) {
      Range(0, cols).foreach(j => out(i, j) = clsBoxes(i).valueAt(j + 1))
      out(i, cols) = clsScores(i)
    }
    out
  }

  def visDetection(d: ImageWithRoi, clsname: String, clsDets: DenseMatrix[Float]): Unit = {
    throw new UnsupportedOperationException()
  }

  def loadFeatures(s: String, size: Array[Int]): Tensor[Float] = {
    Tensor(Storage(Source.fromFile(s).getLines().map(x => x.toFloat).toArray)).reshape(size)
  }

  def main(args: Array[String]) {
    val param = parser.parse(args, new PascolVocLocalParam()).get
    // parser.parse(args, new PascolVocLocalParam()).map(param => {
    val year = "2007"

    val validationDataSource = new PascolVocDataSource(year, "testcode", param.folder, false)
    val trainDataSource = new PascolVocDataSource(year, imageSet = "train", param.folder, false)

    val imageScaler = ImageScalerAndMeanSubstractor(validationDataSource)
    val imageToTensor = new ImageToTensor(batchSize = 1)

    val data = validationDataSource -> imageScaler

    //      val model = param.net match {
    //        case "vgg" => VggCaffeModel.Vgg_16_RPN
    //        case _ => throw new IllegalArgumentException
    //      }

    // todo: need refactor
    val imdb = validationDataSource.imdb
    val allBoxes: Array[Array[DenseMatrix[Float]]] = {
      var out = new Array[Array[DenseMatrix[Float]]](imdb.numClasses)
      Range(0, imdb.numClasses).foreach(x => {
        var arr = new Array[DenseMatrix[Float]](imdb.numImages)
        out(x) = arr
      })
      out
    }
    var start = 0L
    var end = 0L
    def imDetect(d: ImageWithRoi): (DenseMatrix[Float], DenseMatrix[Float]) = {
      val imgTensor = imageToTensor(d)
//      println("===================================== start generating features")
//      println(s"------- input size: ${imgTensor.size().mkString(",")}")
//      start = System.nanoTime()
//      val featureModel = VggCaffeModel.vgg16
//      val featureOut = featureModel.forward(imgTensor)
//      println(s"------- output size: ${featureOut.size().mkString(",")}")
//      println(s"------- time: ${(System.nanoTime() - start) / 1e9}s")
//      println()

      val featureOut = loadFeatures("/home/xianyan/code/intel/pvanet/roi_poolbottom0.dat",
        Array(1, 512, 54, 38))


      println("===================================== start rpn ")
      start = System.nanoTime()
      val rpnModel = VggCaffeModel.rpn
      val clsRegOut = rpnModel.forward(featureOut)

      val rpnBboxPred = clsRegOut(2).asInstanceOf[Tensor[Float]]
      val rpnClsScore = clsRegOut(1).asInstanceOf[Tensor[Float]]
      val clsProc = new Sequential[Tensor[Float], Tensor[Float], Float]()
      clsProc.add(new Reshape2[Float](Array(2, -1), Some(false)))
      clsProc.add(new SoftMax[Float]())
      clsProc.add(new Reshape2[Float](Array(1, 2 * A, -1, rpnBboxPred.size(4)), Some(false)))
      val rpnClsScoreReshape = clsProc.forward(rpnClsScore)
      println(s"------- output size: cls (${rpnClsScoreReshape.size().mkString(",")}), " +
        s"bbox (${rpnBboxPred.size().mkString(",")})")
      println(s"------- time:  ${(System.nanoTime() - start) / 1e9}s")

      println("=====================================  proposal")
      start = System.nanoTime()
      val proposalInput = new Table
      proposalInput.insert(rpnClsScoreReshape)
      proposalInput.insert(rpnBboxPred)
      proposalInput.insert(d.imInfo.get)

      val propoal = new Proposal[Float](phase = 1)
      val proposalOut = propoal.forward(proposalInput)

      val rois = proposalOut(1).asInstanceOf[Tensor[Float]]
      println(s"------- output size: rois (${rois.size().mkString(",")})")
      println(s"------- time:  ${(System.nanoTime() - start) / 1e9}s")

      println("=====================================  fast rcnn")
      start = System.nanoTime()
      val propDecInput = new Table()
//        propDecInput.insert(featureOut.resize(featureOut.size(2),
// featureOut.nElement() / featureOut.size(2)))
//        propDecInput.insert(rois.resize(rois.size(2), rois.nElement() / rois.size(2)))
      propDecInput.insert(featureOut)
      propDecInput.insert(rois)

//      println(s"featureOut: ${featureOut.size().mkString(", ")}")
//      println(s"rois: ${proposalOut(1).asInstanceOf[Tensor[Float]].size().mkString(",")}")

      val propDecModel = VggCaffeModel.fastRcnn

      val result = propDecModel.forward(propDecInput)
      println(s"------- time: ${(System.nanoTime() - start) / 1e9}s")

      val scores = result(1).asInstanceOf[Tensor[Float]]
      val boxDeltas = result(2).asInstanceOf[Tensor[Float]]


      // post process
      // unscale back to raw image space
      println("=====================================  post process")
      val boxes = rois.narrow(2, 2, 4).div(d.imInfo.get(2))
      //      println(s"------- rois: ${rois.size().mkString(",")}")
//      println(s"------- boxes: ${boxes.size().mkString(",")}")
//      println(s"------- boxDeltas: ${boxDeltas.size().mkString(",")}")
// Apply bounding-box regression deltas
var predBoxes = Bbox.bboxTransformInv(boxes.toBreezeMatrix(), boxDeltas.toBreezeMatrix())
      predBoxes = Bbox.clipBoxes(predBoxes, d.height(), d.width())
      println(s"------- scores: (${scores.size().mkString(",")})")
      println(s"------- predBoxes: (${predBoxes.rows}, ${predBoxes.cols})")
      (scores.toBreezeMatrix(), predBoxes)
    }

    for (i <- 0 until imdb.numImages) {
      val d = data.next()
      println(s"process ${d.imagePath} ...............")

      val (scores: DenseMatrix[Float], boxes: DenseMatrix[Float]) = imDetect(d)
//      MatrixUtil.printSelectMatrix("scores", scores)
//      MatrixUtil.printSelectMatrix("boxes", boxes)

      // todo: parameter of testNet
      val thresh = 0.5
      val vis = false
      // skip j = 0, because it's the background class
      for (j <- 1 until imdb.numClasses) {
        def getClsDet: DenseMatrix[Float] = {
          val inds = Range(0, scores.rows).filter(ind => scores(ind, j) > thresh).toArray
          if (inds.length == 0) return new DenseMatrix[Float](0, 5)
          println(s"class ${j}, inds: ${inds.mkString(",")}")
          val clsScores = MatrixUtil.selectMatrix2(scores, inds, Array(j))
          val clsBoxes = MatrixUtil.selectMatrix2(boxes,
            inds, Range(j * 4, (j + 1) * 4).toArray)

          //          println(s"clsScores: ${clsScores.rows}, ${clsScores.cols}")
//          println(s"bbox: ${clsBoxes.rows}, ${clsBoxes.cols}")
var clsDets = DenseMatrix.horzcat(clsBoxes, clsScores)
          println("becore nms====================", clsDets.rows, clsDets.cols)
          val keep = Nms.nms(clsDets, Config.TEST.NMS.toFloat)

          val detsNMSed = MatrixUtil.selectMatrix(clsDets, keep, 0)

          if (Config.TEST.BBOX_VOTE) {
            clsDets = Bbox.bboxVote(detsNMSed, clsDets)
          }
          else {
            clsDets = detsNMSed
          }
          clsDets
        }
        val clsDets = getClsDet
        if (vis) {
          visDetection(d, imdb.classes(j - 1), clsDets)
        }
        // this i need to be modified to image index
        allBoxes(j)(i) = clsDets
      }
      // todo:
      val maxPerImage = 100
      // Limit to max_per_image detections *over all classes*
      if (maxPerImage > 0) {
        // todo
      }
//      println(result)

    }

    println("Evaluating detections")

    val outputDir = Config.getOutputDir(imdb, "VGG16")
    imdb.evaluateDetections(allBoxes, outputDir)
//    })
  }


//  def fullModelTest(imageToTensor: ImageToTensor,
//    model: Module[Tensor[Float], Table, Float], d: ImageWithRoi): Unit = {
//    // get rpn_cls_score and rpn_bbox_pred
//    val res = model.forward(imageToTensor(d))
//    val rpnClsScore = res(1).asInstanceOf[Tensor[Float]]
//    val clsProc = new Sequential[Tensor[Float], Tensor[Float], Float]()
//    clsProc.add(new Reshape2[Float](Array(0, 2, -1, 0)))
//    clsProc.add(new SoftMax[Float]())
//    clsProc.add(new Reshape2[Float](Array(0, 2 * A, -1, 0)))
//    val rpn_bbox_pred = res(2).asInstanceOf[Tensor[Float]]
//
//    val proposalInput = new Table
//    proposalInput.insert(rpnClsScore)
//    proposalInput.insert(rpn_bbox_pred)
//    proposalInput.insert(Tensor(Storage(d.imInfo.get)))
//
//    val roiPoolInput = new ConcatTable[Table, Float]()
//    roiPoolInput
//
//  }


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
