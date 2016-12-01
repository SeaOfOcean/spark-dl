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
import com.intel.analytics.bigdl.pvanet.datasets.Roidb.ImageWithRoi
import com.intel.analytics.bigdl.pvanet.datasets.{ImageScalerAndMeanSubstractor}
import com.intel.analytics.bigdl.pvanet.model._
import com.intel.analytics.bigdl.pvanet.utils.{Bbox, FileUtil, MatrixUtil, Nms}
import com.intel.analytics.bigdl.utils.Timer
import scopt.OptionParser

object Demo {

  val classes = Array[String](
    "__background__", // always index 0
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
  )

  case class PascolVocLocalParam(folder: String = "/home/xianyan/objectRelated/Pedestrain_1",
    net: String = "vgg")

  private val parser = new OptionParser[PascolVocLocalParam]("Object Detection Local Example") {
    head("Spark-DL PascolVoc Local Example")
    opt[String]('f', "demo image folder")
      .text("where you put the demo image data")
      .action((x, c) => c.copy(folder = x))
    opt[String]('n', "net")
      .text("net type : vgg | pvanet")
      .action((x, c) => c.copy(net = x.toLowerCase))
  }

  

  def main(args: Array[String]): Unit = {
    val param = parser.parse(args, PascolVocLocalParam()).get
    val imgNames = Array("1.jpg", "20.jpg", "30.jpg", "40.jpg",
      "50.jpg", "60.jpg", "70.jpg", "80.jpg", "90.jpg", "100.jpg")
    var net: FasterRCNN[Float] = null
    param.net match {
      case "vgg" => net = FasterVgg.model()
      case "pvanet" => net = FasterPvanet.model()
    }
    
    val imageScaler = new ImageScalerAndMeanSubstractor(null, param = net.param)
    
    imgNames.foreach(imaName => {
      val img = ImageWithRoi()
      img.imagePath = param.folder + "/" + imaName
      println(s"process ${img.imagePath} ...")
      val scaledImage = imageScaler.apply(img)
      val timer = new Timer
      timer.tic()
      val (scores: DenseMatrix[Float], boxes: DenseMatrix[Float]) =
        Test.imDetect(net, scaledImage)
      timer.toc()
      println(s"Detection took ${"%.3f".format(timer.totalTime / 1e9)}s " +
        s"for ${boxes.rows} object proposals")
      // Visualize detections for each class
      val CONF_THRESH = 0.8f
      val NMS_THRESH = 0.3f
      for (j <- 1 until classes.length) {
        def getClsDet: DenseMatrix[Float] = {
          val inds = Range(0, scores.rows).toArray
          if (inds.length == 0) return new DenseMatrix[Float](0, 5)
          val clsScores = MatrixUtil.selectMatrix2(scores, inds, Array(j))
          val clsBoxes = MatrixUtil.selectMatrix2(boxes,
            inds, Range(j * 4, (j + 1) * 4).toArray)

          var clsDets = DenseMatrix.horzcat(clsBoxes, clsScores)
          val keep = Nms.nms(clsDets, NMS_THRESH)

          val detsNMSed = MatrixUtil.selectMatrix(clsDets, keep, 0)

          if (net.param.BBOX_VOTE) {
            clsDets = Bbox.bboxVote(detsNMSed, clsDets)
          } else {
            clsDets = detsNMSed
          }
          clsDets
        }
        val clsDets = getClsDet
        Test.visDetection(scaledImage, classes(j), clsDets, thresh = CONF_THRESH)
      }
    })
  }
}
