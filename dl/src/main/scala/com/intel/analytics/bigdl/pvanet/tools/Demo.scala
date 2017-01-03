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

import com.intel.analytics.bigdl.pvanet.dataset.{ImageScalerWithNormalizer, Roidb}
import com.intel.analytics.bigdl.pvanet.model.Model._
import com.intel.analytics.bigdl.pvanet.model._
import com.intel.analytics.bigdl.pvanet.utils.{Bbox, Nms, TensorUtil}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Timer
import org.apache.log4j.Logger
import scopt.OptionParser

object Demo {
  val logger = Logger.getLogger(getClass)

  val classes = Array[String](
    "__background__", // always index 0
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
  )

  case class PascolVocLocalParam(folder: String = "data/Pedestrain_1",
    net: ModelType = Model.VGG16, nThread: Int = 4,
    caffeDefPath: String = "data/faster_rcnn_models/VGG16/" +
      "faster_rcnn_alt_opt/rpn_test.pt",
    caffeModelPath: String = "data/faster_rcnn_models/VGG16/" +
      "VGG16_faster_rcnn_final.caffemodel")

  private val parser = new OptionParser[PascolVocLocalParam]("Spark-DL PascolVoc Local Example") {
    head("Spark-DL PascolVoc Local Example")
    opt[String]('f', "folder")
      .text("where you put the PascolVoc data")
      .action((x, c) => c.copy(folder = x))
    opt[String]('n', "net")
      .text("net type : VGG16 | PVANET")
      .action((x, c) => c.copy(net = Model.withName(x)))
    opt[String]("caffeDefPath")
      .text("caffe prototxt")
      .action((x, c) => c.copy(caffeDefPath = x))
    opt[String]("caffeModelPath")
      .text("caffe model path")
    opt[String]('t', "mkl thread number")
      .action((x, c) => c.copy(nThread = x.toInt))
  }

  def main(args: Array[String]): Unit = {
    val param = parser.parse(args, PascolVocLocalParam()).get
    val imgNames = Array("1.jpg", "20.jpg", "30.jpg", "40.jpg",
      "50.jpg", "60.jpg", "70.jpg", "80.jpg", "90.jpg", "100.jpg")
    val net = FasterRcnn(param.net, caffeModel = Some((param.caffeDefPath, param.caffeModelPath)))

    val model = net.getTestModel

    val imageScaler = new ImageScalerWithNormalizer(net.param)

    imgNames.foreach(imaName => {
      val img = Roidb(param.folder + "/" + imaName)
      logger.info(s"process ${img.imagePath} ...")
      val scaledImage = imageScaler.apply(img)
      val timer = new Timer
      timer.tic()
      val (scores: Tensor[Float], boxes: Tensor[Float])
      = Test.imDetect(model, scaledImage)
      timer.toc()
      logger.info(s"Detection took ${"%.3f".format(timer.totalTime / 1e9)}s " +
        s"for ${boxes.size(1)} object proposals")
      // Visualize detections for each class
      val CONF_THRESH = 0.8f
      val NMS_THRESH = 0.3f
      for (j <- 1 until classes.length) {
        def getClsDet: Tensor[Float] = {
          val inds = (1 to scores.size(1)).toArray
          if (inds.length == 0) return Tensor[Float](0, 5)
          val clsScores = TensorUtil.selectMatrix2(scores, inds, Array(j + 1))
          val clsBoxes = TensorUtil.selectMatrix2(boxes,
            inds, (j * 4 + 1 to (j + 1) * 4).toArray)

          var clsDets = TensorUtil.horzcat(clsBoxes, clsScores)
          val keep = Nms.nms(clsDets, NMS_THRESH)

          val detsNMSed = TensorUtil.selectMatrix(clsDets, keep, 1)

          if (net.param.BBOX_VOTE) {
            clsDets = Bbox.bboxVote(detsNMSed, clsDets)
          } else {
            clsDets = detsNMSed
          }
          clsDets
        }
        val clsDets = getClsDet
        Test.visDetection(img, classes(j), clsDets, thresh = CONF_THRESH)
      }
    })
  }
}
