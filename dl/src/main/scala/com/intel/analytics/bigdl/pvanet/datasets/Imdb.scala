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

import java.io.FileInputStream
import javax.imageio.ImageIO

import breeze.linalg.DenseMatrix
import com.intel.analytics.bigdl.pvanet.model.FasterRcnnParam
import com.intel.analytics.bigdl.pvanet.utils.{FileUtil, MatrixUtil}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.File

abstract class Imdb(val param: FasterRcnnParam) {
  val name: String
  val classes: Array[String]
  var imageIndex: Array[String] = _
  var roidb: Array[Roidb] = _
  private var sizes: Array[Array[Int]] = _

  private def getWidth(i: Int) = {
    if (sizes == null) getImageSizes
    widths(i)
  }

  def widths: Array[Int] = {
    if (sizes == null) getImageSizes
    sizes(0)
  }

  def heights: Array[Int] = {
    if (sizes == null) getImageSizes
    sizes(1)
  }

  /**
   * A roidb is a list of dictionaries, each with the following keys:
   * boxes
   * gt_overlaps
   * gt_classes
   * flipped
   */
  def getRoidb: Array[Roidb] = {
    if (roidb != null && roidb.length > 0) return roidb
    roidb = loadRoidb
    if (param.USE_FLIPPED) {
      appendFlippedImages()
    }
    roidb
  }

  protected def loadRoidb: Array[Roidb]

  def numClasses: Int = classes.length

  def numImages: Int = imageIndex.length

  protected def imagePathAt(i: Int): String

  def imagePathFromIndex(index: String): String

  def appendFlippedImages(): Unit = {
    val isFlip = true
    for (i <- 0 until numImages) {
      val roi = roidb(i)
      val boxes = roi.boxes.copy
      val oldx1 = MatrixUtil.selectCol(boxes, 0).toDenseVector
      val oldx2 = MatrixUtil.selectCol(boxes, 2).toDenseVector
      (0 until boxes.rows).foreach(r => {
        boxes(r, 0) = getWidth(i) - oldx2(i) - 1
        boxes(r, 2) = getWidth(i) - oldx1(i) - 1
      })
      roidb :+ Roidb(roi.imagePath, boxes, roi.gtOverlaps, roi.gtClasses, isFlip)
    }
    val newImageIndex = new Array[String](imageIndex.length * 2)
    imageIndex.copyToArray(newImageIndex, 0)
    imageIndex.copyToArray(newImageIndex, imageIndex.length)
    imageIndex = newImageIndex
  }

  def evaluateDetections(allBoxes: Array[Array[DenseMatrix[Float]]], outputDir: String): Unit

  private def getImageSizes: Array[Array[Int]] = {
    if (sizes != null) return sizes
    val cache_file = FileUtil.cachePath + "/" + name + "_image_sizes.pkl"
    if (FileUtil.existFile(cache_file)) {
      sizes = File.load[Array[Array[Int]]](cache_file)
    } else {
      sizes = Array.ofDim[Int](2, numImages)
      for (i <- 0 until numImages) {
        val (width, height) = getImageSize(imagePathAt(i))
        sizes(0)(i) = width
        sizes(1)(i) = height
      }
      File.save(sizes, cache_file)
    }
    sizes
  }

  def loadAnnotation(index: String): Roidb

  /**
   * load image width and height without loading the entire image
   *
   * @param path image path
   * @return (width, height) tuple
   */
  def getImageSize(path: String): (Int, Int) = {
    var width = 0
    var height = 0
    val in = ImageIO.createImageInputStream(new FileInputStream(path))
    val readers = ImageIO.getImageReaders(in)
    if (readers.hasNext) {
      val reader = readers.next()
      reader.setInput(in)
      width = reader.getWidth(0)
      height = reader.getHeight(0)
      reader.dispose()
    }
    in.close()
    (width, height)
  }
}

case class Roidb(
  imagePath: String,
  boxes: DenseMatrix[Float] = null,
  gtClasses: Tensor[Float] = null,
  gtOverlaps: Tensor[Float] = null,
  flipped: Boolean = false) {
  var maxClasses = None: Option[Tensor[Float]]
  var maxOverlaps = None: Option[Tensor[Float]]
  var bboxTargets = None: Option[Tensor[Float]]
}

case class ImageWithRoi(
  oriWidth: Int,
  oriHeight: Int,
  gtBoxes: Option[Tensor[Float]],
  scaledImage: RGBImageOD,
  imInfo: Option[Tensor[Float]]) {
}

object Imdb {
  /**
   * Get an imdb (image database) by name
   *
   */
  def getImdb(name: String, param: FasterRcnnParam, devkitPath: Option[String] = None): Imdb = {
    val items = name.split("_")
    if (items.length != 3) throw new Exception("dataset name error")
    if (items(0) == "voc") {
      devkitPath match {
        case Some(path) => new PascalVoc(items(1), items(2), path, param = param)
        case _ => new PascalVoc(items(1), items(2), param = param)
      }
    } else {
      throw new UnsupportedOperationException("unsupported dataset")
    }
  }
}
