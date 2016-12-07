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

import javax.imageio.ImageIO

import breeze.linalg.DenseMatrix
import com.intel.analytics.bigdl.pvanet.model.FasterRcnnParam
import com.intel.analytics.bigdl.pvanet.utils.FileUtil
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.File

abstract class Imdb(val param: FasterRcnnParam) {
  val name: String
  val classes: Array[String]
  val imageIndex: Array[String]
  var roidb: Array[ImageWithRoi] = _

  /**
   * A roidb is a list of dictionaries, each with the following keys:
   * boxes
   * gt_overlaps
   * gt_classes
   * flipped
   */
  def getRoidb: Array[ImageWithRoi] = {
    if (roidb != null && roidb.length > 0) return roidb
    prepareRoidb()
    roidb
  }

  def getGroundTruth: Array[ImageWithRoi]


  def numClasses: Int = classes.length


  def numImages: Int = imageIndex.length

  def imagePathAt(i: Int): String

  def appendFlippedImages(): Unit = {
    throw new UnsupportedOperationException
  }

  def evaluateDetections(allBoxes: Array[Array[DenseMatrix[Float]]], outputDir: String): Unit

  /**
   * Enrich the imdb"s roidb by adding some derived quantities that
   * are useful for training. This function precomputes the maximum
   * overlap, taken over ground-truth boxes, between each ROI and
   * each ground-truth box. The class with maximum overlap is also
   * recorded.
   *
   */
  def prepareRoidb() = {
    val sizes = getImageSizes
    if (roidb == null || roidb.length == 0) roidb = getGroundTruth
    for (i <- 0 until numImages) {
      roidb(i).imagePath = imagePathAt(i)
      roidb(i).oriWidth = sizes(i)(0)
      roidb(i).oriHeight = sizes(i)(1)
      // max overlap with gt over classes(columns)
//      val maxRes = roidb(i).gt_overlaps.max(2) // gt class that had the max overlap
//      maxRes match {
//        case (max_overlaps, max_classes) =>
//          roidb(i).max_overlaps = Some(max_overlaps)
//          roidb(i).max_classes = Some(max_classes)
//      }
//      var zero_inds = None: Option[Iterable[(Float, Float)]]
      // sanity check, max overlap of 0 => class should be zero (background)
//      zero_inds = Some(roidb(i).max_overlaps.get.storage() zip roidb(i).max_classes.get.storage())
//      for ((maxOverlap, cls) <- zero_inds.get) {
//        if (maxOverlap == 0) assert(cls == 0)
//        if (maxOverlap > 0) assert(cls != 0)
//      }
    }
  }

  def getImageSizes: Array[Array[Int]] = {
    val cache_file = FileUtil.cachePath + "/" + name + "_image_sizes.pkl"
    if (FileUtil.existFile(cache_file)) {
      return File.load[Array[Array[Int]]](cache_file)
    }
    val sizes = Array.ofDim[Int](numImages, 2)
    for (i <- 0 until numImages) {
      val bimg = ImageIO.read(new java.io.File(imagePathAt(i)))
      sizes(i)(0) = bimg.getWidth()
      sizes(i)(1) = bimg.getHeight()
    }
    File.save(sizes, cache_file)
    sizes
  }

  def getImageSize(path: String): Array[Int] = {
    val bimg = ImageIO.read(new java.io.File(path))
    Array[Int](bimg.getWidth(), bimg.getHeight())
  }
}

case class ImageWithRoi(boxes: DenseMatrix[Float] = null,
  gt_classes: Tensor[Float] = null,
//  gt_overlaps: Tensor[Float] = null,
  flipped: Boolean = false) {
  var gtBoxes = None: Option[Tensor[Float]]
  var scaledImage: RGBImageOD = _
  var oriWidth = 0
  var oriHeight = 0
  var imagePath: String = null
  var imInfo = None: Option[Tensor[Float]]
//  var max_classes = None: Option[Tensor[Float]]
//  var max_overlaps = None: Option[Tensor[Float]]
}

object Imdb {
  /**
   * Get an imdb (image database) by name
   *
   */
  def getImdb(name: String, devkitPath: String, param: FasterRcnnParam): Imdb = {
    val items = name.split("_")
    if (items.length != 3) throw new Exception("dataset name error")
    if (items(0) == "voc") {
      new PascalVoc(items(1), items(2), devkitPath, param = param)
    } else {
      throw new UnsupportedOperationException("unsupported dataset")
    }
  }


}
