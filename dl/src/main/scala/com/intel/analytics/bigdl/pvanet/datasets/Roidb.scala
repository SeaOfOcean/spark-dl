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

import java.util.logging.Logger
import javax.imageio.ImageIO

import breeze.linalg.DenseMatrix
import com.intel.analytics.bigdl.pvanet.layers.AnchorTarget
import com.intel.analytics.bigdl.pvanet.utils.FileUtil
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.File

object Roidb {

  val logger = Logger.getLogger(this.getClass.getName)

  case class ImageWithRoi(boxes: DenseMatrix[Float] = null,
    gt_classes: Tensor[Float] = null,
    gt_overlaps: Tensor[Float] = null,
    flipped: Boolean = false,
    var seg_areas: Tensor[Float] = null) {

    var gtBoxes = None: Option[DenseMatrix[Float]]

    var scaledImage: RGBImageOD = _

    var oriWidth = 0
    var oriHeight = 0
    var imagePath = ""
    var imInfo = None: Option[Tensor[Float]]
    var max_classes = None: Option[Tensor[Float]]
    var max_overlaps = None: Option[Tensor[Float]]
    var anchorTarget = None: Option[AnchorTarget]

    override def toString: String = {
      "boxes: " + boxes +
        "\ngt_classes: " + gt_classes +
        "\ngt_overlaps: " + gt_overlaps +
        "\nflipped: " + flipped +
        "\nseg_areas: " + seg_areas +
        "\nimage: " + imagePath +
        "\nwidth: " + scaledImage.width() +
        "\nheight: " + scaledImage.height() +
        "\nmax_classes: " + max_classes +
        "\nmax_overlaps: " + max_overlaps
    }
  }

  def getImageSizes(imdb: Imdb): Array[Array[Int]] = {
    val cache_file = FileUtil.cachePath + "/" + imdb.name + "_image_sizes.pkl"
    if (FileUtil.existFile(cache_file)) {
      return File.load[Array[Array[Int]]](cache_file)
    }
    val sizes = Array.ofDim[Int](imdb.numImages, 2)
    for (i <- 0 until imdb.numImages) {
      val bimg = ImageIO.read(new java.io.File(imdb.imagePathAt(i)))
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

  /**
   * Enrich the imdb"s roidb by adding some derived quantities that
   * are useful for training. This function precomputes the maximum
   * overlap, taken over ground-truth boxes, between each ROI and
   * each ground-truth box. The class with maximum overlap is also
   * recorded.
   *
   */
  def prepareRoidb(imdb: Imdb): Imdb = {
    val sizes = getImageSizes(imdb)
    val roidb = imdb.roidb()
    for (i <- 0 until imdb.numImages) {
      roidb(i).imagePath = imdb.imagePathAt(i)
      roidb(i).oriWidth = sizes(i)(0)
      roidb(i).oriHeight = sizes(i)(1)
      // max overlap with gt over classes(columns)
      val maxRes = roidb(i).gt_overlaps.max(2) // gt class that had the max overlap
      maxRes match {
        case (max_overlaps, max_classes) =>
          roidb(i).max_overlaps = Some(max_overlaps)
          roidb(i).max_classes = Some(max_classes)
      }
      var zero_inds = None: Option[Iterable[(Float, Float)]]
      // sanity check, max overlap of 0 => class should be zero (background)
      zero_inds = Some(roidb(i).max_overlaps.get.storage() zip roidb(i).max_classes.get.storage())
      for ((maxOverlap, cls) <- zero_inds.get) {
        if (maxOverlap == 0) assert(cls == 0)
        if (maxOverlap > 0) assert(cls != 0)
      }
    }
    imdb
  }

}
