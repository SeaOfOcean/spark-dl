/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.pvanet.dataset

import java.awt.Color
import java.awt.image.{BufferedImage, DataBufferByte}
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.pvanet.model.FasterRcnnParam
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.util.Random

object ImageScalerWithNormalizer {
  def apply(param: FasterRcnnParam): ImageScalerWithNormalizer =
    new ImageScalerWithNormalizer(param)
}

class ImageScalerWithNormalizer(param: FasterRcnnParam)
  extends Transformer[Roidb, ImageWithRoi] {
  def byte2Float(x: Byte): Float = x & 0xff

  def apply(data: Roidb): ImageWithRoi = {
    val scaleTo = param.SCALES(Random.nextInt(param.SCALES.length))
    val img = ImageIO.read(new java.io.File(data.imagePath))
    val imSizeMin = Math.min(img.getWidth, img.getHeight)
    val imSizeMax = Math.max(img.getWidth, img.getHeight)
    var im_scale = scaleTo.toFloat / imSizeMin.toFloat
    // Prevent the biggest axis from being more than MAX_SIZE
    if (Math.round(im_scale * imSizeMax) > param.MAX_SIZE) {
      im_scale = param.MAX_SIZE.toFloat / imSizeMax.toFloat
    }

    val im_scale_x = (Math.floor(img.getHeight * im_scale / param.SCALE_MULTIPLE_OF) *
      param.SCALE_MULTIPLE_OF / img.getHeight).toFloat
    val im_scale_y = (Math.floor(img.getWidth * im_scale / param.SCALE_MULTIPLE_OF) *
      param.SCALE_MULTIPLE_OF / img.getWidth).toFloat

    val scaledImage1: java.awt.Image =
      img.getScaledInstance((im_scale_y * img.getWidth).toInt,
        (im_scale_x * img.getHeight()).toInt, java.awt.Image.SCALE_SMOOTH)

    val imageBuff: BufferedImage =
      new BufferedImage((im_scale_y * img.getWidth).toInt, (im_scale_x * img.getHeight()).toInt,
        BufferedImage.TYPE_3BYTE_BGR)
    imageBuff.getGraphics.drawImage(scaledImage1, 0, 0, new Color(0, 0, 0), null)
    val pixels: Array[Float] = imageBuff.getRaster.getDataBuffer
      .asInstanceOf[DataBufferByte].getData.map(x => byte2Float(x))
    require(pixels.length % 3 == 0)
    // mean subtract
    val meanPixels = pixels.zipWithIndex.map(x =>
      (pixels(x._2) - FasterRcnnParam.PIXEL_MEANS.head.head(x._2 % 3)).toFloat
    )

    val scaledImage = new RGBImageOD(meanPixels, imageBuff.getWidth, imageBuff.getHeight)
    val imScales = Tensor(Storage(Array(im_scale_x, im_scale_y, im_scale_x, im_scale_y)))
    val imInfo = Some(Tensor(Storage(
      Array(imageBuff.getHeight(), imageBuff.getWidth, im_scale_x))))

    var gtBoxes: Option[Tensor[Float]] = None

    if (data.gtClasses != null) {
      val gt_inds = data.gtClasses.storage().array().zip(Stream from 1)
        .filter(x => x._1 != 0).map(x => x._2)
      val gt_boxes = Tensor[Float](gt_inds.length, 5)
      gt_inds.zip(Stream from 1).foreach(ind => {
        val scaled = data.boxes.select(1, ind._1).narrow(1, 1, 4).clone().cmul(imScales)
        gt_boxes.setValue(ind._2, 1, scaled.valueAt(1))
        gt_boxes.setValue(ind._2, 2, scaled.valueAt(2))
        gt_boxes.setValue(ind._2, 3, scaled.valueAt(3))
        gt_boxes.setValue(ind._2, 4, scaled.valueAt(4))
        gt_boxes.setValue(ind._2, 5, data.gtClasses.valueAt(ind._1))
      })
      gtBoxes = Some(gt_boxes)
    }
    ImageWithRoi(img.getWidth, img.getHeight, gtBoxes, scaledImage, imInfo)
  }

  override def apply(prev: Iterator[Roidb]): Iterator[ImageWithRoi] = {
    // generate a serious of random scales
    prev.map(data => apply(data))
  }
}
