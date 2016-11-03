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

package com.intel.analytics.sparkdl.pvanet.datasets

import java.awt.Color
import java.awt.image.{BufferedImage, DataBufferByte}
import javax.imageio.ImageIO

import breeze.linalg.{DenseMatrix, DenseVector}
import com.intel.analytics.sparkdl.dataset.{DataSource, Transformer}
import com.intel.analytics.sparkdl.pvanet.Roidb.ImageWithRoi
import com.intel.analytics.sparkdl.pvanet.{Config, Roidb}
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}

import scala.util.Random


class PascolVocDataSource(year: String = "2007", imageSet: String, devkitPath: String = Config.DATA_DIR + "/VOCdevkit", looped: Boolean = true)
  extends DataSource[ImageWithRoi] {

  val dataset: Imdb = new PascalVoc(year, imageSet, devkitPath)

  val data: Array[ImageWithRoi] = Roidb.prepareRoidb(dataset).roidb()

  //permutation of the data index
  var perm: Array[Int] = Array()

  private var offset = 0


  override def reset(): Unit = {
    offset = 0
  }

  override def finished(): Boolean = (offset >= data.length)

  override def hasNext: Boolean = {
    if (looped) {
      true
    } else {
      offset < data.length
    }
  }

  override def total(): Long = data.length

  override def shuffle(): Unit = {
    def shuffleWithAspectGrouping(widths: Array[Int], heights: Array[Int]): Array[Int] = {
      val horz = (widths zip heights).map(x => x._1 >= x._2)
      val vert = horz.map(x => !x)
      val horz_inds = horz.zipWithIndex.filter(x => x._1).map(x => x._2)
      val vert_inds = vert.zipWithIndex.filter(x => x._1).map(x => x._2)
      val indsArr = (Random.shuffle(horz_inds.toSeq) ++ Random.shuffle(vert_inds.toSeq)).toArray
      val row_perm = Random.shuffle(Seq.range(0, indsArr.size / 2))
      var newInds = new Array[Int](indsArr.length)
      row_perm.zipWithIndex.foreach(r => {
        newInds(r._1 * 2) = indsArr(r._2 * 2)
        newInds(r._1 * 2 + 1) = indsArr(r._2 * 2 + 1)
      })
      newInds
    }
    if (Config.TRAIN.ASPECT_GROUPING) {
      val widths = data.map(r => r.oriWidth)
      val heights = data.map(r => r.oriHeight)
      perm = shuffleWithAspectGrouping(widths, heights)
    } else {
      perm = Random.shuffle(Array.range(0, data.length).toSeq).toArray
    }
  }


  override def next(): ImageWithRoi = {
    val r = (if (looped) (offset % data.length) else offset)
    offset += 1
    data(perm(r))
  }

}

/**
  *
  */
object ImageSizeUniformer extends Transformer[ImageWithRoi, ImageWithRoi] {
  override def transform(prev: Iterator[ImageWithRoi]): Iterator[ImageWithRoi] = {
    //this is standard python version
    //    val maxWidth = prev.maxBy(_.width()).width()
    //    val maxHeight = prev.maxBy(_.height()).height()
    //this is not standard python version. But actually this may not be used
    //because the image batch is 1
    val maxWidth = Config.TRAIN.MAX_SIZE
    val maxHeight = Config.TRAIN.MAX_SIZE
    val numImages = prev.length
    prev.map(data => {
      data.scaledImage = Some(new RGBImageOD(maxWidth, maxHeight).copyContent(data.scaledImage.get))
      data
    })
  }
}

class ImageScalerAndMeanSubstractor(dataSource: PascolVocDataSource) extends Transformer[ImageWithRoi, ImageWithRoi] {
  def byte2Float(x: Byte): Float = x & 0xff

  dataSource.shuffle()

  def apply(data: ImageWithRoi, scaleTo: Int): ImageWithRoi = {
    val img = ImageIO.read(new java.io.File(data.imagePath))
    val imSizeMin = Math.min(img.getWidth, img.getHeight())
    val imSizeMax = Math.max(img.getWidth, img.getHeight())
    var im_scale = scaleTo.toFloat / imSizeMin.toFloat
    // Prevent the biggest axis from being more than MAX_SIZE
    if (Math.round(im_scale * imSizeMax) > Config.TRAIN.MAX_SIZE) {
      im_scale = Config.TRAIN.MAX_SIZE.toFloat / imSizeMax.toFloat
    }

    val im_scale_x = (Math.floor(img.getHeight * im_scale / Config.TRAIN.SCALE_MULTIPLE_OF) * Config.TRAIN.SCALE_MULTIPLE_OF / img.getHeight).toFloat
    val im_scale_y = (Math.floor(img.getWidth * im_scale / Config.TRAIN.SCALE_MULTIPLE_OF) * Config.TRAIN.SCALE_MULTIPLE_OF / img.getWidth).toFloat

    val scaledImage: java.awt.Image =
      img.getScaledInstance((im_scale_y * img.getWidth).toInt, (im_scale_x * img.getHeight()).toInt, java.awt.Image.SCALE_SMOOTH)

    val imageBuff: BufferedImage =
      new BufferedImage((im_scale_y * img.getWidth).toInt, (im_scale_x * img.getHeight()).toInt, BufferedImage.TYPE_3BYTE_BGR)
    imageBuff.getGraphics.drawImage(scaledImage, 0, 0, new Color(0, 0, 0), null)
    val pixels: Array[Float] = (imageBuff.getRaster.getDataBuffer.asInstanceOf[DataBufferByte]).getData.map(x => byte2Float(x))
    require(pixels.length % 3 == 0)
    // mean subtract
    val meanPixels = pixels.zipWithIndex.map(x =>
      (pixels(x._2) - Config.PIXEL_MEANS(0)(0)(x._2 % 3)).toFloat
    )

    data.scaledImage = Some(new RGBImageOD(meanPixels, imageBuff.getWidth, imageBuff.getHeight))
    val imScales = Array(im_scale_x, im_scale_y, im_scale_x, im_scale_y)
    data.imInfo = Some(imScales)

    val gt_inds = data.gt_classes.storage().array().zipWithIndex.filter(x => x._1 != 0).map(x => x._2)
    var gt_boxes = new DenseMatrix[Float](gt_inds.length, 5)
    gt_inds.zipWithIndex.foreach(ind => {
      val tmp = data.boxes(ind._1, 0 until 4)
      val scaled = data.boxes(ind._1, 0 until 4).t :* DenseVector(imScales)
      gt_boxes(ind._2, 0) = scaled(0)
      gt_boxes(ind._2, 1) = scaled(1)
      gt_boxes(ind._2, 2) = scaled(2)
      gt_boxes(ind._2, 3) = scaled(3)
      gt_boxes(ind._2, 4) = data.gt_classes.valueAt(ind._1 + 1)
    })
    data.gt_boxes = Some(gt_boxes)
    data
  }

  override def transform(prev: Iterator[ImageWithRoi]): Iterator[ImageWithRoi] = {
    // generate a serious of random scales
    val scaleLenth = Config.TRAIN.SCALES.length
    prev.map(data => apply(data, Config.TRAIN.SCALES(Random.nextInt(scaleLenth))))
  }
}


class ToTensor(batchSize: Int = 1) extends Transformer[ImageWithRoi, (Tensor[Float], Tensor[Float],
  Tensor[Float])] {
  require(batchSize == 1)

  private def copyImage(img: RGBImageOD, storage: Array[Float], offset: Int): Unit = {
    val content = img.content
    val frameLength = img.width() * img.height()
    require(content.length == frameLength * 3)
    var j = 0
    while (j < frameLength) {
      storage(offset + j) = content(j * 3)
      storage(offset + j + frameLength) = content(j * 3 + 1)
      storage(offset + j + frameLength * 2) = content(j * 3 + 2)
      j += 1
    }
  }

  override def transform(prev: Iterator[ImageWithRoi]): Iterator[(Tensor[Float], Tensor[Float], Tensor[Float])] = {
    new Iterator[(Tensor[Float], Tensor[Float], Tensor[Float])] {
      private val featureTensor: Tensor[Float] = Tensor[Float]()
      private val labelTensor: Tensor[Float] = Tensor[Float]()
      private val roiLabelTensor: Tensor[Float] = Tensor[Float]()
      private var featureData: Array[Float] = null
      private var labelData: Array[Float] = null
      private var roiLabelData: Array[Float] = null
      private var width = 0
      private var height = 0

      override def hasNext: Boolean = prev.hasNext

      override def next(): (Tensor[Float], Tensor[Float], Tensor[Float]) = {
        if (prev.hasNext) {
          var i = 0
          var k = 0
          val imgWithRoi = prev.next()
          if (featureData == null) {
            featureData = new Array[Float](batchSize * 3 * imgWithRoi.scaledImage.get.height() * imgWithRoi.scaledImage.get.width())
            roiLabelData = new Array[Float](batchSize * 3 * 4 * imgWithRoi.anchorTarget.get.labels.length)
            labelData = new Array[Float](batchSize * imgWithRoi.anchorTarget.get.labels.length)
            height = imgWithRoi.scaledImage.get.height()
            width = imgWithRoi.scaledImage.get.width()
          }
          copyImage(imgWithRoi.scaledImage.get, featureData, i * imgWithRoi.scaledImage.get.width() * imgWithRoi.scaledImage.get.height() * 3)

          for (r <- 0 until imgWithRoi.anchorTarget.get.bboxTargets.rows) {
            labelData(r) = imgWithRoi.anchorTarget.get.labels(r)
            for (c <- 0 until imgWithRoi.anchorTarget.get.bboxTargets.cols) {
              roiLabelData(k) = imgWithRoi.anchorTarget.get.bboxTargets.valueAt(r, c)
              k += 1
              roiLabelData(k) = imgWithRoi.anchorTarget.get.bboxInsideWeights.valueAt(r, c)
              k += 1
              roiLabelData(k) = imgWithRoi.anchorTarget.get.bboxOutsideWeights.valueAt(r, c)
              k += 1
            }
          }
          featureTensor.set(Storage[Float](featureData),
            storageOffset = 1, sizes = Array(1, 3, height, width))
          labelTensor.set(Storage[Float](labelData),
            storageOffset = 1, sizes = Array(1, labelData.length))
          roiLabelTensor.set(Storage[Float](roiLabelData),
            storageOffset = 1, sizes = Array(1, 3, roiLabelData.length / 3))
          if (Config.DEBUG) {
            println("<-------------to tensor result------------->")
            println("image size: (" + featureTensor.size().mkString(", ") + ")")
            println("label tensor size: (" + labelTensor.size().mkString(", ") + ")")
            println("roiLabel tensor size: (" + roiLabelTensor.size().mkString(", ") + ")")
          }
          (featureTensor, labelTensor, roiLabelTensor)
        } else {
          null
        }
      }
    }
  }
}


class RGBImageOD(protected var data: Array[Float], protected var _width: Int,
                 protected var _height: Int) {

  def width(): Int = _width

  def height(): Int = _height

  def content: Array[Float] = data

  def this() = this(new Array[Float](0), 0, 0)

  def this(_width: Int, _height: Int) =
    this(new Array[Float](_width * _height * 3), _width, _height)

  def copyContent(other: RGBImageOD): RGBImageOD = {
    if (this.data.length < this._width * this._height * 3) {
      this.data = new Array[Float](this._width * this._height * 3)
    }

    var k = 0
    for (i <- 0 until other._height) {
      for (j <- 0 until other._width) {
        this.data((i * width() + j) * 3) = other.data(k)
        this.data((i * width() + j) * 3 + 1) = other.data(k + 1)
        this.data((i * width() + j) * 3 + 2) = other.data(k + 2)
        k += 3
      }
    }
    this
  }
}