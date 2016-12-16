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

import java.awt.Color
import java.awt.image.{BufferedImage, DataBufferByte}
import java.util.concurrent.atomic.AtomicInteger
import javax.imageio.ImageIO

import breeze.linalg.DenseVector
import com.intel.analytics.bigdl.dataset.{LocalDataSet, Transformer}
import com.intel.analytics.bigdl.pvanet.model.FasterRcnnParam
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator

import scala.util.Random


class ObjectDataSource(val imdb: Imdb, looped: Boolean = true)
  extends LocalDataSet[Roidb] {

  val roidbs = imdb.getRoidb

  // permutation of the data index
  var perm: Array[Int] = _

  override def shuffle(): Unit = {
    def shuffleWithAspectGrouping(widths: Array[Int], heights: Array[Int]): Unit = {
      val horz = (widths zip heights).map(x => x._1 >= x._2)
      val vert = horz.map(x => !x)
      val horzInds = horz.zipWithIndex.filter(x => x._1).map(x => x._2)
      val vertInds = vert.zipWithIndex.filter(x => x._1).map(x => x._2)
      val indsArr = (Random.shuffle(horzInds.toSeq) ++ Random.shuffle(vertInds.toSeq)).toArray
      val rowPerm = Random.shuffle(Seq.range(0, indsArr.length / 2))
      rowPerm.zipWithIndex.foreach(r => {
        perm(r._1 * 2) = indsArr(r._2 * 2)
        perm(r._1 * 2 + 1) = indsArr(r._2 * 2 + 1)
      })
    }
    if (imdb.param.ASPECT_GROUPING) {
      shuffleWithAspectGrouping(imdb.widths, imdb.heights)
    } else {
      RandomGenerator.shuffle(perm)
    }
  }

  /**
   * Get a sequence of data
   *
   * @return
   */
  override def data(): Iterator[Roidb] = {
    perm = imdb.getRoidb.indices.toArray
    new Iterator[Roidb] {
      private val index = new AtomicInteger()

      override def hasNext: Boolean = {
        if (looped) {
          true
        } else {
          index.get() < perm.length
        }
      }

      override def next(): Roidb = {
        val curIndex = index.getAndIncrement()
        if (looped || curIndex < perm.length) {
          roidbs(perm(if (looped) (curIndex % perm.length) else curIndex))
        } else {
          null.asInstanceOf[Roidb]
        }
      }
    }
  }

  /**
   * Return the total size of the data set
   *
   * @return
   */
  override def size(): Long = roidbs.length
}

object ObjectDataSource {
  def apply(imdb: Imdb, looped: Boolean = true): ObjectDataSource =
    new ObjectDataSource(imdb, looped)

  def apply(name: String, devkitPath: String, param: FasterRcnnParam,
    looped: Boolean): ObjectDataSource =
    new ObjectDataSource(Imdb.getImdb(name, param, Some(devkitPath)), looped)

  def apply(name: String, param: FasterRcnnParam, looped: Boolean): ObjectDataSource =
    new ObjectDataSource(Imdb.getImdb(name, param), looped)
}

class ImageScalerAndMeanSubstractor(param: FasterRcnnParam)
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
      (pixels(x._2) - param.PIXEL_MEANS.head.head(x._2 % 3)).toFloat
    )

    val scaledImage = new RGBImageOD(meanPixels, imageBuff.getWidth, imageBuff.getHeight)
    val imScales = Array(im_scale_x, im_scale_y, im_scale_x, im_scale_y)
    val imInfo = Some(Tensor(Storage(
      Array(imageBuff.getHeight(), imageBuff.getWidth, im_scale_x))))

    var gtBoxes: Option[Tensor[Float]] = None

    if (data.gtClasses != null) {
      val gt_inds = data.gtClasses.storage().array().zipWithIndex
        .filter(x => x._1 != 0).map(x => x._2)
      val gt_boxes = Tensor[Float](gt_inds.length, 5)
      gt_inds.zipWithIndex.foreach(ind => {
        val scaled = data.boxes(ind._1, 0 until 4).t :* DenseVector(imScales)
        gt_boxes.setValue(ind._2 + 1, 1, scaled(0))
        gt_boxes.setValue(ind._2 + 1, 2, scaled(1))
        gt_boxes.setValue(ind._2 + 1, 3, scaled(2))
        gt_boxes.setValue(ind._2 + 1, 4, scaled(3))
        gt_boxes.setValue(ind._2 + 1, 5, data.gtClasses.valueAt(ind._1 + 1))
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

object ImageToTensor {
  val imgToTensor = new ImageToTensor(1)

  def apply(img: ImageWithRoi): Tensor[Float] = imgToTensor.apply(img)
}

class ImageToTensor(batchSize: Int = 1) extends Transformer[ImageWithRoi, Tensor[Float]] {
  require(batchSize == 1)

  private val featureTensor: Tensor[Float] = Tensor[Float]()

  def apply(imgWithRoi: ImageWithRoi): Tensor[Float] = {
    assert(batchSize == 1)
    val img = imgWithRoi.scaledImage
    featureTensor.set(Storage[Float](imgWithRoi.scaledImage.content),
      storageOffset = 1, sizes = Array(batchSize, img.height(), img.width(), 3))
    featureTensor.transpose(2, 3).transpose(2, 4).contiguous()
  }

  override def apply(prev: Iterator[ImageWithRoi]): Iterator[Tensor[Float]] = {
    new Iterator[Tensor[Float]] {


      override def hasNext: Boolean = prev.hasNext

      override def next(): Tensor[Float] = {
        if (prev.hasNext) {
          val imgWithRoi = prev.next()
          apply(imgWithRoi)
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
