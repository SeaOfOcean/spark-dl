package com.intel.analytics.bigdl.pvanet.caffe.model

import java.awt.color.ColorSpace
import java.nio.file.Path

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image.{LocalImageFiles, _}
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.Iterator

object AlexNetUtil {
  object RGBImgNormalizer {
    def apply(means: Tensor[Float]): RGBImgNormalizer = new RGBImgNormalizer(means)
  }

  class RGBImgNormalizer(means: Tensor[Float])
    extends Transformer[LabeledRGBImage, LabeledRGBImage] {

    override def apply(prev: Iterator[LabeledRGBImage]): Iterator[LabeledRGBImage] = {
      prev.map(img => {
        val content = img.content
        val meansData = means.storage().array()
        require(content.length % 3 == 0)
        require(content.length == means.nElement())
        var i = 0
        while (i < content.length) {
          content(i + 2) = ((content(i + 2) - meansData(i + 2)))
          content(i + 1) = ((content(i + 1) - meansData(i + 1)))
          content(i + 0) = ((content(i + 0) - meansData(i + 0)))
          i += 3
        }
        img
      })
    }
  }

  class LocalImgReader(scaleTo: Int, normalize: Float)
    extends Transformer[LabeledImageLocalPath, LabeledRGBImage] {
    Class.forName("javax.imageio.ImageIO")
    Class.forName("java.awt.color.ICC_ColorSpace")
    Class.forName("sun.java2d.cmm.lcms.LCMS")
    ColorSpace.getInstance(ColorSpace.CS_sRGB).toRGB(Array[Float](0, 0, 0))

    private val buffer = new LabeledRGBImage()

    override def apply(prev: Iterator[LabeledImageLocalPath]): Iterator[LabeledRGBImage] = {
      prev.map(data => {
        val imgData = RGBImage.readImage(data.path, scaleTo, scaleTo)
        val label = data.label
        buffer.copy(imgData, normalize).setLabel(label)
      })
    }
  }

  def localDataSet(path: Path, imageSize: Int, batchSize: Int, parallel: Int, looped: Boolean,
    means: Tensor[Float])
  : LocalDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.localPathDataSet(path, looped)
    val imgReader = new LocalImgReader(scaleTo = 256, normalize = 1f)
    val normalizer = RGBImgNormalizer(means)
    val cropper = RGBImgCropper(cropWidth = imageSize, cropHeight = imageSize)

    val multiThreadToTensor = MTLabeledRGBImgToTensor[LabeledImageLocalPath](
      width = imageSize,
      height = imageSize,
      threadNum = parallel,
      batchSize = batchSize,
      transformer = imgReader -> normalizer -> cropper
    )
    ds -> multiThreadToTensor
  }
}
