package com.intel.analytics.bigdl.pvanet.caffe.model

import java.nio.file.Path

import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset.{Batch, LocalDataSet}


object GoogleNetUtil {

  def localDataSet(path: Path, imageSize: Int, batchSize: Int, parallel: Int, looped: Boolean)
  : LocalDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.localPathDataSet(path, looped)
    val imgReader = LocalImgReader(scaleTo = 256, normalize = 1f)
    val cropper = RGBImgCropper(cropWidth = imageSize, cropHeight = imageSize)
    val normalizer = RGBImgNormalizer(123, 117, 104, 1, 1, 1)
    val multiThreadToTensor = MTLabeledRGBImgToTensor[LabeledImageLocalPath](
      width = imageSize,
      height = imageSize,
      threadNum = parallel,
      batchSize = batchSize,
      transformer = imgReader -> cropper -> normalizer
    )
    ds -> multiThreadToTensor
  }
}
