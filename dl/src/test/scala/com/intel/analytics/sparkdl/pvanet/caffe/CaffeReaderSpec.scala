package com.intel.analytics.sparkdl.pvanet.caffe

import com.intel.analytics.sparkdl.nn.SpatialConvolution
import org.scalatest.{FlatSpec, Matchers}

class CaffeReaderSpec extends FlatSpec with Matchers {

  val defName = "/home/xianyan/objectRelated/faster_rcnn_models/VGG16/faster_rcnn_alt_opt/rpn_test.pt"
  val modelName = "/home/xianyan/objectRelated/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel"
  "load convolution" should "work properly " in {
    val caffeReader = new CaffeReader[Float](defName, modelName)
    val conv = caffeReader.mapConvolution("conv1_1")
    val conv1_1 = new SpatialConvolution[Float](3, 64, 3, 3, 1, 1, 1, 1)
    assert(conv.kernelH == conv1_1.kernelH)
    assert(conv.kernelW == conv1_1.kernelW)
    assert(conv.padH == conv1_1.padH)
    assert(conv.padW == conv1_1.padW)
    assert(conv.strideH == conv1_1.strideH)
    assert(conv.strideW == conv1_1.strideW)
    assert(conv.nInputPlane == conv1_1.nInputPlane)
    assert(conv.nOutputPlane == conv1_1.nOutputPlane)

    assert(conv.weight.nElement() == 64 * 3 * 3 * 3)
    assert(conv.bias.nElement() == 64)
  }
}
