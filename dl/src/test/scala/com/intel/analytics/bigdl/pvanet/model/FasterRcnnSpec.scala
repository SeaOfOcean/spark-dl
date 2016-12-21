package com.intel.analytics.bigdl.pvanet.model

import com.intel.analytics.bigdl.pvanet.caffe.CaffeReader
import org.scalatest.{FlatSpec, Matchers}

/**
 * Created by xianyan on 12/17/16.
 */
class FasterRcnnSpec extends FlatSpec with Matchers {
  val defFile = "/home/xianyan/objectRelated/faster_rcnn_models/VGG16/" +
    "faster_rcnn_alt_opt/rpn_test.pt"
  val modelFile = "/home/xianyan/objectRelated/faster_rcnn_models/" +
    "VGG16_faster_rcnn_final.caffemodel"
  val model1 = new VggFRcnn(Phase.TEST)
    .copyFromCaffe(new CaffeReader[Float](defFile, modelFile))
  val model2 = new VggFRcnn(Phase.TEST).loadFromCaffeOrCache(defFile, modelFile)

  model1.getTestModel.getParameters() should be(model2.getTestModel.getParameters())
}
