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

package com.intel.analytics.bigdl.pvanet.model

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.pvanet.caffe.CaffeReader
import com.intel.analytics.bigdl.pvanet.layers.{Reshape2, RoiPooling}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class FasterVgg[T: ClassTag](caffeReader: CaffeReader[T] = null)(implicit ev: TensorNumeric[T])
  extends FasterRCNN[T](caffeReader) {

  def vgg16: Module[Tensor[T], Tensor[T], T] = {
    val vggNet = new Sequential[Tensor[T], Tensor[T], T]()
    vggNet.add(conv((3, 64, 3, 1, 1), "conv1_1"))
    vggNet.add(new ReLU[T](true))
    vggNet.add(conv((64, 64, 3, 1, 1), "conv1_2"))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    vggNet.add(conv((64, 128, 3, 1, 1), "conv2_1"))
    vggNet.add(new ReLU[T](true))
    vggNet.add(conv((128, 128, 3, 1, 1), "conv2_2"))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    vggNet.add(conv((128, 256, 3, 1, 1), "conv3_1"))
    vggNet.add(new ReLU[T](true))
    vggNet.add(conv((256, 256, 3, 1, 1), "conv3_2"))
    vggNet.add(new ReLU[T](true))
    vggNet.add(conv((256, 256, 3, 1, 1), "conv3_3"))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    vggNet.add(conv((256, 512, 3, 1, 1), "conv4_1"))
    vggNet.add(new ReLU[T](true))
    vggNet.add(conv((512, 512, 3, 1, 1), "conv4_2"))
    vggNet.add(new ReLU[T](true))
    vggNet.add(conv((512, 512, 3, 1, 1), "conv4_3"))
    vggNet.add(new ReLU[T](true))
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    vggNet.add(conv((512, 512, 3, 1, 1), "conv5_1"))
    vggNet.add(new ReLU[T](true))
    vggNet.add(conv((512, 512, 3, 1, 1), "conv5_2"))
    vggNet.add(new ReLU[T](true))
    vggNet.add(conv((512, 512, 3, 1, 1), "conv5_3"))
    vggNet.add(new ReLU[T](true))
    vggNet
  }

  def rpn: Module[Tensor[T], Table, T] = {
    val rpnModel = new Sequential[Tensor[T], Table, T]()
    rpnModel.add(conv((512, 512, 3, 1, 1), "rpn_conv/3x3"))
    rpnModel.add(new ReLU[T](true))
    val clsAndReg = new ConcatTable[Table, T]()
    clsAndReg.add(conv((512, 18, 1, 1, 0), "rpn_cls_score"))
      .add(conv((512, 36, 1, 1, 0), "rpn_bbox_pred"))
    rpnModel.add(clsAndReg)
    rpnModel
  }

  def vgg16rpn: Module[Tensor[T], Table, T] = {
    val vggRpnModel = new Sequential[Tensor[T], Table, T]()
    vggRpnModel.add(vgg16)
    vggRpnModel.add(rpn)
    vggRpnModel
  }

  def fastRcnn: Module[Table, Table, T] = {
    val model = new Sequential[Table, Table, T]()
    model.add(new RoiPooling[T](7, 7, ev.fromType(0.0625f)))
    model.add(new Reshape2[T](Array(-1, 25088), Some(false)))

    model.add(linear((25088, 4096), "fc6"))
    model.add(new ReLU[T]())

    model.add(linear((4096, 4096), "fc7"))
    model.add(new ReLU[T]())

    val clsReg = new ConcatTable[Table, T]()

    val cls = new Sequential[Tensor[T], Tensor[T], T]()
    cls.add(linear((4096, 21), "cls_score"))
    cls.add(new SoftMax[T]())
    clsReg.add(cls)

    val bboxPred = linear((4096, 84), "bbox_pred")
    clsReg.add(bboxPred)

    model.add(clsReg)
    model
  }

  override def featureNet: Module[Tensor[T], Tensor[T], T] = vgg16

  override val modelName: String = "vgg16"
}

object VggCaffeModel {
  val scales = Array[Float](8, 16, 32)
  val ratios = Array(0.5f, 1.0f, 2.0f)
  val defName = "/home/xianyan/objectRelated/faster_rcnn_models/VGG16/" +
    "faster_rcnn_alt_opt/rpn_test.pt"
  val modelName = "/home/xianyan/objectRelated/faster_rcnn_models/" +
    "VGG16_faster_rcnn_final.caffemodel"

  var caffeReader: CaffeReader[Float] = new CaffeReader[Float](defName, modelName, "vgg16")

  var modelWithCaffeWeight: FasterRCNN[Float] = null

  def getModelWithCaffeWeight: FasterRCNN[Float] = {
    if (modelWithCaffeWeight == null) modelWithCaffeWeight = new FasterVgg[Float](caffeReader)
    modelWithCaffeWeight
  }

  def main(args: Array[String]): Unit = {
    val vgg = getModelWithCaffeWeight
    vgg.featureNetWithCache
    vgg.rpnWithCache
    vgg.fastRcnnWithCache
  }
}

