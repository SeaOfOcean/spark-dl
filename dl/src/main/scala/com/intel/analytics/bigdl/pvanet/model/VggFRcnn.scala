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
import com.intel.analytics.bigdl.pvanet.layers._
import com.intel.analytics.bigdl.pvanet.model.Model._
import com.intel.analytics.bigdl.pvanet.model.Phase._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class VggFRcnn[T: ClassTag](phase: Phase = TEST)(implicit ev: TensorNumeric[T])
  extends FasterRcnn[T](phase) {

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

  def rpn(): Module[Tensor[T], Table, T] = {
    val rpnModel = new Sequential[Tensor[T], Table, T]()
    rpnModel.add(conv((512, 512, 3, 1, 1), "rpn_conv/3x3"))
    rpnModel.add(new ReLU[T](true))
    val clsAndReg = new ConcatTable[Table, T]()
    val clsSeq = new Sequential[Tensor[T], Tensor[T], T]()
    clsSeq.add(conv((512, 18, 1, 1, 0), "rpn_cls_score"))
    phase match {
      case TRAIN => clsSeq.add(new Reshape2[T](Array(0, 2, -1, 0), Some(false)))
      case TEST =>
        clsSeq.add(new Reshape2[T](Array(0, 2, -1, 0), Some(false)))
          .add(new SoftMax[T]())
          .add(new Reshape2[T](Array(1, 2 * param.anchorNum, -1, 0), Some(false)))
    }
    clsAndReg.add(clsSeq)
      .add(conv((512, 36, 1, 1, 0), "rpn_bbox_pred"))
    rpnModel.add(clsAndReg)
    rpnModel
  }

  def featureAndRpnNet(): Module[Tensor[T], Table, T] = {
    val compose = new Sequential[Tensor[T], Table, T]()
    compose.add(vgg16)
    phase match {
      case TRAIN => compose.add(rpn)
      case TEST =>
        val vggRpnModel = new ConcatTable[Tensor[T], T]()
        vggRpnModel.add(rpn)
        vggRpnModel.add(new Identity[T]())
        compose.add(vggRpnModel)
      case FINETUNE => throw new NotImplementedError()
    }
    compose
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

  override val model: Model = VGG16
  override val param: FasterRcnnParam = new VggParam(phase)

  override def rpnCriterion: ParallelCriterion[T] = {
    val rpn_loss_bbox = new SmoothL1Criterion2[T](ev.fromType(3.0), 1)
    val rpn_loss_cls = new SoftmaxWithCriterion[T](ignoreLabel = Some(-1))
    val pc = new ParallelCriterion[T]()
    pc.add(rpn_loss_cls, 1)
    pc.add(rpn_loss_bbox, 1)
    pc
  }

  override def fastRcnnCriterion: ParallelCriterion[T] = {
    val loss_bbox = new SmoothL1Criterion2[T](ev.fromType(1.0), 1)
    val loss_cls = new SoftmaxWithCriterion[T]()
    val pc = new ParallelCriterion[T]()
    pc.add(loss_cls, 1)
    pc.add(loss_bbox, 1)
    pc
  }

  override def fullModel: Module[Table, Table, T] = {
    val model = new Sequential[Table, Table, T]()
    val model1 = new ParallelTable[T]()
    model1.add(featureAndRpnNet)
    model1.add(new Identity[T]())
    model.add(model1)
    // connect rpn and fast-rcnn
    val middle = new ConcatTable[Tensor[T], T]()
    val left = new Sequential[Table, Table, T]()
    val left1 = new ConcatTable[Tensor[T], T]()
    left1.add(new Sequential[Table, Tensor[T], T]()
      .add(new SelectTable[Table, T](1))
      .add(new SelectTable[Table, T](1))
      .add(new SelectTable[Tensor[T], T](1)))
    left1.add(new Sequential[Table, Tensor[T], T]()
      .add(new SelectTable[Table, T](1))
      .add(new SelectTable[Table, T](1))
      .add(new SelectTable[Tensor[T], T](2)))
    left1.add(new Sequential[Table, Tensor[T], T]()
      .add(new SelectTable[Tensor[T], T](2)))
    left.add(left1)
    left.add(new Proposal[T](param))
    left.add(new SelectTable[Tensor[T], T](1))
    // first add feature from feature net
    middle.add(new Sequential[Table, Tensor[T], T]()
      .add(new SelectTable[Table, T](1))
      .add(new SelectTable[Tensor[T], T](2)))
    // then add rois from proposal
    middle.add(left)
    model.add(middle)
    // get the fast rcnn results and rois
    model.add(new ConcatTable[Table, T]().add(fastRcnn).add(new SelectTable[Tensor[T], T](2)))
    model
  }
}

object VggFRcnn {
  private val defName = "/home/xianyan/objectRelated/faster_rcnn_models/VGG16/" +
    "faster_rcnn_alt_opt/rpn_test.pt"
  private val modelName = "/home/xianyan/objectRelated/faster_rcnn_models/" +
    "VGG16_faster_rcnn_final.caffemodel"

  private val caffeReader: CaffeReader[Float] = new CaffeReader[Float](defName, modelName, "vgg16")

  private var modelWithCaffeWeight: FasterRcnn[Float] = null

  def model(isTrain: Boolean = false): FasterRcnn[Float] = {
    if (modelWithCaffeWeight == null) modelWithCaffeWeight = new VggFRcnn[Float]()
    modelWithCaffeWeight.setCaffeReader(caffeReader)
    modelWithCaffeWeight
  }

  def main(args: Array[String]): Unit = {
    val vgg = model()
    vgg.featureAndRpnNet
    vgg.fastRcnn
  }
}

