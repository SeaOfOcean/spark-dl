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

class VggFRcnn[T: ClassTag](phase: PhaseType = TEST)(implicit ev: TensorNumeric[T])
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
    val vggRpnModel = new ConcatTable[Tensor[T], T]()
    vggRpnModel.add(rpn())
    vggRpnModel.add(new Identity[T]())
    compose.add(vggRpnModel)
    compose
  }

  def fastRcnn(): Module[Table, Table, T] = {
    val model = new STT()
      .add(new RoiPooling[T](7, 7, ev.fromType(0.0625f)))
      .add(new Reshape2[T](Array(-1, 25088), Some(false)))
      .add(linear((25088, 4096), "fc6"))
      .add(new ReLU[T]())
      .add(linear((4096, 4096), "fc7"))
      .add(new ReLU[T]())

    val clsReg = new CT()
      .add(new Stt()
        .add(linear((4096, 21), "cls_score"))
        .add(new SoftMax[T]()))
      .add(linear((4096, 84), "bbox_pred"))

    model.add(clsReg)
    model
  }

  override val modelType: ModelType = VGG16
  override val param: FasterRcnnParam = new VggParam(phase)

  override def criterion4: ParallelCriterion[T] = {
    val rpn_loss_bbox = new SmoothL1Criterion2[T](ev.fromType(3.0), 1)
    val rpn_loss_cls = new SoftmaxWithCriterion[T](ignoreLabel = Some(-1))
    val loss_bbox = new SmoothL1Criterion2[T](ev.fromType(1.0), 1)
    val loss_cls = new SoftmaxWithCriterion[T]()
    val pc = new ParallelCriterion[T]()
    pc.add(rpn_loss_cls, 1.0f)
    pc.add(rpn_loss_bbox, 1.0f)
    pc.add(loss_cls, 1.0f)
    pc.add(loss_bbox, 1.0f)
    pc
  }

  /**
   * select tensor from nested tables
   *
   * @param depths a serious of depth to use when fetching certain tensor
   * @return a wanted tensor
   */
  def selectTensor(depths: Int*): Sequential[Table, Tensor[T], T] = {
    val module = new Sequential[Table, Tensor[T], T]()
    depths.slice(0, depths.length - 1).foreach(depth =>
      module.add(new SelectTable[Table, T](depth)))
    module.add(new SelectTable[Tensor[T], T](depths(depths.length - 1)))
  }

  def selectTensor1(depth: Int): SelectTable[Tensor[T], T] = {
    new SelectTable[Tensor[T], T](depth)
  }

  def selectTable(depths: Int*): Sequential[Table, Table, T] = {
    val module = new Sequential[Table, Table, T]()
    depths.slice(0, depths.length).foreach(depth =>
      module.add(new SelectTable[Table, T](depth)))
    module
  }

  def selectTable1(depth: Int): SelectTable[Table, T] = {
    new SelectTable[Table, T](depth)
  }

  def fullModelTest(): Module[Table, Table, T] = {
    val model = new Sequential[Table, Table, T]()
    val model1 = new ParallelTable[T]()
    model1.add(featureAndRpnNet())
    model1.add(new Identity[T]())
    model.add(model1)
    // connect rpn and fast-rcnn
    val middle = new ConcatTable[Tensor[T], T]()
    val left = new Sequential[Table, Table, T]()
    val left1 = new ConcatTable[Tensor[T], T]()
    left1.add(selectTensor(1, 1, 1))
    left1.add(selectTensor(1, 1, 2))
    left1.add(selectTensor1(2))
    left.add(left1)
    left.add(new Proposal[T](param))
    left.add(selectTensor1(1))
    // first add feature from feature net
    middle.add(selectTensor(1, 2))
    // then add rois from proposal
    middle.add(left)
    model.add(middle)
    // get the fast rcnn results and rois
    model.add(new ConcatTable[Table, T]().add(fastRcnn()).add(selectTensor(2)))
    model
  }


  def fullModelTrain2(): Module[Table, Table, T] = {
    val model = new Sequential[Table, Table, T]()
    val rpnFeatureWithInfoGt = new ParallelTable[T]()
    // (rcls, rreg), feature)
    rpnFeatureWithInfoGt.add(featureAndRpnNet())
    // im_info
    rpnFeatureWithInfoGt.add(new Identity[T]())
    // gt_boxes
    rpnFeatureWithInfoGt.add(new Identity[T]())
    // (((rcls, rreg), feature), im_info, gt_boxes)
    model.add(rpnFeatureWithInfoGt)
    val rpnAndFastRcnn = new ConcatTable[Tensor[T], T]()
    // rpn cls
    rpnAndFastRcnn.add(selectTensor(1, 1, 1))
    // rpn reg
    rpnAndFastRcnn.add(selectTensor(1, 1, 2))
    val modelWithGt = new ConcatTable[Tensor[T], T]()
    // connect rpn and fast-rcnn
    val middle = new ConcatTable[Tensor[T], T]()
    val left = new Sequential[Table, Table, T]()
    val left1 = new ConcatTable[Tensor[T], T]()
    left1.add(new Sequential[Table, Tensor[T], T]()
      .add(selectTensor(1, 1, 1))
      .add(new SoftMax[T]())
      .add(new Reshape2[T](Array(1, 2 * param.anchorNum, -1, 0), Some(false))))
    left1.add(selectTensor(1, 1, 2))
    left1.add(selectTensor1(2))
    left.add(left1)
    left.add(new Proposal[T](param))
    // add rois
    left.add(selectTensor1(1))
    // first add feature from feature net
    middle.add(selectTensor(1, 2))
    // then add rois from proposal
    middle.add(left)
    // add gt_boxes
    modelWithGt.add(middle).add(selectTensor1(3))

    rpnAndFastRcnn.add(new STT().add(middle)
      .add(new ConcatTable[Table, T]().add(fastRcnn()).add(selectTensor(2))))
    model.add(rpnAndFastRcnn)
    // make each res a tensor
    model.add(new ConcatTable[Tensor[T], T]()
      .add(selectTensor1(1))
      .add(selectTensor1(2))
      .add(selectTensor(3, 1, 1))
      .add(selectTensor(3, 1, 2))
      .add(selectTensor(3, 2)))
    model
  }

  def fullModelTrain(): Module[Table, Table, T] = {
    val model = new STT()

    val rpnFeatureWithInfoGt = new ParallelTable[T]()
    rpnFeatureWithInfoGt.add(featureAndRpnNet())
    // im_info
    rpnFeatureWithInfoGt.add(new Identity[T]())
    // gt_boxes
    rpnFeatureWithInfoGt.add(new Identity[T]())
    model.add(rpnFeatureWithInfoGt)

    val lossModels = new CT()
    model.add(lossModels)

    lossModels.add(selectTensor(1, 1, 1).setName("rpn_cls"))
    lossModels.add(selectTensor(1, 1, 2).setName("rpn_reg"))
    val fastRcnnLossModel = new STT().setName("loss from fast rcnn")
    // get ((rois, otherProposalTargets), features
    val fastRcnnInputModel = new CT().setName("fast-rcnn")
    fastRcnnInputModel.add(selectTensor(1, 2).setName("features"))
    val sampleRoisModel = new STT
    fastRcnnInputModel.add(sampleRoisModel)
    // add sample rois
    lossModels.add(fastRcnnLossModel)

    val proposalTargetInput = new CT()
    // get rois from proposal layer
    val proposalModel = new STt()
      .add(new Ct()
        .add(new STt()
          .add(selectTensor(1, 1, 1))
          .add(new SoftMax[T]())
          .add(new Reshape2[T](Array(1, 2 * param.anchorNum, -1, 0), Some(false)))
          .setName("rpn_cls_softmax_reshape"))
        .add(selectTensor(1, 1, 2).setName("rpn_reg"))
        .add(selectTensor1(2).setName("im_info")))
      .add(new Proposal[T](param))
      .add(selectTensor1(1).setName("rois"))

    proposalTargetInput.add(proposalModel)
    proposalTargetInput.add(selectTensor1(3).setName("gtBoxes"))
    sampleRoisModel.add(proposalTargetInput)
    sampleRoisModel.add(new ProposalTarget[T](param))

    // ( features, (rois, otherProposalTargets))
    fastRcnnLossModel.add(fastRcnnInputModel)
    fastRcnnLossModel.add(
      new CT()
        .add(new CT()
          .add(selectTensor1(1).setName("features"))
          .add(selectTensor(2, 1).setName("rois")))
        .add(selectTable(2, 2).setName("other targets info")))
    fastRcnnLossModel.add(new ParallelTable[T]()
      .add(fastRcnn())
      .add(new Identity[T]()))
    // make each res a tensor
    model.add(new ConcatTable[Tensor[T], T]()
      .add(selectTensor1(1))
      .add(selectTensor1(2))
      .add(selectTensor(3, 1, 1))
      .add(selectTensor(3, 1, 2))
      .add(selectTensor(3, 2)))
    model
  }

  override def fullModel(): Module[Table, Table, T] = {
    phase match {
      case TEST => fullModelTest()
      case TRAIN => fullModelTrain()
      case _ => throw new UnsupportedOperationException
    }
  }
}

object VggFRcnn {
  private val defName = "/home/xianyan/objectRelated/faster_rcnn_models/VGG16/" +
    "faster_rcnn_alt_opt/rpn_test.pt"
  private val modelName = "/home/xianyan/objectRelated/faster_rcnn_models/" +
    "VGG16_faster_rcnn_final.caffemodel"

  private val caffeReader: CaffeReader[Float] = new CaffeReader[Float](defName, modelName, "vgg16")

  private var modelWithCaffeWeight: FasterRcnn[Float] = _

  def model(phase: PhaseType = TEST): FasterRcnn[Float] = {
    if (modelWithCaffeWeight == null) modelWithCaffeWeight = new VggFRcnn[Float](phase)
    modelWithCaffeWeight.setCaffeReader(caffeReader)
    modelWithCaffeWeight
  }

  def main(args: Array[String]): Unit = {
    val vgg = model()
    vgg.featureAndRpnNet()
    vgg.fastRcnn()
  }
}

