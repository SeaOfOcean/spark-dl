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
import com.intel.analytics.bigdl.pvanet.layers._
import com.intel.analytics.bigdl.pvanet.model.Model._
import com.intel.analytics.bigdl.pvanet.model.Phase._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class VggFRcnn[T: ClassTag](phase: PhaseType = TEST)(implicit ev: TensorNumeric[T])
  extends FasterRcnn[T](phase) {

  def createVgg16(): Sequential[T] = {
    val vggNet = new Sequential[T]()
    def addConvRelu(param: (Int, Int, Int, Int, Int), name: String, isBack: Boolean = true)
    : Unit = {
      vggNet.add(conv(param, s"conv$name", isBack))
      vggNet.add(new ReLU[T](true).setName(s"relu$name").setPropagateBack(isBack))
    }
    addConvRelu((3, 64, 3, 1, 1), "1_1", isBack = false)
    addConvRelu((64, 64, 3, 1, 1), "1_2", isBack = false)
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil().setPropagateBack(false).setName("pool1"))

    addConvRelu((64, 128, 3, 1, 1), "2_1", isBack = false)
    addConvRelu((128, 128, 3, 1, 1), "2_2", isBack = false)
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil().setPropagateBack(false).setName("pool2"))

    addConvRelu((128, 256, 3, 1, 1), "3_1")
    addConvRelu((256, 256, 3, 1, 1), "3_2")
    addConvRelu((256, 256, 3, 1, 1), "3_3")
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil().setName("pool3"))

    addConvRelu((256, 512, 3, 1, 1), "4_1")
    addConvRelu((512, 512, 3, 1, 1), "4_2")
    addConvRelu((512, 512, 3, 1, 1), "4_3")
    vggNet.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil().setName("pool4"))

    addConvRelu((512, 512, 3, 1, 1), "5_1")
    addConvRelu((512, 512, 3, 1, 1), "5_2")
    addConvRelu((512, 512, 3, 1, 1), "5_3")
    vggNet
  }

  def createRpn(): Sequential[T] = {
    val rpnModel = new Sequential[T]()
    rpnModel.add(conv((512, 512, 3, 1, 1), "rpn_conv/3x3", init = (0.01, 0.0)))
    rpnModel.add(new ReLU[T](true).setName("rpn_relu/3x3"))
    val clsAndReg = new ConcatTable[T]()
    val clsSeq = new Sequential[T]()
    clsSeq.add(conv((512, 18, 1, 1, 0), "rpn_cls_score", init = (0.01, 0.0)))
    phase match {
      case TRAIN => clsSeq.add(new ReshapeInfer[T](Array(0, 2, -1, 0)))
      case TEST =>
        clsSeq.add(new ReshapeInfer[T](Array(0, 2, -1, 0)))
          .add(new SoftMax[T]())
          .add(new ReshapeInfer[T](Array(1, 2 * param.anchorNum, -1, 0)))
    }
    clsAndReg.add(clsSeq)
      .add(conv((512, 36, 1, 1, 0), "rpn_bbox_pred", init = (0.01, 0.0)))
    rpnModel.add(clsAndReg)
    rpnModel
  }

  def createFeatureAndRpnNet(): Sequential[T] = {
    val compose = new Sequential[T]()
    compose.add(createVgg16())
    val vggRpnModel = new ConcatTable[T]()
    vggRpnModel.add(createRpn())
    vggRpnModel.add(new Identity[T]())
    compose.add(vggRpnModel)
    compose
  }

  protected def createFastRcnn(): Sequential[T] = {
    val model = new Sequential[T]()
      .add(new RoiPooling[T](pool, pool, ev.fromType(0.0625f)).setName("pool5"))
      .add(new ReshapeInfer[T](Array(-1, 512 * pool * pool)))
      .add(linear((512 * pool * pool, 4096), "fc6"))
      .add(new ReLU[T]())
      .add(new Dropout[T]().setName("drop6"))
      .add(linear((4096, 4096), "fc7"))
      .add(new ReLU[T]())
      .add(new Dropout[T]().setName("drop7"))

    val cls = new Sequential[T]().add(linear((4096, 21), "cls_score", (0.01, 0.0)))
    if (isTest) cls.add(new SoftMax[T]())
    val clsReg = new ConcatTable[T]()
      .add(cls)
      .add(linear((4096, 84), "bbox_pred", (0.001, 0.0)))

    model.add(clsReg)
    model
  }

  override val pool: Int = 7
  override val modelType: ModelType = VGG16
  override val param: FasterRcnnParam = new VggParam(phase)

  override def criterion4: ParallelCriterion[T] = {
    val rpn_loss_bbox = new SmoothL1Criterion2[T](3.0)
    val rpn_loss_cls = new SoftmaxWithCriterion[T](ignoreLabel = Some(-1))
    val loss_bbox = new SmoothL1Criterion2[T](1.0)
    val loss_cls = new SoftmaxWithCriterion[T]()
    val pc = new ParallelCriterion[T]()
    pc.add(rpn_loss_cls, 1.0f)
    pc.add(rpn_loss_bbox, 1.0f)
    pc.add(loss_cls, 1.0f)
    pc.add(loss_bbox, 1.0f)
    pc
  }

  def createTestModel(): Sequential[T] = {
    val model = new Sequential[T]()
    val model1 = new ParallelTable[T]()
    model1.add(featureAndRpnNet)
    model1.add(new Identity[T]())
    model.add(model1)
    // connect rpn and fast-rcnn
    val middle = new ConcatTable[T]()
    val left = new Sequential[T]()
    val left1 = new ConcatTable[T]()
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
    model.add(new ConcatTable[T]().add(fastRcnn).add(selectTensor(2)))
    model
  }


  def createTrainModel(): Sequential[T] = {
    val model = new Sequential[T]()

    val rpnFeatureWithInfoGt = new ParallelTable[T]()
    rpnFeatureWithInfoGt.add(featureAndRpnNet)
    // im_info
    rpnFeatureWithInfoGt.add(new Identity[T]())
    // gt_boxes
    rpnFeatureWithInfoGt.add(new Identity[T]())
    model.add(rpnFeatureWithInfoGt)

    val lossModels = new ConcatTable[T]()
    model.add(lossModels)

    lossModels.add(selectTensor(1, 1, 1).setName("rpn_cls"))
    lossModels.add(selectTensor(1, 1, 2).setName("rpn_reg"))
    val fastRcnnLossModel = new Sequential[T]().setName("loss from fast rcnn")
    // get ((rois, otherProposalTargets), features
    val fastRcnnInputModel = new ConcatTable[T]().setName("fast-rcnn")
    fastRcnnInputModel.add(selectTensor(1, 2).setName("features"))
    val sampleRoisModel = new Sequential[T]().setPropagateBack(false)
    fastRcnnInputModel.add(sampleRoisModel)
    // add sample rois
    lossModels.add(fastRcnnLossModel)

    val proposalTargetInput = new ConcatTable[T]()
    // get rois from proposal layer
    val proposalModel = new Sequential[T]()
      .add(new ConcatTable[T]()
        .add(new Sequential[T]()
          .add(selectTensor(1, 1, 1).setName("rpn_cls"))
          .add(new SoftMax[T]())
          .add(new ReshapeInfer[T](Array(1, 2 * param.anchorNum, -1, 0)))
          .setName("rpn_cls_softmax_reshape"))
        .add(selectTensor(1, 1, 2).setName("rpn_reg"))
        .add(selectTensor1NoBack(2).setName("im_info")))
      .add(new Proposal[T](param))
      .add(selectTensor1NoBack(1).setName("rpn_rois"))

    proposalTargetInput.add(proposalModel)
    proposalTargetInput.add(selectTensor1NoBack(3).setName("gtBoxes"))
    sampleRoisModel.add(proposalTargetInput)
    sampleRoisModel.add(new ProposalTarget[T](param))

    // ( features, (rois, otherProposalTargets))
    fastRcnnLossModel.add(fastRcnnInputModel)
    fastRcnnLossModel.add(
      new ConcatTable[T]()
        .add(new ConcatTable[T]()
          .add(selectTensor1(1).setName("features"))
          .add(selectTensorNoBack(2, 1).setName("rois")))
        .add(selectTableNoBack(2, 2).setName("other targets info")))
    fastRcnnLossModel.add(new ParallelTable[T]()
      .add(fastRcnn.setName("fast rcnn"))
      .add(new Identity[T]()).setName("other targets info"))
    // make each res a tensor
    model.add(new ConcatTable[T]()
      .add(selectTensor1(1).setName("rpn cls"))
      .add(selectTensor1(2).setName("rpn reg"))
      .add(selectTensor(3, 1, 1).setName("cls"))
      .add(selectTensor(3, 1, 2).setName("reg"))
      .add(selectTensorNoBack(3, 2).setName("other target info")))
    model
  }
}

