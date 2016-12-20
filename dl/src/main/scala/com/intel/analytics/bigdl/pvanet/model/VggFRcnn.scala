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
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.pvanet.layers._
import com.intel.analytics.bigdl.pvanet.model.Model._
import com.intel.analytics.bigdl.pvanet.model.Phase._


class VggFRcnn(phase: PhaseType = TEST) extends FasterRcnn(phase) {

  def createVgg16(): Sequential[Float] = {
    val vggNet = new Sequential()
    def addConvRelu(param: (Int, Int, Int, Int, Int), name: String, isBack: Boolean = true)
    : Unit = {
      vggNet.add(conv(param, s"conv$name", isBack))
      vggNet.add(new ReLU(true).setName(s"relu$name").setPropagateBack(isBack))
    }
    addConvRelu((3, 64, 3, 1, 1), "1_1", isBack = false)
    addConvRelu((64, 64, 3, 1, 1), "1_2", isBack = false)
    vggNet.add(new SpatialMaxPooling(2, 2, 2, 2).ceil().setPropagateBack(false).setName("pool1"))

    addConvRelu((64, 128, 3, 1, 1), "2_1", isBack = false)
    addConvRelu((128, 128, 3, 1, 1), "2_2", isBack = false)
    vggNet.add(new SpatialMaxPooling(2, 2, 2, 2).ceil().setPropagateBack(false).setName("pool2"))

    addConvRelu((128, 256, 3, 1, 1), "3_1")
    addConvRelu((256, 256, 3, 1, 1), "3_2")
    addConvRelu((256, 256, 3, 1, 1), "3_3")
    vggNet.add(new SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool3"))

    addConvRelu((256, 512, 3, 1, 1), "4_1")
    addConvRelu((512, 512, 3, 1, 1), "4_2")
    addConvRelu((512, 512, 3, 1, 1), "4_3")
    vggNet.add(new SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool4"))

    addConvRelu((512, 512, 3, 1, 1), "5_1")
    addConvRelu((512, 512, 3, 1, 1), "5_2")
    addConvRelu((512, 512, 3, 1, 1), "5_3")
    vggNet
  }

  def createRpn(): Sequential[Float] = {
    val rpnModel = new Sequential()
    rpnModel.add(conv((512, 512, 3, 1, 1), "rpn_conv/3x3", init = (0.01, 0.0)))
    rpnModel.add(new ReLU(true).setName("rpn_relu/3x3"))
    val clsAndReg = new ConcatTable()
    val clsSeq = new Sequential()
    clsSeq.add(conv((512, 18, 1, 1, 0), "rpn_cls_score", init = (0.01, 0.0)))
    phase match {
      case TRAIN => clsSeq.add(new ReshapeInfer(Array(0, 2, -1, 0)))
      case TEST =>
        clsSeq.add(new ReshapeInfer(Array(0, 2, -1, 0)))
          .add(new SoftMax())
          .add(new ReshapeInfer(Array(1, 2 * param.anchorNum, -1, 0)))
    }
    clsAndReg.add(clsSeq)
      .add(conv((512, 36, 1, 1, 0), "rpn_bbox_pred", init = (0.01, 0.0)))
    rpnModel.add(clsAndReg)
    rpnModel
  }

  def createFeatureAndRpnNet(): Sequential[Float] = {
    val compose = new Sequential()
    compose.add(createVgg16())
    val vggRpnModel = new ConcatTable()
    vggRpnModel.add(createRpn())
    vggRpnModel.add(new Identity())
    compose.add(vggRpnModel)
    compose
  }

  protected def createFastRcnn(): Sequential[Float] = {
    val model = new Sequential()
      .add(new RoiPooling(pool, pool, 0.0625f).setName("pool5"))
      .add(new ReshapeInfer(Array(-1, 512 * pool * pool)))
      .add(linear((512 * pool * pool, 4096), "fc6"))
      .add(new ReLU())
      .add(new Dropout().setName("drop6"))
      .add(linear((4096, 4096), "fc7"))
      .add(new ReLU())
      .add(new Dropout().setName("drop7"))

    val cls = new Sequential().add(linear((4096, 21), "cls_score", (0.01, 0.0)))
    if (isTest) cls.add(new SoftMax().setName("cls_prob"))
    val clsReg = new ConcatTable()
      .add(cls)
      .add(linear((4096, 84), "bbox_pred", (0.001, 0.0)))

    model.add(clsReg)
    model
  }

  override val pool: Int = 7
  override val modelType: ModelType = VGG16
  override val param: FasterRcnnParam = new VggParam(phase)

  override def criterion4: ParallelCriterion[Float] = {
    val rpn_loss_bbox = new SmoothL1Criterion2(3.0)
    val rpn_loss_cls = new SoftmaxWithCriterion(ignoreLabel = Some(-1))
    val loss_bbox = new SmoothL1Criterion2(1.0)
    val loss_cls = new SoftmaxWithCriterion()
    val pc = new ParallelCriterion()
    pc.add(rpn_loss_cls, 1.0f)
    pc.add(rpn_loss_bbox, 1.0f)
    pc.add(loss_cls, 1.0f)
    pc.add(loss_bbox, 1.0f)
    pc
  }
}

