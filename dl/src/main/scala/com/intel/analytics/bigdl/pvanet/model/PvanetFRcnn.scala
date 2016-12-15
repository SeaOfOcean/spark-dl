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

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.pvanet.layers.{ReshapeInfer, RoiPooling, SmoothL1Criterion2, SoftmaxWithCriterion}
import com.intel.analytics.bigdl.pvanet.model.Model._
import com.intel.analytics.bigdl.pvanet.model.Phase._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric


class PvanetFRcnn(phase: PhaseType = TEST)
  (implicit ev: TensorNumeric[Float])
  extends FasterRcnn(phase) {

  var pvanet: Sequential[Float] = _

  private def concatNeg(name: String): Concat[Float] = {
    val concat = new Concat(2)
    concat.add(new Identity())
    concat.add(new Power(1, -1, 0).setName(s"$name/neg"))
    concat.setName(s"$name/concat")
    concat
  }

  private def addScale(module: Sequential[Float],
    sizes: Array[Int], name: String): Unit = {
    val sc = scale(sizes, name)
    module.add(sc._1)
    module.add(sc._2)
    module.add(new ReLU())
  }

  private def addConvComponent(compId: Int, index: Int, p: Array[(Int, Int, Int, Int, Int)]) = {
    val label = s"${compId}_$index"
    val convTable = new ConcatTable[Float]
    val conv_left = new Sequential()
    var i = 0
    if (index == 1) {
      conv_left.add(conv(p(i), s"conv$label/1/conv"))
      i += 1
    } else {
      conv_left.add(conv(p(i), s"conv$label/1/conv"))
      i += 1
    }

    conv_left.add(new ReLU())
    conv_left.add(conv(p(i), s"conv$label/2/conv"))
    i += 1
    conv_left.add(concatNeg(s"conv$label/2"))
    if (compId == 2) {
      addScale(conv_left, Array(48), s"conv$label/2/scale")
    } else {
      addScale(conv_left, Array(96), s"conv$label/2/scale")
    }

    conv_left.add(conv(p(i), s"conv$label/3/conv"))
    i += 1

    convTable.add(conv_left)
    if (index == 1) {
      convTable.add(conv(p(i), s"conv$label/proj"))
      i += 1
    } else {
      convTable.add(new Identity())
    }
    pvanet.add(convTable)
    pvanet.add(new CAddTable().setName(s"conv$label"))
  }

  private def addInception(module: Sequential[Float], label: String, index: Int,
    p: Array[(Int, Int, Int, Int, Int)]): Unit = {
    val left = new Sequential()
    val incep = new Concat(2)

    var i = 0
    val com1 = new Sequential()
    com1.add(conv(p(i), s"conv$label/incep/0/conv")).add(new ReLU())
    i += 1
    incep.add(com1)

    val com2 = new Sequential()
    com2.add(conv(p(i), s"conv$label/incep/1_reduce/conv")).add(new ReLU())
    i += 1
    com2.add(conv(p(i), s"conv$label/incep/1_0/conv")).add(new ReLU())
    i += 1
    incep.add(com2)

    val com3 = new Sequential()
    com3.add(conv(p(i), s"conv$label/incep/2_reduce/conv")).add(new ReLU())
    i += 1
    com3.add(conv(p(i), s"conv$label/incep/2_0/conv")).add(new ReLU())
    i += 1
    com3.add(conv(p(i), s"conv$label/incep/2_1/conv")).add(new ReLU())
    i += 1
    incep.add(com3)

    if (index == 1) {
      val com4 = new Sequential()
      com4.add(new SpatialMaxPooling(3, 3, 2, 2).ceil().setName(s"conv$label/incep/pool"))
      com4.add(conv(p(i), s"conv$label/incep/poolproj/conv")).add(new ReLU())
      i += 1
      incep.add(com4)
    }

    left.add(incep)
    left.add(conv(p(i), s"conv$label/out/conv"))
    i += 1
    val table = new ConcatTable[Float]
    table.add(left)
    if (index == 1) {
      table.add(conv(p(i), s"conv$label/proj"))
      i += 1
    } else {
      table.add(new Identity())
    }
    module.add(table)
    module.add(new CAddTable().setName(s"conv$label"))
  }


  private def getPvanet: Sequential[Float] = {
    if (pvanet != null) return pvanet
    pvanet = new Sequential()
    pvanet.add(conv((3, 16, 7, 2, 3), "conv1_1/conv"))

    pvanet.add(concatNeg("conv1_1"))
    addScale(pvanet, Array(32), "conv1_1/scale")
    pvanet.add(new SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1"))


    addConvComponent(2, 1, Array((32, 24, 1, 1, 0), (24, 24, 3, 1, 1),
      (48, 64, 1, 1, 0), (32, 64, 1, 1, 0)))
    for (i <- 2 to 3) {
      addConvComponent(2, i, Array((64, 24, 1, 1, 0), (24, 24, 3, 1, 1), (48, 64, 1, 1, 0)))
    }

    addConvComponent(3, 1, Array((64, 48, 1, 2, 0), (48, 48, 3, 1, 1),
      (96, 128, 1, 1, 0), (64, 128, 1, 2, 0)))
    for (i <- 2 to 4) {
      addConvComponent(3, i, Array((128, 48, 1, 1, 0), (48, 48, 3, 1, 1), (96, 128, 1, 1, 0)))
    }

    val inceptions4_5 = new Sequential()

    val inceptions4 = new Sequential()
    addInception(inceptions4, "4_1", 1, Array((128, 64, 1, 2, 0), (128, 48, 1, 2, 0),
      (48, 128, 3, 1, 1), (128, 24, 1, 2, 0), (24, 48, 3, 1, 1), (48, 48, 3, 1, 1),
      (128, 128, 1, 1, 0), (368, 256, 1, 1, 0), (128, 256, 1, 2, 0)))
    for (i <- 2 to 4) {
      addInception(inceptions4, s"4_$i", i, Array((256, 64, 1, 1, 0), (256, 64, 1, 1, 0),
        (64, 128, 3, 1, 1), (256, 24, 1, 1, 0), (24, 48, 3, 1, 1),
        (48, 48, 3, 1, 1), (240, 256, 1, 1, 0)))
    }
    inceptions4_5.add(inceptions4)


    val seq5 = new Sequential()
    val inceptions5 = new Sequential()
    addInception(inceptions5, "5_1", 1, Array((256, 64, 1, 2, 0), (256, 96, 1, 2, 0),
      (96, 192, 3, 1, 1), (256, 32, 1, 2, 0), (32, 64, 3, 1, 1), (64, 64, 3, 1, 1),
      (256, 128, 1, 1, 0), (448, 384, 1, 1, 0), (256, 384, 1, 2, 0)))
    for (i <- 2 to 4) {
      addInception(inceptions5, s"5_$i", i, Array((384, 64, 1, 1, 0), (384, 96, 1, 1, 0),
        (96, 192, 3, 1, 1), (384, 32, 1, 1, 0), (32, 64, 3, 1, 1), (64, 64, 3, 1, 1),
        (320, 384, 1, 1, 0)))
    }

    seq5.add(inceptions5)
    seq5.add(spatialFullConv((384, 4, 2, 1, false), "upsample"))

    val concat5 = new Concat(2)
    concat5.add(new Identity())
    concat5.add(seq5)

    inceptions4_5.add(concat5)

    val concatConvf = new Concat(2).setName("concat")
    concatConvf.add(new SpatialMaxPooling(3, 3, 2, 2).ceil().setName("downsample"))
    concatConvf.add(inceptions4_5)
    pvanet.add(concatConvf)

    pvanet
  }

  override def createFeatureAndRpnNet(): Sequential[Float] = {
    val compose = new Sequential()
    compose.add(getPvanet)

    val convTable = new ConcatTable[Float]
    convTable.add(new Sequential()
      .add(conv((768, 128, 1, 1, 0), "convf_rpn"))
      .add(new ReLU()))
    convTable.add(new Sequential()
      .add(conv((768, 384, 1, 1, 0), "convf_2"))
      .add(new ReLU()))
    compose.add(convTable)
    val rpnAndFeature = new ConcatTable()
    rpnAndFeature.add(new Sequential()
      .add(new SelectTable[Tensor[Float], Float](1)).add(createRpn()))
    rpnAndFeature.add(new JoinTable(2, 4))
    compose.add(rpnAndFeature)
    compose
  }

  protected def createFastRcnn(): Sequential[Float] = {
    val model = new Sequential()
      .add(new RoiPooling(pool, pool, ev.fromType(0.0625f)).setName("pool5"))
      .add(new ReshapeInfer(Array(-1, 512 * pool * pool)))
      .add(linear((512 * pool * pool, 4096), "fc6"))
      .add(new ReLU())
      .add(linear((4096, 4096), "fc7"))
      .add(new ReLU())

    val cls = new Sequential().add(linear((4096, 21), "cls_score"))
    if (isTest) cls.add(new SoftMax())
    val clsReg = new ConcatTable()
      .add(cls)
      .add(linear((4096, 84), "bbox_pred"))

    model.add(clsReg)
    model
  }

  def createRpn(): Sequential[Float] = {
    val rpnModel = new Sequential()
    rpnModel.add(conv((128, 384, 3, 1, 1), "rpn_conv1"))
    rpnModel.add(new ReLU())
    val clsAndReg = new ConcatTable()
    clsAndReg.add(conv((384, 100, 1, 1, 0), "rpn_bbox_pred"))
    val clsSeq = new Sequential()
    phase match {
      case TRAIN => clsSeq.add(new ReshapeInfer(Array(0, 2, -1, 0)))
      case TEST =>
        clsSeq.add(new ReshapeInfer(Array(0, 2, -1, 0)))
          .add(new SoftMax())
          .add(new ReshapeInfer(Array(1, 2 * param.anchorNum, -1, 0)))
    }
    clsAndReg.add(clsSeq)
      .add(conv((384, 50, 1, 1, 0), "rpn_cls_score"))
    rpnModel.add(clsAndReg)
    rpnModel
  }

  override val pool: Int = 6
  override val modelType: ModelType = PVANET
  override val modelName: String = modelType.toString
  override val param: FasterRcnnParam = new PvanetParam(phase)

  override def criterion4: ParallelCriterion[Float] = {
    val rpn_loss_bbox = new SmoothL1Criterion2(3.0)
    val rpn_loss_cls = new SoftmaxWithCriterion(ignoreLabel = Some(-1))
    val loss_bbox = new SmoothL1Criterion2(1.0)
    val loss_cls = new SoftmaxWithCriterion(ignoreLabel = Some(-1))
    val pc = new ParallelCriterion()
    pc.add(rpn_loss_cls, 1)
    pc.add(rpn_loss_bbox, 1)
    pc.add(loss_cls, 1)
    pc.add(loss_bbox, 1)
    pc
  }

  override def createTestModel(): Sequential[Float] = ???

  override def createTrainModel(): Sequential[Float] = ???
}
