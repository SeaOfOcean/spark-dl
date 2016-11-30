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

import com.intel.analytics.bigdl.nn.{Sequential, _}
import com.intel.analytics.bigdl.pvanet.caffe.CaffeReader
import com.intel.analytics.bigdl.pvanet.layers.{Reshape2, RoiPooling}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class FasterPvanet[@specialized(Float, Double) T: ClassTag](caffeReader: CaffeReader[T] = null)(implicit ev: TensorNumeric[T])
  extends FasterRCNN[T](caffeReader) {

  var pvanet: Sequential[Tensor[T], Tensor[T], T] = _

  def concatNeg(name: String): Concat[T] = {
    val concat = new Concat[T](2)
    concat.add(new Identity[T]())
    concat.add(new Power[T](1, -1, 0).setName(name))
    concat
  }

  def addScale(module: Sequential[Tensor[T], Tensor[T], T], sizes: Array[Int], name: String): Unit = {
    val sc = scale(sizes, name)
    module.add(sc._1)
    module.add(sc._2)
    module.add(new ReLU[T]())
  }

  def addConvComponent(compId: Int, index: Int, p: Array[(Int, Int, Int, Int, Int)]) = {
    val label = s"${compId}_$index"
    val convTable = new ConcatTable[Tensor[T], T]
    val conv_left = new Sequential[Tensor[T], Tensor[T], T]()
    var i = 0
    if (index == 1) {
      conv_left.add(conv(p(i), s"conv$label/1/conv"))
      i += 1
    } else {
      conv_left.add(conv(p(i), s"conv$label/1/conv"))
      i += 1
    }

    conv_left.add(new ReLU[T]())
    conv_left.add(conv(p(i), s"conv$label/2/conv"))
    i += 1
    conv_left.add(concatNeg(s"conv$label/2/neg"))
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
      convTable.add(new Identity[T]())
    }
    pvanet.add(convTable)
    pvanet.add(new CAddTable[T]())
  }

  def addInception(module: Sequential[Tensor[T], Tensor[T], T], label: String, index: Int, p: Array[(Int, Int, Int, Int, Int)]): Unit = {
    val left = new Sequential[Tensor[T], Tensor[T], T]()
    val incep = new Concat[T](4)

    var i = 0
    val com1 = new Sequential[Tensor[T], Tensor[T], T]()
    com1.add(conv(p(i), s"conv$label/incep/0/conv"))
    i += 1
    com1.add(new ReLU[T]())
    incep.add(com1)

    val com2 = new Sequential[Tensor[T], Tensor[T], T]()
    com2.add(conv(p(i), s"conv$label/incep/1_reduce/conv"))
    i += 1
    com2.add(new ReLU[T]())
    com2.add(conv(p(i), s"conv$label/incep/1_0/conv"))
    i += 1
    com2.add(new ReLU[T]())
    incep.add(com2)

    val com3 = new Sequential[Tensor[T], Tensor[T], T]()
    com3.add(conv(p(i), s"conv$label/incep/2_reduce/conv"))
    i += 1
    com3.add(new ReLU[T]())
    com3.add(conv(p(i), s"conv$label/incep/2_0/conv"))
    i += 1
    com3.add(new ReLU[T]())
    com3.add(conv(p(i), s"conv$label/incep/2_1/conv"))
    i += 1
    com3.add(new ReLU[T]())
    incep.add(com3)

    if (index == 1) {
      val com4 = new Sequential[Tensor[T], Tensor[T], T]()
      com4.add(new SpatialMaxPooling[T](3, 3, 2, 2, 0, 0).setName(s"conv$label/incep/pool"))
      com4.add(conv(p(i), s"conv$label/incep/poolproj/conv"))
      i += 1
      com4.add(new ReLU[T]())
      incep.add(com4)
    }

    left.add(incep)
    left.add(conv(p(i), s"conv$label/out/conv"))
    i += 1
    val table = new ConcatTable[Tensor[T], T]
    table.add(left)
    if (index == 1) {
      table.add(conv(p(i), s"conv$label/proj"))
      i += 1
    } else {
      table.add(new Identity[T]())
    }

    val elewiseAdd = new CAddTable[T]()
    module.add(table)
    module.add(elewiseAdd)
  }


  def getModel: Module[Tensor[T], Tensor[T], T] = {
    if (pvanet != null) return pvanet
    pvanet = new Sequential[Tensor[T], Tensor[T], T]()
    pvanet.add(conv((3, 16, 7, 2, 3), "conv1_1/conv"))

    pvanet.add(concatNeg("conv1_1/neg"))
    // todo: sizes
    addScale(pvanet, Array(32), "conv1_1/scale")
    pvanet.add(new SpatialMaxPooling[T](3, 3, 2, 2, 0, 0).setName("pool1"))


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

    val inceptions4_5 = new Sequential[Tensor[T], Tensor[T], T]()

    val inceptions4 = new Sequential[Tensor[T], Tensor[T], T]()
    addInception(inceptions4, "4_1", 1, Array((128, 64, 1, 2, 0), (128, 48, 1, 2, 0),
      (48, 128, 3, 1, 1), (128, 24, 1, 2, 0), (24, 48, 3, 1, 1), (48, 48, 3, 1, 1),
      (128, 128, 1, 1, 0), (368, 256, 1, 1, 0), (128, 256, 1, 2, 0)))
    for (i <- 2 to 4) {
      addInception(inceptions4, s"4_$i", i, Array((256, 64, 1, 1, 0), (256, 64, 1, 1, 0),
        (64, 128, 3, 1, 1), (256, 24, 1, 1, 0), (24, 48, 3, 1, 1),
        (48, 48, 3, 1, 1), (240, 256, 1, 1, 0)))
    }
    inceptions4_5.add(inceptions4)


    val seq5 = new Sequential[Tensor[T], Tensor[T], T]()
    val inceptions5 = new Sequential[Tensor[T], Tensor[T], T]()
    addInception(inceptions5, "5_1", 1, Array((256, 64, 1, 2, 0), (256, 96, 1, 2, 0),
      (96, 192, 3, 1, 1), (256, 32, 1, 2, 0), (32, 64, 3, 1, 1), (64, 64, 3, 1, 1),
      (256, 128, 1, 1, 0), (448, 384, 1, 1, 0), (256, 384, 1, 2, 0)))
    for (i <- 2 to 4) {
      addInception(inceptions5, s"5_$i", i, Array((384, 64, 1, 1, 0), (384, 96, 1, 1, 0),
        (96, 192, 3, 1, 1), (384, 32, 1, 1, 0), (32, 64, 3, 1, 1), (64, 64, 3, 1, 1),
        (320, 384, 1, 1, 0)))
    }
    seq5.add(inceptions5)
    seq5.add(spatialFullConv((384, 384, 4, 2, 1), "upsample"))

    val concat5 = new Concat[T](2)
    concat5.add(new Identity[T]())
    concat5.add(seq5)

    inceptions4_5.add(concat5)

    val concatIncept = new Concat[T](2)
    concatIncept.add(new SpatialMaxPooling[T](3, 3, 2, 2, 0, 0))
    concatIncept.add(inceptions4_5)

    pvanet.add(concatIncept)
    pvanet
  }

  override def featureNet = getModel

  def rpn: Module[Tensor[T], Table, T] = {
    val rpnModel = new Sequential[Tensor[T], Table, T]()
    rpnModel.add(conv((128, 384, 3, 1, 1), "rpn_conv1"))
    rpnModel.add(new ReLU[T]())
    val clsAndReg = new ConcatTable[Table, T]()
    clsAndReg.add(conv((384, 50, 1, 1, 0), "rpn_cls_score"))
    clsAndReg.add(conv((384, 100, 1, 1, 0), "rpn_bbox_pred"))
    rpnModel.add(clsAndReg)
    rpnModel
  }

  def fastRcnn: Module[Table, Table, T] = {
    val model = new Sequential[Table, Table, T]()
    model.add(new RoiPooling[T](6, 6, ev.fromType(0.0625)))
    model.add(new Reshape2[T](Array(-1, 18432), Some(false)))

    model.add(linear((18432, 4096), "fc6"))
    model.add(new ReLU[T]())
    model.add(linear((4096, 4096), "fc7"))
    model.add(new ReLU[T]())

    val clsReg = new ConcatTable[Table, T]()

    val cls = new Sequential[Tensor[T], Tensor[T], T]()
    cls.add(linear((4096, 21), "cls_score"))
    cls.add(new SoftMax[T]())
    clsReg.add(cls)
    clsReg.add(linear((4096, 84), "bbox_pred"))

    model.add(clsReg)
    model
  }

  override val modelName: String = "pvanet"
}

object FasterPvanet {
  val defName = "/home/xianyan/objectRelated/pvanet/full/test.pt"
  val modelName = "/home/xianyan/objectRelated/pvanet/full/test.model"
  val caffeReader: CaffeReader[Float] = new CaffeReader(defName, modelName, "pvanet")
  var modelWithCaffeWeight: FasterRCNN[Float] = null

  def getModelWithCaffeWeight: FasterRCNN[Float] = {
    if (modelWithCaffeWeight == null) modelWithCaffeWeight = new FasterPvanet[Float](caffeReader)
    modelWithCaffeWeight
  }

  def main(args: Array[String]): Unit = {
    val pvanet = getModelWithCaffeWeight
    pvanet.featureNetWithCache
    pvanet.rpnWithCache
    pvanet.fastRcnnWithCache
  }
}
