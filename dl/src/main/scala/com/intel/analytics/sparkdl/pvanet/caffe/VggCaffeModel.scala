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

package com.intel.analytics.sparkdl.pvanet.caffe

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.pvanet.Config
import com.intel.analytics.sparkdl.pvanet.layers.RoiPooling
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.{File, Table}

object VggCaffeModel {
  val scales = Array[Float](8, 16, 32)
  val ratios = Array(0.5f, 1.0f, 2.0f)
  var caffeReader: CaffeReader[Float] = null
  val defName = "/home/xianyan/objectRelated/faster_rcnn_models/VGG16/" +
    "faster_rcnn_alt_opt/rpn_test.pt"
  val modelName = "/home/xianyan/objectRelated/faster_rcnn_models/" +
    "VGG16_faster_rcnn_final.caffemodel"


  def vgg16: Module[Tensor[Float], Tensor[Float], Float] = {
    val cache = Config.modelPath + "/" + "vgg16WithCaffeParams.obj"
    if (Config.existFile(cache)) {
      return File.loadObj[Module[Tensor[Float], Tensor[Float], Float]](cache)
    }
    if (caffeReader == null) {
      caffeReader = new CaffeReader[Float](defName, modelName)
    }
    val vggNet = new Sequential[Tensor[Float], Tensor[Float], Float]()
    val conv1_1 = caffeReader.mapConvolution("conv1_1")
    vggNet.add(conv1_1)
    vggNet.add(new ReLU[Float](true))
    val conv1_2 = caffeReader.mapConvolution("conv1_2")
    vggNet.add(conv1_2)
    assert(conv1_2.nInputPlane == 64)
    assert(conv1_2.nOutputPlane == 64)
    //    vggNet.add(new SpatialConvolution[Float](64, 64, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[Float](true))
    vggNet.add(new SpatialMaxPooling[Float](2, 2, 2, 2))

    val conv2_1 = caffeReader.mapConvolution("conv2_1")
    vggNet.add(conv2_1)
    assert(conv2_1.nInputPlane == 64)
    assert(conv2_1.nOutputPlane == 128)
    //    vggNet.add(new SpatialConvolution[Float](64, 128, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[Float](true))
    val conv2_2 = caffeReader.mapConvolution("conv2_2")
    vggNet.add(conv2_2)
    assert(conv2_2.nInputPlane == 128)
    assert(conv2_2.nOutputPlane == 128)
    //    vggNet.add(new SpatialConvolution[Float](128, 128, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[Float](true))
    vggNet.add(new SpatialMaxPooling[Float](2, 2, 2, 2))

    val conv3_1 = caffeReader.mapConvolution("conv3_1")
    vggNet.add(conv3_1)
    assert(conv3_1.nInputPlane == 128)
    assert(conv3_1.nOutputPlane == 256)
    //    vggNet.add(new SpatialConvolution[Float](128, 256, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[Float](true))
    val conv3_2 = caffeReader.mapConvolution("conv3_2")
    vggNet.add(conv3_2)
    assert(conv3_2.nInputPlane == 256)
    assert(conv3_2.nOutputPlane == 256)
    //    vggNet.add(new SpatialConvolution[Float](256, 256, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[Float](true))
    val conv3_3 = caffeReader.mapConvolution("conv3_3")
    vggNet.add(conv3_3)
    assert(conv3_3.nInputPlane == 256)
    assert(conv3_3.nOutputPlane == 256)
    //    vggNet.add(new SpatialConvolution[Float](256, 256, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[Float](true))
    vggNet.add(new SpatialMaxPooling[Float](2, 2, 2, 2))

    val conv4_1 = caffeReader.mapConvolution("conv4_1")
    vggNet.add(conv4_1)
    assert(conv4_1.nInputPlane == 256)
    assert(conv4_1.nOutputPlane == 512)
    //    vggNet.add(new SpatialConvolution[Float](256, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[Float](true))
    val conv4_2 = caffeReader.mapConvolution("conv4_2")
    vggNet.add(conv4_2)
    assert(conv4_2.nInputPlane == 512)
    assert(conv4_2.nOutputPlane == 512)
    //    vggNet.add(new SpatialConvolution[Float](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[Float](true))
    val conv4_3 = caffeReader.mapConvolution("conv4_3")
    vggNet.add(conv4_3)
    assert(conv4_3.nInputPlane == 512)
    assert(conv4_3.nOutputPlane == 512)
    //    vggNet.add(new SpatialConvolution[Float](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[Float](true))
    vggNet.add(new SpatialMaxPooling[Float](2, 2, 2, 2))

    val conv5_1 = caffeReader.mapConvolution("conv5_1")
    vggNet.add(conv5_1)
    assert(conv5_1.nInputPlane == 512)
    assert(conv5_1.nOutputPlane == 512)
    //    vggNet.add(new SpatialConvolution[Float](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[Float](true))
    val conv5_2 = caffeReader.mapConvolution("conv5_2")
    vggNet.add(conv5_2)
    assert(conv5_2.nInputPlane == 512)
    assert(conv5_2.nOutputPlane == 512)
    //    vggNet.add(new SpatialConvolution[Float](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[Float](true))
    val conv5_3 = caffeReader.mapConvolution("conv5_3")
    vggNet.add(conv5_3)
    assert(conv5_3.nInputPlane == 512)
    assert(conv5_3.nOutputPlane == 512)
    //    vggNet.add(new SpatialConvolution[Float](512, 512, 3, 3, 1, 1, 1, 1))
    vggNet.add(new ReLU[Float](true))

    File.save(vggNet, cache, true)
    vggNet
  }

  def rpn: Module[Tensor[Float], Table, Float] = {
    val cache = Config.modelPath + "/" + "RPNWithCaffeParams.obj"
    if (Config.existFile(cache)) {
      return File.loadObj[Module[Tensor[Float], Table, Float]](cache)
    }
    if (caffeReader == null) {
      caffeReader = new CaffeReader[Float](defName, modelName)
    }
    val rpnModel = new Sequential[Tensor[Float], Table, Float]()
    rpnModel.add(caffeReader.mapConvolution("rpn_conv/3x3"))
    rpnModel.add(new ReLU[Float](true))
    val clsAndReg = new ConcatTable[Table, Float]()
    clsAndReg.add(caffeReader.mapConvolution("rpn_cls_score"))
      .add(caffeReader.mapConvolution("rpn_bbox_pred"))
    rpnModel.add(clsAndReg)
    File.save(rpnModel, cache, true)
    rpnModel
  }

  def vgg16rpn: Module[Tensor[Float], Table, Float] = {
    val cache = Config.modelPath + "/" + "vgg16RPNWithCaffeParams.obj"
    if (Config.existFile(cache)) {
      return File.loadObj[Module[Tensor[Float], Table, Float]](cache)
    }
    if (caffeReader == null) {
      caffeReader = new CaffeReader[Float](defName, modelName)
    }
    val vggRpnModel = new Sequential[Tensor[Float], Table, Float]()
    vggRpnModel.add(vgg16)
    vggRpnModel.add(rpn)
    File.save(vggRpnModel, cache, true)
    vggRpnModel
  }

  def vggFull: Module[Tensor[Float], Table, Float] = {
    val cache = Config.modelPath + "/" + "vgg16FullWithCaffeParams.obj"
    if (Config.existFile(cache)) {
      return File.loadObj[Module[Tensor[Float], Table, Float]](cache)
    }
    if (caffeReader == null) {
      caffeReader = new CaffeReader[Float](defName, modelName)
    }
    val clsAndReg = new ConcatTable[Table, Float]()
    val cls = new Sequential[Tensor[Float], Tensor[Float], Float]()
    cls.add(caffeReader.mapConvolution("rpn_cls_score"))
    //    cls.add(new Reshape[Float](Array(0,2, -1, 0)))
    cls.add(new SoftMax[Float]())
    //    cls.add(new Reshape[Float](Array(0, 18, -1, 0)))
    clsAndReg.add(cls)
      .add(caffeReader.mapConvolution("rpn_bbox_pred"))
    val fullModel = new Sequential[Tensor[Float], Table, Float]()
    fullModel.add(vgg16)
    fullModel.add(clsAndReg)
    val proposalInput = new ConcatTable[Table, Float]()
    proposalInput.add(fullModel)
    //    proposalInput.add()

    File.save(fullModel, cache, true)
    fullModel
  }

  def fastRcnn: Module[Table, Table, Float] = {
    val cache = Config.modelPath + "/" + "vgg16FastRcnnWithCaffeParams.obj"
    if (Config.existFile(cache)) {
      return File.loadObj[Module[Table, Table, Float]](cache)
    }
    if (caffeReader == null) {
      caffeReader = new CaffeReader[Float](defName, modelName)
    }
    val model = new Sequential[Table, Table, Float]()
    model.add(new RoiPooling[Float](7, 7, 0.0625f))
    //    model.add(new Reshape[Float]())

    //    model.add(new Linear[Float](4096, 4096).setName("fc6"))
    val fc6 = caffeReader.mapInnerProduct("fc6")
    model.add(fc6)
    println("fc6", "out", fc6.weight.size(1), "in", fc6.weight.size(2))
    model.add(new ReLU[Float]())
    model.add(new Dropout[Float]())

    //    model.add(new Linear[Float](4096, 4096).setName("fc7"))
    val fc7 = caffeReader.mapInnerProduct("fc7")
    println("fc7", "out", fc7.weight.size(1), "in", fc7.weight.size(2))
    model.add(fc7)
    model.add(new ReLU[Float]())
    model.add(new Dropout[Float]())

    val clsReg = new ConcatTable[Table, Float]()

    val cls = new Sequential[Tensor[Float], Tensor[Float], Float]()
    //    cls.add(new Linear[Float](4096, 21).setName("cls_score"))
    val cls_score = caffeReader.mapInnerProduct("cls_score")
    println("cls_score", "out", cls_score.weight.size(1), "in", cls_score.weight.size(2))
    cls.add(cls_score)
    cls.add(new SoftMax[Float]())
    clsReg.add(cls)
    //    clsReg.add(new Linear[Float](4096, 84))
    cls.add(caffeReader.mapInnerProduct("bbox_pred"))

    model.add(clsReg)

    File.save(model, cache, true)
    model
  }

  def main(args: Array[String]): Unit = {
    fastRcnn
  }


}
