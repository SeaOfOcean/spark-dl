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
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.Table

object Vgg_16_RPN {
  val defName = "/home/xianyan/objectRelated/faster_rcnn_models/VGG16/faster_rcnn_alt_opt/rpn_test.pt"
  val modelName = "/home/xianyan/objectRelated/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel"
  val caffeReader = new CaffeReader[Float](defName, modelName)

  def vggNet() = {
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

    val rpn_conv = caffeReader
    vggNet.add(caffeReader.mapConvolution("rpn_conv/3x3"))
    vggNet.add(new ReLU[Float](true))

    vggNet
  }

  def apply(): Module[Tensor[Float], Table, Float] = {
    val clsAndReg = new ConcatTable[Table, Float]()
    clsAndReg.add(caffeReader.mapConvolution("rpn_cls_score"))
      .add(caffeReader.mapConvolution("rpn_bbox_pred"))
    val rpnModel = new Sequential[Tensor[Float], Table, Float]()
    rpnModel.add(vggNet)
    rpnModel.add(clsAndReg)

    rpnModel
  }
}



