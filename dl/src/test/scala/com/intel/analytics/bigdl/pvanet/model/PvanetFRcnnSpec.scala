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

import com.intel.analytics.bigdl.nn.Utils
import com.intel.analytics.bigdl.pvanet.utils.FileUtil
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table
import org.scalatest.FlatSpec

class PvanetFRcnnSpec extends FlatSpec {
  "faster rcnn with pvanet net" should "work properly" in {
    val caffeDef = "/home/xianyan/objectRelated/pvanet/full/test.pt"
    val caffeModel = "/home/xianyan/objectRelated/pvanet/full/test.model"
    val vggFrcnn = FasterRcnn(Model.PVANET, Phase.TEST, pretrained = (caffeDef, caffeModel))
    val model = vggFrcnn.getModel
    val input = new Table()
    FileUtil.middleRoot = FileUtil.getFile("middle/pvanet/14/")
    input.insert(FileUtil.loadFeatures[Float]("data"))
    input.insert(FileUtil.loadFeatures[Float]("im_info").resize(3))
    val result = model.forward(input).asInstanceOf[Table]

    val name2Module = Utils.getNamedModules[Float](model)

    def compare(name: String): Unit = compare2(name, name)

    def compare2(name1: String, name2: String): Unit = FileUtil.assertEqual[Float](name1,
      name2Module(name2).output.asInstanceOf[Tensor[Float]], 1e-2)

    compare2("conv3_4", "conv4_1/incep/pool")

    compare2("conv5_4", "conv5_4")
  }
}
