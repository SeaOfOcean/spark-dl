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

import com.intel.analytics.bigdl.pvanet.caffe.CaffeReader
import org.scalatest.{FlatSpec, Matchers}

/**
 * Created by xianyan on 12/17/16.
 */
class FasterRcnnSpec extends FlatSpec with Matchers {
  val defFile = "/home/xianyan/objectRelated/faster_rcnn_models/VGG16/" +
    "faster_rcnn_alt_opt/rpn_test.pt"
  val modelFile = "/home/xianyan/objectRelated/faster_rcnn_models/" +
    "VGG16_faster_rcnn_final.caffemodel"
  val model1 = new VggFRcnn(Phase.TEST)
    .copyFromCaffe(new CaffeReader[Float](defFile, modelFile))
  val model2 = new VggFRcnn(Phase.TEST).loadFromCaffeOrCache(defFile, modelFile)

  model1.getTestModel.getParameters() should be(model2.getTestModel.getParameters())
}
