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

package com.intel.analytics.bigdl.pvanet.tools

import com.intel.analytics.bigdl.nn.{Module, ParallelCriterion}
import com.intel.analytics.bigdl.pvanet.datasets.Roidb.ImageWithRoi
import com.intel.analytics.bigdl.pvanet.datasets.{AnchorToTensor, ImageToTensor}
import com.intel.analytics.bigdl.pvanet.layers.{AnchorTargetLayer, SmoothL1Criterion2, SoftmaxWithCriterion}
import com.intel.analytics.bigdl.pvanet.model.VggParam
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table

object Train {
  val param = new VggParam

  def rpnTest(imageToTensor: ImageToTensor,
    model: Module[Tensor[Float], Table, Float], d: ImageWithRoi): Unit = {
    val res = model.forward(imageToTensor.apply(d))
    require(res.length() == 2)
    val (height: Int, width: Int, nAnchors: Int, target: Table) = getRpnTarget(d, res)
    res(1).asInstanceOf[Tensor[Float]].resize(2, nAnchors * height * width)
    res(2).asInstanceOf[Tensor[Float]].resize(nAnchors * 4 * height * width)
    val output: Float = rpnLoss(res, target)
    println("output from parallel criterion: " + output)
  }

  def getRpnTarget(d: ImageWithRoi, res: Table): (Int, Int, Int, Table) = {
    val clsOut = res(1).asInstanceOf[Tensor[Float]]
    val sizes = clsOut.size()
    //    val regOut = res(2).asInstanceOf[Tensor[Float]]
    val height = sizes(sizes.length - 2)
    val width = sizes(sizes.length - 1)
    val anchorTargetLayer = new AnchorTargetLayer(param)
    val anchors = anchorTargetLayer.generateAnchors(d, height, width)
    val anchorToTensor = new AnchorToTensor(1, height, width)
    val anchorTensors = anchorToTensor.apply(anchors)
    val target = new Table
    target.insert(anchorTensors._1)
    target.insert(anchorTensors._2)
    (height, width, param.anchorNum, target)
  }

  def rpnLoss(res: Table, target: Table): Float = {
    val slc = new SmoothL1Criterion2[Float](3f, 1)
    val sfm = new SoftmaxWithCriterion[Float](ignoreLabel = Some(-1))
    val pc = new ParallelCriterion[Float]()
    pc.add(sfm, 1.0f)
    pc.add(slc, 1.0f)
    val output = pc.forward(res, target)
    output
  }

}
