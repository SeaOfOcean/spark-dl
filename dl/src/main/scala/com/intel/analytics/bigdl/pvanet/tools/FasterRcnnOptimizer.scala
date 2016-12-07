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

import com.intel.analytics.bigdl.dataset.LocalDataSource
import com.intel.analytics.bigdl.nn.{Module, ParallelCriterion}
import com.intel.analytics.bigdl.optim.{OptimMethod, Trigger}
import com.intel.analytics.bigdl.pvanet.datasets.{AnchorToTensor, ImageToTensor, ImageWithRoi, ObjectDataSource}
import com.intel.analytics.bigdl.pvanet.layers.AnchorTargetLayer
import com.intel.analytics.bigdl.pvanet.model.FasterRcnn
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table

class FasterRcnnOptimizer(data: LocalDataSource[ImageWithRoi],
  validationData: ObjectDataSource,
  net: FasterRcnn[Float],
  model: Module[Table, Table, Float],
  optimMethod: OptimMethod[Float],
  criterion: ParallelCriterion[Float],
  state: Table,
  endWhen: Trigger) {

  // todo: extends
  protected var cacheTrigger: Option[Trigger] = None
  protected var cachePath: Option[String] = None
  protected var isOverWrite: Boolean = false
  protected var validationTrigger: Option[Trigger] = None

  def setCache(path: String, trigger: Trigger): this.type = {
    this.cachePath = Some(path)
    this.cacheTrigger = Some(trigger)
    this
  }

  protected def saveModel(postfix: String = ""): this.type = {
    if (this.cachePath.isDefined) {
      model.save(s"${cachePath.get}.model$postfix", isOverWrite)
    }
    this
  }

  protected def saveState(state: Table, postfix: String = ""): this.type = {
    if (this.cachePath.isDefined) {
      state.save(s"${cachePath.get}.state$postfix", isOverWrite)
    }
    this
  }

  def setValidationTrigger(trigger: Trigger): this.type = {
    this.validationTrigger = Some(trigger)
    this
  }

  def overWriteCache(): this.type = {
    isOverWrite = true
    this
  }

  /**
   *
   * @param d      image data class
   * @param output (rpn_cls, rpn_reg, cls, reg, rois)
   * @return
   */
  def getAnchorTarget(target: Table, d: ImageWithRoi, output: Table): Unit = {
    val sizes = output(2).asInstanceOf[Tensor[Float]].size()
    val height = sizes(sizes.length - 2)
    val width = sizes(sizes.length - 1)
    val anchorTargetLayer = new AnchorTargetLayer(net.param)
    val anchors = anchorTargetLayer.generateAnchors(d, height, width)
    val anchorTensors = new AnchorToTensor(1, height, width).apply(anchors)
    target.insert(anchorTensors._1)
    target.insert(anchorTensors._2)
  }

  def getProposalTarget(target: Table, d: ImageWithRoi, output: Table): Unit = {
    println(output.length())
    val pTargets = output(5).asInstanceOf[Table]
    target.insert(pTargets(1))
    target.insert(pTargets(2))
  }

  val imageToTensor = new ImageToTensor(batchSize = 1)

  def optimize(): Module[Table, Table, Float] = {
    val (weights, grad) = model.getParameters()
    var wallClockTime = 0L
    var count = 0

    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("neval") = state.get[Int]("neval").getOrElse(1)
    data.reset()
    data.shuffle()
    while (!endWhen(state)) {
      val start = System.nanoTime()
      val d = data.next()
      val input = new Table
      input.insert(ImageToTensor(d))
      input.insert(d.imInfo.get)
      input.insert(d.gtBoxes.get)
      val dataFetchTime = System.nanoTime()
      model.zeroGradParameters()
      // (rpn_cls, rpn_reg, cls, reg, proposalTargets)
      val output = model.forward(input)
      val target = new Table
      getAnchorTarget(target, d, output)
      getProposalTarget(target, d, output)

      val loss = criterion.forward(output, target)
      val gradOutput = criterion.backward(output, target)
      model.backward(input, gradOutput)
      optimMethod.optimize(_ => (loss, grad), weights, state)
      val end = System.nanoTime()
      wallClockTime += end - start
      count += 1
      println(s"[Epoch ${state[Int]("epoch")} $count/${data.total()}][Iteration ${
        state[Int]("neval")
      }][Wall Clock ${
        wallClockTime / 1e9
      }s] loss is $loss, iteration time is ${(end - start) / 1e9}s data " +
        s"fetch time is " +
        s"${(dataFetchTime - start) / 1e9}s, train time ${(end - dataFetchTime) / 1e9}s." +
        s" Throughput is ${1.toDouble / (end - start) * 1e9} img / second")
      state("neval") = state[Int]("neval") + 1

      if (count >= data.total()) {
        state("epoch") = state[Int]("epoch") + 1
        data.reset()
        data.shuffle()
        count = 0
      }
      validate(wallClockTime)
      cacheTrigger.foreach(trigger => {
        if (trigger(state) && cachePath.isDefined) {
          println(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to ${cachePath.get}")
          saveModel(s".${state[Int]("neval")}")
          saveState(state, s".${state[Int]("neval")}")
        }
      })
    }
    validate(wallClockTime)
    model
  }

  private def validate(wallClockTime: Long): Unit = {
    validationTrigger.foreach(trigger => {
      if (trigger(state)) {
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Validate model...")
        net.evaluate
        validationData.reset()
        net.copyParamToTest(model)
        Test.testNet(net, validationData)
        net.train
      }
    })
  }
}


