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

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{OptimMethod, Trigger}
import com.intel.analytics.bigdl.pvanet.datasets.Roidb.ImageWithRoi
import com.intel.analytics.bigdl.pvanet.datasets.{AnchorToTensor, ImageToTensor, PascolVocDataSource}
import com.intel.analytics.bigdl.pvanet.layers.AnchorTargetLayer
import com.intel.analytics.bigdl.pvanet.model.Phase._
import com.intel.analytics.bigdl.pvanet.model.{FasterRcnn, FasterRcnnParam, Model}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.ArrayBuffer

class RpnOptimizer(data: PascolVocDataSource,
  net: FasterRcnn[Float],
  optimMethod: OptimMethod[Float],
  state: Table,
  endWhen: Trigger) {


  // todo: extends
  protected var cacheTrigger: Option[Trigger] = None
  protected var cachePath: Option[String] = None
  protected var isOverWrite: Boolean = false
  protected var validationTrigger: Option[Trigger] = None
  protected val validationMethods: ArrayBuffer[ValidationMethod[Float]] = new ArrayBuffer()

  def setCache(path: String, trigger: Trigger): this.type = {
    this.cachePath = Some(path)
    this.cacheTrigger = Some(trigger)
    this
  }

  protected def saveModel(postfix: String = ""): this.type = {
    if (this.cachePath.isDefined) {
      net.rpn.save(s"${cachePath.get}.rpn_model$postfix", isOverWrite)
      net.fastRcnn.save(s"${cachePath.get}.fastRcnn_model$postfix", isOverWrite)
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

  def addValidation(validationMethod: ValidationMethod[Float]): this.type = {
    validationMethods.append(validationMethod)
    this
  }

  def overWriteCache(): this.type = {
    isOverWrite = true
    this
  }

  def getAnchorTarget(d: ImageWithRoi, output: Table): Table = {
    val sizes = output(2).asInstanceOf[Tensor[Float]].size()
    val height = sizes(sizes.length - 2)
    val width = sizes(sizes.length - 1)
    val anchorTargetLayer = new AnchorTargetLayer(net.param)
    val anchors = anchorTargetLayer.generateAnchors(d, height, width)
    val anchorToTensor = new AnchorToTensor(1, height, width)
    val anchorTensors = anchorToTensor.apply(anchors)
    val target = new Table
    target.insert(anchorTensors._1)
    target.insert(anchorTensors._2)
    target
  }

  val imageToTensor = new ImageToTensor(batchSize = 1)


  def optimizeRpn(): Module[Tensor[Float], Table, Float] = {
    optimizeRpn(net.featureAndRpnNet)
  }

  def optimizeRpn(rpnModel: Module[Tensor[Float], Table, Float]):
  Module[Tensor[Float], Table, Float] = {
    val (weights, grad) = rpnModel.getParameters()
    var wallClockTime = 0L
    var count = 0

    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("neval") = state.get[Int]("neval").getOrElse(1)
    data.reset()
    data.shuffle()
    while (!endWhen(state)) {
      val start = System.nanoTime()
      val d = data.next()
      val input = imageToTensor(d)
      val dataFetchTime = System.nanoTime()
      rpnModel.zeroGradParameters()
      val output = rpnModel.forward(input)
      val target = getAnchorTarget(d, output)
      val loss = net.rpnCriterion.forward(output, target)
      val gradOutput = net.rpnCriterion.backward(output, target)
      rpnModel.backward(input, gradOutput)
      optimMethod.optimize(_ => (loss, grad), weights, state)
      val end = System.nanoTime()
      wallClockTime += end - start
      count += input.size(1)
      println(s"[Epoch ${state[Int]("epoch")} $count/${data.total()}][Iteration ${
        state[Int]("neval")
      }][Wall Clock ${
        wallClockTime / 1e9
      }s] loss is $loss, iteration time is ${(end - start) / 1e9}s data " +
        s"fetch time is " +
        s"${(dataFetchTime - start) / 1e9}s, train time ${(end - dataFetchTime) / 1e9}s." +
        s" Throughput is ${input.size(1).toDouble / (end - start) * 1e9} img / second")
      state("neval") = state[Int]("neval") + 1

      if (count >= data.total()) {
        state("epoch") = state[Int]("epoch") + 1
        data.reset()
        data.shuffle()
        count = 0
      }
      cacheTrigger.foreach(trigger => {
        if (trigger(state) && cachePath.isDefined) {
          println(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to ${cachePath.get}")
          saveModel(s".${state[Int]("neval")}")
          saveState(state, s".${state[Int]("neval")}")
        }
      })
    }
    rpnModel
  }

  def optimizeFastRcnn(fastRcnnModel: Module[Table, Table, Float]): Module[Table, Table, Float] = {
    val (weights, grad) = fastRcnnModel.getParameters()
    var wallClockTime = 0L
    var count = 0

    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("neval") = state.get[Int]("neval").getOrElse(1)
    data.reset()
    data.shuffle()
    while (!endWhen(state)) {
      val start = System.nanoTime()
      val d = data.next()
      val input1 = imageToTensor(d)
      // todo
      val rois = Tensor[Float]
      val input = new Table
      input.insert(input1)
      input.insert(rois)
      val dataFetchTime = System.nanoTime()
      fastRcnnModel.zeroGradParameters()
      val output = fastRcnnModel.forward(input)
      val target = getAnchorTarget(d, output)
      val loss = net.rpnCriterion.forward(output, target)
      val gradOutput = net.rpnCriterion.backward(output, target)
      fastRcnnModel.backward(input, gradOutput)
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
      cacheTrigger.foreach(trigger => {
        if (trigger(state) && cachePath.isDefined) {
          println(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to ${cachePath.get}")
          saveModel(s".${state[Int]("neval")}")
          saveState(state, s".${state[Int]("neval")}")
        }
      })
    }
    fastRcnnModel
  }

  def rpnGenerate(data: PascolVocDataSource, net: FasterRcnn[Float]): Unit = {
    var param = FasterRcnnParam.getNetParam(Model.withName(net.modelName), TEST)
  }
}


