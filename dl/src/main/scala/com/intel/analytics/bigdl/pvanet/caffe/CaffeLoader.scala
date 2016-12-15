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

package com.intel.analytics.bigdl.pvanet.caffe

import java.io.{File, FileInputStream, InputStream, InputStreamReader}

import caffe.Caffe
import caffe.Caffe.{LayerParameter, NetParameter}
import com.google.protobuf.{CodedInputStream, TextFormat}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.imagenet.AlexNet
import com.intel.analytics.bigdl.nn.{Module, Utils}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.reflect.ClassTag

class CaffeLoader[@specialized(Float, Double) T: ClassTag](defName: String, modelName: String,
  matchAll: Boolean = true //if match all modules with parameters
)(implicit ev: TensorNumeric[T]) {
  val isOverwrite = true
  var netparam: Caffe.NetParameter = _
  var name2CaffeLayer: Map[String, LayerParameter] = loadCaffe(defName, modelName)

  private def loadCaffe(prototxtName: String, modelName: String): Map[String, LayerParameter] = {
    if (name2CaffeLayer == null) {
      name2CaffeLayer = Map[String, LayerParameter]()
      netparam = loadBinary(prototxtName, modelName)
      assert(netparam.getLayerCount > 0, "only support proto V2")
      for (i <- 0 until netparam.getLayerCount) {
        val layer = netparam.getLayer(i)
        name2CaffeLayer += (layer.getName -> layer)
      }
    }
    name2CaffeLayer
  }

  private def loadBinary(prototxtName: String, modelName: String): Caffe.NetParameter = {
    val f: File = new File(prototxtName)
    assert(f.exists(), prototxtName + "does not exists")
    val rawInput: InputStream = new FileInputStream(f)
    val reader: InputStreamReader = new InputStreamReader(rawInput, "ASCII")
    val builder: Caffe.NetParameter.Builder = NetParameter.newBuilder
    TextFormat.merge(reader, builder)
    println("start loading caffe model")
    val cis = CodedInputStream.newInstance(new FileInputStream(modelName))
    cis.setSizeLimit(1000000000)
    builder.mergeFrom(cis)
    println("load caffe model done")
    builder.build()
  }

  private def loadModule(name: String, hasBias: Boolean = true): (Tensor[T], Tensor[T]) = {
    val layer = name2CaffeLayer(name)
    var weight: Tensor[T] = null
    var bias: Tensor[T] = null
    if (layer.getBlobsCount == 0) {
      return (null, null)
    }
    val wB = layer.getBlobs(0)
    var nInputPlane, nOutputPlane, kW, kH = 0
    if (wB.hasShape && wB.getShape.getDimCount >= 2) {
      nInputPlane = wB.getShape.getDim(1).toInt
      nOutputPlane = wB.getShape.getDim(0).toInt
      if (layer.getType != "InnerProduct") {
        kW = wB.getShape.getDim(3).toInt
        kH = wB.getShape.getDim(2).toInt
      }
    }
    else {
      if (layer.getType == "InnerProduct") {
        nInputPlane = wB.getWidth
        nOutputPlane = wB.getHeight
      }
      else {
        nInputPlane = wB.getChannels
        nOutputPlane = wB.getNum
        kW = wB.getWidth
        kH = wB.getHeight
      }
    }
    val weightList = layer.getBlobs(0).getDataList
    val weightData: Array[T] = new Array[T](weightList.size())
    for (i <- 0 until weightList.size()) {
      weightData(i) = ev.fromType[Float](weightList.get(i))
    }
    weight = Tensor(Storage(weightData))
    layer.getType match {
      case "InnerProduct" =>
        printf("%s: %d %d (%d, %d)\n", name, 1, 1, nInputPlane, nOutputPlane)
        weight.resize(Array(layer.getConvolutionParam.getGroup, 1, 1, nInputPlane, nOutputPlane))
      case "Scale" =>
      case "Deconvolution" =>
      case _ => weight.resize(Array(layer.getConvolutionParam.getGroup, nOutputPlane,
        nInputPlane, kW, kH))
    }

    if (hasBias) {
      val biasList = layer.getBlobs(1).getDataList
      val biasData: Array[T] = new Array[T](biasList.size())
      for (i <- 0 until biasList.size()) {
        biasData(i) = ev.fromType[Float](biasList.get(i))
      }
      bias = Tensor(Storage(biasData))
    }

    (weight, bias)
  }

  /**
   * copy caffe parameters to module
   * if matchAll, throw an exception if some layers are not mapped
   *
   * @param model
   * @return
   */
  def copyParameters(model: Module[T]): Module[T] = {
    val namedModules = Utils.getNamedModules[T](model)

    def copyParameter(name: String, mod: Module[T]): Unit = {
      if (mod.parameters() == null) return
      if (!name2CaffeLayer.contains(name)) {
        if (matchAll) {
          throw new Exception(s"module $name cannot map a layer in caffe model")
        }
        println(s"$name uses initialized parameters")
        return
      }
      val (weight, bias) = loadModule(name)
      if (weight == null) {
        println(s"$name uses initialized parameters")
        return
      }
      mod.parameters()._1(0).copy(weight)
      if (bias != null) {
        mod.parameters()._1(1).copy(bias)
      }
      println(s"load ${mod.getName()} done")
    }

    namedModules.foreach {
      case (name: String, mod: Module[T]) => {
        copyParameter(name, mod)
      }
    }
    model
  }
}

object CaffeLoader {

  def load[@specialized(Float, Double) T: ClassTag](model: Module[T],
    defPath: String, modelPath: String, matchAll: Boolean = true)
    (implicit ev: TensorNumeric[T]): Module[T] = {
    val caffeLoader = new CaffeLoader[T](defPath, modelPath, matchAll)
    caffeLoader.copyParameters(model)
  }

  def main(args: Array[String]): Unit = {
    val module = Module.loadCaffe[Float](AlexNet(1000),
      "data/model/alexnet/deploy.prototxt",
      "data/model/alexnet/bvlc_alexnet.caffemodel"
    )
  }
}
