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

import java.io._

import caffe.Caffe
import caffe.Caffe.{LayerParameter, NetParameter}
import com.google.protobuf.{CodedInputStream, TextFormat}
import com.intel.analytics.bigdl.nn.{SpatialFullConvolutionMap, _}
import com.intel.analytics.bigdl.pvanet.utils.FileUtil
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{File => DlFile}

import scala.reflect.ClassTag

object ModuleType extends Enumeration {
  type ModuleType = Value
  val TensorModule, Criterion = Value
}

class CaffeReader[T: ClassTag](defName: String, modelName: String, netName: String)
  (implicit ev: TensorNumeric[T]) {
  val isOverwrite = true

  private def cachePath(name: String): String = {
    val folder = FileUtil.cachePath + s"/$netName"
    if (!FileUtil.existFile(folder)) {
      new File(folder).mkdirs()
    }
    folder + "/" + name.replaceAll("/", "_")
  }

  var netparam: Caffe.NetParameter = _

  var name2layer = Map[String, LayerParameter]()

  def loadModuleFromFile[M](name: String): Option[M] = {
    try {
      if (FileUtil.existFile(cachePath(name))) return Some(DlFile.load[M](cachePath(name)))
    } catch {
      case ex: Exception => None
    }
    None
  }

  def mapScale(name: String): (CMul[T], CAdd[T]) = {
    val module = loadModuleFromFile[(CMul[T], CAdd[T])](name)
    module match {
      case Some(m) => return m
      case _ =>
    }
    if (name2layer.isEmpty) {
      loadCaffe(defName, modelName)
    }
    val layer = name2layer(name)
    val param = layer.getScaleParam
    val hasBias = param.getBiasTerm
    var cmul: CMul[T] = null
    var cadd: CAdd[T] = null
    val (weight, bias) = loadModule(name2layer(name), name, hasBias)
    cmul = new CMul[T](weight.size())
    cmul.weight.copy(weight)
    if (hasBias) {
      cadd = new CAdd[T](bias.size())
      cadd.bias.copy(bias)
    }
    DlFile.save((cmul, cadd), cachePath(layer.getName), isOverwrite)
    println(s"$name: size(${weight.size().mkString(",")})")
    cmul.setName(name)
    cadd.setName(name)
    (cmul, cadd)
  }


  def mapDeconvolution(name: String): SpatialFullConvolutionMap[T] = {
    val mod = loadModuleFromFile[SpatialFullConvolutionMap[T]](name)
    mod match {
      case Some(m) => return m
      case _ =>
    }
    if (name2layer.isEmpty) {
      loadCaffe(defName, modelName)
    }
    val layer = name2layer(name)
    val param = layer.getConvolutionParam
    val groups = param.getGroup match {
      case 0 => 1
      case _ => param.getGroup
    }
    if (layer.getBlobsCount == 0) {
      //        println("convolution blob is empty")
      return null
    }
    val wB = layer.getBlobs(0)
    val nInputPlane = ((if (wB.hasShape) wB.getShape.getDim(1) else wB.getChannels) * groups).toInt
    val nOutputPlane = (if (wB.hasShape) wB.getShape.getDim(0) else wB.getNum).toInt

    var kW = param.getKernelW
    var kH = param.getKernelH
    var dW = param.getStrideW
    var dH = param.getStrideH
    var padW = param.getPadW
    var padH = param.getPadH

    if (kW == 0 || kH == 0) {
      kW = param.getKernelSize(0)
      kH = kW
    }
    if (dW == 0 || dH == 0) {
      if (param.getStrideCount == 0) {
        dW = 1
      } else {
        dW = param.getStride(0)
      }
      dH = dW

    }
    if (padW == 0 || padH == 0) {
      if (param.getPadCount != 0) {
        padW = param.getPad(0)
        padH = padW
      }
    }
    val hasBias = param.getBiasTerm
    val module = new SpatialFullConvolutionMap[T](SpatialConvolutionMap.oneToOne[T](groups),
      kW, kH, dW, dH, padW, padH, noBias = !hasBias)
    val (weight, bias) = loadModule(name2layer(name), name, hasBias)
    module.weight.copy(weight)
    if (hasBias) module.bias.copy(bias)
    DlFile.save(module, cachePath(layer.getName), isOverwrite)
    module.setName(name)
    println(s"$name: ($nInputPlane, $nOutputPlane, $kW, $kH, $dW, $dH, $padW, $padH)")
    module
  }

  def mapInnerProduct(name: String): Linear[T] = {
    val mod = loadModuleFromFile[Linear[T]](name)
    mod match {
      case Some(m) => return m
      case _ =>
    }
    if (name2layer.isEmpty) {
      loadCaffe(defName, modelName)
    }

    val layer = name2layer(name)
    val param = layer.getInnerProductParam
    val wB = layer.getBlobs(0)
    val nInputPlane = if (wB.hasShape) wB.getShape.getDim(1) else wB.getWidth
    val nOutputPlane = param.getNumOutput
    val module = new Linear[T](nInputPlane.toInt, nOutputPlane)
    val (weight, bias) = loadModule(name2layer(name), name)
    module.weight.copy(weight)
    module.bias.copy(bias)
    module.setName(name)
    DlFile.save(module, cachePath(layer.getName), isOverwrite)
    module
  }

  def mapConvolution(name: String): SpatialConvolution[T] = {
    val mod = loadModuleFromFile[SpatialConvolution[T]](name)
    mod match {
      case Some(m) => return m
      case _ =>
    }
    if (name2layer.isEmpty) {
      loadCaffe(defName, modelName)
    }
    val layer = name2layer(name)
    val param = layer.getConvolutionParam
    val groups = param.getGroup match {
      case 0 => 1
      case _ => param.getGroup
    }
    if (layer.getBlobsCount == 0) {
      //        println("convolution blob is empty")
      return null
    }
    val wB = layer.getBlobs(0)
    val nInputPlane = ((if (wB.hasShape) wB.getShape.getDim(1) else wB.getChannels) * groups).toInt
    val nOutputPlane = (if (wB.hasShape) wB.getShape.getDim(0) else wB.getNum).toInt

    var kW = param.getKernelW
    var kH = param.getKernelH
    var dW = param.getStrideW
    var dH = param.getStrideH
    var padW = param.getPadW
    var padH = param.getPadH

    if (kW == 0 || kH == 0) {
      kW = param.getKernelSize(0)
      kH = kW
    }
    if (dW == 0 || dH == 0) {
      if (param.getStrideCount == 0) {
        dW = 1
      } else {
        dW = param.getStride(0)
      }
      dH = dW

    }
    if (padW == 0 || padH == 0) {
      if (param.getPadCount != 0) {
        padW = param.getPad(0)
        padH = padW
      }
    }

    if (groups != 1) {
      println("nn supports no groups!")
      return null
    }
    val module = new SpatialConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    val (weight, bias) = loadModule(name2layer(name), name)
    module.weight.copy(weight)
    module.bias.copy(bias)
    module.setName(name)
    DlFile.save(module, cachePath(layer.getName), isOverwrite)
    println(s"$name: ($nInputPlane, $nOutputPlane, $kW, $kH, $dW, $dH, $padW, $padH)")
    module
  }


  private def loadCaffe(prototxtName: String, modelName: String): Map[String, LayerParameter] = {
    netparam = loadBinary(prototxtName, modelName)
    assert(netparam.getLayerCount > 0, "only support proto V2")
    for (i <- 0 until netparam.getLayerCount) {
      val layer = netparam.getLayer(i)
      name2layer += (layer.getName -> layer)
    }
    name2layer
  }


  private def loadModule(layer: LayerParameter, name: String, hasBias: Boolean = true)
  : (Tensor[T], Tensor[T]) = {
    var weight: Tensor[T] = null
    var bias: Tensor[T] = null
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


  private def loadBinary(prototxtName: String, modelName: String): Caffe.NetParameter = {
    val f: File = new File(prototxtName)
    assert(f.exists(), prototxtName + "does not exists")
    val rawInput: InputStream = new FileInputStream(f)
    val reader: InputStreamReader = new InputStreamReader(rawInput, "ASCII")
    val builder: Caffe.NetParameter.Builder = NetParameter.newBuilder
    TextFormat.merge(reader, builder)
    println("start loading caffe model")
    val cis = CodedInputStream.newInstance(new FileInputStream(modelName))
    cis.setSizeLimit(800000000)
    builder.mergeFrom(cis)
    println("load caffe model done")
    builder.build()
  }
}
