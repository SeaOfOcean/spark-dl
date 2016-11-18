package com.intel.analytics.sparkdl.pvanet.caffe

import java.io.{File, FileInputStream, InputStream, InputStreamReader}

import caffe.Caffe
import caffe.Caffe.{LayerParameter, NetParameter}
import com.google.protobuf.{CodedInputStream, TextFormat}
import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}

import scala.reflect.ClassTag

object ModuleType extends Enumeration {
  type ModuleType = Value
  val TensorModule, Criterion = Value
}

object CaffeReader {

  def main(args: Array[String]): Unit = {
    val defName = "/home/xianyan/objectRelated/faster_rcnn_models/VGG16/faster_rcnn_alt_opt/rpn_test.pt"
    val modelName = "/home/xianyan/objectRelated/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel"
    val caffeReader = new CaffeReader[Float](defName, modelName)
    val conv = caffeReader.mapConvolution("conv1_1")
  }
}

class CaffeReader[T: ClassTag](defName: String, modelName: String)(implicit ev: TensorNumeric[T]) {
  var netparam: Caffe.NetParameter = null
  var numOutput = 0

  val name2layer = loadCaffe(defName, modelName)

  def mapPooling(layer: LayerParameter): TensorModule[T] = {
    val param = layer.getPoolingParam
    val ptype = if (param.getPool == caffe.Caffe.PoolingParameter.PoolMethod.MAX) "Max" else "Avg"
    var kW = param.getKernelW
    var kH = param.getKernelH
    var dW = param.getStrideW
    var dH = param.getStrideH
    var padW = param.getPadW
    var padH = param.getPadH

    if (kW == 0 || kH == 0) {
      //todo: not sure, with a size list
      kW = param.getKernelSize
      kH = kW
    }
    if (dW == 0 || dH == 0) {
      //todo: not sure
      dW = param.getStride()
      dH = dW
    }
    if (padW == 0 || padH == 0) {
      //todo: not sure, with a size list
      padW = param.getPad()
      padH = padW
    }

    ptype match {
      case "Max" => new SpatialMaxPooling[T](kW, kH, dW, dH, padW, padH)
      case "Avg" => new SpatialAveragePooling[T](kW, kH, dW, dH, padW, padH)
      case _ => throw new NotImplementedError(ptype + " pooling is not supported")
    }
  }

  def mapInnerProduct(name: String): Linear[T] = {
    val layer = name2layer(name)
    val param = layer.getInnerProductParam
    val wB = layer.getBlobs(0)
    val nInputPlane = if (wB.hasShape) wB.getShape.getDim(1) else wB.getWidth
    val nOutputPlane = param.getNumOutput
    val module = new Linear[T](nInputPlane.toInt, nOutputPlane)
    val (weight, bias) = loadModule(name2layer(name), name)
    module.weight.copy(weight)
    module.bias.copy(bias)
    module
  }

  def mapConvolution(name: String): SpatialConvolution[T] = {
    val layer = name2layer(name)
    val param = layer.getConvolutionParam
    val groups = param.getGroup() match {
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

    numOutput = nOutputPlane

    var kW = param.getKernelW
    var kH = param.getKernelH
    var dW = param.getStrideW
    var dH = param.getStrideH
    var padW = param.getPadW
    var padH = param.getPadH

    if (kW == 0 || kH == 0) {
      //todo: not sure, with a size list
      kW = param.getKernelSize(0)
      kH = kW
    }
    if (dW == 0 || dH == 0) {
      //todo: not sure
      if (param.getStrideCount == 0) {
        dW = 1
      } else {
        dW = param.getStride(0)
      }
      dH = dW

    }
    if (padW == 0 || padH == 0) {
      //todo: not sure, with a size list
      if (param.getPadCount == 0) {
        padW = 1
      } else {
        padW = param.getPad(0)
      }
      padH = padW
    }

    if (groups != 1) {
      println("nn supports no groups!")
      return null
    }
    val module = new SpatialConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    val (weight, bias) = loadModule(name2layer(name), name)
    module.weight.copy(weight)
    module.bias.copy(bias)
    module
  }

  private def loadCaffe(prototxtName: String, modelName: String): Map[String, LayerParameter] = {
    netparam = loadBinary(prototxtName, modelName)

    numOutput = netparam.getInputShapeCount() * 4

    assert(netparam.getLayerCount > 0, "only support proto V2")
    var name2param = Map[String, LayerParameter]()
    for (i <- 0 until netparam.getLayerCount) {
      val layer = netparam.getLayer(i)
      name2param += (layer.getName -> layer)
    }
    name2param
  }


  private def loadModule(layer: LayerParameter, name: String): (Tensor[T], Tensor[T]) = {
    var weight: Tensor[T] = null
    var bias: Tensor[T] = null
    val wB = layer.getBlobs(0)
    var nInputPlane, nOutputPlane, kW, kH = 0
    if (wB.hasShape) {
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
    if (layer.getType == "InnerProduct") {
      printf("%s: %d %d %d %d\n", name, 1, 1, nInputPlane, nOutputPlane)
      weight.resize(Array(layer.getConvolutionParam.getGroup, 1, 1, nInputPlane, nOutputPlane))
    }
    else {
      printf("%s: %d %d %d %d\n", name, nOutputPlane, nInputPlane, kW, kH);
      weight.resize(Array(layer.getConvolutionParam.getGroup, nOutputPlane, nInputPlane, kW, kH))
    }
    val biasList = layer.getBlobs(1).getDataList
    val biasData: Array[T] = new Array[T](biasList.size())
    for (i <- 0 until biasList.size()) {
      biasData(i) = ev.fromType[Float](biasList.get(i))
    }
    bias = Tensor(Storage(biasData))
    (weight, bias)
  }


  private def loadBinary(prototxtName: String, modelName: String): Caffe.NetParameter = {
    val f: File = new File(prototxtName)
    assert(f.exists(), prototxtName + "does not exists")
    val raw_input: InputStream = new FileInputStream(f)
    val reader: InputStreamReader = new InputStreamReader(raw_input, "ASCII")
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
