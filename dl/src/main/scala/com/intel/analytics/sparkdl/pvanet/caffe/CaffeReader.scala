package com.intel.analytics.sparkdl.pvanet.caffe

import java.io._

import caffe.Caffe
import caffe.Caffe.NetParameter
import com.google.protobuf.{CodedInputStream, TextFormat}

import scala.collection.JavaConversions._

object CaffeReader {
  def main(args: Array[String]): Unit = {
    val modelDef = getCaffeParams("/home/xianyan/objectRelated/pvanet/full/original.pt")
    //   modelDef.
    //    modelDef.MergeFromString(open(self.data_path, 'rb').read())
    System.out.println(modelDef.getName)
    for (layer <- modelDef.getLayerList) {
      println(layer)
    }
    //    System.out.println(modelDef.getLayerList)
  }

  def getCaffeParams(defPath: String): Caffe.NetParameter = {
    val f: File = new File(defPath)
    assert(f.exists(), defPath + "does not exists")
    val raw_input: InputStream = new FileInputStream(f)
    val reader: InputStreamReader = new InputStreamReader(raw_input, "ASCII")
    val builder: Caffe.NetParameter.Builder = NetParameter.newBuilder
    TextFormat.merge(reader, builder)
    println("start loading caffe model")
    val cis = CodedInputStream.newInstance(new FileInputStream("/home/xianyan/objectRelated/pvanet/full/original.model"))
    cis.setSizeLimit(800000000)
    builder.mergeFrom(cis)
    println("load caffe model done")
    builder.build()
  }
}