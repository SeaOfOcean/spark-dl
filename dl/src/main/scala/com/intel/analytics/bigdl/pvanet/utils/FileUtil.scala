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

package com.intel.analytics.bigdl.pvanet.utils

import java.io.{BufferedWriter, File, FileNotFoundException, FileWriter}

import breeze.linalg.DenseMatrix
import com.intel.analytics.bigdl.pvanet.datasets.Imdb
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{File => DlFile}

import scala.io.Source
import scala.reflect.ClassTag

object FileUtil {


  // Root directory of project
  var ROOT_DIR = System.getProperty("user.dir")

  // Data directory
  var DATA_DIR = ROOT_DIR + "/data"

  // Place outputs under an experiments directory
  var EXP_DIR = "default"


  def getOutputDir(imdb: Imdb, netName: String): String = {
    // Return the directory where experimental artifacts are placed.
    // If the directory does not exist, it is created.

    // A canonical path is built using the name from an imdb and a network
    // (if not None).
    var outdir = ROOT_DIR + "/" + EXP_DIR + "/" + imdb.name
    if (!netName.isEmpty) outdir = outdir + "/" + netName
    if (!new File(outdir).exists()) {
      new File(outdir).mkdirs()
    }
    outdir
  }

  def getOutputDir(imdb: Imdb): String = {
    getOutputDir(imdb, "")
  }

  def cachePath: String = {
    val path = DATA_DIR + "/cache"
    if (!existFile(path)) new File(path.toString).mkdirs()
    path
  }

  def modelPath: String = {
    val path = DATA_DIR + "/model"
    if (!existFile(path)) new File(path).mkdirs()
    path
  }

  def existFile(f: String): Boolean = new java.io.File(f).exists()

  def demoPath: String = {
    val path = DATA_DIR + "/demo"
    if (!existFile(path)) new File(path).mkdirs()
    path
  }


  def loadFeaturesFullName[T: ClassTag](s: String, middleRoot: String = middleRoot)
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    println(s"load $s from file")
    val size = s.substring(s.lastIndexOf("-") + 1, s.lastIndexOf(".")).split("_").map(x => x.toInt)
    Tensor(Storage(Source.fromFile(middleRoot + s).getLines()
      .map(x => ev.fromType[Double](x.toDouble)).toArray)).reshape(size)
  }

  var middleRoot = "dl/data/middle/vgg16/step1/"

  def loadFeatures[T: ClassTag](s: String, middleRoot: String = middleRoot)
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    if (s.contains(".txt")) {
      loadFeaturesFullName[T](s, middleRoot)
    } else {
      val list = new File(middleRoot).listFiles()
      list.foreach(x => {
        if (x.getName.matches(s"$s-.*txt")) {
          return loadFeaturesFullName[T](x.getName, middleRoot)
        }
      })
      throw new FileNotFoundException(s"cannot map $s")
    }
  }


  def saveDenseMatrix(detsNMS: DenseMatrix[Float], s: String): Unit = {
    val savePath = s"/home/xianyan/code/intel/pvanet/spark-dl/data/middle/pvanet/" +
      s"$s-${detsNMS.rows}_${detsNMS.cols}"
    val writer = new BufferedWriter(new FileWriter(savePath))
    for (i <- 0 until detsNMS.rows) {
      for (j <- 0 until detsNMS.cols) {
        writer.write(detsNMS(i, j) + "\n")
      }
    }
    writer.close()
  }


  def assertEqual[T: ClassTag](expectedName: String, output: Tensor[T])
    (implicit ev: TensorNumeric[T]): Unit = {
    val expected = loadFeatures[T](expectedName)
    assertEqual(expected, output, expectedName)
  }

  def assertEqual[T: ClassTag](expected: Tensor[T], output: Tensor[T], info: String = "")
    (implicit ev: TensorNumeric[T]): Unit = {
    if (!info.isEmpty) {
      println(s"compare $info ...")
    }
    require(expected.size().mkString(",") == output.size().mkString(","), "size mismatch: " +
      s"expected size ${expected.size().mkString(",")} " +
      s"does not match output ${output.size().mkString(",")}")
    (expected.storage().array() zip output.storage().array()).foreach(x =>
      require(ev.toType[Double](ev.abs(ev.minus(x._1, x._2))) < 5,
        s"${x._1} does not equal ${x._2}"))
    if (!info.isEmpty) {
      println(s"$info pass")
    }
  }

  def loadModuleFromFile[M](filename: String): Option[M] = {
    try {
      if (existFile(filename)) return Some(DlFile.load[M](filename))
    } catch {
      case ex: Exception => None
    }
    None
  }
}
