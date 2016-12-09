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

package com.intel.analytics.bigdl.pvanet.datasets

import java.io.{File, PrintWriter}
import java.util.UUID

import breeze.linalg.DenseMatrix
import com.intel.analytics.bigdl.pvanet.model.FasterRcnnParam
import com.intel.analytics.bigdl.pvanet.tools.VocEval
import com.intel.analytics.bigdl.pvanet.utils.FileUtil
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{File => DlFile}

import scala.Array._
import scala.io.Source
import scala.xml.XML


class PascalVoc(val year: String = "2007", val imageSet: String,
  val devkitPath: String = FileUtil.DATA_DIR + "/VOCdevkit", param: FasterRcnnParam)
  extends Imdb(param) {

  override val name = "voc_" + year + "_" + imageSet
  val dataPath = devkitPath + "/VOC" + year
  override val classes = Array[String](
    "__background__", // always index 0
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
  )
  val classToInd = (classes zip (Stream from 1)).toMap
  val imageExt = ".jpg"
  imageIndex = loadImageSetIndex()

  val compId = "comp4"
  val salt = UUID.randomUUID().toString
  // PASCAL specific config options
  val config = Map("cleanup" -> true,
    "use_salt" -> true,
    "use_diff" -> false,
    "matlab_eval" -> false,
    "rpn_file" -> None,
    "min_size" -> 2)
  assert(FileUtil.existFile(devkitPath),
    "VOCdevkit path does not exist: " + devkitPath)
  assert(FileUtil.existFile(dataPath),
    "Path does not exist: {}" + dataPath)


  /**
   * Return the absolute path to image i in the image sequence.
   */
  def imagePathAt(i: Int): String = imagePathFromIndex(imageIndex(i))

  // Construct an image path from the image"s "index" identifier.
  def imagePathFromIndex(index: String): String = dataPath + "/JPEGImages/" + index + imageExt


  /**
   * Load the indexes listed in this dataset's image set file.
   *
   * Example path to image set file: devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
   *
   * @return
   */
  def loadImageSetIndex(): Array[String] = {
    val imageSetFile = dataPath + "/ImageSets/Main/" + imageSet + ".txt"
    assert(FileUtil.existFile(imageSetFile), "Path does not exist " + imageSetFile)
    Source.fromFile(imageSetFile).getLines().map(line => line.trim).toArray
  }

  /**
   * Load image and bounding boxes info from XML file in the PASCAL VOC
   * format.
   */
  def loadPascalAnnotation(index: String): Roidb = {
    val xml = XML.loadFile(dataPath + "/Annotations/" + index + ".xml")
    var objs = xml \\ "object"

    if (config("use_diff") == false) {
      // Exclude the samples labeled as difficult
      val non_diff_objs = objs.filter(obj => (obj \ "difficult").text.toInt == 0)
      objs = non_diff_objs
    }

    val boxes = new DenseMatrix[Float](objs.length, 4)
    val gt_classes = Tensor[Float](objs.length)
    // Load object bounding boxes into a data frame.
    for ((obj, ix) <- objs.zip(Stream from 1)) {
      // pixel indexes 1-based
      val bndbox = obj \ "bndbox"
      val x1 = (bndbox \ "xmin").text.toFloat
      val y1 = (bndbox \ "ymin").text.toFloat
      val x2 = (bndbox \ "xmax").text.toFloat
      val y2 = (bndbox \ "ymax").text.toFloat
      val cls = classToInd((obj \ "name").text)
      boxes(ix - 1, 0) = x1
      boxes(ix - 1, 1) = y1
      boxes(ix - 1, 2) = x2
      boxes(ix - 1, 3) = y2
      gt_classes.setValue(ix, cls)
    }
    Roidb(imagePathFromIndex(index), boxes, gt_classes, flipped = false)
  }


  /**
   * This function loads/saves from/to a cache file to speed up future calls.
   *
   * @return the database of ground-truth regions of interest.
   */
  def loadRoidb: Array[Roidb] = {
    val cache_file = FileUtil.cachePath + "/" + name + "_gt_roidb.pkl"
    if (FileUtil.existFile(cache_file)) {
      println("%s gt roidb loaded from %s".format(name, cache_file))
      try {
        DlFile.load[Array[Roidb]](cache_file)
      } catch {
        case e: Exception =>
          val gtRoidb = imageIndex.map(index => loadPascalAnnotation(index))
          new java.io.File(cache_file).delete()
          DlFile.save(gtRoidb, cache_file)
          gtRoidb
      }
    } else {
      val gtRoidb = imageIndex.map(index => loadPascalAnnotation(index))
      DlFile.save(gtRoidb, cache_file)
      gtRoidb
    }
  }

  def getCompId: String = {
    if (config("use_salt").asInstanceOf[Boolean]) s"${compId}_$salt" else compId
  }

  /**
   * VOCdevkit / results / VOC2007 / Main /< comp_id > _det_test_aeroplane.txt
   */
  def getVocResultsFileTemplate: String = {
    devkitPath + s"/results/VOC$year/Main/${compId}_det_${imageSet}_%s.txt"
  }

  private def writeVocResultsFile(allBoxes: Array[Array[DenseMatrix[Float]]]) = {
    classToInd.foreach {
      case (cls, clsInd) =>
        if (cls != "__background__") {
          println(s"writing $cls VOC results file")
          val filename = getVocResultsFileTemplate.format(cls)
          val of = new PrintWriter(new java.io.File(filename))
          imageIndex.zipWithIndex.foreach {
            case (imInd, index) =>
              val dets = allBoxes(clsInd - 1)(index)
              if (dets.size > 0) {
                // the VOCdevkit expects 1-based indices
                for (k <- 0 until dets.rows) {
                  of.write("%s %.3f %.1f %.1f %.1f %.1f\n".format(
                    imInd, dets(k, dets.cols - 1),
                    dets(k, 0) + 1, dets(k, 1) + 1,
                    dets(k, 2) + 1, dets(k, 3) + 1
                  ))
                }
              }
          }
          of.close()
        }
    }
  }

  def eval(outputDir: String = "output"): Unit = {
    val annopath = s"$devkitPath/VOC$year/Annotations/%s.xml"
    val imagesetfile = s"$devkitPath/VOC$year/ImageSets/Main/$imageSet.txt"
    val cachedir = s"$devkitPath/annotations_cache"
    var aps = List[Double]()
    // The PASCAL VOC metric changed in 2010
    val use_07_metric = if (year.toInt < 2010) true else false
    println("VOC07 metric ? " + (if (use_07_metric) "yes" else "No"))
    if (!FileUtil.existFile(outputDir)) {
      new File(outputDir).mkdirs()
    }
    classes.zipWithIndex.foreach {
      case (cls, i) =>
        if (cls != "__background__") {
          val filename = getVocResultsFileTemplate.format(cls)
          val (_, _, ap) = VocEval.eval(filename, annopath, imagesetfile, cls,
            cachedir, ovthresh = 0.5, use_07_metric = use_07_metric)
          aps :+= ap
          println(s"AP for $cls = ${"%.4f".format(ap)}")
        }
    }
    println(s"Mean AP = ${"%.4f".format(aps.sum / aps.length)}")
    println("~~~~~~~~")
    println("Results:")
    aps.foreach(ap => println(s"${"%.3f".format(ap)}"))
    println(s"${"%.3f".format(aps.sum / aps.length)}")
    println("~~~~~~~~")
    cleanup(cachedir)
  }

  def cleanup(cachedir: String) = new File(cachedir).listFiles().foreach(f => f.delete())

  def evaluateDetections(allBoxes: Array[Array[DenseMatrix[Float]]],
    outputDir: String): Unit = {
    writeVocResultsFile(allBoxes)
    eval(outputDir)
  }
}
