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

package com.intel.analytics.sparkdl.pvanet.datasets

import java.io.{File, PrintWriter}

import breeze.linalg.DenseMatrix
import com.intel.analytics.sparkdl.pvanet.Roidb.ImageWithRoi
import com.intel.analytics.sparkdl.pvanet.{Config, VocEval}
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.{File => DlFile}

import scala.Array._
import scala.io.Source
import scala.xml.XML


class PascalVoc(val year: String = "2007", val imageSet: String,
  var devkitPath: String = Config.DATA_DIR + "/VOCdevkit") extends Imdb {
  def this(year: String, imageSet: String) {
    this(year, imageSet, Config.DATA_DIR + "/VOCdevkit")
  }

  name = "voc_" + year + "_" + imageSet
  if (devkitPath == None) devkitPath = getDefaultPath
  val dataPath = devkitPath + "/VOC" + year
  classes = Array[String](
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
  // PASCAL specific config options
  val config = Map("cleanup" -> true,
    "use_salt" -> true,
    "use_diff" -> false,
    "matlab_eval" -> false,
    "rpn_file" -> None,
    "min_size" -> 2)
  assert(Config.existFile(devkitPath),
    "VOCdevkit path does not exist: " + devkitPath)
  assert(Config.existFile(dataPath),
    "Path does not exist: {}" + dataPath)


  /**
   * Return the absolute path to image i in the image sequence.
   *
   * @param i
   * @return
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
  def loadImageSetIndex(): List[String] = {
    println(dataPath)
    val imageSetFile = dataPath + "/ImageSets/Main/" + imageSet + ".txt"
    assert(Config.existFile(imageSetFile), "Path does not exist " + imageSetFile)
    Source.fromFile(imageSetFile).getLines().map(line => line.trim).toList
  }


  // Return the default path where PASCAL VOC is expected to be installed.
  private def getDefaultPath = Config.DATA_DIR + "/VOCdevkit"

  /**
   * Load image and bounding boxes info from XML file in the PASCAL VOC
   * format.
   *
   * @param index
   */
  def loadPascalAnnotation(index: String): ImageWithRoi = {
    val xml = XML.loadFile(dataPath + "/Annotations/" + index + ".xml")
    var objs = xml \\ "object"

    if (config("use_diff") == false) {
      // Exclude the samples labeled as difficult
      val non_diff_objs = objs.filter(obj => (obj \ "difficult").text.toInt == 0)
      objs = non_diff_objs
    }

    val boxes = new DenseMatrix[Float](objs.length, 4)
    val gt_classes = Tensor[Float](objs.length)
    val overlaps = Tensor[Float](objs.length, classes.length)
    // "Seg" area for pascal is just the box area
    var seg_areas = Tensor[Float](objs.length)
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
      overlaps.setValue(ix, cls, 1)
      seg_areas.setValue(ix, (x2 - x1 + 1) * (y2 - y1 + 1))
    }
    // todo: overlaps = scipy.sparse.csr_matrix(overlaps)
    return ImageWithRoi(boxes, gt_classes, overlaps, false, seg_areas)
  }


  /**
   * This function loads/saves from/to a cache file to speed up future calls.
   *
   * @return the database of ground-truth regions of interest.
   */
  def getGroundTruth(): Array[ImageWithRoi] = {
    val cache_file = Config.cachePath + "/" + name + "_gt_roidb.pkl"
    if (Config.existFile(cache_file)) {
      println("%s gt roidb loaded from %s".format(name, cache_file))
      try {
        return DlFile.load[Array[ImageWithRoi]](cache_file)
      } catch {
        case e: Exception =>
          val gtRoidb = imageIndex.map(index => loadPascalAnnotation(index)).toArray
          new java.io.File(cache_file).delete()
          DlFile.save(gtRoidb, cache_file)
          gtRoidb
      }
    } else {
      val gtRoidb = imageIndex.map(index => loadPascalAnnotation(index)).toArray
      DlFile.save(gtRoidb, cache_file)
      gtRoidb
    }
  }

  /**
   * VOCdevkit / results / VOC2007 / Main /< comp_id > _det_test_aeroplane.txt
   */
  def getVocResultsFileTemplate(): String = {
    val filename = compId + "_det_" + imageSet + "_%s.txt"
    devkitPath + s"/results/VOC$year/Main/$filename"
  }

  private def writeVocResultsFile(allBoxes: Array[Array[DenseMatrix[Float]]]) = {
    def writeResult(clsInd: Int, imInd: Int, of: PrintWriter): Unit = {
      val dets = allBoxes(clsInd)(imInd)
      if (dets.size == 0) return
      // the VOCdevkit expects 1-based indices
      for (k <- 0 until dets.rows) {
        of.write("%s %.3f %.1f %.1f %.1f %.1f\n".format())
      }
    }
    classToInd.foreach {
      case (cls, clsInd) =>
        if (cls != "__background__") {
          println(s"writing ${cls} VOC results file")
          val filename = getVocResultsFileTemplate().format(cls)
          val of = new PrintWriter(new java.io.File(filename))
          imageIndex.zipWithIndex.foreach {
            case (imInd, index) =>
              val dets = allBoxes(clsInd - 1)(index)
              if (dets.size > 0) {
                // the VOCdevkit expects 1-based indices
                for (k <- 0 until dets.rows) {
                  of.write("%s %.3f %.1f %.1f %.1f %.1f\n".format(
                    index, dets(k, dets.cols - 1),
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
    if (!Config.existFile(outputDir)) {
      new File(outputDir).mkdirs()
    }
    classes.zipWithIndex.foreach {
      case (cls, i) =>
        if (cls != "__background__") {
          val filename = getVocResultsFileTemplate().format(cls)
          val (rec, prec, ap) = VocEval.eval(filename, annopath, imagesetfile, cls,
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

  }

  def evaluateDetections(allBoxes: Array[Array[DenseMatrix[Float]]],
    outputDir: String): Unit = {
    writeVocResultsFile(allBoxes)
    eval(outputDir)
  }
}
