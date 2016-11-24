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

package com.intel.analytics.bigdl.pvanet

import java.io.File
import java.util

import breeze.linalg.{DenseMatrix, DenseVector, argmax, argsort, convert, max}
import com.intel.analytics.bigdl.pvanet.datasets.PascalVoc
import com.intel.analytics.bigdl.utils.{File => DlFile}

import scala.io.Source
import scala.xml.XML

object VocEval {

  def cumsum(arr: Array[Int]): Array[Int] = {
    var sum = 0
    arr.map { x => sum += x; sum }
  }

  /**
   * ap = voc_ap(rec, prec, [use_07_metric])
   * Compute VOC AP given precision and recall.
   * If use_07_metric is true, uses the
   * VOC 07 11 point method (default:False)
   *
   * @param rec
   * @param prec
   * @param use_07_metric
   * @return
   */
  def vocAp(rec: Array[Double], prec: Array[Double], use_07_metric: Boolean): Double = {
    var ap = 0.0
    if (use_07_metric) {
      // 11 point metric
      var p = 0.0
      for (t <- 0.0 until 1.1 by 0.1) {
        val xgt = rec.map(x => if (x >= t) 1 else 0)
        if (xgt.sum == 0) {
          p = 0
        } else {
          p = (prec zip xgt).filter(x => x._2 == 1).map(x => x._1).max
        }
        ap = ap + p / 11.0
      }
    } else {
      // correct AP calculation
      // first append sentinel values at the end
      var mrec = rec.clone()
      mrec +:= 0.0
      mrec :+= 1.0
      var mpre = prec.clone()
      mpre +:= 0.0
      mpre :+= 0.0

      // compute the precision envelope
      for (i <- mpre.size - 1 until 0 by -1) {
        mpre(i - 1) = Math.max(mpre(i - 1), mpre(i))
      }
      // to calculate area under PR curve, look for points
      // where X axis (recall) changes value
      val inds = (mrec.slice(1, mrec.length) zip mrec.slice(0, mrec.length - 1)).map(
        x => x._1 != x._2).zipWithIndex.map(x => x._2)


      // and sum (\Delta recall) * prec
      ap = inds.map(i => (mrec(i + 1) - mrec(i)) * mpre(i + 1)).sum
    }
    ap
  }

  /**
   * rec, prec, ap = voc_eval(detpath,
   * annopath,
   * imagesetfile,
   * classname,
   * [ovthresh],
   * [use_07_metric])
   * Top level function that does the PASCAL VOC evaluation.
   *
   * @param detpath       Path to detections
   *                      detpath.format(classname) should produce the detection results file.
   * @param annopath      Path to annotations
   *                      annopath.format(imagename) should be the xml annotations file.
   * @param imagesetfile  Text file containing the list of images, one image per line.
   * @param classname     Category name (duh)
   * @param cachedir      Directory for caching the annotations
   * @param ovthresh      Overlap threshold (default = 0.5)
   * @param use_07_metric Whether to use VOC07's 11 point AP computation
   * @return
   */
  def eval(detpath: String, annopath: String, imagesetfile: String, classname: String,
    cachedir: String, ovthresh: Double = 0.5, use_07_metric: Boolean = false)
  : (Array[Double], Array[Double], Double) = {
    // assumes detections are in detpath.format(classname)
    // assumes annotations are in annopath.format(imagename)
    // assumes imagesetfile is a text file with each line an image name
    // cachedir caches the annotations in a pickle file

    // first load gt
    if (!Config.existFile(cachedir)) {
      new File(cachedir).mkdirs()
    }
    val cachefile = s"${cachedir}/annots.pkl"
    // read list of images
    val imagenames = Source.fromFile(imagesetfile).getLines().toList.map(x => x.trim)

    var recs: util.HashMap[Int, List[Object]] = null
    def loadAnnots: Unit = {
// load annots
      recs = new util.HashMap[Int, List[Object]]()
      imagenames.zipWithIndex.foreach {
        case (imagename, i) =>
          recs.put(imagename.toInt, parseRec(annopath.format(imagename)))
      }
      DlFile.save(recs, cachefile)
    }
    if (!Config.existFile(cachefile)) {
      loadAnnots
    } else {
      try {
        recs = DlFile.load[util.HashMap[Int, List[Object]]](cachefile)
      } catch {
        case e: Exception =>
          new File(cachefile).delete()
          loadAnnots
      }

    }

    // extract gt objects for this class
    var npos = 0
    var classRecs = Map[Int, (DenseMatrix[Int], List[Boolean], Array[Boolean])]()
    imagenames.foreach { imagename =>
      val R = recs.get(imagename.toInt).filter(obj => obj.name == classname)
      var bbox = new DenseMatrix[Int](R.length, 4)
      R.zipWithIndex.foreach(x => Range(0, 4).foreach(j => bbox(x._2, j) = x._1.bbox(j)))
      val difficult = R.map(x => x.difficult)
      val det = new Array[Boolean](R.length)
      npos = npos + difficult.map(x => if (!x) 1 else 0).sum
      classRecs += (imagename.toInt -> (bbox, difficult, det))
    }
    // read dets
    val detfile = detpath.format(classname)
    val splitlines = Source.fromFile(detfile).getLines().map(x =>
      x.trim.split(" ")).toList
    var imageIds = splitlines.map(x => x(0).toInt)
    val confidence = splitlines.map(x => x(1).toFloat)
    var BB = splitlines.map(x => {
      x.slice(2, x.length).map(z => z.toFloat)
    })
    // sort by confidence
    val sortedIds = argsort(DenseVector(confidence.toArray)).reverse
    BB = sortedIds.map(id => BB(id)).toList
    imageIds = sortedIds.map(x => imageIds(x)).toList
    // go down dets and mark TPs and FPs
    val nd = imageIds.length
    var tp = new Array[Int](nd)
    var fp = new Array[Int](nd)
    for (d <- 0 until nd) {
      val R = classRecs(imageIds(d))
      val bb = BB.slice(d, d + 1)(0)
      var ovmax = -Float.MaxValue
      val BBGT = R._1
      var jmax = 0
      if (BBGT.size > 0) {
        // compute overlaps intersection
        val ixmin = BBGT(::, 0).toArray.map(x => Math.max(x, bb(0)))
        val iymin = BBGT(::, 1).toArray.map(x => Math.max(x, bb(1)))
        val ixmax = BBGT(::, 2).toArray.map(x => Math.min(x, bb(2)))
        val iymax = BBGT(::, 3).toArray.map(x => Math.min(x, bb(3)))

        val iw = (ixmax zip ixmin).map(x => Math.max(x._1 - x._2 + 1, 0))
        val ih = (iymax zip iymin).map(x => Math.max(x._1 - x._2 + 1, 0))
        val inters = DenseVector(iw) :* DenseVector(ih)

        // union
        val xx = convert((BBGT(::, 2) - BBGT(::, 0) + 1) :* (BBGT(::, 3) :- BBGT(::, 1) :+ 1), Float)
        val tmp = (bb(2) - bb(0) + 1f) * (bb(3) - bb(1) + 1f)
        val uni = xx :- inters :+ tmp


        val overlaps = inters :/ uni
        ovmax = max(overlaps)
        jmax = argmax(overlaps)
      }

      if (ovmax > ovthresh) {
        if (!R._2(jmax)) {
          if (!R._3(jmax)) {
            tp(d) = 1
            R._3(jmax) = true
          } else {
            fp(d) = 1
          }
        }
      } else {
        fp(d) = 1
      }
    }

    // compute precision recall
    fp = cumsum(fp)
    tp = cumsum(tp)
    val rec = tp.map(x => x / npos.toDouble)
    // avoid divide by zero in case the first detection matches a difficult
    // ground truth
    val prec = (tp zip (tp zip fp).map(x => x._1 + x._2)
      .map(x => Math.max(x, 2.2204460492503131e-16)))
      .map(x => x._1 / x._2)
    val ap = vocAp(rec, prec, use_07_metric)
    (rec, prec, ap)
  }

  def parseRec(path: String): List[Object] = {
    val xml = XML.loadFile(path)
    var objs = xml \\ "object"
    val boxes = new Array[Int](4)
    var objects = List[Object]()
    // Load object bounding boxes into a data frame.
    for (obj <- objs) {
      // pixel indexes 1-based
      val bndbox = obj \ "bndbox"
      val x1 = (bndbox \ "xmin").text.toInt
      val y1 = (bndbox \ "ymin").text.toInt
      val x2 = (bndbox \ "xmax").text.toInt
      val y2 = (bndbox \ "ymax").text.toInt
      objects :+= new Object((obj \ "name").text, (obj \ "pose").text,
        (obj \ "truncated").text.toInt, (obj \ "difficult").text == "1", Array(x1, y1, x2, y2))
    }
    objects
  }

  def main(args: Array[String]): Unit = {
    val dataset = new PascalVoc(year = "2007", imageSet = "testcode")
    dataset.eval()
  }

}

case class Object(name: String, pose: String, truncated: Int, difficult: Boolean, bbox: Array[Int]
) extends Serializable
