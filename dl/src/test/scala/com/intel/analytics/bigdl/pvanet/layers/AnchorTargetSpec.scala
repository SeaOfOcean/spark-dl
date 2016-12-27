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

package com.intel.analytics.bigdl.pvanet.layers

import com.intel.analytics.bigdl.pvanet.datasets.{ImageScalerAndMeanSubstractor, Imdb}
import com.intel.analytics.bigdl.pvanet.model.{Phase, VggParam}
import com.intel.analytics.bigdl.pvanet.utils.FileUtil
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class AnchorTargetSpec extends FlatSpec with Matchers {
  FileUtil.middleRoot = "data/middle/vgg16/step1/"
  val param = new VggParam(Phase.TRAIN)
  val datasource = Imdb.getImdb("voc_2007_testcode1")
  val roidb = datasource.loadAnnotation("000014")
  val scaler = new ImageScalerAndMeanSubstractor(param)
  val img = scaler.apply(roidb)
  println(img.gtBoxes.get)
  val anchorTarget = new AnchorTarget(param)
  val insideAnchorsGtOverlaps =
    FileUtil.loadFeatures("insideAnchorsGtOverlaps", "data/middle/vgg16/step1")
  val featureW = 57
  val featureH = 38
  val (indsInside, insideAnchors, totalAnchors)
  = anchorTarget.getAnchors(featureW, featureH, 901, 600)

  "getAllLabels" should "work properly" in {
    val labels = anchorTarget.getAllLabels(indsInside, insideAnchorsGtOverlaps)
    compare("labelbeforesample", labels, 1e-6)
  }

  "computeTargets" should "work properly" in {
    val gtBoxes = img.gtBoxes.get
    val targets = anchorTarget.computeTargets(insideAnchors, gtBoxes, insideAnchorsGtOverlaps)
    // todo: precision may not be enough
    compare("targetBefore3354", targets, 0.01)
  }

  "get weights" should "work properly" in {
    val labels = FileUtil.loadFeatures("labelsBefore3354")
    val bboxInsideWeights = anchorTarget.getBboxInsideWeights(indsInside, labels)
    compare("inwBefore3354", bboxInsideWeights, 1e-6)
    val bboxOutSideWeights = anchorTarget.getBboxOutsideWeights(indsInside, labels)
    compare("outWBefore3354", bboxOutSideWeights, 1e-6)
  }

  "unmap" should "work properly" in {
    var labels = FileUtil.loadFeatures("labelsBefore3354")
    labels = anchorTarget.unmap(labels, totalAnchors, indsInside, -1)
    compare("labelUnmap", labels, 1e-6)
  }

  def compare(name: String, vec: Tensor[Float], prec: Double): Unit = {
    val exp = FileUtil.loadFeatures(name, "data/middle/vgg16/step1")
    FileUtil.assertEqualIgnoreSize(exp, vec, name, prec)
  }
}
