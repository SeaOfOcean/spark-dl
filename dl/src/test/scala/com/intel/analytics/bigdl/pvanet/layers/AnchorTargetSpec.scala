package com.intel.analytics.bigdl.pvanet.layers

import com.intel.analytics.bigdl.pvanet.datasets.{ImageScalerAndMeanSubstractor, Imdb}
import com.intel.analytics.bigdl.pvanet.model.{Phase, VggParam}
import com.intel.analytics.bigdl.pvanet.utils.FileUtil
import org.scalatest.{FlatSpec, Matchers}

class AnchorTargetSpec extends FlatSpec with Matchers {
  "getAnchorTarget" should "work properly" in {
    val param = new VggParam(Phase.TRAIN)
    val datasource = Imdb.getImdb("voc_2007_testcode1", param)
    val roidb = datasource.loadAnnotation("000014")
    val scaler = new ImageScalerAndMeanSubstractor(param)
    val img = scaler.apply(roidb)
    println(img.gtBoxes.get)
    val anchorTarget = new AnchorTarget(param)
    val height = 38
    val width = 57
//    val targets = anchorTarget.getAnchorTarget(height, width,
//      img.scaledImage.height(), img.scaledImage.width(), img.gtBoxes.get)
    val targets = anchorTarget.getAnchorTarget(height, width,
      600, 901, img.gtBoxes.get)
    val expected1 = BboxTarget(
      FileUtil.loadFeatures[Float]("rpn_labels"),
      FileUtil.loadFeatures[Float]("rpn_bbox_targets"),
      FileUtil.loadFeatures[Float]("rpn_bbox_inside_weights"),
      FileUtil.loadFeatures[Float]("rpn_bbox_outside_weights"))
    FileUtil.assertEqualIgnoreSize[Float](expected1.labels, targets.labels, "compare targets label")
    FileUtil.assertEqualTable[Float](expected1.targetsTable, targets.targetsTable)
  }
}
