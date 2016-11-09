package com.intel.analytics.sparkdl.pvanet

import breeze.linalg.{DenseMatrix, DenseVector, convert}
import breeze.numerics.abs
import com.intel.analytics.sparkdl.pvanet.datasets.{AnchorToTensor, ImageScalerAndMeanSubstractor, ImdbFactory, PascolVocDataSource}
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random


class AnchorTargetLayerSpec extends FlatSpec with Matchers {
  Config.TRAIN.SCALES = List(100)
  val anchorTargetLayer = new AnchorTargetLayer()
  val width = 133
  val height = 100
  val shifts = anchorTargetLayer.generateShifts(width, height, 16).get
  "generateShifts" should "work properly" in {
    assert(shifts.rows == 13300)
    assert(shifts.cols == 4)
    val expectedHead = DenseMatrix((0, 0, 0, 0), (16, 0, 16, 0), (32, 0, 32, 0))
    assert(expectedHead === shifts(0 until expectedHead.rows, ::))
    val expectedTail = DenseMatrix((2080, 1584, 2080, 1584),
      (2096, 1584, 2096, 1584),
      (2112, 1584, 2112, 1584))
    for (i <- 0 until expectedTail.rows) {
      for (j <- 0 until expectedTail.cols) {
        assert(abs(expectedTail(i, j) - shifts(shifts.rows - 3 + i, j)) < 1e-6)
      }
    }
  }
  val anchors = anchorTargetLayer.anchors
  val allAnchors = anchorTargetLayer.getAllAnchors(shifts, anchors)
  checkAllAnchors

  def checkAllAnchors: Unit = {
    "getAllAnchors" should "return right values" in {
      assert(allAnchors.rows == 332500)
      assert(allAnchors.cols == 4)
      val expectedHead = DenseMatrix(
        (-26.5, -10.0, 41.5, 25. ), (-26.5, -10.0, 41.5, 25.0), (-61.0, -28.0, 76.0, 43.0), (-95.5, -46.0, 110.5, 61.0),
        (-176.0, -88.0, 191.0, 103.0), (-360.0, -184.0, 375.0, 199.0), (-22.0, -11.5, 37.0, 26.5), (-52.0, -31.0, 67.0, 46.0),
        (-82.0, -50.5, 97.0, 65.5), (-152.0, -96.0, 167.0, 111.0), (-312.0, -200.0, 327.0, 215.0), (-16.0, -16.0, 31.0, 31.0),
        (-40.0, -40.0, 55.0, 55.0), (-64.0, -64.0, 79.0, 79.0), (-120.0, -120.0, 135.0, 135.0), (-248.0, -248.0, 263.0, 263.0),
        (-11.5, -22.0, 26.5, 37.0), (-31.0, -52.0, 46.0, 67.0), (-50.5, -82.0, 65.5, 97.0), (-96.0, -152.0, 111.0, 167.0))
      convert(expectedHead, Float)
      for (i <- 0 until expectedHead.rows) {
        for (j <- 0 until expectedHead.cols) {
          assert(abs(expectedHead(i, j) - allAnchors(i, j)) < 1e-6)
        }
      }
      val expectedTail = DenseMatrix((2090.0, 1572.5, 2149.0, 1610.5), (2060.0, 1553.0, 2179.0, 1630.0), (2030.0, 1533.5, 2209.0, 1649.5), (1960.0, 1488.0, 2279.0, 1695.0)
        , (1800.0, 1384.0, 2439.0, 1799.0), (2096.0, 1568.0, 2143.0, 1615.0), (2072.0, 1544.0, 2167.0, 1639.0), (2048.0, 1520.0, 2191.0, 1663.0)
        , (1992.0, 1464.0, 2247.0, 1719.0), (1864.0, 1336.0, 2375.0, 1847.0), (2100.5, 1562.0, 2138.5, 1621.0), (2081.0, 1532.0, 2158.0, 1651.0)
        , (2061.5, 1502.0, 2177.5, 1681.0), (2016.0, 1432.0, 2223.0, 1751.0), (1912.0, 1272.0, 2327.0, 1911.0), (2103.5, 1559.0, 2135.5, 1624.0)
        , (2087.0, 1526.0, 2152.0, 1657.0), (2070.5, 1493.0, 2168.5, 1690.0), (2032.0, 1416.0, 2207.0, 1767.0), (1944., 1240., 2295., 1943. ))
      for (i <- 0 until expectedTail.rows) {
        for (j <- 0 until expectedTail.cols) {
          assert(abs(expectedTail(i, j) - allAnchors(allAnchors.rows - 3 + i, j)) < 1e-6)
        }
      }
    }
  }

  val indsInside = anchorTargetLayer.getIndsInside(width, height, allAnchors, anchorTargetLayer.allowedBorder)
  "getIndsInside" should "return right values" in {
    assert(indsInside.length == 82)
    val expected = Array(3360, 3375, 3380, 3385, 3400, 3405, 3410, 3425, 3430, 3435,
      3450, 3455, 3460, 3485, 6685, 6690, 6695, 6700, 6705, 6710, 6715, 6720, 6725, 6730,
      6735, 6740, 6745, 6750, 6755, 6756, 6760, 6765, 6770, 6775, 6780, 6785, 6790, 6795,
      6810, 6815, 6820, 10010, 10015, 10020, 10025, 10030, 10035, 10040, 10045, 10050, 10055,
      10060, 10065, 10070, 10075, 10080, 10081, 10085, 10090, 10095, 10100, 10105, 10110, 10115,
      10120, 10135, 10140, 10145, 13335, 13350, 13355, 13360, 13375, 13380, 13385, 13400, 13405,
      13410, 13425, 13430, 13435, 13460)
    expected.zip(indsInside).foreach(x => assert(abs(x._1 - x._2) < 1e-6))


  }


  //keep only inside anchors
  val insideAnchors: DenseMatrix[Float] = anchorTargetLayer.getInsideAnchors(indsInside, allAnchors)
  assert(insideAnchors.size == 82 * 4)
  val expectedHead = DenseMatrix((0., 0., 47., 47. ), (5.5, 6., 73.5, 41. ), (10., 4.5, 69., 42.5))
  convert(expectedHead, Float)
  for (i <- 0 until expectedHead.rows) {
    for (j <- 0 until expectedHead.cols) {
      assert(abs(expectedHead(i, j) - insideAnchors(i, j)) < 1e-6)
    }
  }
  val expectedTail = DenseMatrix((58., 52.5, 117., 90.5), (64., 48., 111., 95. ), (80., 48., 127., 95. ))
  for (i <- 0 until expectedTail.rows) {
    for (j <- 0 until expectedTail.cols) {
      assert(abs(expectedTail(i, j) - insideAnchors(insideAnchors.rows - 3 + i, j)) < 1e-6)
    }
  }
  // label: 1 is positive, 0 is negative, -1 is dont care
  var labels = DenseVector.fill(indsInside.length, -1)

  val pascal = ImdbFactory.get_imdb("voc_2007_testcode")
  val trainDataSource = new PascolVocDataSource(imageSet = "testcode")
  val imageScaler = new ImageScalerAndMeanSubstractor(trainDataSource)
  val sc = trainDataSource -> imageScaler
  var data = sc.next()
  while (data.imagePath.substring(data.imagePath.indexOf("VOCdevkit")) != "VOCdevkit/VOC2007/JPEGImages/000003.jpg") {
    data = sc.next()
  }

  anchorTargetLayer.generateAnchors(data, data.scaledImage.get.height(), data.scaledImage.get.height())

  data.imagePath.substring(data.imagePath.indexOf("VOCdevkit")) should be("VOCdevkit/VOC2007/JPEGImages/000003.jpg")
  val overlaps = Bbox.bboxOverlap(insideAnchors, data.gt_boxes.get)
  assert(overlaps.size == 164)
  assert(abs(overlaps(0, 0) - 0.04323438) < 1e-2)
  assert(abs(overlaps(1, 1) - 0.00271812) < 1e-2)
  assert(abs(overlaps(-2, -1) - 0.0558742) < 1e-2)
  assert(abs(overlaps(-1, -1) - 0.00694012) < 1e-2)
  assert(abs(overlaps(-1, 0)) < 1e-6)
  assert(abs(overlaps(-1, 0)) < 1e-6)

  val argmax_overlaps = MatrixUtil.argmax2(overlaps, 1).get

  val expected_argmax_ol = Array(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
  //0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,1,1,0,1,1,1,1,1,1,1
  //0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1
  //  todo: have one bit differnece
  //   argmax_overlaps.zip(expected_argmax_ol).foreach(x => assert(x._2 == x._1))
  var max_overlaps = argmax_overlaps.zipWithIndex.map(x => overlaps(x._2, x._1))
  var gt_argmax_overlaps = MatrixUtil.argmax2(overlaps, 0).get
  gt_argmax_overlaps.zip(Array(19, 32)).foreach(x => assert(x._1 == x._2))

  val gt_max_overlaps = gt_argmax_overlaps.zipWithIndex.map(x => {
    overlaps(x._1, x._2)
  })


  gt_max_overlaps.zip(Array(0.1289815, 0.12328038)).foreach(x => assert(abs(x._2 - x._1) < 1e-3))
  gt_argmax_overlaps = Array.range(0, overlaps.rows).filter(r => overlaps(r, 0) == gt_max_overlaps(0) || overlaps(r, 1) == gt_max_overlaps(1))
  if (!Config.TRAIN.RPN_CLOBBER_POSITIVES) {
    // assign bg labels first so that positive labels can clobber them
    max_overlaps.zipWithIndex.foreach(x => {
      if (x._1 < Config.TRAIN.RPN_NEGATIVE_OVERLAP) labels(x._2) = 0
    })
  }

  // fg label: for each gt, anchor with highest overlap
  gt_argmax_overlaps.foreach(x => labels(x) = 1)

  // fg label: above threshold IOU
  max_overlaps.zipWithIndex.foreach(x => {
    if (x._1 >= Config.TRAIN.RPN_POSITIVE_OVERLAP) max_overlaps(x._2) = 1
  })

  if (Config.TRAIN.RPN_CLOBBER_POSITIVES) {
    //assign bg labels last so that negative labels can clobber positives
    max_overlaps.zipWithIndex.foreach(x => {
      if (x._1 < Config.TRAIN.RPN_NEGATIVE_OVERLAP) max_overlaps(x._2) = 0
    })
  }

  // subsample positive labels if we have too many
  val num_fg = Config.TRAIN.RPN_FG_FRACTION * Config.TRAIN.RPN_BATCHSIZE
  val fg_inds = labels.findAll(_ == 1)
  if (fg_inds.length > num_fg) {
    val disable_inds = Random.shuffle(fg_inds).take(fg_inds.length - num_fg.toInt)
    disable_inds.foreach(x => labels(x) = -1)
  }

  var bbox_targets = anchorTargetLayer._compute_targets(insideAnchors, MatrixUtil.select(data.gt_boxes.get, argmax_overlaps, 0).get)

  var bbox_inside_weights = DenseMatrix.zeros[Float](indsInside.length, 4)
  labels.foreachPair((k, v) => {
    if (v == 1) {
      bbox_inside_weights(k, ::) := convert(DenseVector(Config.TRAIN.RPN_BBOX_INSIDE_WEIGHTS), Float).t
    }
  })

  var bbox_outside_weights = DenseMatrix.zeros[Float](indsInside.length, 4)

  val labelGe0 = labels.findAll(x => x >= 0).toArray
  val labelE1 = labels.findAll(x => x == 1).toArray
  val labelE0 = labels.findAll(x => x == 0).toArray
  var positive_weights = None: Option[DenseMatrix[Float]]
  var negative_weights = None: Option[DenseMatrix[Float]]
  if (Config.TRAIN.RPN_POSITIVE_WEIGHT < 0) {
    // uniform weighting of examples (given non -uniform sampling)
    val num_examples = labelGe0.length
    positive_weights = Some(DenseMatrix.ones[Float](1, 4) * (1.0f / num_examples))
    negative_weights = Some(DenseMatrix.ones[Float](1, 4) * (1.0f / num_examples))
  }
  else {
    require((Config.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
      (Config.TRAIN.RPN_POSITIVE_WEIGHT < 1))
    positive_weights = Some(DenseMatrix(Config.TRAIN.RPN_POSITIVE_WEIGHT.toFloat / labelE1.length))
    negative_weights = Some(DenseMatrix((1.0f - Config.TRAIN.RPN_POSITIVE_WEIGHT.toFloat) / labelE0.length))
  }

  labelE1.foreach(x => bbox_outside_weights(x, ::) := positive_weights.get.toDenseVector.t)
  labelE0.foreach(x => bbox_outside_weights(x, ::) := negative_weights.get.toDenseVector.t)

  assert(labels.length == 82)

  val expectedLabels = DenseVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for (i <- 0 until expectedLabels.length) {
    assert(labels(i) == expectedLabels(i))
  }

  assert(bbox_targets.rows == 82)
  assert(bbox_targets.cols == 4)

  assert(bbox_inside_weights(19, 0) == 1)
  assert(bbox_inside_weights(24, 0) == 1)
  assert(bbox_inside_weights(32, 0) == 1)
  assert(bbox_inside_weights(46, 0) == 1)
  assert(bbox_inside_weights(51, 0) == 1)
  assert(bbox_inside_weights(59, 0) == 1)
  assert(abs(bbox_outside_weights(0, 0) - 0.01219512) < 1e-6)

  val toTensor = new AnchorToTensor(1, 4, 5)
  toTensor.apply(new AnchorTarget(labels, bbox_targets, bbox_inside_weights, bbox_outside_weights))

}
