package com.intel.analytics.sparkdl.pvanet

import breeze.linalg.{DenseMatrix, convert}
import org.scalatest.{FlatSpec, Matchers}


class MatrixUtilSpec extends FlatSpec with Matchers {

  var arr = DenseMatrix((0.39796564, 0.09962627, 0.38716339, 0.78216441),
    (0.8748918, 0.24124542, 0.34264925, 0.28663851),
    (0.35269534, 0.7103468, 0.5326144, 0.03050023))

  val ar = convert(arr, Float)

  behavior of "MatrixUtilSpec"

  it should "argmax" in {
    MatrixUtil.argmax2(ar, 0).get should be(Array[Int](1, 2, 2, 0))
    MatrixUtil.argmax2(ar, 1).get should be(Array[Int](3, 0, 1))
  }

  "select " should "work properly" in {
    val gt = DenseMatrix((0.39796564, 0.09962627, 0.38716339, 0.78216441),
      (0.35269534, 0.7103468, 0.5326144, 0.03050023))
    MatrixUtil.select(ar, Array(0, 2), 0).get should be (convert(gt, Float))

    val gt2 = DenseMatrix((0.39796564, 0.38716339, 0.78216441),
      (0.8748918, 0.34264925, 0.28663851),
      (0.35269534, 0.5326144, 0.03050023))
    MatrixUtil.select(ar, Array(0, 2, 3),1 ).get should be (convert(gt2, Float))
  }

}
