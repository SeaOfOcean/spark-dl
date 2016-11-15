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
    MatrixUtil.argmax2(ar, 0) should be(Array[Int](1, 2, 2, 0))
    MatrixUtil.argmax2(ar, 1) should be(Array[Int](3, 0, 1))
  }

  "select " should "work properly" in {
    val gt = DenseMatrix((0.39796564, 0.09962627, 0.38716339, 0.78216441),
      (0.35269534, 0.7103468, 0.5326144, 0.03050023))
    MatrixUtil.selectMatrix(ar, Array(0, 2), 0) should be (convert(gt, Float))

    val gt2 = DenseMatrix((0.39796564, 0.38716339, 0.78216441),
      (0.8748918, 0.34264925, 0.28663851),
      (0.35269534, 0.5326144, 0.03050023))
    MatrixUtil.selectMatrix(ar, Array(0, 2, 3),1 ) should be (convert(gt2, Float))
  }

  it should "selectCols" in {

    val mat = DenseMatrix((0, 1, 2, 3, 4, 5, 6, 7),
      (8, 9, 10, 11, 12, 13, 14, 15),
      (16, 17, 18, 19, 20, 21, 22, 23),
      (24, 25, 26, 27, 28, 29, 30, 31),
      (32, 33, 34, 35, 36, 37, 38, 39),
      (40, 41, 42, 43, 44, 45, 46, 47),
      (48, 49, 50, 51, 52, 53, 54, 55),
      (56, 57, 58, 59, 60, 61, 62, 63),
      (64, 65, 66, 67, 68, 69, 70, 71),
      (72, 73, 74, 75, 76, 77, 78, 79))
    val mat2 = mat.map(x => x.toFloat)
    val res1 = MatrixUtil.selectCols(mat2, 0, 4)

    val expectedRes = DenseMatrix(
      (0.0, 4.0),
      (8.0, 12.0),
      (16.0, 20.0),
      (24.0, 28.0),
      (32.0, 36.0),
      (40.0, 44.0),
      (48.0, 52.0),
      (56.0, 60.0),
      (64.0, 68.0),
      (72.0, 76.0))
    assert(res1 === expectedRes)

    val res2 = MatrixUtil.selectCols(mat2, 2, 4)

    val expectedRes2 = DenseMatrix(
      (2.0, 6.0),
      (10.0, 14.0),
      (18.0, 22.0),
      (26.0, 30.0),
      (34.0, 38.0),
      (42.0, 46.0),
      (50.0, 54.0),
      (58.0, 62.0),
      (66.0, 70.0),
      (74.0, 78.0))
    assert(res2 === expectedRes2)
  }

}
