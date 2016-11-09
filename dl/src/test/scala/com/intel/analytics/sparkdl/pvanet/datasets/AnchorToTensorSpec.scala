package com.intel.analytics.sparkdl.pvanet.datasets

import breeze.linalg.{DenseMatrix, DenseVector, convert}
import com.intel.analytics.sparkdl.pvanet.AnchorTarget
import org.scalatest.{FlatSpec, Matchers}

class AnchorToTensorSpec extends FlatSpec with Matchers {


  "apply" should "work properly " in {
    val att = new AnchorToTensor(1, 1, 1)
    val labels: DenseVector[Int] = DenseVector(1, 2, 3, 4)
    val bboxTargets: DenseMatrix[Float] = convert(
      DenseMatrix((0.3, 0.4, 0.3, 0.3),
        (0.3, 0.4, 0.3, 0.3),
        (0.3, 0.4, 0.3, 0.3),
        (0.3, 0.4, 0.3, 0.3)), Float)
    val bboxInsideWeights: DenseMatrix[Float] = convert(
      DenseMatrix((0.1, 0.2, 0.1, 0.1),
        (0.4, 0.2, 0.1, 0.1),
        (0.7, 0.2, 0.1, 0.1),
        (0.6, 0.2, 0.1, 0.1)), Float)
    val bboxOutsideWeights: DenseMatrix[Float] = convert(
      DenseMatrix((0.6, 0.4, 0.6, 0.5),
        (0.5, 0.4, 0.6, 0.5),
        (0.4, 0.4, 0.6, 0.5),
        (0.7, 0.4, 0.6, 0.5)), Float)
    val at = new AnchorTarget(labels, bboxTargets, bboxInsideWeights, bboxOutsideWeights)
    val expectedTargets = convert(
      DenseMatrix(0.3, 0.4, 0.3, 0.3,
        0.3, 0.4, 0.3, 0.3,
        0.3, 0.4, 0.3, 0.3,
        0.3, 0.4, 0.3, 0.3,
        0.1, 0.2, 0.1, 0.1,
        0.4, 0.2, 0.1, 0.1,
        0.7, 0.2, 0.1, 0.1,
        0.6, 0.2, 0.1, 0.1,
        0.6, 0.4, 0.6, 0.5,
        0.5, 0.4, 0.6, 0.5,
        0.4, 0.4, 0.6, 0.5,
        0.7, 0.4, 0.6, 0.5), Float)
    val (label, target) = att.apply(at)
    for (i <- 1 to target.nElement()) {
      assert(expectedTargets.valueAt(i - 1) == target.resize(target.nElement()).valueAt(i))
    }

  }
}
