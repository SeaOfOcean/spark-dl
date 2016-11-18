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
