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

import breeze.linalg.DenseMatrix
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.FlatSpec

class PascolVocSpec extends FlatSpec {

  behavior of "PascolVocSpec"

  it should "hstack" in {
    val clsBoxes = new Array[Tensor[Float]](3)
    val arr1 = Array(1, 2, 3).map(x => x.toFloat)
    val arr2 = Array(4, 5, 6).map(x => x.toFloat)
    val arr3 = Array(7, 8, 9).map(x => x.toFloat)
    val scores = Array(4, 3, 2).map(x => x.toFloat)
    clsBoxes(0) = Tensor(Storage(arr1))
    clsBoxes(1) = Tensor(Storage(arr2))
    clsBoxes(2) = Tensor(Storage(arr3))
    val out = PascolVoc.hstack(clsBoxes, scores)
    val expected = DenseMatrix(
      (1.0, 2.0, 3.0, 4.0),
      (4.0, 5.0, 6.0, 3.0),
      (7.0, 8.0, 9.0, 2.0))
    assert(out.rows == expected.rows && out.cols == expected.cols)
    for (i <- 0 until out.rows) {
      for (j <- 0 until out.cols) {
        assert(out(i, j) == expected(i, j))
      }
    }
  }

}
