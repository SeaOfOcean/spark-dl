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

package com.intel.analytics.sparkdl.pvanet

import breeze.linalg.DenseMatrix
import breeze.numerics._
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}

import scala.io.Source

object TestUtil {
  def assertMatrixEqualTM(actual: Tensor[Float],
    expected: DenseMatrix[Double], diff: Double): Unit = {
    if (actual.dim() == 1) {
      assert(actual.nElement() == expected.size)
      var d = 1
      for (r <- 0 until expected.rows) {
        for (c <- 0 until expected.cols) {
          assert(abs(expected(r, c) - actual.valueAt(d)) < diff)
          d += 1
        }
      }
    } else {
      assert(actual.size(1) == expected.rows && actual.size(2) == expected.cols)
      for (r <- 0 until expected.rows) {
        for (c <- 0 until expected.cols) {
          assert(abs(expected(r, c) - actual.valueAt(r + 1, c + 1)) < diff)
        }
      }
    }

  }


  def assertMatrixEqual(actual: DenseMatrix[Float],
    expected: DenseMatrix[Float], diff: Float): Unit = {
    for (r <- 0 until expected.rows) {
      for (c <- 0 until expected.cols) {
        assert(abs(expected(r, c) - actual(r, c)) < diff)
      }
    }
  }

  def assertMatrixEqualFD(actual: DenseMatrix[Float],
    expected: DenseMatrix[Double], diff: Double): Unit = {
    assert((actual.rows == expected.rows) && (actual.cols == expected.cols),
      s"actual shape is (${actual.rows}, ${actual.cols}), " +
        s"while expected shape is (${expected.rows}, ${expected.cols})")
    for (r <- 0 until expected.rows) {
      for (c <- 0 until expected.cols) {
        assert(abs(expected(r, c) - actual(r, c)) < diff)
      }
    }
  }


  def loadDataFromFile(fileName: String, sizes: Array[Int]): Tensor[Float] = {
    val lines = Source.fromFile(fileName).getLines().toArray.map(x => x.toFloat)
    Tensor(Storage(lines)).resize(sizes)
  }
}
