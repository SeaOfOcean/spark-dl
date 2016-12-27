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

import breeze.linalg.DenseMatrix
import breeze.numerics.abs
import com.intel.analytics.bigdl.pvanet.TestUtil._
import com.intel.analytics.bigdl.pvanet.model.{Phase, VggParam}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table
import org.scalatest.{FlatSpec, Matchers}

class ProposalSpec extends FlatSpec with Matchers {
  val param = new VggParam(Phase.TRAIN)
  val classLoader = getClass().getClassLoader()
  val input = new Table
  input.insert(loadDataFromFile(
    classLoader.getResource("pvanet/data1.dat").getFile, Array(1, 18, 30, 40)))
  input.insert(loadDataFromFile(
    classLoader.getResource("pvanet/data2.dat").getFile, Array(1, 36, 30, 40)))
  input.insert(Tensor(Storage(Array(100f, 200f, 6.0f))))

  val expected1 = DenseMatrix((0.0, 0.0, 0.0, 199.0, 99.0),
    (0.0, 0.0, 2.69759297, 127.94831848, 99.0),
    (0.0, 71.59012604, 0.0, 199.0, 99.0),
    (0.0, 47.27521133, 0.0, 152.9140625, 99.0))
  val expected2 = Array(
    0.99929377,
    0.99398681,
    0.9928103,
    0.78120455)
  val proposal = new Proposal[Float](param)
  "testUpdateOutput" should "be correct" in {
    val out = proposal.forward(input)
    assert(out.length() == 2)
    val out1 = out(1).asInstanceOf[Tensor[Float]]
    val out2 = out(2).asInstanceOf[Tensor[Float]]
    for (i <- 0 until expected1.rows) {
      for (j <- 0 until expected1.cols) {
        assert(abs(expected1(i, j) - out1.valueAt(i + 1, j + 1)) < 1e-4)
      }
    }
    expected2.zipWithIndex.foreach(x => assert(abs(out2.valueAt(x._2 + 1, 1) - x._1) < 1e-4))
  }

  "testUpdateOutput twice" should "be correct" in {
    val out = proposal.forward(input)
    assert(out.length() == 2)
    val out1 = out(1).asInstanceOf[Tensor[Float]]
    val out2 = out(2).asInstanceOf[Tensor[Float]]
    for (i <- 0 until expected1.rows) {
      for (j <- 0 until expected1.cols) {
        assert(abs(expected1(i, j) - out1.valueAt(i + 1, j + 1)) < 1e-4)
      }
    }
    expected2.zipWithIndex.foreach(x => assert(abs(out2.valueAt(x._2 + 1, 1) - x._1) < 1e-4))
  }
}
