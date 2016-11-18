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

package com.intel.analytics.sparkdl.pvanet.layers

import com.intel.analytics.sparkdl.tensor.Tensor
import org.scalatest.FlatSpec

class ReshapeSpec extends FlatSpec {

  "A Reshape Module " should "generate correct output and grad" in {
    val module = new Reshape2[Double](Array(3, 2))
    for (batchSize <- 1 to 4) {
      val input = Tensor[Double](batchSize, 1, 6)
      input.rand()
      val inputOrg = input.clone()
      val output = module.forward(input)
      val gradOutput = Tensor[Double](batchSize, 3, 2)
      gradOutput.rand()
      val gradOutputOrg = gradOutput.clone()
      val gradInput = module.backward(input, gradOutput)
      assert(output.nDimension() == 3)
      assert(output.size(1) == batchSize)
      assert(output.size(2) == 3)
      assert(output.size(3) == 2)
      assert(gradInput.isSameSizeAs(input))
      for (i <- 1 to batchSize) {
        for (j <- 0 to 5) {
          assert(input(Array(i, 1, j + 1)) == output(Array(i, j / 2 + 1, j % 2 + 1)))
          assert(gradInput(Array(i, 1, j + 1)) == gradOutput(Array(i, j / 2 + 1, j % 2 + 1)))
        }
      }
      assert(input == inputOrg)
      assert(gradOutput == gradOutputOrg)
    }

    intercept[IllegalArgumentException] {
      module.forward(Tensor[Double](2, 2))
    }

    intercept[IllegalArgumentException] {
      module.forward(Tensor[Double](3, 2, 2))
    }
  }

  "A Reshape Module default batch" should "generate correct output and grad" in {
    val module = new Reshape2[Double](Array(3, 2))
    val input = Tensor[Double](2, 3)
    input.rand()
    val inputOrg = input.clone()
    val output = module.forward(input)
    val gradOutput = Tensor[Double](3, 2)
    gradOutput.rand()
    val gradOutputOrg = gradOutput.clone()
    val gradInput = module.backward(input, gradOutput)
    assert(output.nDimension() == 2)
    assert(output.size(1) == 3)
    assert(output.size(2) == 2)
    for (j <- 0 to 5) {
      assert(input(Array(j / 3 + 1, j % 3 + 1)) == output(Array(j / 2 + 1, j % 2 + 1)))
      assert(gradInput(Array(j / 3 + 1, j % 3 + 1)) == gradOutput(Array(j / 2 + 1, j % 2 + 1)))
    }
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)
  }

  "A Reshape Module disable batch" should "generate correct output and grad" in {
    val module = new Reshape2[Double](Array(3, 2), Some(false))
    val input = Tensor[Double](1, 2, 3)
    input.rand()
    val inputOrg = input.clone()
    val output = module.forward(input)
    val gradOutput = Tensor[Double](3, 2)
    gradOutput.rand()
    val gradOutputOrg = gradOutput.clone()
    val gradInput = module.backward(input, gradOutput)
    assert(output.nDimension() == 2)
    assert(output.size(1) == 3)
    assert(output.size(2) == 2)
    for (j <- 0 to 5) {
      assert(input(Array(1, j / 3 + 1, j % 3 + 1)) == output(Array(j / 2 + 1, j % 2 + 1)))
      assert(gradInput(Array(1, j / 3 + 1, j % 3 + 1)) == gradOutput(Array(j / 2 + 1, j % 2 + 1)))
    }
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)

    intercept[IllegalArgumentException] {
      module.forward(Tensor[Double](2, 3, 2))
    }
  }

  "A Reshape Module enable batch" should "generate correct output and grad" in {
    val module = new Reshape2[Double](Array(3, 2), Some(true))
    for (batchSize <- 1 to 4) {
      val input = Tensor[Double](batchSize, 1, 6)
      input.rand()
      val inputOrg = input.clone()
      val output = module.forward(input)
      val gradOutput = Tensor[Double](batchSize, 3, 2)
      gradOutput.rand()
      val gradOutputOrg = gradOutput.clone()
      val gradInput = module.backward(input, gradOutput)
      assert(output.nDimension() == 3)
      assert(output.size(1) == batchSize)
      assert(output.size(2) == 3)
      assert(output.size(3) == 2)
      assert(gradInput.isSameSizeAs(input))
      for (i <- 1 to batchSize) {
        for (j <- 0 to 5) {
          assert(input(Array(i, 1, j + 1)) == output(Array(i, j / 2 + 1, j % 2 + 1)))
          assert(gradInput(Array(i, 1, j + 1)) == gradOutput(Array(i, j / 2 + 1, j % 2 + 1)))
        }
      }
      assert(input == inputOrg)
      assert(gradOutput == gradOutputOrg)
    }

    intercept[IllegalArgumentException] {
      module.forward(Tensor[Double](3, 2))
    }
  }

  "A Reshape Module with infer" should "generate correct output and grad" in {
    val module = new Reshape2[Double](Array(3, -1))
    for (batchSize <- 1 to 4) {
      val input = Tensor[Double](batchSize, 1, 6)
      input.rand()
      val inputOrg = input.clone()
      val output = module.forward(input)
      val gradOutput = Tensor[Double](batchSize, 3, 2)
      gradOutput.rand()
      val gradOutputOrg = gradOutput.clone()
      val gradInput = module.backward(input, gradOutput)
      assert(output.nDimension() == 3)
      assert(output.size(1) == batchSize)
      assert(output.size(2) == 3)
      assert(output.size(3) == 2)
      assert(gradInput.isSameSizeAs(input))
      for (i <- 1 to batchSize) {
        for (j <- 0 to 5) {
          assert(input(Array(i, 1, j + 1)) == output(Array(i, j / 2 + 1, j % 2 + 1)))
          assert(gradInput(Array(i, 1, j + 1)) == gradOutput(Array(i, j / 2 + 1, j % 2 + 1)))
        }
      }
      assert(input == inputOrg)
      assert(gradOutput == gradOutputOrg)
    }

    intercept[IllegalArgumentException] {
      module.forward(Tensor[Double](2, 2))
    }

    intercept[IllegalArgumentException] {
      module.forward(Tensor[Double](3, 2, 2))
    }
  }

  "A Reshape Module default batch with infer" should "generate correct output and grad" in {
    val module = new Reshape2[Double](Array(-1, 2))
    val input = Tensor[Double](2, 3)
    input.rand()
    val inputOrg = input.clone()
    val output = module.forward(input)
    val gradOutput = Tensor[Double](3, 2)
    gradOutput.rand()
    val gradOutputOrg = gradOutput.clone()
    val gradInput = module.backward(input, gradOutput)
    assert(output.nDimension() == 2)
    assert(output.size(1) == 3)
    assert(output.size(2) == 2)
    for (j <- 0 to 5) {
      assert(input(Array(j / 3 + 1, j % 3 + 1)) == output(Array(j / 2 + 1, j % 2 + 1)))
      assert(gradInput(Array(j / 3 + 1, j % 3 + 1)) == gradOutput(Array(j / 2 + 1, j % 2 + 1)))
    }
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)
  }

  "A Reshape Module disable batch with infer" should "generate correct output and grad" in {
    val module = new Reshape2[Double](Array(3, -1), Some(false))
    val input = Tensor[Double](1, 2, 3)
    input.rand()
    val inputOrg = input.clone()
    val output = module.forward(input)
    val gradOutput = Tensor[Double](3, 2)
    gradOutput.rand()
    val gradOutputOrg = gradOutput.clone()
    val gradInput = module.backward(input, gradOutput)
    assert(output.nDimension() == 2)
    assert(output.size(1) == 3)
    assert(output.size(2) == 2)
    for (j <- 0 to 5) {
      assert(input(Array(1, j / 3 + 1, j % 3 + 1)) == output(Array(j / 2 + 1, j % 2 + 1)))
      assert(gradInput(Array(1, j / 3 + 1, j % 3 + 1)) == gradOutput(Array(j / 2 + 1, j % 2 + 1)))
    }
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)

    intercept[IllegalArgumentException] {
      module.forward(Tensor[Double](2, 3, 2))
    }
  }

  "A Reshape Module enable batch with infer" should "generate correct output and grad" in {
    val module = new Reshape2[Double](Array(-1, 2), Some(true))
    for (batchSize <- 1 to 4) {
      val input = Tensor[Double](batchSize, 1, 6)
      input.rand()
      val inputOrg = input.clone()
      val output = module.forward(input)
      val gradOutput = Tensor[Double](batchSize, 3, 2)
      gradOutput.rand()
      val gradOutputOrg = gradOutput.clone()
      val gradInput = module.backward(input, gradOutput)
      assert(output.nDimension() == 3)
      assert(output.size(1) == batchSize)
      assert(output.size(2) == 3)
      assert(output.size(3) == 2)
      assert(gradInput.isSameSizeAs(input))
      for (i <- 1 to batchSize) {
        for (j <- 0 to 5) {
          assert(input(Array(i, 1, j + 1)) == output(Array(i, j / 2 + 1, j % 2 + 1)))
          assert(gradInput(Array(i, 1, j + 1)) == gradOutput(Array(i, j / 2 + 1, j % 2 + 1)))
        }
      }
      assert(input == inputOrg)
      assert(gradOutput == gradOutputOrg)
    }

    intercept[IllegalArgumentException] {
      module.forward(Tensor[Double](3, 2))
    }
  }

  "reshape with 0 and -1" should "work well" in {
    val tensor = Tensor.randperm[Float](1024)
    tensor.resize(2, 16, 4, 8)
    val model = new Reshape2[Float](Array(0, 4, -1, 0))
    val expectedShape = Array(2, 4, 16, 8)
    val out = model.forward(tensor).size()
    (out zip expectedShape).foreach(x => x._1 == x._2)
  }

}
