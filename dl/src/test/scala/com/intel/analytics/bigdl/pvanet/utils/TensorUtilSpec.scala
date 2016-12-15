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

package com.intel.analytics.bigdl.pvanet.utils

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class TensorUtilSpec extends FlatSpec with Matchers {

  "concat" should "work properly" in {
    val t1 = Tensor(Storage(Array[Double](1, 2, 3)))
    val t2 = Tensor(Storage(Array[Double](4, 5, 6)))
    val t3 = Tensor(Storage(Array[Double](7, 8, 9)))

    val res = TensorUtil.concat[Double](t1, t2, t3)
    val expected = Tensor(Storage(Array[Double](1, 2, 3, 4, 5, 6, 7, 8, 9)))
    res should be(expected)
  }

  "updateRange" should "work properly" in {
    val dest = Tensor(Storage(Array(1, 2, 3, 4, 5, 6, 7, 8, 9).map(x => x.toFloat))).resize(3, 3)
    val src = Tensor(Storage(Array(-1, -2, -3, -4).map(x => x.toFloat))).resize(2, 2)
    TensorUtil.updateRange(dest, 1, 2, 1, 2, src)
    val expected = Tensor(Storage(Array(-1, -2, 3, -3, -4, 6, 7, 8, 9)
      .map(x => x.toFloat))).resize(3, 3)
    dest should be(expected)
    TensorUtil.updateRange(dest, 2, 3, 2, 3, src)
    val expected2 = Tensor(Storage(Array(-1, -2, 3, -3, -1, -2, 7, -3, -4)
      .map(x => x.toFloat))).resize(3, 3)
    dest should be(expected2)
  }

}
