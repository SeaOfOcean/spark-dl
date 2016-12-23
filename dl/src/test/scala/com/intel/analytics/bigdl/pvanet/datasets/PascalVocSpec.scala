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

import java.nio.file.Paths

import org.scalatest.{FlatSpec, Matchers}

class PascalVocSpec extends FlatSpec with Matchers {
  val voc = new PascalVoc("2007", "test")
  behavior of "PascalVocSpec"
  val resource = getClass().getClassLoader().getResource("pvanet")
  it should "eval" in {
    val result = voc.eval(Some(Paths.get(resource.getPath, "result").toString))
    val expected = Array(
      ("aeroplane", 0.8767),
      ("bicycle", 0.8721),
      ("bird", 0.8655),
      ("boat", 0.8166),
      ("bottle", 0.7330),
      ("bus", 0.8882),
      ("car", 0.8907),
      ("cat", 0.8992),
      ("chair", 0.7032),
      ("cow", 0.8817),
      ("diningtable", 0.8057),
      ("dog", 0.8927),
      ("horse", 0.8598),
      ("motorbike", 0.8678),
      ("person", 0.8594),
      ("pottedplant", 0.6093),
      ("sheep", 0.8642),
      ("sofa", 0.8412),
      ("train", 0.8875),
      ("tvmonitor", 0.8523),
      ("Mean AP", 0.8383)
    )
    (result zip expected).foreach(x => {
      assert(x._1._1 == x._2._1)
      // todo: the result is not exactly the same
      assert(Math.abs(x._1._2 - x._2._2) < 1e-2)
    })
  }
}
