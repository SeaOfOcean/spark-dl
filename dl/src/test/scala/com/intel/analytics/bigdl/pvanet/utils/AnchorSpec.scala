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

import breeze.linalg.{DenseMatrix, convert}
import com.intel.analytics.bigdl.pvanet.TestUtil
import com.intel.analytics.bigdl.pvanet.model.PvanetParam
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class AnchorSpec extends FlatSpec with Matchers {

  "mkanchors" should "work properly" in {
    val hs = Tensor(Storage(Array[Float](12, 16, 22)))
    val ws = Tensor(Storage(Array[Float](23, 16, 11)))
    val anchors = Anchor.mkanchors(ws, hs, 7.5f, 7.5f)
    anchors.storage().toArray should be(Array(-3.5, 2.0, 18.5, 13.0,
      0.0, 0.0, 15.0, 15.0,
      2.5, -3.0, 12.5, 18.0))
  }

  "_whctrs" should "work properly" in {
    Anchor.whctrs(Tensor(Storage(Array[Float](0, 0, 15, 15)))) should be
    (Array(16, 16, 7.5, 7.5))
    Anchor.whctrs(Tensor(Storage(Array[Float](-3.5f, 2.0f, 18.5f, 13f)))) should be
    (Array(23.0, 12.0, 7.5, 7.5))
  }

  "_ratio_enum" should "work properly" in {
    Anchor.ratioEnum(Tensor(Storage(Array(0f, 0, 15, 15))),
      Tensor(Storage(Array(0.5f, 1, 2)))).storage().toArray should be(
      Array(-3.5, 2.0, 18.5, 13.0, 0.0, 0.0, 15.0, 15.0, 2.5, -3.0, 12.5, 18.0))
  }

  "_scale_enum" should "work well" in {
    Anchor.scaleEnum(Tensor(Storage(Array(2.5f, -3.0f, 12.5f, 18.0f))),
      Tensor(Storage(Array[Float](8, 16, 32)))).storage().toArray should be(
      Array(-36.0, -80.0, 51.0, 95.0, -80.0, -168.0, 95.0, 183.0, -168.0, -344.0, 183.0, 359.0)
    )
  }

  "_generate anchors" should "work well" in {
    val act1 = Anchor.generateBasicAnchors(Array[Float](0.5f, 1, 2),
      Array[Float](8, 16, 32))
    val act1Tensor = Anchor.generateBasicAnchors2(Array[Float](0.5f, 1, 2),
      Array[Float](8, 16, 32))
    val expected1 = DenseMatrix((-84.0, -40.0, 99.0, 55.0),
      (-176.0, -88.0, 191.0, 103.0),
      (-360.0, -184.0, 375.0, 199.0),
      (-56.0, -56.0, 71.0, 71.0),
      (-120.0, -120.0, 135.0, 135.0),
      (-248.0, -248.0, 263.0, 263.0),
      (-36.0, -80.0, 51.0, 95.0),
      (-80.0, -168.0, 95.0, 183.0),
      (-168.0, -344.0, 183.0, 359.0))
    assert(act1 === expected1)
    TestUtil.assertMatrixEqualTM(act1Tensor, expected1, 1e-6)
    val act2 = Anchor.generateBasicAnchors(Array[Float](0.5f, 1, 2, 4, 8),
      Array[Float](8, 16, 32, 64, 128))
    val act2Tensor = Anchor.generateBasicAnchors2(Array[Float](0.5f, 1, 2, 4, 8),
      Array[Float](8, 16, 32, 64, 128))
    val expected2 = DenseMatrix(
      (-84.0, -40.0, 99.0, 55.0),
      (-176.0, -88.0, 191.0, 103.0),
      (-360.0, -184.0, 375.0, 199.0),
      (-728.0, -376.0, 743.0, 391.0),
      (-1464.0, -760.0, 1479.0, 775.0),
      (-56.0, -56.0, 71.0, 71.0),
      (-120.0, -120.0, 135.0, 135.0),
      (-248.0, -248.0, 263.0, 263.0),
      (-504.0, -504.0, 519.0, 519.0),
      (-1016.0, -1016.0, 1031.0, 1031.0),
      (-36.0, -80.0, 51.0, 95.0),
      (-80.0, -168.0, 95.0, 183.0),
      (-168.0, -344.0, 183.0, 359.0),
      (-344.0, -696.0, 359.0, 711.0),
      (-696.0, -1400.0, 711.0, 1415.0),
      (-24.0, -120.0, 39.0, 135.0),
      (-56.0, -248.0, 71.0, 263.0),
      (-120.0, -504.0, 135.0, 519.0),
      (-248.0, -1016.0, 263.0, 1031.0),
      (-504.0, -2040.0, 519.0, 2055.0),
      (-16.0, -184.0, 31.0, 199.0),
      (-40.0, -376.0, 55.0, 391.0),
      (-88.0, -760.0, 103.0, 775.0),
      (-184.0, -1528.0, 199.0, 1543.0),
      (-376.0, -3064.0, 391.0, 3079.0)
    )
    assert(act2 === expected2)
    TestUtil.assertMatrixEqualTM(act2Tensor, expected2, 1e-6)
  }

  val param = new PvanetParam()
  val root = "data/middle/"
  val width = 37
  val height = 50


  val anchors = Anchor.generateBasicAnchors(param.anchorRatios, param.anchorScales)

  "generateAnchors" should "work properly" in {
    TestUtil.assertMatrixEqualTM(FileUtil.loadFeatures[Float]("anchors", root),
      convert(anchors, Double), 1e-6)
  }

  var shifts: DenseMatrix[Float] = Anchor.generateShifts(width, height, 16)

  "generateShifts" should "work properly" in {
    val expected = FileUtil.loadFeatures[Float]("shifts", root)
    TestUtil.assertMatrixEqualTM(expected, convert(shifts, Double), 1e-6)
  }

  val allAnchors = Anchor.getAllAnchors(shifts, anchors)
  val allAnchors2 = Anchor.getAllAnchors(Tensor(shifts), Tensor(anchors))
  "generateAllAnchors" should "work properly" in {
    val expected = FileUtil.loadFeatures[Float]("allAnchors", root)
    TestUtil.assertMatrixEqualTM(expected, convert(allAnchors, Double), 1e-6)
    TestUtil.assertMatrixEqualTM(allAnchors2, convert(allAnchors, Double), 1e-6)
  }

  "generate shifts with tensor " should "work properly" in {
    val shifts2 = Anchor.generateShifts2(2, 3, 2)
    val shift = Anchor.generateShifts(2, 3, 2)
    val expected = Tensor(Storage(Array(0.0, 0.0, 0.0, 0.0,
      2.0, 0.0, 2.0, 0.0,
      0.0, 2.0, 0.0, 2.0,
      2.0, 2.0, 2.0, 2.0,
      0.0, 4.0, 0.0, 4.0,
      2.0, 4.0, 2.0, 4.0).map(x => x.toFloat))).resize(6, 4)
    shifts2 should be(expected)
    TestUtil.assertMatrixEqualTM(shifts2, convert(shift, Double), 1e-6)
  }

  "getAllAnchors" should "work properly" in {
    val shifts = Tensor.randperm[Float](12).resize(3, 4)
    val anchors = Tensor.randperm[Float](8).resize(2, 4)
    val allAnchors = Anchor.getAllAnchors(shifts.toBreezeMatrix(), anchors.toBreezeMatrix())
    val allAnchors2 = Anchor.getAllAnchors(shifts, anchors)
    TestUtil.assertMatrixEqualTM(allAnchors2, convert(allAnchors, Double), 1e-6)
  }
}
