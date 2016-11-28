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

package com.intel.analytics.bigdl.pvanet

import breeze.linalg.DenseMatrix
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}


object Anchor {
  /**
   * Generate anchor (reference) windows by enumerating aspect ratios X
   * scales wrt a reference (0, 0, 15, 15) window.
   *
   * @param baseSize
   * @return
   */
  def generateAnchors(baseSize: Float = 16,
    ratios: Array[Float],
    scales: Array[Float]): DenseMatrix[Float] = {
    val baseAnchor = Tensor(Storage(Array(1 - 1, 1 - 1, baseSize - 1, baseSize - 1)))
    val ratioAnchors = ratioEnum(baseAnchor, Tensor(Storage(ratios)))
    var anchors = new DenseMatrix[Float](scales.length * ratioAnchors.size(1), 4)
    //    var anchors = Tensor[Float]()
    var idx = 0
    for (i <- 0 until ratioAnchors.size(1)) {
      val scaleAnchors = scaleEnum(ratioAnchors(i + 1), Tensor(Storage(scales)))
      for (j <- 0 until scaleAnchors.size(1)) {
        anchors(idx, 0) = scaleAnchors(j + 1).valueAt(1)
        anchors(idx, 1) = scaleAnchors(j + 1).valueAt(2)
        anchors(idx, 2) = scaleAnchors(j + 1).valueAt(3)
        anchors(idx, 3) = scaleAnchors(j + 1).valueAt(4)
        idx = idx + 1
      }
    }
    anchors
  }

  /**
   * Given a vector of widths (ws) and heights (hs) around a center
   * (x_ctr, y_ctr), output a set of anchors (windows).
   *
   * @param ws
   * @param hs
   * @param xCtr
   * @param yCtr
   * @return
   */
  def mkanchors(ws: Tensor[Float], hs: Tensor[Float],
    xCtr: Float, yCtr: Float): Tensor[Float] = {
    val a1 = (ws.-(1)).mul(-0.5f).add(xCtr)
    val a2 = (hs.-(1)).mul(-0.5f).add(yCtr)
    val a3 = (ws.-(1)).mul(0.5f).add(xCtr)
    val a4 = (hs.-(1)).mul(0.5f).add(yCtr)
    var anchors = Tensor[Float](a1.nElement(), 4)
    for (i <- 1 to a1.nElement()) {
      anchors.setValue(i, 1, a1.valueAt(i))
      anchors.setValue(i, 2, a2.valueAt(i))
      anchors.setValue(i, 3, a3.valueAt(i))
      anchors.setValue(i, 4, a4.valueAt(i))
    }
    anchors
  }

  /**
   * Return width, height, x center, and y center for an anchor (window).
   *
   * @param anchor
   * @return
   */
  def whctrs(anchor: Tensor[Float]): Array[Float] = {
    val w: Float = anchor.valueAt(3) - anchor.valueAt(1) + 1
    val h: Float = anchor.valueAt(4) - anchor.valueAt(2) + 1
    val xCtr: Float = (anchor.valueAt(1) + 0.5f * (w - 1))
    val yCtr: Float = (anchor.valueAt(2) + 0.5f * (h - 1))
    Array[Float](w, h, xCtr, yCtr)
  }

  /**
   * Enumerate a set of anchors for each aspect ratio wrt an anchor.
   *
   * @param anchor
   * @param ratios
   * @return
   */
  def ratioEnum(anchor: Tensor[Float], ratios: Tensor[Float]): Tensor[Float] = {
    // w, h, x_ctr, y_ctr
    val out = whctrs(anchor)
    val size = out(0) * out(1)
    var sizeRatios = ratios.clone().apply1(x => size / x)
    val ws = sizeRatios.apply1(x => Math.sqrt(x).round)
    var hs = Tensor[Float](ws.nElement())
    for (i <- 1 to ws.nElement()) {
      hs.setValue(i, Math.round(ws.valueAt(i) * ratios.valueAt(i)))
    }
    mkanchors(ws, hs, out(2), out(3))
  }

  /**
   * Enumerate a set of anchors for each scale wrt an anchor.
   *
   * @param anchor
   * @param scales
   * @return
   */
  def scaleEnum(anchor: Tensor[Float], scales: Tensor[Float]): Tensor[Float] = {
    val out = whctrs(anchor)
    val ws = scales.clone().apply1(x => x * out(0))
    val hs = scales.clone().apply1(x => x * out(1))
    mkanchors(ws, hs, out(2), out(3))
  }

}
