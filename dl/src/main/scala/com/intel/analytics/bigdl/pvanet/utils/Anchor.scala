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

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}


object Anchor {
  /**
   * Generate anchor (reference) windows by enumerating aspect ratios X
   * scales wrt a reference (0, 0, 15, 15) window.
   *
   */
  def generateBasicAnchors(ratios: Array[Float], scales: Array[Float],
    baseSize: Float = 16): DenseMatrix[Float] = {
    val baseAnchor = Tensor(Storage(Array(1 - 1, 1 - 1, baseSize - 1, baseSize - 1)))
    val ratioAnchors = ratioEnum(baseAnchor, Tensor(Storage(ratios)))
    val anchors = new DenseMatrix[Float](scales.length * ratioAnchors.size(1), 4)
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
   * Generate anchor (reference) windows by enumerating aspect ratios X
   * scales wrt a reference (0, 0, 15, 15) window.
   *
   */
  def generateBasicAnchors2(ratios: Array[Float], scales: Array[Float],
    baseSize: Float = 16): Tensor[Float] = {
    val baseAnchor = Tensor(Storage(Array(1 - 1, 1 - 1, baseSize - 1, baseSize - 1)))
    val ratioAnchors = ratioEnum(baseAnchor, Tensor(Storage(ratios)))
    val anchors = Tensor[Float](scales.length * ratioAnchors.size(1), 4)
    var idx = 1
    for (i <- 1 to ratioAnchors.size(1)) {
      val scaleAnchors = scaleEnum(ratioAnchors(i), Tensor(Storage(scales)))
      for (j <- 1 to scaleAnchors.size(1)) {
        anchors.setValue(idx, 1, scaleAnchors(j).valueAt(1))
        anchors.setValue(idx, 2, scaleAnchors(j).valueAt(2))
        anchors.setValue(idx, 3, scaleAnchors(j).valueAt(3))
        anchors.setValue(idx, 4, scaleAnchors(j).valueAt(4))
        idx = idx + 1
      }
    }
    anchors
  }

  /**
   * Given a vector of widths (ws) and heights (hs) around a center
   * (x_ctr, y_ctr), output a set of anchors (windows).
   *
   */
  def mkanchors(ws: Tensor[Float], hs: Tensor[Float],
    xCtr: Float, yCtr: Float): Tensor[Float] = {
    val a1 = (ws.-(1)).mul(-0.5f).add(xCtr)
    val a2 = (hs.-(1)).mul(-0.5f).add(yCtr)
    val a3 = (ws.-(1)).mul(0.5f).add(xCtr)
    val a4 = (hs.-(1)).mul(0.5f).add(yCtr)
    val anchors = Tensor[Float](a1.nElement(), 4)
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
   */
  def whctrs(anchor: Tensor[Float]): Array[Float] = {
    val w: Float = anchor.valueAt(3) - anchor.valueAt(1) + 1
    val h: Float = anchor.valueAt(4) - anchor.valueAt(2) + 1
    val xCtr: Float = anchor.valueAt(1) + 0.5f * (w - 1)
    val yCtr: Float = anchor.valueAt(2) + 0.5f * (h - 1)
    Array[Float](w, h, xCtr, yCtr)
  }

  /**
   * Enumerate a set of anchors for each aspect ratio wrt an anchor.
   *
   */
  def ratioEnum(anchor: Tensor[Float], ratios: Tensor[Float]): Tensor[Float] = {
    // w, h, x_ctr, y_ctr
    val out = whctrs(anchor)
    val size = out(0) * out(1)
    val sizeRatios = ratios.clone().apply1(x => size / x)
    val ws = sizeRatios.apply1(x => Math.sqrt(x).round)
    val hs = Tensor[Float](ws.nElement())
    for (i <- 1 to ws.nElement()) {
      hs.setValue(i, Math.round(ws.valueAt(i) * ratios.valueAt(i)))
    }
    mkanchors(ws, hs, out(2), out(3))
  }

  /**
   * Enumerate a set of anchors for each scale wrt an anchor.
   *
   */
  def scaleEnum(anchor: Tensor[Float], scales: Tensor[Float]): Tensor[Float] = {
    val out = whctrs(anchor)
    val ws = scales.clone().apply1(x => x * out(0))
    val hs = scales.clone().apply1(x => x * out(1))
    mkanchors(ws, hs, out(2), out(3))
  }

  def generateShifts(width: Int, height: Int, featStride: Float): DenseMatrix[Float] = {
    val shiftX = DenseVector.range(0, width).map(x => x * featStride)
    val shiftY = DenseVector.range(0, height).map(x => x * featStride)
    MatrixUtil.meshgrid(shiftX, shiftY) match {
      case (x1Mesh, x2Mesh) =>
        return DenseMatrix.vertcat(x1Mesh.t.toDenseVector.toDenseMatrix,
          x2Mesh.t.toDenseVector.toDenseMatrix,
          x1Mesh.t.toDenseVector.toDenseMatrix,
          x2Mesh.t.toDenseVector.toDenseMatrix).t
    }
    new DenseMatrix(0, 0)
  }

  def generateShifts2(width: Int, height: Int, featStride: Float): Tensor[Float] = {
    val shiftX = Tensor[Float].range(0, width - 1).apply1(x => x * featStride)
    val shiftY = Tensor[Float].range(0, height - 1).apply1(x => x * featStride)
    TensorUtil.meshgrid(shiftX, shiftY) match {
      case (x1Mesh, x2Mesh) =>
        return TensorUtil.concat(
          x1Mesh.resize(x1Mesh.nElement()),
          x2Mesh.resize(x2Mesh.nElement()),
          x1Mesh.resize(x1Mesh.nElement()),
          x2Mesh.resize(x2Mesh.nElement())).resize(4, x1Mesh.nElement()).t().contiguous()
    }
    Tensor[Float](0, 0)
  }

  def getAllAnchors(shifts: DenseMatrix[Float],
    anchors: DenseMatrix[Float]): DenseMatrix[Float] = {
    val allAnchors = new DenseMatrix[Float](shifts.rows * anchors.rows, 4)
    for (s <- 0 until shifts.rows) {
      allAnchors(s * anchors.rows until (s + 1) * anchors.rows, 0 until 4) :=
        (anchors.t(::, *) + shifts.t(::, s)).t
    }
    allAnchors
  }

  /**
   * each row of shifts add each row of anchors
   * and return shifts.size(1) * anchors.size(1) rows
   *
   * @param shifts
   * @param anchors
   * @return
   */
  def getAllAnchors(shifts: Tensor[Float],
    anchors: Tensor[Float]): Tensor[Float] = {
    assert(shifts.size(2) == 4 && anchors.size(2) == 4)
    val allAnchors = Tensor[Float](shifts.size(1) * anchors.size(1), 4)
    var r = 1
    for (s <- 1 to shifts.size(1)) {
      for (a <- 1 to anchors.size(1)) {
        allAnchors.update(r, shifts(s).clone().add(anchors(a)))
        r = r + 1
      }
    }
    allAnchors
  }
}
