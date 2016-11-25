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

import breeze.linalg.{max, min}
import breeze.numerics.{ceil, floor, round}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.pvanet.datasets.PascolVoc
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.io.Source
import scala.reflect.ClassTag

class RoiPooling[@specialized(Float, Double) T: ClassTag]
(val pooled_w: Int, val pooled_h: Int, val spatial_scale: T)
  (implicit ev: TensorNumeric[T]) extends Module[Table, Tensor[T], T] {
  @transient var channels = 0
  @transient var height = 0
  @transient var width = 0
  @transient var argmax: Tensor[T] = null
  var gradInput1: Tensor[T] = null

  override def updateOutput(input: Table): Tensor[T] = {
    assert(input.length() == 2, "there must have two tensors in the table")
    val data = input(1).asInstanceOf[Tensor[T]]
    val rois = input(2).asInstanceOf[Tensor[T]]
    assert(rois.size().length > 1 && rois.size()(1) == 5, "roi input shape should be (R, 5)")
    assert(rois.size()(0) * rois.size()(1) == rois.nElement(), "roi input shape should be (R, 5)")
    channels = data.size()(1)
    height = data.size()(2)
    width = data.size()(3)

    val numRois = rois.size()(0)
    val batchSize = data.size()(0)
    val roiData = rois.storage().array()
    output = Tensor(numRois, channels, pooled_h, pooled_w)

    output.fill(ev.fromType[Double](-Double.MaxValue))
    var outputData = output.storage().array()
    if (argmax == null) {
      argmax = Tensor[T]()
    }
    argmax.resizeAs(output)
    argmax.fill(ev.fromType(-1))
    var argmax_data = argmax.storage().array()

    val bottom_data = data.storage().array()
    val bottom_rois = rois.storage().array()

    //    val topData = output.storage().array()

    var topDataIndex = 0
    var argmaxIndex = 0

    var bottomRoisIndex = 0
    for (n <- 0 until numRois) {
      val roiBatchInd = roiData(bottomRoisIndex)
      def roundRoi(ind: Int) = round(ev.toType[Double](roiData(bottomRoisIndex + ind))
        * ev.toType[Double](spatial_scale)).toInt
      val roi_start_w = roundRoi(1)
      val roi_start_h = roundRoi(2)
      val roi_end_w = roundRoi(3)
      val roi_end_h = roundRoi(4)

      assert(ev.isGreaterEq(roiBatchInd, ev.fromType(0)))
      assert(ev.isGreater(ev.fromType(batchSize), roiBatchInd))

      val roiHeight = max(roi_end_h - roi_start_h + 1, 1)
      val roiWidth = max(roi_end_w - roi_start_w + 1, 1)
      val binSizeH = ev.divide(ev.fromType[Double](roiHeight), ev.fromType[Int](pooled_h))
      val binSizeW = ev.divide(ev.fromType[Double](roiWidth), ev.fromType[Int](pooled_w))
      var batchDataIndex = offset(ev.toType[Int](roiBatchInd), sizes = data.size())

      for (c <- 0 until channels) {
        for (ph <- 0 until pooled_h) {
          for (pw <- 0 until pooled_w) {
            // Compute pooling region for this output unit:
            //  start (included) = floor(ph * roi_height / pooled_height_)
            //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
            var hstart = floor(ph * ev.toType[Double](binSizeH)).toInt
            var wstart = floor(pw * ev.toType[Double](binSizeW)).toInt
            var hend = ceil((ph + 1) * ev.toType[Double](binSizeH)).toInt
            var wend = ceil((pw + 1) * ev.toType[Double](binSizeW)).toInt

            hstart = min(max(hstart + roi_start_h, 0), height)
            hend = min(max(hend + roi_start_h, 0), height)
            wstart = min(max(wstart + roi_start_w, 0), width)
            wend = min(max(wend + roi_start_w, 0), width)

            val is_empty = (hend <= hstart) || (wend <= wstart)

            val pool_index = ph * pooled_w + pw
            if (is_empty) {
              outputData(topDataIndex + pool_index) = ev.fromType(0)
              argmax_data(argmaxIndex + pool_index) = ev.fromType(-1)
            }

            for (h <- hstart until hend) {
              for (w <- wstart until wend) {
                val index = h * width + w
                if (ev.isGreater(bottom_data(batchDataIndex + index),
                  outputData(topDataIndex + pool_index))) {
                  outputData(topDataIndex + pool_index) = bottom_data(batchDataIndex + index)
                  argmax_data(argmaxIndex + pool_index) = ev.fromType(index)
//                  println("pool here",bottom_data(batchDataIndex + index), ev.fromType(index))
                }
              }
            }
            outputData(topDataIndex + pool_index)
          }
        }
        // Increment all data pointers by one channel
        batchDataIndex += offset(0, 1, sizes = data.size())
        topDataIndex += offset(0, 1, sizes = output.size())
        argmaxIndex += offset(0, 1, sizes = argmax.size())
      }
      bottomRoisIndex += offset(1, sizes = rois.size())
    }
    output
//    loadFeatures("pool5-300_512_7_7.txt")
  }

  def loadFeatures(s: String): Tensor[T] = {
    val middleRoot = "/home/xianyan/code/intel/pvanet/spark-dl/middle/"
    val size = s.substring(s.lastIndexOf("-") + 1, s.lastIndexOf(".")).split("_").map(x => x.toInt)
    Tensor(Storage(Source.fromFile(middleRoot + s).getLines()
      .map(x => ev.fromType(x.toFloat)).toArray)).reshape(size)
  }

  def offset(n: Int, c: Int = 0, h: Int = 0, w: Int = 0, sizes: Array[Int]): Int = {
    assert(sizes.length == 2 || sizes.length >= 4)
    if (sizes.length == 2) ((n * sizes(1) + c) + h) + w
    else ((n * sizes(1) + c) * sizes(2) + h) * sizes(3) + w
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val data = input(1).asInstanceOf[Tensor[T]]
    val roisData = input(2).asInstanceOf[Tensor[T]].storage().array()
    val argmaxData = argmax.storage().array()
    val numRois = output.size()(0)
    if (gradInput1 == null) {
      gradInput1 = Tensor[T]()
      gradInput.insert(gradInput1)
    }
    var gradInputData = gradInput1.resizeAs(data).fill(ev.fromType(0)).storage().array()
    val gradOutputData = gradOutput.storage().array()
    // Accumulate gradient over all ROIs
    for (roiN <- 0 until numRois) {
      val roiBatchInd = roisData(roiN * 5)
      // Accumulate gradients over each bin in this ROI
      for (c <- 0 until channels) {
        for (ph <- 0 until pooled_h) {
          for (pw <- 0 until pooled_w) {
            val outputOffset = ((roiN * channels + c) * pooled_h + ph) * pooled_w + pw
            val argmaxIndex = argmaxData(outputOffset)
            if (ev.toType[Double](argmaxIndex) >= 0) {
              val inputOffset = (ev.toType[Int](roiBatchInd) * channels + c) * height * width
              +ev.toType[Int](argmaxIndex)
              gradInputData(inputOffset) =
                ev.plus(gradInputData(inputOffset), gradOutputData(outputOffset))
            }
          }
        }
      }
    }
    gradInput
  }
}
